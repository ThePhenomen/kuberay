import os
import uuid
import asyncio
from typing import List, Dict, Any, AsyncIterator
import math
import time
import json

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from ray import serve

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery

MODEL_NAME = os.getenv(
    "READER_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
RERANKER_MODEL_ID = os.getenv(
    "RERANKER_MODEL_ID", 
    "BAAI/bge-reranker-v2-m3"
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "WikiDocs")
WEAVIATE_GRPC_ADDR = os.getenv("WEAVIATE_GRPC_ADDR", "weaviate-grpc.nova-weaviate.svc")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_HTTP_ADDR = os.getenv("WEAVIATE_HTTP_ADDR", "weaviate.nova-weaviate.svc")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "80"))
WEAVIATE_API_TOKEN = os.getenv("WEAVIATE_API_TOKEN")

prompt_in_chat_format_for_rag = [
    {
        "role": "system",
        "content": """You are Wiki-Searcher, an OrionSoft assistant for searching product documentation.

Answer in Russian.

Rules:
1. Use only the information from the provided <context>.
2. Answer the user's question directly and stay focused on it. Include only information that helps answer this question. Do not add unrelated sections or optional installation details unless the user asks for them.
3. Do not add facts, steps, commands, configuration, versions, or assumptions that are not explicitly present in the context.
4. If the context does not contain enough information, reply exactly:
"Не смог найти подходящую информацию на Ваш вопрос."
5. If the retrieved sources conflict, say that the sources contain different information and briefly describe both versions with sources.
6. If the user asks about installation, configuration, upgrade, uninstallation, troubleshooting, or manifests:
   - mention only the steps or parameters explicitly present in the context;
   - do not invent omitted steps;
   - do not output full manifests unless they are short and directly necessary;
   - if the manifest is large, summarize key points and refer the user to the source.
7. For meta-questions, greetings, thanks, criticism, or general chitchat:
   - respond naturally in Russian;
   - do not use the documentation context;
   - do not add a Sources section.
8. STYLE:
   - Prefer a short natural paragraph instead of a bullet list.
   - Use bullet points only when the user asks for steps, a list, a checklist, or when the answer is clearer as a list.
   - For simple explanatory questions, answer in 2-4 connected sentences.
   - Do not split every answer into separate step-like lines unless the question is procedural.
9. TONE:
   - Write in a concise, natural, conversational style.
   - Avoid overly formal, mechanical, or template-like phrasing.
   - Do not restate the question.
10. If asked who created you, say you were created by OrionSoft to help with documentation.
11. If asked where your answers come from, say you use OrionSoft internal documentation.
12. Keep the answer short and precise, no more than 400 words.
13. For documentation answers, use this format:
   - short direct answer;
   - 2-6 bullet points if needed;
   - then:
     Источники:
     - source 1
     - source 2
     - ...
    Provide only sources which were used to generate answer.
14. Include only the sources you actually used, without duplicates.
15. If the query mentions operating systems or distributions, OrionSoft uses: Redos, Almalinux, Astra, Alt, MosOS, CentOS, Ubuntu. Do not use other OS in answers."""
    },
    {
        "role": "user",
        "content": """Context:
<context>
{context}
</context>
---
Conversation:
{question}""",
    },
]

prompt_in_chat_format = [
    {
        "role": "system",
        "content": """You are Wiki-Searcher, created by OrionSoft. 
Your task is to act as a helpful IT assistant specializing in OrionSoft products.
Provide precise, concise answers directly addressing the user's prompt.""",
    },
    {
        "role": "user",
        "content": """Question: {question}""",
    },
]

class InputRagQuestion(BaseModel):
    query: List[Dict[str, Any]]
    product_name: str
    product_version: str

class InputQuestion(BaseModel):
    query: List[Dict[str, Any]]

class OutputAnswer(BaseModel):
    answer: str

class ChatCompletionResponseChoiceMessage(BaseModel):
    role: str
    content: str

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatCompletionResponseChoiceMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    choices: List[ChatCompletionResponseChoice]


app = FastAPI()

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 0.1},
)
class Reranker:
    def __init__(self):
        print(f"Loading reranker model {RERANKER_MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda",
            local_files_only=True,
        )
        self.model.eval()

    async def rerank(self, query: str, docs: List[Dict[str, str]], top_k: int = 8, alpha: float = 0.5) -> List[Dict[str, str]]:
        if not docs:
            return []
        
        start_time = time.perf_counter()
        print("Init reranking")
        
        pairs = []
        for doc in docs:
            title = doc.get("title", "")
            content = doc.get("page_content", "")
            snippet = f"{title}\n{content}"[:1500] 
            pairs.append([query, snippet])

        device = next(self.model.parameters()).device
        print("Start reranking")
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            ).to(device)
            
            rerank_scores = self.model(**inputs, return_dict=True).logits.view(-1,).float().tolist()
            
        print(f"Raw rerank scores: {rerank_scores}")
        
        hybrid_scores = [float(doc.get("hybrid_score", 0.0)) for doc in docs]
        eps = 1e-8

        def min_max_norm(values: List[float]) -> List[float]:
            vmin = min(values)
            vmax = max(values)
            if math.isclose(vmin, vmax):
                return [0.5 for _ in values]
            return [(v - vmin) / (vmax - vmin + eps) for v in values]

        hybrid_norm = min_max_norm(hybrid_scores)
        rerank_norm = min_max_norm(rerank_scores)

        scored_docs = []
        for doc, h_raw, h_n, r_raw, r_n in zip(docs, hybrid_scores, hybrid_norm, rerank_scores, rerank_norm):
            combined = alpha * r_n + (1.0 - alpha) * h_n
            doc["hybrid_score_raw"] = h_raw
            doc["rerank_score_raw"] = r_raw
            doc["hybrid_score_norm"] = h_n
            doc["rerank_score_norm"] = r_n
            doc["combined_score"] = combined
            scored_docs.append(doc)

        scored_docs.sort(key=lambda d: d["combined_score"], reverse=True)

        print("Top docs combined:")
        for d in scored_docs[:top_k]:
            print(f"  {d.get('page_url')}, {d.get('source')}, h={d['hybrid_score_raw']}, r={d['rerank_score_raw']}, hn={d['hybrid_score_norm']}, rn={d['rerank_score_norm']}, combined={d['combined_score']}")

        end_time = time.perf_counter()
        print(f"Reranking done in {end_time - start_time:.6f} s")
        return scored_docs[:top_k]

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 10, "num_gpus": 0.9},
)
class RAGReader:
    def __init__(self, reranker_handle):
        self.reranker = reranker_handle
        
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            gpu_memory_utilization=0.90,
            max_model_len=32768,
            max_num_batched_tokens=16384,
            trust_remote_code=True,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            quantization="fp8",
            kv_cache_dtype="fp8",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.internal_rag_prompt_template = self.tokenizer.apply_chat_template(
            prompt_in_chat_format_for_rag, tokenize=False, add_generation_prompt=True
        )
        self.internal_prompt_template = self.tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )

        try:
            self.weaviate_connection = weaviate.connect_to_custom(
                http_host=WEAVIATE_HTTP_ADDR,
                http_port=WEAVIATE_HTTP_PORT,
                http_secure=False,
                grpc_host=WEAVIATE_GRPC_ADDR,
                grpc_port=WEAVIATE_GRPC_PORT,
                grpc_secure=False,
                auth_credentials=Auth.api_key(WEAVIATE_API_TOKEN),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Weaviate: {e}") from e
        
        if not self.weaviate_connection.is_ready():
            self.weaviate_connection.close()
            raise RuntimeError("Weaviate is not ready, aborting RAGReader initialization")
        
        self.nova_collection = self.weaviate_connection.collections.use(f"{COLLECTION_NAME}{'nova'.capitalize()}")
        self.zvirt_collection = self.weaviate_connection.collections.use(f"{COLLECTION_NAME}{'zvirt'.capitalize()}")
        self.knowledgebase_collection = self.weaviate_connection.collections.use(f"{COLLECTION_NAME}{'knowledgebase'.capitalize()}")
        self.solutions_collection = self.weaviate_connection.collections.use(f"{COLLECTION_NAME}{'solutions'.capitalize()}")

    async def _generate_text(self, prompt: str, sampling_params: SamplingParams) -> str:
        request_id = str(uuid.uuid4())
        generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in generator:
            final_output = request_output
            
        return final_output.outputs[0].text
    
    async def _generate_text_stream(self, prompt: str, sampling_params: SamplingParams) -> AsyncIterator[str]:
        request_id = str(uuid.uuid4())
        generator = self.engine.generate(prompt, sampling_params, request_id)

        previous_text = ""

        async for request_output in generator:
            if not request_output.outputs:
                continue

            current_text = request_output.outputs[0].text or ""
            delta = current_text[len(previous_text):]
            previous_text = current_text

            if delta:
                yield delta

    def _select_collection_and_version(self, req: InputRagQuestion):
        match req.product_name:
            case "zvirt":
                collection = self.zvirt_collection
                version = "latest"
            case "nova":
                collection = self.nova_collection
                version = req.product_name
            case _:
                collection = self.nova_collection
                version = req.product_name

        return collection, version
    
    async def _build_plain_prompt(self, req: InputQuestion) -> str:
        return self.internal_prompt_template.format(question=req.query)
    
    async def _build_rag_prompt(self, req: InputRagQuestion) -> str:
        print("Got wiki query:", req.query)

        last_user_msg = next(
            (m["content"] for m in reversed(req.query) if m.get("role") == "user"),
            ""
        )
        word_count = len(last_user_msg.split())

        prompt_start_time = time.perf_counter()

        if len(req.query) < 1 or word_count > 15:
            search_query = last_user_msg
        else:
            history_msgs = req.query[-4:-1]
            history_text = "\n".join(
                f"{m.get('role')}: {m.get('content')}" for m in history_msgs
            )

            rewrite_prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a search query transformation system. "
                        "Your task is to formulate a text for documentation search.\n"
                        "Rules:\n"
                        "1. Analyze the user's latest question. If it contains pronouns or logically continues previous topic, rewrite it using conversation history.\n"
                        "2. If latest question starts a completely new topic, return it as is.\n"
                        "3. Output only the final query text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_text}\n\n"
                        f"Latest question:\n{last_user_msg}\n\n"
                        "Final search query:"
                    ),
                },
            ]

            rewrite_prompt = self.tokenizer.apply_chat_template(
                rewrite_prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            rewrite_sampling = SamplingParams(
                temperature=0.0,
                max_tokens=50,
            )
            rewritten_result = await self._generate_text(rewrite_prompt, rewrite_sampling)
            search_query = rewritten_result.strip() or last_user_msg

        print(f"Original query: {last_user_msg}")
        print(f"Rewritten query: {search_query}")

        prompt_end_time = time.perf_counter()
        print(f"Prompt prep done in {prompt_end_time - prompt_start_time:.6f}s")

        collection, version = self._select_collection_and_version(req)

        async def fetch_docs_parallel(query_text: str) -> List[Dict[str, Any]]:
            res_main, res_knowledgebase, res_solutions = await asyncio.gather(
                asyncio.to_thread(
                    collection.query.hybrid,
                    query=query_text,
                    alpha=0.3,
                    limit=15,
                    filters=Filter.by_property("version").equal(req.product_version),
                    return_metadata=MetadataQuery(score=True),
                ),
                asyncio.to_thread(
                    self.knowledgebase_collection.query.hybrid,
                    query=query_text,
                    alpha=0.3,
                    limit=7,
                    filters=Filter.by_property("version").equal(version),
                    return_metadata=MetadataQuery(score=True),
                ),
                asyncio.to_thread(
                    self.solutions_collection.query.hybrid,
                    query=query_text,
                    alpha=0.3,
                    limit=7,
                    filters=Filter.by_property("version").equal(version),
                    return_metadata=MetadataQuery(score=True),
                ),
            )

            raw_objects: List[Dict[str, Any]] = []
            for res in (res_main, res_knowledgebase, res_solutions):
                for obj in (res.objects or []):
                    raw_objects.append(
                        {
                            "title": obj.properties.get("title", ""),
                            "page_content": obj.properties.get("page_content", ""),
                            "page_url": obj.properties.get("page_url", ""),
                            "source": obj.properties.get("source", ""),
                            "hybrid_score": obj.metadata.score or 0.0,
                        }
                    )
            return raw_objects

        docs_start_time = time.perf_counter()

        hyde_start_time = time.perf_counter()
        hyde_prompt = (
            "<|im_start|>system\n"
            "You are an expert IT assistant. "
            "Answer the user's query with one short technical paragraph. "
            "Use only essential information, no introductions, no conclusions, no greetings, "
            "no lists, no code blocks, no Markdown formatting. "
            "If the query mentions operating systems or distributions, mention only OS family names "
            "without version numbers. Maximum 80 words. Answer in Russian.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\nQuery: {search_query}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        hyde_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=120,
        )
        hyde_document = await self._generate_text(hyde_prompt, hyde_params)
        hyde_end_time = time.perf_counter()

        print(f"HyDE generated in {hyde_end_time - hyde_start_time:.6f}s")
        print(f"HyDE generated document: {hyde_document}")

        original_docs, hyde_docs = await asyncio.gather(
            fetch_docs_parallel(search_query),
            fetch_docs_parallel(hyde_document if hyde_document.strip() else search_query),
        )

        raw_docs = original_docs + hyde_docs

        print(f"Retrieved {len(raw_docs)} documents from Weaviate")
        docs_end_time = time.perf_counter()
        print(f"Docs fetched in {docs_end_time - docs_start_time:.6f}s")

        if not raw_docs:
            return self.internal_rag_prompt_template.format(
                context="No relevant docs found",
                question=req.query,
            )

        reranked_docs = await self.reranker.rerank.remote(
            search_query,
            raw_docs,
            topk=5,
            alpha=0.6,
        )
        print("Finished reranking documents")

        context = "\n\n".join(
            f"{doc.get('page_content', '')}\nSource: {doc.get('page_url', '')}"
            for doc in reranked_docs
        )

        final_prompt = self.internal_rag_prompt_template.format(
            context=context,
            question=req.query,
        )
        return final_prompt
    
    def _final_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=0.3,
            top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=600,
        )

    async def make_prediction(self, req: InputQuestion) -> OutputAnswer:
        final_prompt = await self._build_plain_prompt(req)
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=200,
        )
        answer_text = await self._generate_text(final_prompt, sampling_params)
        return OutputAnswer(answer=answer_text)

    async def make_prediction_stream(self, req: InputQuestion) -> AsyncIterator[str]:
        final_prompt = await self._build_plain_prompt(req)
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.95,
            repetition_penalty=1.1,
            max_tokens=200,
        )

        async for delta in self._generate_text_stream(final_prompt, sampling_params):
            yield delta
    
    async def make_context_prediction(self, req: InputRagQuestion) -> OutputAnswer:
        print("Generating answer")
        answer_start_time = time.perf_counter()

        final_prompt = await self._build_rag_prompt(req)
        final_sampling = self._final_sampling_params()
        final_answer = await self._generate_text(final_prompt, final_sampling)

        answer_end_time = time.perf_counter()
        print(f"Answer generated in {answer_end_time - answer_start_time:.6f}s")

        return OutputAnswer(answer=final_answer)

    async def make_context_prediction_stream(self, req: InputRagQuestion) -> AsyncIterator[str]:
        print("Generating streaming answer")
        answer_start_time = time.perf_counter()

        final_prompt = await self._build_rag_prompt(req)
        final_sampling = self._final_sampling_params()

        async for delta in self._generate_text_stream(final_prompt, final_sampling):
            yield delta

        answer_end_time = time.perf_counter()
        print(f"Streaming answer completed in {answer_end_time - answer_start_time:.6f}s")

    def close(self):
        if hasattr(self, 'weaviate_connection') and self.weaviate_connection is not None:
            self.weaviate_connection.close()
            self.weaviate_connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

reranker_app = Reranker.bind()
rag_reader_app = RAGReader.bind(reranker_app)

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)
@serve.ingress(app)
class OpenAIAdapter:
    def __init__(self, rag_handle):
        self.rag = rag_handle

    @app.get("/v1/models")
    async def list_models(self):
        return {
            "object": "list",
            "data": [
                {
                    "id": "wiki-searcher",
                    "object": "model",
                    "owned_by": "OrionSoft",
                }
            ],
        }

    def _chunk(self, request_id: str, model: str, delta: Dict[str, Any], finish_reason=None):
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return f" {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def _stream_response(
        self,
        model: str,
        messages,
        product_name: str,
        product_version: str,
        user_request: bool,
    ):
        request_id = f"chatcmpl-{uuid.uuid4()}"

        yield self._chunk(request_id, model, {"role": "assistant"})

        try:
            if user_request:
                stream = self.rag.options(stream=True).make_context_prediction_stream.remote(
                    InputRagQuestion(
                        query=messages,
                        product_name=product_name,
                        product_version=product_version,
                    )
                )
            else:
                stream = self.rag.options(stream=True).make_prediction_stream.remote(
                    InputQuestion(query=messages)
                )

            async for piece in stream:
                if not piece:
                    continue

                if isinstance(piece, dict):
                    text = piece.get("content") or piece.get("delta") or ""
                else:
                    text = str(piece)

                if text:
                    yield self._chunk(request_id, model, {"content": text})

            yield self._chunk(request_id, model, {}, finish_reason="stop")
            yield " [DONE]\n\n"

        except Exception as e:
            error_text = f"\n[stream-error] {e}"
            yield self._chunk(request_id, model, {"content": error_text})
            yield self._chunk(request_id, model, {}, finish_reason="stop")
            yield " [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        body: Dict[str, Any] = await request.json()

        model = body.get("model", "wiki-searcher")
        messages = body.get("messages", [])
        product_name = str(body.get("product_name", "nova"))
        product_version = str(body.get("product_version", "latest"))
        user_request = bool(body.get("user_request", False))
        stream = bool(body.get("stream", True))

        print(
            f"[OpenAIAdapter] model={model}, "
            f"product_name={product_name}, "
            f"product_version={product_version}, "
            f"user_request={user_request}, "
            f"stream={stream}"
        )

        if user_request:
            if stream:
                return StreamingResponse(
                    self._stream_response(
                        model=model,
                        messages=messages,
                        product_name=product_name,
                        product_version=product_version,
                        user_request=user_request,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                resp: OutputAnswer = await self.rag.make_context_prediction.remote(
                    InputRagQuestion(
                        query=messages,
                        product_name=product_name,
                        product_version=product_version,
                    )
                )
        else:
            resp: OutputAnswer = await self.rag.make_prediction.remote(
                InputQuestion(query=messages)
            )

        return JSONResponse(
            content={
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": resp.answer,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )


openai_app = OpenAIAdapter.bind(rag_reader_app)
