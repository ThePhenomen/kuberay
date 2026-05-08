import os
import uuid
import asyncio
from typing import List, Dict, Any
import math
import time
import torch
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from ray import serve
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from openai import AsyncOpenAI
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery

MODEL_NAME = os.getenv("READER_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "BAAI/bge-reranker-v2-m3")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "WikiDocs")
WEAVIATE_GRPC_ADDR = os.getenv("WEAVIATE_GRPC_ADDR", "weaviate-grpc.nova-weaviate.svc")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_HTTP_ADDR = os.getenv("WEAVIATE_HTTP_ADDR", "weaviate.nova-weaviate.svc")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "80"))
WEAVIATE_API_TOKEN = os.getenv("WEAVIATE_API_TOKEN")

RAG_EXTERNAL_LLM_ENDPOINT = os.getenv("RAG_EXTERNAL_LLM_ENDPOINT")
RAG_EXTERNAL_LLM_API_KEY = os.getenv("RAG_EXTERNAL_LLM_API_KEY", "EMPTY")

# ВАЖНО: отключим reasoning по умолчанию, если он явно не нужен, 
# чтобы не получать "analysis... The user writes..." в поток.
RAG_EXTERNAL_LLM_HAS_REASONING = os.getenv("RAG_EXTERNAL_LLM_HAS_REASONING", "false").lower() in ("true", "1", "yes")
RAG_EXTERNAL_LLM_MODEL = os.getenv("RAG_EXTERNAL_LLM_MODEL", "nvidia/gpt-oss-puzzle-88B")
RAG_EXTERNAL_LLM_REASONING_EFFORT = os.getenv("RAG_EXTERNAL_LLM_REASONING_EFFORT", "low")

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
    stream: bool = False

class InputQuestion(BaseModel):
    query: List[Dict[str, Any]]
    stream: bool = False

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local-vllm"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]

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
        pairs = []
        for doc in docs:
            title = doc.get("title", "")
            content = doc.get("page_content", "")
            snippet = f"{title}\n{content}"[:1500]
            pairs.append([query, snippet])

        device = next(self.model.parameters()).device
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
            rerank_scores = self.model(**inputs, return_dict=True).logits.view(-1,).float().tolist()

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
            doc["combined_score"] = combined
            scored_docs.append(doc)

        scored_docs.sort(key=lambda d: d["combined_score"], reverse=True)
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
            max_model_len=8192,
            max_num_batched_tokens=8192,
            trust_remote_code=True,
            enable_chunked_prefill=False,
            enable_prefix_caching=True,
            quantization="fp8",
            kv_cache_dtype="fp8",
            enforce_eager=False,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.internal_promt_template = self.tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True,
        )
        self.rag_answer_messages_template = prompt_in_chat_format_for_rag

        if not RAG_EXTERNAL_LLM_ENDPOINT:
            raise RuntimeError("RAG_EXTERNAL_LLM_ENDPOINT is not set")

        self.external_llm_client = AsyncOpenAI(
            base_url=RAG_EXTERNAL_LLM_ENDPOINT.rstrip("/"),
            api_key=RAG_EXTERNAL_LLM_API_KEY,
        )

        self.weaviate_connection = weaviate.connect_to_custom(
            http_host=WEAVIATE_HTTP_ADDR,
            http_port=WEAVIATE_HTTP_PORT,
            http_secure=False,
            grpc_host=WEAVIATE_GRPC_ADDR,
            grpc_port=WEAVIATE_GRPC_PORT,
            grpc_secure=False,
            auth_credentials=Auth.api_key(WEAVIATE_API_TOKEN),
        )
        if not self.weaviate_connection.is_ready():
            self.weaviate_connection.close()
            raise RuntimeError("Weaviate is not ready")

        self.nova_collection = self.weaviate_connection.collections.use(f"{COLLECTION_NAME}Nova")
        self.zvirt_collection = self.weaviate_connection.collections.use(f"{COLLECTION_NAME}Zvirt")
        self.knowledgebase_collection = self.weaviate_connection.collections.use(f"{COLLECTION_NAME}Knowledgebase")
        self.solutions_collection = self.weaviate_connection.collections.use(f"{COLLECTION_NAME}Solutions")

    async def _generate_text(self, prompt: str, sampling_params: SamplingParams) -> str:
        request_id = str(uuid.uuid4())
        generator = self.engine.generate(prompt, sampling_params, request_id)
        final_output = None
        async for request_output in generator:
            final_output = request_output
        return final_output.outputs[0].text

    async def _generate_answer_external_stream(self, messages: List[Dict[str, str]], max_tokens: int = 4096):
        request_kwargs = {
            "model": RAG_EXTERNAL_LLM_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if RAG_EXTERNAL_LLM_HAS_REASONING:
            request_kwargs["extra_body"] = {"reasoning_effort": RAG_EXTERNAL_LLM_REASONING_EFFORT}

        response = await self.external_llm_client.chat.completions.create(**request_kwargs)
        async for chunk in response:
            # Отфильтровываем возможный reasoning. 
            # У многих моделей reasoning лежит в `chunk.choices[0].delta.reasoning_content`
            # Но мы отдаем дальше только обычный content!
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # Забираем только финальный контент
            if hasattr(delta, 'content') and delta.content:
                yield delta.content

    async def _generate_answer_external(self, messages: List[Dict[str, str]], max_tokens: int = 4096, stream: bool = False):
        request_kwargs = {
            "model": RAG_EXTERNAL_LLM_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        if RAG_EXTERNAL_LLM_HAS_REASONING:
            request_kwargs["extra_body"] = {"reasoning_effort": RAG_EXTERNAL_LLM_REASONING_EFFORT}

        response = await self.external_llm_client.chat.completions.create(**request_kwargs)

        if not stream:
            content = response.choices[0].message.content
            if isinstance(content, str): return content.strip()
            if isinstance(content, list):
                parts = [part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "") for part in content]
                return "".join(parts).strip()
            return str(content).strip()
        else:
            async def chunk_generator():
                async for chunk in response:
                    if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return chunk_generator()

    async def make_prediction(self, req: InputQuestion) -> OutputAnswer:
        final_prompt = self.internal_promt_template.format(question=req.query)
        sampling_params = SamplingParams(temperature=0.3, top_p=0.95, repetition_penalty=1.1, max_tokens=200)
        answer_text = await self._generate_text(final_prompt, sampling_params)
        return OutputAnswer(answer=answer_text)

    async def make_context_prediction(self, req: InputRagQuestion):
        last_user_msg = next((m["content"] for m in reversed(req.query) if m.get("role") == "user"), "")
        word_count = len(last_user_msg.split())

        if len(req.query) <= 1 or word_count > 15:
            search_query = last_user_msg
        else:
            search_query = last_user_msg

        match req.product_name:
            case "zvirt": collection = self.zvirt_collection
            case _: collection = self.nova_collection

        response = collection.query.hybrid(
            query=search_query,
            limit=8,
            filters=Filter.by_property("version").equal(req.product_version),
            return_metadata=MetadataQuery(score=True),
        )

        docs = []
        for obj in response.objects or []:
            docs.append({
                "title": obj.properties.get("title", ""),
                "page_content": obj.properties.get("page_content", ""),
                "page_url": obj.properties.get("page_url", ""),
                "source": obj.properties.get("source", ""),
                "hybrid_score": getattr(obj.metadata, "score", 0.0),
            })

        if not docs:
            return OutputAnswer(answer="No relevant docs found") if not req.stream else self._generate_answer_external_stream([], max_tokens=10)

        reranked_docs = await self.reranker.rerank.remote(search_query, docs, top_k=5, alpha=0.7)

        texts_with_links = []
        for doc in reranked_docs:
            texts_with_links.append(f"{doc['page_content']}\n\nИсточник: {doc['page_url']}")

        context = "\n\n---\n\n".join(texts_with_links)

        rag_messages = [
            {"role": item["role"], "content": item["content"].format(question=req.query, context=context)}
            for item in self.rag_answer_messages_template
        ]

        if not req.stream:
            final_answer = await self._generate_answer_external(rag_messages, max_tokens=4096, stream=False)
            return OutputAnswer(answer=final_answer)
        else:
            return await self._generate_answer_external(rag_messages, max_tokens=4096, stream=True)

    def close(self):
        if hasattr(self, "weaviate_connection") and self.weaviate_connection:
            self.weaviate_connection.close()

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()

reranker_app = Reranker.bind()
rag_reader_app = RAGReader.bind(reranker_app)

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 1, "num_gpus": 0})
@serve.ingress(app)
class OpenAIAdapter:
    def __init__(self, rag_handle):
        self.rag = rag_handle

    @app.get("/v1/models")
    async def list_models(self):
        return {
            "object": "list",
            "data": [
                {"id": "wiki-searcher", "object": "model", "owned_by": "OrionSoft"}
            ]
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        body: Dict[str, Any] = await request.json()
        model = body.get("model", "wiki-searcher")
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        product_name = str(body.get("product_name", "nova"))
        product_version = str(body.get("product_version", "latest"))

        # Parse user_request correctly (can be string or bool from clients)
        ur_raw = body.get("user_request", False)
        user_request = str(ur_raw).lower() in ("true", "1", "yes")

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())

        if stream:
            if user_request:
                req = InputRagQuestion(query=messages, product_name=product_name, product_version=product_version, stream=True)
                resp_gen = self.rag.options(stream=True).make_context_prediction.remote(req)
            else:
                req = InputQuestion(query=messages, stream=True)
                resp_gen = self.rag.options(stream=True).make_prediction.remote(req)

            async def sse_generator():
                try:
                    init_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
                    }
                    yield f"data: {json.dumps(init_chunk, ensure_ascii=False)}\n\n"

                    async for chunk_text in resp_gen:
                        if await request.is_disconnected():
                            break
                        content = chunk_text if isinstance(chunk_text, str) else str(chunk_text)
                        if not content:
                            continue

                        chunk_data = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                    }
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                except asyncio.CancelledError:
                    return

            return StreamingResponse(
                sse_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            if user_request:
                req = InputRagQuestion(query=messages, product_name=product_name, product_version=product_version, stream=False)
                resp = await self.rag.make_context_prediction.remote(req)
            else:
                req = InputQuestion(query=messages, stream=False)
                resp = await self.rag.make_prediction.remote(req)

            return ChatCompletionResponse(
                id=request_id,
                object="chat.completion",
                created=created_time,
                model=model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatCompletionResponseChoiceMessage(role="assistant", content=resp.answer),
                        finish_reason="stop"
                    )
                ]
            )

openai_app = OpenAIAdapter.bind(rag_reader_app)
