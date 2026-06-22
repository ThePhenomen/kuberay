import os
import uuid
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Optional
import math
import time
import torch
import httpx
import re

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
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

import mlflow

import logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

def init_logger():
    """Get the root logger"""
    return logging.getLogger("rag_service")

logger = init_logger()

PRODUCTS = [
    "nova",
    "zvirt",
    "zvirt-containers",
    "zvirt-metrics",
    "zvirt-dc-manager",
    "terraform",
    "termit",
    "cloudlink",
    "nova-se",
    "starvault",
    "knowledgebase", 
    "solutions",
]

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
RAG_EXTERNAL_LLM_HAS_REASONING = os.getenv("RAG_EXTERNAL_LLM_HAS_REASONING", "true").lower() in ("1", "true", "yes")
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
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]

class SearchRequest(BaseModel):
    query: str
    product_name: str = "nova"
    product_version: str = "latest"
    top_k: int = 5

class SearchResultDocument(BaseModel):
    url: str
    score: float
    content: str
    product_name: str
    product_version: str

class SearchResponse(BaseModel):
    results: List[SearchResultDocument]

app = FastAPI()

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 0.1},
)
class Reranker:
    def __init__(self):
        self.logger = init_logger()
        self.logger.info(f"Loading reranker model {RERANKER_MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda",
            local_files_only=True,
        )
        self.model.eval()

    async def rerank(
        self,
        query: str,
        request_id: str,
        docs: List[Dict[str, str]],
        top_k: int = 8,
        alpha: float = 0.5,

    ) -> List[Dict[str, str]]:
        if not docs:
            self.logger.info(f"[req: {request_id}] Rerank skipped: no documents")
            return []

        start_time = time.perf_counter()
        self.logger.info(
            f"[req: {request_id}] Rerank started, docs_count=%d, top_k=%d, alpha=%.2f",
            len(docs),
            top_k,
            alpha,
        )

        pairs = []
        for doc in docs:
            title = doc.get("title", "")
            content = doc.get("page_content", "")
            snippet = f"{title}\n{content}"[:1500]
            pairs.append([query, snippet])
        device = next(self.model.parameters()).device

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            ).to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                rerank_scores = self.model(**inputs).logits.view(-1).float().tolist()

        self.logger.debug(f"[req: {request_id}] Raw rerank scores computed, calucalted scores: {rerank_scores}")

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

        self.logger.debug(
            f"[req: {request_id}] Top reranked docs (count=%d, top_k=%d)",
            len(scored_docs),
            top_k,
        )

        elapsed = time.perf_counter() - start_time
        self.logger.info(
            f"[req: {request_id}] Rerank finished (docs=%d, returned=%d, elapsed=%.6f sec)",
            len(docs),
            min(len(scored_docs), top_k),
            round(elapsed, 6),
        )

        return scored_docs[:top_k]
    
@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 2, "num_gpus": 0},
)
class Searcher:
    def __init__(self, reranker_handle):
        self.reranker = reranker_handle
        self.logger = init_logger()
        self.logger.info("Initializing Searcher & Weaviate connection")
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
            raise RuntimeError("Weaviate is not ready, aborting Searcher initialization")
        
        self.product_collections = {}
        for product in PRODUCTS:
            product_alias = f"{COLLECTION_NAME}{product.replace('-', '_').capitalize()}"
            self.product_collections[product] = self.weaviate_connection.collections.use(product_alias)

    async def _fetch_docs_parallel(self, query_text: str, product_name: str, product_version: str, request_id: str) -> List[Dict[str, Any]]:
        collection = self.product_collections.get(product_name) or self.product_collections["nova"]
        version = "latest" if product_name == "zvirt" else product_name

        self.logger.info(f"[req: {request_id}] Execute remote document search in db {product_name}")
        res_main, res_knowledge_base, res_solutions = await asyncio.gather(
            asyncio.to_thread(
                collection.query.hybrid,
                query=query_text,
                alpha=0.3,
                limit=15,
                filters=Filter.by_property("version").equal(product_version),
                return_metadata=MetadataQuery(score=True),
            ),
            asyncio.to_thread(
                self.product_collections["knowledgebase"].query.hybrid,
                query=query_text,
                alpha=0.3,
                limit=7,
                filters=Filter.by_property("version").equal(version),
                return_metadata=MetadataQuery(score=True),
            ),
            asyncio.to_thread(
                self.product_collections["solutions"].query.hybrid,
                query=query_text,
                alpha=0.3,
                limit=7,
                filters=Filter.by_property("version").equal(version),
                return_metadata=MetadataQuery(score=True),
            ),
        )
        
        raw_objects = []
        self.logger.info(
            f"Found following docs: {product_name} - {len(res_main.objects)}, "
            f"knowledgebase - {len(res_knowledge_base.objects)}, "
            f"solutions - {len(res_solutions.objects)}"
        )
        for res in (res_main, res_knowledge_base, res_solutions):
            for obj in res.objects:
                raw_objects.append({
                    "title": obj.properties.get("title", ""),
                    "page_content": obj.properties.get("page_content", ""),
                    "page_url": obj.properties.get("page_url", ""),
                    "source": obj.properties.get("source", ""),
                    "hybrid_score": obj.metadata.score or 0.0,
                })
        return raw_objects
    
    async def search(
        self, 
        queries: List[str], 
        product_name: str, 
        product_version: str, 
        request_id: str,
        top_k: int = 5,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        docs_start_time = time.perf_counter()
        
        tasks = [self._fetch_docs_parallel(q, product_name, product_version, request_id) for q in queries]
        all_docs_lists = await asyncio.gather(*tasks)

        seen_urls = set()
        raw_docs = []
        for docs in all_docs_lists:
            for doc in docs:
                url = doc.get("source")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    raw_docs.append(doc)

        docs_end_time = time.perf_counter()
        self.logger.info(f"[req: {request_id}] Retrieved {len(raw_docs)} unique documents in {docs_end_time - docs_start_time:.6f}s")

        if not raw_docs:
            return []

        main_query = queries[0]
        reranked_docs = await self.reranker.rerank.remote(main_query, request_id, raw_docs, top_k=top_k, alpha=alpha)
        return reranked_docs

    def close(self):
        if hasattr(self, "weaviate_connection") and self.weaviate_connection is not None:
            self.weaviate_connection.close()
            self.weaviate_connection = None

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 10, "num_gpus": 0.9},
)
class RAGSystem:
    def __init__(self, searcher_handle):
        self.searcher = searcher_handle
        self.logger = init_logger()
        self.logger.info("Initializing RAGSystem")
        
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
        self.external_llm_raw_base = RAG_EXTERNAL_LLM_ENDPOINT.rstrip("/")

    async def _generate_text(self, prompt: str, sampling_params: SamplingParams) -> str:
        request_id = str(uuid.uuid4())
        generator = self.engine.generate(prompt, sampling_params, request_id)
        final_output = None
        async for request_output in generator:
            final_output = request_output
        return final_output.outputs[0].text if final_output else ""

    async def _generate_answer_external_stream_raw(self, messages: List[Dict[str, str]], max_tokens: int = 4096):
        url = f"{self.external_llm_raw_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {RAG_EXTERNAL_LLM_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        payload: Dict[str, Any] = {
            "model": RAG_EXTERNAL_LLM_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if RAG_EXTERNAL_LLM_HAS_REASONING:
            payload["reasoning_effort"] = RAG_EXTERNAL_LLM_REASONING_EFFORT

        timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=None)

        async def generator() -> AsyncGenerator[bytes, None]:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        error_text = (await response.aread()).decode("utf-8", errors="replace")
                        yield f'data: {{"error": {{"message": "LLM error {response.status_code}: {error_text}"}}}}\n\n'.encode("utf-8")
                        yield b"data: [DONE]\n\n"
                        return
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        yield (line + "\n\n").encode("utf-8")
        return generator()
    
    async def _generate_answer_external(self, messages: List[Dict[str, str]], max_tokens: int = 4096, stream: bool = False):
        if stream:
            return await self._generate_answer_external_stream_raw(messages=messages, max_tokens=max_tokens)
            
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
        content = response.choices[0].message.content
        if isinstance(content, list):
            return "".join([p.get("text", "") if isinstance(p, dict) else getattr(p, "text", "") for p in content]).strip()
        return str(content).strip()

    async def make_prediction(self, req: InputQuestion, request_id: str) -> OutputAnswer:
        final_prompt = self.internal_promt_template.format(question=req.query)
        sampling_params = SamplingParams(temperature=0.3, top_p=0.95, repetition_penalty=1.1, max_tokens=200)
        answer_text = await self._generate_text(final_prompt, sampling_params)
        self.logger.info(f"[req: {request_id}] Generating remote answer...")
        return OutputAnswer(answer=answer_text)
    
    async def _compress_history(self, messages: List[Dict[str, Any]]) -> str:

        last_user_msg = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )

        history = messages[-7:-1]
        if not history:
            return last_user_msg

        last_words = last_user_msg.split()
        if len(last_words) > 20:
            return last_user_msg

        history_lines = []
        for m in history:
            role = m.get("role", "")
            content = m.get("content", "")
            if not content:
                continue

            if role == "user":
                snippet = content[:400] if len(content) > 400 else content
                history_lines.append(f"User: {snippet}")

            elif role == "assistant":
                clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                snippet = clean[:250] if len(clean) > 400 else clean
                if snippet:
                    history_lines.append(f"Assistant: {snippet}")

        if not history_lines:
            return last_user_msg
        
        self.logger.debug(f"History: {history_lines}")

        history_text = "\n".join(history_lines)

        rewrite_prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are a search query transformation system for a technical documentation assistant.\n"
                    "You will receive a conversation (user questions and assistant answers), "
                    "followed by the user's latest question.\n\n"
                    "Rules:\n"
                    "1. TOPIC CONTINUATION — if the latest question uses pronouns (it, this, they, there), "
                    "   refers to something mentioned before, or is clearly a follow-up: "
                    "   rewrite it into a precise standalone search query, "
                    "   incorporating relevant technical terms from BOTH user questions AND assistant answers.\n"
                    "2. NEW TOPIC — if the latest question starts a completely unrelated topic: "
                    "   return it exactly as-is, do not modify.\n"
                    "3. QUERY QUALITY — the output must be a good documentation search query: "
                    "   specific, technical, free of conversational filler. "
                    "   Preserve key product names, component names, and action verbs. "
                    "   Do not over-compress — it is fine to use 20-30 words if needed for clarity.\n"
                    "4. OUTPUT FORMAT — output ONLY the final search query. "
                    "   No quotes, no explanations, no prefixes like 'Query:'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_text}\n\n"
                    f"Latest question: {last_user_msg}\n\n"
                    f"Final search query:"
                ),
            },
        ]

        rewrite_prompt = self.tokenizer.apply_chat_template(
            rewrite_prompt_messages, tokenize=False, add_generation_prompt=True,
        )
        rewritten = await self._generate_text(
            rewrite_prompt, SamplingParams(temperature=0.0, max_tokens=50)
        )
        result = rewritten.strip()
        self.logger.debug(f"Query rewrite: [{last_user_msg}] → [{result}]")
        return result or last_user_msg
    
    def _strip_thinking(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _build_rag_messages_hybrid(
        self,
        history: List[Dict[str, Any]],
        context: str,
    ) -> List[Dict[str, str]]:
        system_prompt = self.rag_answer_messages_template[0]["content"]

        augmented_system = (
            f"{system_prompt}\n\n"
            f"## Relevant documentation\n"
            f"<context>\n{context}\n</context>"
        )

        messages = [{"role": "system", "content": augmented_system}]

        past_messages = history[-7:-1]

        deduped = []
        prev_key = None
        for m in past_messages:
            key = (m.get("role"), (m.get("content") or "").strip())
            if key != prev_key:
                deduped.append(m)
                prev_key = key

        for m in deduped:
            role = m.get("role", "")
            content = (m.get("content") or "").strip()
            if not content or role not in ("user", "assistant"):
                continue
            if role == "assistant":
                content = self._strip_thinking(content)
            if content:
                messages.append({"role": role, "content": content})

        last_user = next(
            (m["content"] for m in reversed(history) if m.get("role") == "user"), ""
        )
        messages.append({"role": "user", "content": last_user})

        self.logger.debug(f"Built RAG messages: {len(messages)} total (1 system + {len(messages)-2} history + 1 user)")
        return messages

    async def make_context_prediction(self, req: InputRagQuestion, request_id: str):
        last_user_msg = next((m["content"] for m in reversed(req.query) if m.get("role") == "user"), "")

        self.logger.debug(f"[req: {request_id}] Input request: {req.query}")
        self.logger.debug(f"[req: {request_id}] Last user message: {last_user_msg}")
        
        start_time = time.perf_counter()

        search_query = await self._compress_history(req.query)
        hyde_prompt = (
            f"<|im_start|>system\n"
            f"You are an expert IT assistant. "
            f"Answer the user's query with one short technical paragraph. "
            f"Use only essential information, no introductions, no conclusions, no greetings, "
            f"no lists, no code blocks, no Markdown formatting. "
            f"If the query mentions operating systems or distributions, "
            f"mention only OS family names without version numbers "
            f"(for example: Redos, Almalinux, Astra, Alt, MosOS, CentOS, Ubuntu). "
            f"Never write specific version numbers or minor releases. "
            f"Maximum 60 words. Answer in Russian.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\nQuery: {search_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        h_doc = await self._generate_text(hyde_prompt, SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=80))

        queries_to_search = [search_query, h_doc]
        reranked_docs = await self.searcher.search.remote(
            queries=queries_to_search,
            product_name=req.product_name,
            product_version=req.product_version,
            top_k=5,
            request_id=request_id,
        )

        if not reranked_docs:
            if req.stream:
                async def empty_stream():
                    yield b'data: {"choices":[{"delta":{"content":"Failed to find relevant docs."}}]}\n\n'
                    yield b"data: [DONE]\n\n"
                return empty_stream()
            return OutputAnswer(answer="Не смог найти подходящую информацию на Ваш вопрос.")

        texts_with_links = [f"{doc['page_content']}\n\nИсточник: {doc['page_url']}" for doc in reranked_docs]
        context = "\n\n---\n\n".join(texts_with_links)

        rag_messages = self._build_rag_messages_hybrid(req.query, context)
        self.logger.info(f"[req: {request_id}] Raw messages: {rag_messages}")

        end_time = time.perf_counter()
        self.logger.info(f"[req: {request_id}] Init actions for request {request_id} done in {end_time - start_time:.6f}s")

        self.logger.info(f"[req: {request_id}] Generating remote answer...")
        if not req.stream:
            final_answer = await self._generate_answer_external(rag_messages, max_tokens=4096, stream=False)
            return OutputAnswer(answer=final_answer)

        return await self._generate_answer_external(rag_messages, max_tokens=4096, stream=True)
    
@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1, "num_gpus": 0},
)
@serve.ingress(app)
class SmartRouter:
    def __init__(self, rag_handle, searcher_handle):
        self.rag = rag_handle
        self.searcher = searcher_handle

    @app.get("/v1/models")
    async def list_models(self):
        return {
            "object": "list",
            "data": [{"id": "wiki-searcher", "object": "model", "owned_by": "OrionSoft"}],
        }
    
    @app.post("/v1/chat/completions")
    @mlflow.trace
    async def chat_completions(self, request: Request):
        body: Dict[str, Any] = await request.json()
        model = body.get("model", "wiki-searcher")
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        product_name = str(body.get("product_name", "nova"))
        product_version = str(body.get("product_version", "latest"))
        user_request = body.get("user_request", False)

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_time = int(time.time())

        print(f"[{request_id}] Got query: {messages}")

        mlflow.update_current_trace(
            request_id=request_id,
            model=model,
            product_name=product_name,
            product_version=product_version,
            user_request=user_request,
            tags={
                "environment": "production"
            },
        )

        if stream:
            if user_request:
                req = InputRagQuestion(
                    query=messages, product_name=product_name, product_version=product_version, stream=True,
                )
                resp_gen = self.rag.options(stream=True).make_context_prediction.remote(req, request_id)

                async def passthrough_sse():
                    try:
                        async for chunk in resp_gen:
                            if await request.is_disconnected(): break
                            yield chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8")
                    except asyncio.CancelledError: return

                return StreamingResponse(
                    passthrough_sse(), media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                )
            
        if user_request:
            req = InputRagQuestion(query=messages, product_name=product_name, product_version=product_version, stream=False)
            resp = await self.rag.make_context_prediction.remote(req, request_id)
        else:
            req = InputQuestion(query=messages, stream=False)
            resp = await self.rag.make_prediction.remote(req, request_id)

        return ChatCompletionResponse(
            id=request_id, object="chat.completion", created=created_time, model=model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatCompletionResponseChoiceMessage(role="assistant", content=resp.answer),
                    finish_reason="stop"
                )
            ]
        )
    
    @app.post("/search", response_model=SearchResponse)
    async def search_endpoint(self, req: SearchRequest):
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        docs = await self.searcher.search.remote(
            queries=[req.query],
            product_name=req.product_name,
            product_version=req.product_version,
            top_k=req.top_k,
            request_id=request_id,
        )
        
        results = []
        for doc in docs:
            results.append(SearchResultDocument(
                url=doc.get("page_url", ""),
                score=doc.get("combined_score", 0.0),
                content=doc.get("page_content", ""),
                product_name=req.product_name,
                product_version=req.product_version
            ))
            
        return SearchResponse(results=results)

reranker_app = Reranker.bind()
searcher_app = Searcher.bind(reranker_app)
rag_reader_app = RAGSystem.bind(searcher_app)
smart_router_app = SmartRouter.bind(rag_reader_app, searcher_app)
