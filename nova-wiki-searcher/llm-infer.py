import os
import uuid
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Optional
import json
import math
import time
import torch
import re

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ray import serve
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery

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

PRODUCTS = [ "starguard" ]

# PRODUCTS = [
#     "nova",
#     "zvirt",
#     "zvirt-containers",
#     "zvirt-metrics",
#     "zvirt-dc-manager",
#     "terraform",
#     "termit",
#     "cloudlink",
#     "nova-se",
#     "starvault",
#     "knowledgebase", 
#     "solutions",
# ]

RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "BAAI/bge-reranker-v2-m3")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "NewWikiDocs")
WEAVIATE_GRPC_ADDR = os.getenv("WEAVIATE_GRPC_ADDR", "weaviate-grpc.nova-weaviate.svc")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_HTTP_ADDR = os.getenv("WEAVIATE_HTTP_ADDR", "weaviate.nova-weaviate.svc")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "80"))
WEAVIATE_API_TOKEN = os.getenv("WEAVIATE_API_TOKEN")


RAG_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
LLM_NUM_GPUS = float(os.getenv("LLM_NUM_GPUS", "0.9"))
LLM_NUM_CPUS = float(os.getenv("LLM_NUM_CPUS", "10"))
RERANKER_NUM_GPUS = float(os.getenv("RERANKER_NUM_GPUS", "0.1"))

LLM_MAX_MODEL_LEN = int(os.getenv("LLM_MAX_MODEL_LEN", "262144"))
LLM_MAX_NUM_BATCHED_TOKENS = int(os.getenv("LLM_MAX_NUM_BATCHED_TOKENS", "8192"))
LLM_GPU_MEMORY_UTILIZATION = float(os.getenv("LLM_GPU_MEMORY_UTILIZATION", "0.95"))
LLM_KV_CACHE_DTYPE = os.getenv("LLM_KV_CACHE_DTYPE", "auto")

# Для rewrite и HyDE рассуждения выключены всегда: там нужен короткий литерал.
# Для финального ответа режим управляется флагом.
RAG_LLM_ANSWER_THINKING = os.getenv("RAG_LLM_ANSWER_THINKING", "false").lower() in ("1", "true", "yes")

RAG_ANSWER_MAX_TOKENS = int(os.getenv("RAG_ANSWER_MAX_TOKENS", "4096"))
SIMPLE_ANSWER_MAX_TOKENS = int(os.getenv("SIMPLE_ANSWER_MAX_TOKENS", "512"))
REWRITE_MAX_TOKENS = int(os.getenv("REWRITE_MAX_TOKENS", "64"))
HYDE_MAX_TOKENS = int(os.getenv("HYDE_MAX_TOKENS", "96"))

# Реранкер: длину контролирует токенизатор, батчи держим небольшими,
# чтобы не ловить пики памяти на GPU, который делится с vLLM.
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "2048"))
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "6000"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))

# Retrieval: RRF-слияние списков, ограничение кандидатов на реранк,
# лимит чанков с одной страницы в финальном контексте.
RRF_K = int(os.getenv("RRF_K", "60"))
SEARCH_CANDIDATES_LIMIT = int(os.getenv("SEARCH_CANDIDATES_LIMIT", "40"))
SEARCH_PER_PAGE_CAP = int(os.getenv("SEARCH_PER_PAGE_CAP", "2"))

# HyDE вызывается адаптивно: только если первый проход дал слабый top-1.
# Порог сравнивается с сырым логитом реранкера (для bge-reranker >0 ≈ релевантно).
HYDE_RERANK_THRESHOLD = float(os.getenv("HYDE_RERANK_THRESHOLD", "0.0"))

RAG_SYSTEM_PROMPT_MESSAGES = [
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
15. If the query mentions operating systems or distributions, OrionSoft uses: Redos, Almalinux, Astra, Alt, MosOS, CentOS, Ubuntu. Do not use other OS in answers.""",
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

DEFAULT_SYSTEM_PROMPT_MESSAGES = [
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

REWRITE_PROMPT_MESSAGES = [
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
            "Conversation history:\n{history_text}\n\n"
            "Latest question: {last_user_msg}\n\n"
            "Final search query:"
        ),
    },
]

HYDE_PROMPT_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are an expert IT assistant. "
            "Answer the user's query with one short technical paragraph. "
            "Use only essential information, no introductions, no conclusions, no greetings, "
            "no lists, no code blocks, no Markdown formatting. "
            "If the query mentions operating systems or distributions, "
            "mention only OS family names without version numbers "
            "(for example: Redos, Almalinux, Astra, Alt, MosOS, CentOS, Ubuntu). "
            "Never write specific version numbers or minor releases. "
            "Maximum 60 words. Answer in Russian."
        ),
    },
    {
        "role": "user",
        "content": "Query: {search_query}",
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
    ray_actor_options={"num_cpus": 4, "num_gpus": RERANKER_NUM_GPUS},
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

    def _score_pairs(self, pairs: List[List[str]]) -> List[float]:
        """Скорит пары батчами, сортируя по длине, чтобы не раздувать padding."""
        device = next(self.model.parameters()).device
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
        scores = [0.0] * len(pairs)

        with torch.no_grad():
            for batch_start in range(0, len(order), RERANK_BATCH_SIZE):
                batch_idx = order[batch_start:batch_start + RERANK_BATCH_SIZE]
                inputs = self.tokenizer(
                    [pairs[i] for i in batch_idx],
                    padding=True,
                    truncation=True,
                    max_length=RERANK_MAX_LENGTH,
                    return_tensors="pt",
                ).to(device)

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    batch_scores = self.model(**inputs).logits.view(-1).float().tolist()

                for i, score in zip(batch_idx, batch_scores):
                    scores[i] = score

        return scores

    async def rerank(
        self,
        query: str,
        request_id: str,
        docs: List[Dict[str, str]],
        top_k: int = 8,
        alpha: float = 0.8,

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
            # длину режет токенизатор по RERANK_MAX_LENGTH; тут только страховка
            # от аномально больших страниц, чтобы не тормозить токенизацию
            snippet = f"{title}\n{content}"[:RERANK_MAX_CHARS]
            pairs.append([query, snippet])

        rerank_scores = self._score_pairs(pairs)

        self.logger.debug(f"[req: {request_id}] Raw rerank scores computed, calucalted scores: {rerank_scores}")

        # Приор от retrieval: RRF-ранг сопоставим между коллекциями, в отличие
        # от сырого hybrid_score, поэтому он предпочтительнее при наличии.
        prior_scores = [
            float(doc.get("rrf_score", doc.get("hybrid_score", 0.0))) for doc in docs
        ]
        eps = 1e-8

        def min_max_norm(values: List[float]) -> List[float]:
            vmin = min(values)
            vmax = max(values)
            if math.isclose(vmin, vmax):
                return [0.5 for _ in values]
            return [(v - vmin) / (vmax - vmin + eps) for v in values]

        prior_norm = min_max_norm(prior_scores)
        rerank_norm = min_max_norm(rerank_scores)

        scored_docs = []
        for doc, p_raw, p_n, r_raw, r_n in zip(docs, prior_scores, prior_norm, rerank_scores, rerank_norm):
            combined = alpha * r_n + (1.0 - alpha) * p_n
            doc["prior_score_raw"] = p_raw
            doc["rerank_score_raw"] = r_raw
            doc["prior_score_norm"] = p_n
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

    async def _fetch_docs_parallel(self, query_text: str, product_name: str, product_version: str, request_id: str) -> List[List[Dict[str, Any]]]:
        """Возвращает ранжированные списки по коллекциям — отдельно, для RRF."""
        collection = self.product_collections.get(product_name) or self.product_collections["nova"]
        version = "latest" if product_name == "zvirt" else product_name

        return_properties = ["title", "page_content", "page_url", "source"]

        self.logger.info(f"[req: {request_id}] Execute remote document search in db {product_name}")
        # Список пар (имя, запрос): коллекцию можно закомментировать,
        # не правя распаковку результатов и логирование.
        pending = [
            (
                product_name,
                asyncio.to_thread(
                    collection.query.hybrid,
                    query=query_text,
                    alpha=0.3,
                    limit=15,
                    filters=Filter.by_property("version").equal(product_version),
                    return_metadata=MetadataQuery(score=True),
                    return_properties=return_properties,
                ),
            ),
            # (
            #     "knowledgebase",
            #     asyncio.to_thread(
            #         self.product_collections["knowledgebase"].query.hybrid,
            #         query=query_text,
            #         alpha=0.3,
            #         limit=7,
            #         filters=Filter.by_property("version").equal(version),
            #         return_metadata=MetadataQuery(score=True),
            #         return_properties=return_properties,
            #     ),
            # ),
            # (
            #     "solutions",
            #     asyncio.to_thread(
            #         self.product_collections["solutions"].query.hybrid,
            #         query=query_text,
            #         alpha=0.3,
            #         limit=7,
            #         filters=Filter.by_property("version").equal(version),
            #         return_metadata=MetadataQuery(score=True),
            #         return_properties=return_properties,
            #     ),
            # ),
        ]

        results = await asyncio.gather(*(task for _, task in pending))

        self.logger.info(
            f"[req: {request_id}] Found following docs: "
            + ", ".join(
                f"{name} - {len(res.objects or [])}"
                for (name, _), res in zip(pending, results)
            )
        )

        ranked_lists = []
        for res in results:
            ranked_lists.append([
                {
                    "title": obj.properties.get("title", ""),
                    "page_content": obj.properties.get("page_content", ""),
                    "page_url": obj.properties.get("page_url", ""),
                    "source": obj.properties.get("source", ""),
                    "hybrid_score": obj.metadata.score or 0.0,
                }
                for obj in (res.objects or [])
            ])
        return ranked_lists

    @staticmethod
    def _fuse_rrf(ranked_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Reciprocal rank fusion: ранги сопоставимы между коллекциями и вариантами запроса."""
        fused: Dict[str, Dict[str, Any]] = {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                key = doc.get("source") or doc.get("page_url")
                if not key:
                    continue
                entry = fused.get(key)
                if entry is None:
                    entry = dict(doc)
                    entry["rrf_score"] = 0.0
                    fused[key] = entry
                entry["rrf_score"] += 1.0 / (RRF_K + rank)
                entry["hybrid_score"] = max(
                    float(entry.get("hybrid_score", 0.0)),
                    float(doc.get("hybrid_score", 0.0)),
                )
        return sorted(fused.values(), key=lambda d: d["rrf_score"], reverse=True)

    @staticmethod
    def _chunk_index(source: str) -> int:
        match = re.search(r"#chunk-(\d+)$", source or "")
        return int(match.group(1)) if match else -1

    @staticmethod
    def _select_diverse(docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Не даёт одной странице занять весь контекст."""
        selected: List[Dict[str, Any]] = []
        per_page: Dict[str, int] = {}
        for doc in docs:
            page = doc.get("page_url") or doc.get("source", "")
            if per_page.get(page, 0) >= SEARCH_PER_PAGE_CAP:
                continue
            per_page[page] = per_page.get(page, 0) + 1
            selected.append(doc)
            if len(selected) >= top_k:
                break
        return selected

    def _merge_adjacent_chunks(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Склеивает соседние чанки одной страницы, сохраняя порядок по релевантности."""
        by_page: Dict[str, List[Any]] = {}
        for pos, doc in enumerate(docs):
            by_page.setdefault(doc.get("page_url", ""), []).append((pos, doc))

        def flush(group: List[Any]) -> Any:
            best_pos = min(pos for pos, _ in group)
            base = dict(max(group, key=lambda item: item[1].get("combined_score", 0.0))[1])
            if len(group) > 1:
                base["page_content"] = "\n\n".join(doc.get("page_content", "") for _, doc in group)
            return (best_pos, base)

        merged: List[Any] = []
        for items in by_page.values():
            items.sort(key=lambda item: self._chunk_index(item[1].get("source", "")))
            group: List[Any] = []
            for pos, doc in items:
                idx = self._chunk_index(doc.get("source", ""))
                prev_idx = self._chunk_index(group[-1][1].get("source", "")) if group else -99
                if group and idx >= 0 and prev_idx == idx - 1:
                    group.append((pos, doc))
                    continue
                if group:
                    merged.append(flush(group))
                group = [(pos, doc)]
            if group:
                merged.append(flush(group))

        merged.sort(key=lambda item: item[0])
        return [doc for _, doc in merged]

    async def search(
        self, 
        queries: List[str], 
        product_name: str, 
        product_version: str, 
        request_id: str,
        top_k: int = 5,
        alpha: float = 0.8
    ) -> List[Dict[str, Any]]:
        docs_start_time = time.perf_counter()

        tasks = [self._fetch_docs_parallel(q, product_name, product_version, request_id) for q in queries]
        per_query_lists = await asyncio.gather(*tasks)

        ranked_lists = [ranked for lists in per_query_lists for ranked in lists]
        raw_docs = self._fuse_rrf(ranked_lists)[:SEARCH_CANDIDATES_LIMIT]

        docs_end_time = time.perf_counter()
        self.logger.info(f"[req: {request_id}] Retrieved {len(raw_docs)} unique documents in {docs_end_time - docs_start_time:.6f}s")

        if not raw_docs:
            return []

        main_query = queries[0]
        scored_docs = await self.reranker.rerank.remote(
            main_query, request_id, raw_docs, top_k=len(raw_docs), alpha=alpha
        )

        selected = self._select_diverse(scored_docs, top_k)
        return self._merge_adjacent_chunks(selected)

    def close(self):
        if hasattr(self, "weaviate_connection") and self.weaviate_connection is not None:
            self.weaviate_connection.close()
            self.weaviate_connection = None

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": LLM_NUM_CPUS, "num_gpus": LLM_NUM_GPUS},
)
class RAGSystem:
    def __init__(self, searcher_handle):
        self.searcher = searcher_handle
        self.logger = init_logger()
        self.logger.info(f"Initializing RAGSystem with local LLM {RAG_LLM_MODEL}")

        self.rag_answer_messages_template = RAG_SYSTEM_PROMPT_MESSAGES
        self.rewrite_prompt_messages = REWRITE_PROMPT_MESSAGES
        self.hyde_prompt_messages = HYDE_PROMPT_MESSAGES

        engine_kwargs: Dict[str, Any] = {
            "model": RAG_LLM_MODEL,
            "gpu_memory_utilization": LLM_GPU_MEMORY_UTILIZATION,
            "max_model_len": LLM_MAX_MODEL_LEN,
            "max_num_batched_tokens": LLM_MAX_NUM_BATCHED_TOKENS,
            "trust_remote_code": True,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "kv_cache_dtype": LLM_KV_CACHE_DTYPE,
            "enforce_eager": False,
        }

        self.engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))
        self.tokenizer = AutoTokenizer.from_pretrained(RAG_LLM_MODEL, trust_remote_code=True)

    def _render_prompt(self, messages: List[Dict[str, str]], thinking: bool) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        except TypeError:
            # шаблон модели не знает про enable_thinking
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    async def _generate_text(self, prompt: str, sampling_params: SamplingParams) -> str:
        request_id = str(uuid.uuid4())
        generator = self.engine.generate(prompt, sampling_params, request_id)
        final_output = None
        async for request_output in generator:
            final_output = request_output
        return final_output.outputs[0].text if final_output else ""

    async def _chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.3,
        top_p: float = 0.95,
        thinking: bool = False,
    ) -> str:
        prompt = self._render_prompt(messages, thinking)
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        text = await self._generate_text(prompt, sampling_params)
        return self._strip_thinking(text)

    @staticmethod
    def _visible_text(text: str) -> str:
        """Скрывает блок рассуждений: пока </think> не пришёл, показывать нечего."""
        if "<think>" not in text:
            return text
        end = text.find("</think>")
        if end == -1:
            return ""
        return text[end + len("</think>"):].lstrip()

    async def _stream_sse(self, messages: List[Dict[str, str]], max_tokens: int):
        """Отдаёт ответ движка чанками в формате OpenAI chat.completion.chunk."""
        prompt = self._render_prompt(messages, RAG_LLM_ANSWER_THINKING)
        sampling_params = SamplingParams(
            temperature=0.3, top_p=0.95, max_tokens=max_tokens
        )
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        request_id = str(uuid.uuid4())

        def sse(delta: Optional[str] = None, finish_reason: Optional[str] = None,
                role: Optional[str] = None) -> bytes:
            payload: Dict[str, Any] = {}
            if role is not None:
                payload["role"] = role
            if delta is not None:
                payload["content"] = delta
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": RAG_LLM_MODEL,
                "choices": [{"index": 0, "delta": payload, "finish_reason": finish_reason}],
            }
            return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")

        async def generator() -> AsyncGenerator[bytes, None]:
            yield sse(role="assistant")
            sent = 0
            finish_reason = "stop"
            try:
                async for request_output in self.engine.generate(
                    prompt, sampling_params, request_id
                ):
                    output = request_output.outputs[0]
                    visible = self._visible_text(output.text)
                    if len(visible) > sent:
                        yield sse(delta=visible[sent:])
                        sent = len(visible)
                    if output.finish_reason:
                        finish_reason = output.finish_reason
            except asyncio.CancelledError:
                await self.engine.abort(request_id)
                raise
            except Exception as e:
                self.logger.exception("Streaming generation failed")
                yield f'data: {{"error": {{"message": "LLM error: {e}"}}}}\n\n'.encode("utf-8")
                yield b"data: [DONE]\n\n"
                return

            yield sse(finish_reason=finish_reason)
            yield b"data: [DONE]\n\n"

        return generator()

    async def _generate_answer(self, messages: List[Dict[str, str]], max_tokens: int, stream: bool = False):
        if stream:
            return await self._stream_sse(messages=messages, max_tokens=max_tokens)
        return await self._chat(
            messages, max_tokens=max_tokens, thinking=RAG_LLM_ANSWER_THINKING
        )

    async def make_prediction(self, req: InputQuestion, request_id: str) -> OutputAnswer:
        messages = [DEFAULT_SYSTEM_PROMPT_MESSAGES[0]]
        for m in req.query:
            role = m.get("role", "")
            content = (m.get("content") or "").strip()
            if content and role in ("user", "assistant"):
                messages.append({"role": role, "content": self._strip_thinking(content)})

        self.logger.info(f"[req: {request_id}] Generating remote answer...")
        answer_text = await self._chat(
            messages,
            max_tokens=SIMPLE_ANSWER_MAX_TOKENS,
            thinking=RAG_LLM_ANSWER_THINKING,
        )
        return OutputAnswer(answer=self._strip_thinking(answer_text))
    
    async def _compress_history(self, messages: List[Dict[str, Any]], request_id: str) -> str:

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
        
        self.logger.debug(f"[req: {request_id}] History: {history_lines}")

        history_text = "\n".join(history_lines)

        filled_messages = [
            {
                "role": m["role"],
                "content": m["content"].format(history_text=history_text, last_user_msg=last_user_msg),
            }
            for m in self.rewrite_prompt_messages
        ]
        try:
            rewritten = await self._chat(
                filled_messages,
                max_tokens=REWRITE_MAX_TOKENS,
                temperature=0.0,
                top_p=1.0,
                thinking=False,
            )
        except Exception as e:
            self.logger.warning(f"[req: {request_id}] Query rewrite failed: {e}")
            return last_user_msg

        # модель может вернуть <think> даже при выключенных рассуждениях
        result = self._strip_thinking(rewritten).strip().strip('"')
        self.logger.debug(f"[req: {request_id}] Query rewrite: [{last_user_msg}] → [{result}]")
        return result or last_user_msg
    
    async def _generate_hyde(self, search_query: str) -> str:
        filled_hyde_messages = [
            {
                "role": m["role"],
                "content": m["content"].format(search_query=search_query),
            }
            for m in self.hyde_prompt_messages
        ]
        try:
            h_doc = await self._chat(
                filled_hyde_messages,
                max_tokens=HYDE_MAX_TOKENS,
                temperature=0.0,
                top_p=1.0,
                thinking=False,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.warning(f"HyDE generation failed: {e}")
            return search_query

        return self._strip_thinking(h_doc) or search_query

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

        search_query = await self._compress_history(req.query, request_id)

        # HyDE считается параллельно с первым поиском, поэтому его латентность
        # скрывается за retrieval. Второй проход делаем только если top-1 слабый.
        hyde_task = asyncio.create_task(self._generate_hyde(search_query))
        reranked_docs = await self.searcher.search.remote(
            queries=[search_query],
            product_name=req.product_name,
            product_version=req.product_version,
            top_k=5,
            request_id=request_id,
        )

        top_score = max(
            (float(doc.get("rerank_score_raw", 0.0)) for doc in reranked_docs),
            default=float("-inf"),
        )

        if top_score < HYDE_RERANK_THRESHOLD:
            h_doc = await hyde_task
            self.logger.info(
                f"[req: {request_id}] Weak top-1 ({top_score:.3f} < {HYDE_RERANK_THRESHOLD}), retrying with HyDE"
            )
            reranked_docs = await self.searcher.search.remote(
                queries=[search_query, h_doc],
                product_name=req.product_name,
                product_version=req.product_version,
                top_k=5,
                request_id=request_id,
            )
        else:
            hyde_task.cancel()
            self.logger.debug(
                f"[req: {request_id}] HyDE skipped, top-1 rerank score {top_score:.3f}"
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
        self.logger.debug(f"[req: {request_id}] Raw messages: {rag_messages}")

        end_time = time.perf_counter()
        self.logger.info(f"[req: {request_id}] Init actions for request {request_id} done in {end_time - start_time:.6f}s")

        self.logger.info(f"[req: {request_id}] Generating remote answer...")
        if not req.stream:
            final_answer = await self._generate_answer(
                rag_messages, max_tokens=RAG_ANSWER_MAX_TOKENS, stream=False
            )
            return OutputAnswer(answer=final_answer)

        return await self._generate_answer(
            rag_messages, max_tokens=RAG_ANSWER_MAX_TOKENS, stream=True
        )
    
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

        if stream and user_request:
            req = InputRagQuestion(
                query=messages,
                product_name=product_name,
                product_version=product_version,
                stream=True,
            )
            resp_gen = self.rag.options(stream=True).make_context_prediction.remote(req, request_id)

            async def passthrough_sse():
                async for chunk in resp_gen:
                    if await request.is_disconnected():
                        break
                    yield chunk if isinstance(chunk, bytes) else str(chunk).encode("utf-8")

            return StreamingResponse(
                passthrough_sse(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        if user_request:
            req = InputRagQuestion(
                query=messages,
                product_name=product_name,
                product_version=product_version,
                stream=False,
            )
            resp = await self.rag.make_context_prediction.remote(req, request_id)
        else:
            req = InputQuestion(query=messages, stream=False)
            resp = await self.rag.make_prediction.remote(req, request_id)

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
