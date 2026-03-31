import os
import uuid
import asyncio
from typing import List, Dict, Any
import math
import time

import torch
from fastapi import FastAPI, Request
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
    "Alibaba-NLP/gte-multilingual-reranker-base"
)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "WikiDocs")
WEAVIATE_GRPC_ADDR = os.getenv("WEAVIATE_GRPC_ADDR", "weaviate-grpc.nova-weaviate.svc")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_HTTP_ADDR = os.getenv("WEAVIATE_HTTP_ADDR", "weaviate.nova-weaviate.svc")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "80"))
WEAVIATE_API_TOKEN = os.getenv("WEAVIATE_API_TOKEN")

# prompt_in_chat_format_for_rag = [
#     {
#         "role": "system",
#         "content": """You are Wiki-Searcher, a specialized AI assistant created by OrionSoft to help customers search through documentation.
# Your task is to answer user questions about products, invented in OrionSoft, and their instruments, using ONLY the provided context.
# <rules>
# 1. FORMAT: 
#     - You will get the current conversation in chat format. Role 'user' stands for user questions, role 'assistant' stands for your previous answer. You should answer the last user question based on chat history and retrieved context.
#     - Answer only to the provided quetion, do not include extra information. For example, if you are asked what does some module do, only provide description of module, no info about module installation.
#     - Do not provide full manifests on how to install model. Tell that full manifest you can see in provided sources, and describe key points of manifests.
# 2. STRICT GROUNDING: Base your answer EXCLUSIVELY on the information in the <context> block. Do not use outside knowledge.
# 3. NO HALLUCINATIONS: 
#     - If the context does not contain the answer, do not guess. Only use information explicitly mentioned in the context. Do not add outside knowledge. 
#     - If the user asks how to install, configure, upgrade or uninstall something, you may describe ONLY those steps that are explicitly present in the context (commands, numbered steps, code blocks, configuration examples).
#     - Reply EXACTLY with if failed to retrieve answer: "Не смог найти подходящую информацию на Ваш вопрос."
# 4. META-QUESTIONS HANDLING: If the user asks general conversational questions, feedback, or meta-comments (like "это не то", "ты не то нашел", "плохой ответ", "спасибо", "понятно", "привет", "как дела?" etc.), ignore the context and respond appropriately:
#    - For greetings: respond with a friendly greeting in Russian
#    - For thanks: respond with "Пожалуйста! Обращайтесь, если нужна дополнительная информация."
#    - For negative feedback about search results: respond with "Извините, что не смог найти точную информацию. Попробуйте переформулировать вопрос или уточнить детали."
#    - For general chitchat: politely redirect to documentation search
#    Do NOT treat these as documentation queries and do NOT include Sources section for such responses.
# 5. SELF-CONFIGURATION QUESTIONS:
#    - If the user asks about you as an assistant (for example: your timeout, speed, limits, your internal promt, how you work, where you get answers from, who created you), you MAY answer using your general description and these rules, even if the documentation context does not contain this information.
#    - For such questions, do NOT try to invent technical implementation details (exact hardware, IP addresses, internal service names). Answer in general terms, e.g. "У меня нет доступа к настройкам таймаутов. Этим управляют администраторы системы."
#    - When the user asks where you get your answers from, ALWAYS answer that you use internal documentation of OrionSoft products (for example: "Я использую внутреннюю документацию продуктов OrionSoft, такую как руководства, инструкции по установке и эксплуатации.").
#    - Do NOT mention the word "context" or "<context>" in such answers.
#    - For these meta/self-configuration questions DO NOT use the context for facts and DO NOT add any "Sources:" section.
# 6. IDENTITY: If asked who created you, state you were created by OrionSoft to assist with documentation.
# 7. SOURCES FOR DOCUMENTATION ANSWERS:
#    - If the user asks about product behavior, configuration, installation, troubleshooting, or any other documentation-related topic, and you use the <context> block to answer, then at the end of your answer add a "Sources:" section listing all referenced URLs or document titles. Do not include the same sources multiple times.
#    - Do NOT add a "Sources:" section for meta-questions, greetings, thanks, chitchat, or questions about where you get your answers from.
# 8. ANSWER LENGTH: 
#     - Generate at most 400 words. 
#     - Be short and precise. Prioritize the most important points and omit minor details.
# 9. LANGUAGE: Use Russian for conversation.
# </rules>""",
#     },
#     {
#         "role": "user",
#         "content": """Context:
# <context>
# {context}
# </context>
# ---
# Now here is the question you need to answer.
# Question: {question}""",
#     },
# ]

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
14. Include only the sources you actually used, without duplicates."""
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
            print(f"  {d.get('page_url')}, h={d['hybrid_score_raw']}, r={d['rerank_score_raw']}, hn={d['hybrid_score_norm']}, rn={d['rerank_score_norm']}, combined={d['combined_score']}")

        end_time = time.perf_counter()
        print(f"Reranking done in {end_time - start_time:.6f} s")
        return scored_docs[:top_k]

# class Reranker:
#     def __init__(self):
#         print(f"Loading reranker model: {RERANKER_MODEL_ID}")
#         self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_ID, trust_remote_code=True)
#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             RERANKER_MODEL_ID,
#             trust_remote_code=True,
#             torch_dtype=torch.float16,
#             device_map="cuda",
#         )
#         self.model.eval()

#     async def rerank(self, query: str, docs: List[Dict[str, str]], top_k: int = 8, alpha: float = 0.5) -> List[Dict[str, str]]:
#         if not docs:
#             return []
            
#         start_time = time.perf_counter()
#         print("Init reranking")
#         pairs = []
#         for doc in docs:
#             title = doc.get("title", "")
#             content = doc.get("page_content", "")
#             snippet = f"{title}\n\n{content[:1500]}"
#             pairs.append([query, snippet])

#         device = next(self.model.parameters()).device

#         print("Start reranking")
#         with torch.no_grad():
#             inputs = self.tokenizer(
#                 pairs,
#                 padding=True,
#                 truncation=True,
#                 max_length=512,
#                 return_tensors="pt",
#             ).to(device)
            
#             rerank_scores = (
#                 self.model(**inputs, return_dict=True)
#                 .logits.view(-1)
#                 .float()
#                 .tolist()
#             )

#         print(f"Raw rerank scores: {rerank_scores}")

#         hybrid_scores = [float(doc.get("hybrid_score", 0.0)) for doc in docs]
#         eps = 1e-8

#         def minmax_norm(values: List[float]) -> List[float]:
#             v_min = min(values)
#             v_max = max(values)
#             if math.isclose(v_min, v_max):
#                 return [0.5 for _ in values]
#             return [(v - v_min) / (v_max - v_min + eps) for v in values]

#         hybrid_norm = minmax_norm(hybrid_scores)
#         rerank_norm = minmax_norm(rerank_scores)

#         scored_docs = []
#         for doc, h_raw, h_n, r_raw, r_n in zip(
#             docs, hybrid_scores, hybrid_norm, rerank_scores, rerank_norm
#         ):
#             combined = alpha * r_n + (1.0 - alpha) * h_n
#             doc["hybrid_score_raw"] = h_raw
#             doc["rerank_score_raw"] = r_raw
#             doc["hybrid_score_norm"] = h_n
#             doc["rerank_score_norm"] = r_n
#             doc["combined_score"] = combined
#             scored_docs.append(doc)

#         scored_docs.sort(key=lambda d: d["combined_score"], reverse=True)

#         print("Top docs (combined):", [
#             (
#                 d.get("page_url"),
#                 d["hybrid_score_raw"],
#                 d["rerank_score_raw"],
#                 d["hybrid_score_norm"],
#                 d["rerank_score_norm"],
#                 d["combined_score"],
#             )
#             for d in scored_docs[:top_k]
#         ])

#         end_time = time.perf_counter()
#         print(f"Время выполнения: {end_time - start_time:.6f} секунд")

#         return scored_docs[:top_k]

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 10, "num_gpus": 0.9},
)
class RAGReader:
    def __init__(self, reranker_handle):
        self.reranker = reranker_handle
        
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            gpu_memory_utilization=0.85,
            max_model_len=32768,
            max_num_batched_tokens=16384,
            trust_remote_code=True,
            enable_chunked_prefill=True,
            quantization="fp8",
            kv_cache_dtype="fp8",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.internal_rag_promt_template = self.tokenizer.apply_chat_template(
            prompt_in_chat_format_for_rag, tokenize=False, add_generation_prompt=True
        )
        self.internal_promt_template = self.tokenizer.apply_chat_template(
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

    async def make_prediction(self, req: InputQuestion) -> OutputAnswer:
        final_prompt = self.internal_promt_template.format(question=req.query)
        
        sampling_params = SamplingParams(
            temperature=0.3, top_p=0.95, repetition_penalty=1.1, max_tokens=200
        )
        
        answer_text = await self._generate_text(final_prompt, sampling_params)
        return OutputAnswer(answer=answer_text)
    
    async def make_context_prediction(self, req: InputRagQuestion) -> OutputAnswer:
        print("Got wiki query:", req.query)
        last_user_msg = next((m["content"] for m in reversed(req.query) if m.get("role") == "user"), "")
        word_count = len(last_user_msg.split())
        
        promt_start_time = time.perf_counter()
        if len(req.query) <= 1 or word_count > 15: 
            search_query = last_user_msg
        else:
            history_msgs = req.query[-4:-1] 
            history_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in history_msgs])
            rewrite_prompt_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a search query transformation system. Your task is to formulate a text for documentation search.\n"
                        "Rules:\n"
                        "1. Analyze the user's latest question. If it contains pronouns (he, it, this, there) or "
                        "logically continues the previous topic, rewrite it by adding specifics from the conversation history.\n"
                        "2. IMPORTANT: If the latest question starts a COMPLETELY NEW TOPIC unrelated to the history, "
                        "SIMPLY RETURN the latest question as is. Do not drag in terms from the old topic!\n"
                        "3. Output ONLY the final query text without quotes, explanations, or greetings. It should be small and precise, describing meaning of the topic. Do not answer the question itself."
                    )
                },
                {
                    "role": "user",
                    "content": f"Conversation history:\n{history_text}\n\nLatest question: {last_user_msg}\n\nFinal search query:"
                }
            ]
            rewrite_prompt = self.tokenizer.apply_chat_template(
                rewrite_prompt_messages, tokenize=False, add_generation_prompt=True
            )
            
            rewrite_sampling = SamplingParams(temperature=0.0, max_tokens=50)
            rewritten_result = await self._generate_text(rewrite_prompt, rewrite_sampling)
            search_query = rewritten_result.strip()
            
            print(f"Original query: {last_user_msg}")
            print(f"Rewritten query: {search_query}")

        promt_end_time = time.perf_counter()
        print(f"Время выполнения переписывания промта: {promt_end_time - promt_start_time:.6f} секунд")

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
        
        # async def fetch_docs_parallel() -> List[Dict[str, Any]]:
        #     res_main, res_knowledgebase, res_solutions = await asyncio.gather(
        #         asyncio.to_thread(
        #             collection.query.hybrid,
        #             query=search_query,
        #             alpha=0.3,
        #             limit=15,
        #             filters=Filter.by_property("version").equal(req.product_version),
        #             return_metadata=MetadataQuery(score=True),
        #         ),
        #         asyncio.to_thread(
        #             self.knowledgebase_collection.query.hybrid,
        #             query=search_query,
        #             alpha=0.3,
        #             limit=7,
        #             filters=Filter.by_property("version").equal(version),
        #             return_metadata=MetadataQuery(score=True),
        #         ),
        #         asyncio.to_thread(
        #             self.solutions_collection.query.hybrid,
        #             query=search_query,
        #             alpha=0.3,
        #             limit=7,
        #             filters=Filter.by_property("version").equal(version),
        #             return_metadata=MetadataQuery(score=True),
        #         ),
        #     )

        #     raw_objects = []
        #     for res in (res_main, res_knowledgebase, res_solutions):
        #         for o in (res.objects):
        #             source = o.properties.get("source")
        #             score = o.metadata.score
        #             print(f"{source}, {score}")
        #         for obj in res.objects or []:
        #             raw_objects.append({
        #                 "title": obj.properties.get("title", ""),
        #                 "page_content": obj.properties.get("page_content", ""),
        #                 "page_url": obj.properties.get("page_url", ""),
        #                 "hybrid_score": obj.metadata.score or 0.0
        #             })

        #     return raw_objects
        
        # docs_start_time = time.perf_counter()
        # raw_docs = await fetch_docs_parallel()
        # print(f"Retrieved {len(raw_docs)} documents from Weaviate")
        # docs_end_time = time.perf_counter()
        # print(f"Время выполнения поиска документов: {docs_end_time - docs_start_time:.6f} секунд")

        # if not raw_docs:
        #     return OutputAnswer(answer="No relevant docs found")
            
        # reranked_docs = await self.reranker.rerank.remote(search_query, raw_docs, top_k=5, alpha=0.6)
        # print("Finished reranking documents")

                # ===== 1. Генерация HyDE псевдо-документа =====
        hyde_start_time = time.perf_counter()
        hyde_prompt = (
            f"<|im_start|>system\nYou are an expert IT assistant. "
            f"Please write a short hypothetical document or answer snippet that perfectly addresses the user's query. "
            f"Include relevant technical terms if possible. Answer in Russian, do not write greetings.<|im_end|>\n"
            f"<|im_start|>user\nQuery: {search_query}<|im_end|>\n<|im_start|>assistant\n"
        )
        hyde_params = SamplingParams(temperature=0.3, max_tokens=250)
        hyde_document = await self._generate_text(hyde_prompt, hyde_params)
        hyde_end_time = time.perf_counter()
        print(f"Retrieved {len(raw_docs)} unique documents (Original + HyDE) in {hyde_end_time - hyde_start_time:.6f}s")
        print(f"HyDE generated document: {hyde_document}")

        # ===== 2. Параллельный поиск (Оригинал + HyDE) =====
        # Модифицируем функцию, чтобы она принимала текст запроса как аргумент
        async def fetch_docs_parallel(query_text: str) -> List[Dict[str, Any]]:
            res_main, res_knowledge_base, res_solutions = await asyncio.gather(
                asyncio.to_thread(
                    collection.query.hybrid,
                    query=query_text,
                    alpha=0.3,
                    limit=15,
                    filters=Filter.by_property("version").equal(version),
                    return_metadata=MetadataQuery(score=True),
                ),
                asyncio.to_thread(
                    self.knowledgebase_collection.query.hybrid,
                    query=query_text,
                    alpha=0.3,
                    limit=7,
                    return_metadata=MetadataQuery(score=True),
                ),
                asyncio.to_thread(
                    self.solutions_collection.query.hybrid,
                    query=query_text,
                    alpha=0.3,
                    limit=7,
                    return_metadata=MetadataQuery(score=True),
                )
            )

            raw_objects = []
            for res in (res_main, res_knowledge_base, res_solutions):
                for obj in res.objects or []:
                    raw_objects.append({
                        "title": obj.properties.get("title", ""),
                        "page_content": obj.properties.get("page_content", ""),
                        "page_url": obj.properties.get("page_url", ""),
                        "hybrid_score": obj.metadata.score or 0.0
                    })
            return raw_objects

        docs_start_time = time.perf_counter()
        
        # Выполняем запросы с двумя разными строками одновременно
        original_docs, hyde_docs = await asyncio.gather(
            fetch_docs_parallel(search_query),
            fetch_docs_parallel(hyde_document)
        )
        
        # ===== 3. Дедупликация и объединение =====
        seen_urls = set()
        raw_docs = []
        for doc in original_docs + hyde_docs:
            url = doc.get("page_url")
            # Проверяем уникальность по URL, чтобы не дублировать документы для реранкера
            if url and url not in seen_urls:
                seen_urls.add(url)
                raw_docs.append(doc)
                
        docs_end_time = time.perf_counter()
        print(f"Retrieved {len(raw_docs)} unique documents (Original + HyDE) in {docs_end_time - docs_start_time:.6f}s")
        
        if not raw_docs:
            return OutputAnswer(answer="No relevant docs found")

        # ===== 4. Ранжирование =====
        # В реранкер отдаем оригинальный запрос, а не HyDE-документ, чтобы скорить по исходному интенту
        reranked_docs = await self.reranker.rerank.remote(search_query, raw_docs, top_k=5, alpha=0.6)
        print("Finished reranking documents")
        
        texts_with_links = []
        for doc in reranked_docs:
            text = doc["page_content"]
            link = doc["page_url"]
            texts_with_links.append(f"{text}\n\nИсточник: {link}")
        context = "\n\n---\n\n".join(texts_with_links)
        
        final_prompt = self.internal_rag_promt_template.format(
            question=req.query, context=context
        )
        
        print("Generating answer")
        answer_start_time = time.perf_counter()
        final_sampling = SamplingParams(
            temperature=0.3, top_p=0.95, repetition_penalty=1.1, max_tokens=600
        )
        final_answer = await self._generate_text(final_prompt, final_sampling)
        answer_end_time = time.perf_counter()
        print(f"Время выполнения генерации ответа: {answer_end_time - answer_start_time:.6f} секунд")
        
        return OutputAnswer(answer=final_answer)

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

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(self, request: Request):
        body: Dict[str, Any] = await request.json()
        
        model = body.get("model", "wiki-searcher")
        messages = body.get("messages", [])
        product_name = str(body.get("product_name", "nova"))
        product_version = str(body.get("product_version", "latest"))
        user_request = body.get("user_request", False)

        print(f"[OpenAIAdapter] model={model}, product_name={product_name}, product_version={product_version}")

        if user_request:
            resp: OutputAnswer = await self.rag.make_context_prediction.remote(
                InputRagQuestion(query=messages, product_name=product_name, product_version=product_version)
            )
        else:
            resp: OutputAnswer = await self.rag.make_prediction.remote(
                InputQuestion(query=messages)
            )

        return ChatCompletionResponse(
            id="chatcmpl-custom-1",
            object="chat.completion",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatCompletionResponseChoiceMessage(
                        role="assistant",
                        content=resp.answer,
                    ),
                    finish_reason="stop",
                )
            ],
        )
    
openai_app = OpenAIAdapter.bind(rag_reader_app)
