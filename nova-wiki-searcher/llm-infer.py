import os
import uuid
import asyncio
from typing import List, Dict, Any

from fastapi import FastAPI, Request
from pydantic import BaseModel
from ray import serve

# Оставляем токенизатор для применения Chat Templates (vLLM может использовать его внутри, 
# но для кастомных шаблонов надежнее держать отдельный инстанс)
from transformers import AutoTokenizer

# Импорты vLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter

MODEL_NAME = os.getenv(
    "READER_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct", # При запуске передайте ваш Qwen3-30B-A3B-Instruct-2507-FP8
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
        "content": """You are Wiki-Searcher, a specialized AI assistant created by OrionSoft to help customers search through documentation.
Your task is to answer user questions about Nova Container Platform (NCP) and its instruments, using ONLY the provided context.
<rules>
1. FORMAT: You will get the current conversation in chat format. Role 'user' stands for user questions, role 'assistant' stands for your previoues answer. You should answer last user question based on chat history and retrieved context.
2. RELEVANCE: Answer ONLY questions related to Nova, NCP, and their components. If a question is entirely unrelated, politely decline to answer.
3. STRICT GROUNDING: Base your answer EXCLUSIVELY on the information in the <context> block. Do not use outside knowledge.
4. NO HALLUCINATIONS: If the context does not contain the answer, do not guess. Reply EXACTLY with: "I didn't find any information about this in the documentation."
5. FORMAT: Be concise and direct. Include direct sources at the end of the sentences, where suitable. Include word 'Source' and hyperlink to the url of this document.
6. IDENTITY: If asked who created you, state you were created by OrionSoft to assist with documentation.
</rules>""",
    },
    {
        "role": "user",
        "content": """Context:
{context}
---
Now here is the question you need to answer.
Question: {question}""",
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
    ray_actor_options={"num_cpus": 8, "num_gpus": 1},
)
class RAGReader:
    def __init__(self):
        # 1. Инициализация vLLM Async Engine для H100
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            tensor_parallel_size=1,        # 1 GPU (H100)
            gpu_memory_utilization=0.9,    # Используем 90% VRAM под модель и KV Cache
            max_model_len=8192,            # Ограничиваем максимальный контекст для стабильности
            trust_remote_code=True,
            # quantization="fp8",          # Раскомментируйте, если vLLM не определит FP8 автоматически
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 2. Инициализация токенизатора для Chat Templates
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.internal_rag_promt_template = self.tokenizer.apply_chat_template(
            prompt_in_chat_format_for_rag, tokenize=False, add_generation_prompt=True
        )
        self.internal_promt_template = self.tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True
        )

        # 3. Подключение к Weaviate
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
        
        self.nova_collection = self.weaviate_connection.collections.use(COLLECTION_NAME)

    # Вспомогательный асинхронный метод для генерации текста через vLLM
    async def _generate_text(self, prompt: str, sampling_params: SamplingParams) -> str:
        request_id = str(uuid.uuid4())
        generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in generator:
            final_output = request_output
            
        return final_output.outputs[0].text

    async def make_prediction(self, req: InputQuestion) -> OutputAnswer:
        print("Got query:", req.query)
        final_prompt = self.internal_promt_template.format(question=req.query)
        
        sampling_params = SamplingParams(
            temperature=0.6, top_p=0.95, repetition_penalty=1.1, max_tokens=1000
        )
        
        answer_text = await self._generate_text(final_prompt, sampling_params)
        return OutputAnswer(answer=answer_text)
    
    async def make_context_prediction(self, req: InputRagQuestion) -> OutputAnswer:
        print("Got wiki query:", req.query)

        last_user_msg = next((m["content"] for m in reversed(req.query) if m.get("role") == "user"), "")
        
        # --- 1. Query Rewriting (Асинхронно) ---
        if len(req.query) <= 2: 
            search_query = last_user_msg
        else:
            history_msgs = req.query[-5:-1] 
            history_text = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in history_msgs])
            rewrite_prompt_messages = [
                {
                    "role": "system",
                    "content": "Ты — ИИ, оптимизирующий поисковые запросы. Проанализируй историю диалога и перепиши последний вопрос пользователя так, чтобы он стал самостоятельным и содержал все необходимые существительные (названия продуктов) вместо местоимений (он, это, туда). Верни ТОЛЬКО переформулированный вопрос, не отвечай на него."
                },
                {
                    "role": "user",
                    "content": f"История диалога:\n{history_text}\n\nПоследний вопрос: {last_user_msg}\n\nСамостоятельный поисковый запрос:"
                }
            ]
            rewrite_prompt = self.tokenizer.apply_chat_template(
                rewrite_prompt_messages, tokenize=False, add_generation_prompt=True
            )
            
            rewrite_sampling = SamplingParams(temperature=0.1, max_tokens=50)
            rewritten_result = await self._generate_text(rewrite_prompt, rewrite_sampling)
            search_query = rewritten_result.strip()
            
            print(f"Original query: {last_user_msg}")
            print(f"Rewritten search query: {search_query}")

        print("Searching for relevant documents using query:", search_query)
        version = "latest" if req.product_name == "zvirt" else req.product_name
        
        # --- 2. Поиск в Weaviate (Выносим в отдельный поток, чтобы не блокировать Event Loop) ---
        def fetch_docs():
            return self.nova_collection.query.hybrid(
                query=search_query,
                alpha=0.6,
                limit=5,
                filters=(
                    Filter.any_of([
                        Filter.all_of([
                            Filter.by_property("version").equal(req.product_version),
                            Filter.by_property("product").equal(req.product_name),
                        ]),
                        Filter.all_of([
                            Filter.by_property("version").equal(version),
                            Filter.any_of([
                                Filter.by_property("source").like("*solutions*"),
                                Filter.by_property("source").like("*knowledgebase*"),
                            ]),
                        ]),
                    ])
                ),
            )

        docs = await asyncio.to_thread(fetch_docs)
        print("Finished looking for documents")

        if not docs.objects:
            return OutputAnswer(answer="No relevant docs found")
        
        # --- 3. Подготовка контекста и финальная генерация (Асинхронно) ---
        texts_with_links = []
        for obj in docs.objects:
            text = obj.properties["page_content"]
            link = obj.properties["source"]
            texts_with_links.append(f"{text}\n\nИсточник: {link}")
        context = "\n\n---\n\n".join(texts_with_links)
        
        final_prompt = self.internal_rag_promt_template.format(
            question=req.query, context=context
        )
        
        print("Generating answer")
        final_sampling = SamplingParams(
            temperature=0.6, top_p=0.95, repetition_penalty=1.1, max_tokens=1000
        )
        final_answer = await self._generate_text(final_prompt, final_sampling)
        
        return OutputAnswer(answer=final_answer)

    def close(self):
        if hasattr(self, 'weaviate_connection') and self.weaviate_connection is not None:
            self.weaviate_connection.close()
            self.weaviate_connection = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


rag_reader_app = RAGReader.bind()

@serve.deployment
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
        print("Тело", body)

        model = body.get("model", "wiki-searcher")
        messages = body.get("messages", [])
        product_name = str(body.get("product_name", "nova"))
        product_version = str(body.get("product_version", "latest"))
        user_request = body.get("user_request", False)

        print(f"[OpenAIAdapter] model={model}, product_name={product_name}, product_version={product_version}")

        # Ray Serve корректно обрабатывает await для remote-вызовов
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
