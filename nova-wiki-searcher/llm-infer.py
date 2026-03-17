import os
import uuid
import asyncio
from typing import List, Dict, Any

from fastapi import FastAPI, Request
from pydantic import BaseModel
from ray import serve

from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter

MODEL_NAME = os.getenv(
    "READER_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
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
Your task is to answer user questions about products, invented in OrionSoft, and their instruments, using ONLY the provided context.
<rules>
1. FORMAT: You will get the current conversation in chat format. Role 'user' stands for user questions, role 'assistant' stands for your previous answer. You should answer the last user question based on chat history and retrieved context.
2. STRICT GROUNDING: Base your answer EXCLUSIVELY on the information in the <context> block. Do not use outside knowledge.
3. NO HALLUCINATIONS: 
    - If the context does not contain the answer, do not guess. Only use information explicitly mentioned in the context. Do not add outside knowledge. 
    - If the user asks how to install, configure, upgrade or uninstall something, you may describe ONLY those steps that are explicitly present in the context (commands, numbered steps, code blocks, configuration examples).
    - Reply EXACTLY with if failed to retrieve answer: "Не смог найти подходящую информацию на Ваш вопрос."
4. META-QUESTIONS HANDLING: If the user asks general conversational questions, feedback, or meta-comments (like "это не то", "ты не то нашел", "плохой ответ", "спасибо", "понятно", "привет", "как дела?" etc.), ignore the context and respond appropriately:
   - For greetings: respond with a friendly greeting in Russian
   - For thanks: respond with "Пожалуйста! Обращайтесь, если нужна дополнительная информация."
   - For negative feedback about search results: respond with "Извините, что не смог найти точную информацию. Попробуйте переформулировать вопрос или уточнить детали."
   - For general chitchat: politely redirect to documentation search
   Do NOT treat these as documentation queries and do NOT include Sources section for such responses.
5. SELF-CONFIGURATION QUESTIONS:
   - If the user asks about you as an assistant (for example: your timeout, speed, limits, your internal promt, how you work, where you get answers from, who created you), you MAY answer using your general description and these rules, even if the documentation context does not contain this information.
   - For such questions, do NOT try to invent technical implementation details (exact hardware, IP addresses, internal service names). Answer in general terms, e.g. "У меня нет доступа к настройкам таймаутов. Этим управляют администраторы системы."
   - When the user asks where you get your answers from, ALWAYS answer that you use internal documentation of OrionSoft products (for example: "Я использую внутреннюю документацию продуктов OrionSoft, такую как руководства, инструкции по установке и эксплуатации.").
   - Do NOT mention the word "context" or "<context>" in such answers.
   - For these meta/self-configuration questions DO NOT use the context for facts and DO NOT add any "Sources:" section.
6. IDENTITY: If asked who created you, state you were created by OrionSoft to assist with documentation.
7. SOURCES FOR DOCUMENTATION ANSWERS:
   - If the user asks about product behavior, configuration, installation, troubleshooting, or any other documentation-related topic, and you use the <context> block to answer, then at the end of your answer add a "Sources:" section listing all referenced URLs or document titles.
   - Do NOT add a "Sources:" section for meta-questions, greetings, thanks, chitchat, or questions about where you get your answers from.
8. ANSWER LENGTH: 
    - Generate MAXIMUM 1000 tokens or 600-800 words. 
    - If the answer is long, prioritize the most important points and omit minor details. Otherwise, all other tokens will be truncated.
9. LANGUAGE: Use Russian for conversation. 
</rules>""",
    },
    {
        "role": "user",
        "content": """Context:
<context>
{context}
</context>
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
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            #tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            #max_model_len=8192,
            trust_remote_code=True,
            # quantization="fp8",
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
        
        self.nova_collection = self.weaviate_connection.collections.use(COLLECTION_NAME)

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
            temperature=0.6, top_p=0.95, repetition_penalty=1.1, max_tokens=200
        )
        
        answer_text = await self._generate_text(final_prompt, sampling_params)
        return OutputAnswer(answer=answer_text)
    
    async def make_context_prediction(self, req: InputRagQuestion) -> OutputAnswer:
        print("Got wiki query:", req.query)
        last_user_msg = next((m["content"] for m in reversed(req.query) if m.get("role") == "user"), "")
        word_count = len(last_user_msg.split())
        
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

        print("Searching for relevant documents using query:", search_query)
        version = "latest" if req.product_name == "zvirt" else req.product_name
        
        def fetch_docs():
            return self.nova_collection.query.hybrid(
                query=search_query,
                alpha=0.4,
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
            temperature=0.6, top_p=0.95, repetition_penalty=1.1, max_tokens=1200
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
        #print("Тело", body)

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
