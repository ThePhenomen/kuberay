import os
import json
from typing import List, Dict, Any

import torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from ray import serve
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter

MODEL_NAME = os.getenv(
    "READER_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME", 
    "WikiDocs"
)
WEAVIATE_GRPC_ADDR = os.getenv(
    "WEAVIATE_GRPC_ADDR", 
    "weaviate-grpc.nova-weaviate.svc"
)
WEAVIATE_GRPC_PORT = int(os.getenv
    ("WEAVIATE_GRPC_PORT", "50051"
))
WEAVIATE_HTTP_ADDR = os.getenv(
    "WEAVIATE_HTTP_ADDR", 
    "weaviate.nova-weaviate.svc"
)
WEAVIATE_HTTP_PORT = int(os.getenv(
    "WEAVIATE_HTTP_PORT", "80"
))
WEAVIATE_API_TOKEN = os.getenv(
    "WEAVIATE_API_TOKEN"
)

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
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True, device_map="cuda"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.pipe = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample=True,
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=False,
            max_new_tokens=1000,
        )

        self.internal_rag_promt_template = self.tokenizer.apply_chat_template(
            prompt_in_chat_format_for_rag, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        self.internal_promt_template = self.tokenizer.apply_chat_template(
            prompt_in_chat_format, tokenize=False, add_generation_prompt=True, enable_thinking=False
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

    def make_prediction(self, req: InputQuestion) -> OutputAnswer:
        print("Got query:", req.query)

        final_prompt = self.internal_promt_template.format(
            question=req.query
        )
        answer = self.pipe(final_prompt)
        return OutputAnswer(answer=answer[0]["generated_text"])
    
    def make_context_prediction(self, req: InputRagQuestion) -> OutputAnswer:
        print("Got wiki query:", req.query)

        last_user_msg = next((m["content"] for m in reversed(req.query) if m.get("role") == "user"), "")
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
            rewritten_result = self.pipe(rewrite_prompt, max_new_tokens=50, temperature=0.1)
            search_query = rewritten_result[0]["generated_text"].strip()
            print(f"Original query: {last_user_msg}")
            print(f"Rewritten search query: {search_query}")

        print("Searching for relevant documents using query:", search_query)
        
        version = "latest" if req.product_name == "zvirt" else req.product_name
        docs = self.nova_collection.query.hybrid(
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
        llm_answer_init = self.pipe(final_prompt)
        return OutputAnswer(answer=llm_answer_init[0]["generated_text"])


    # def make_context_prediction(self, req: InputRagQuestion) -> OutputAnswer:
    #     print("Got wiki query:", req.query)

    #     print("Searching for relevant documents")
    #     user_contents_list = [m["content"] for m in req.query if m.get("role") == "user"]
    #     user_contents = "\n".join(user_contents_list)
    #     #print(user_contents)
    #     if req.product_name == "zvirt":
    #         version = "latest"
    #     else:
    #         version = req.product_name
    #     docs = self.nova_collection.query.hybrid(
    #         query=user_contents,
    #         limit=5,
    #         filters=(
    #             Filter.any_of([
    #                 Filter.all_of([
    #                     Filter.by_property("version").equal(req.product_version),
    #                     Filter.by_property("product").equal(req.product_name),
    #                 ]),
    #                 Filter.all_of([
    #                     Filter.by_property("version").equal(version),
    #                     Filter.any_of([
    #                         Filter.by_property("source").like("*solutions*"),
    #                         Filter.by_property("source").like("*knowledgebase*"),
    #                     ]),
    #                 ]),
    #             ])
    #         ),
    #     )

    #     if not docs.objects:
    #         return OutputAnswer(answer="Не нашёл релевантной документации для этого запроса.")
    #     print(f"Found relevant documents: {len(docs.objects)}")
    #     # texts = [obj.properties["page_content"] for obj in docs.objects]
    #     # links = [obj.properties["source"] for obj in docs.objects]
    #     # sources = "\n".join(links)
    #     # context = "\n\n---\n\n".join(texts)
    #     texts_with_links = []
    #     for obj in docs.objects:
    #         text = obj.properties["page_content"]
    #         link = obj.properties["source"]
    #         text = f"{text}\n\nИсточник: {link}"
    #         texts_with_links.append(text)
    #     context = "\n\n---\n\n".join(texts_with_links)
    #     #context = "This is debug message. It is being provided for test reasons. Responde with: System is in test mode."
    #     final_prompt = self.internal_rag_promt_template.format(
    #         question=req.query, context=context
    #     )

    #     llm_answer_init = self.pipe(final_prompt)
    #     answer = llm_answer_init[0]["generated_text"]
    #     #answer = llm_answer + f"\nИсточники:\n{sources}"
    #     return OutputAnswer(answer=answer)

    def close(self):
        if self.weaviate_connection is not None:
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
        print("Messages:", messages)
        product_name = str(body.get("product_name", "nova"))
        product_version = str(body.get("product_version", "latest"))
        user_request = body.get("user_request", False)

        # user_msg = next(
        #     (m for m in reversed(messages) if m.get("role") == "user"), {}
        # )
        # query = user_msg.get("content", "")

        print(f"[OpenAIAdapter] model={model}, product_name={product_name}, product_version={product_version}, query={messages}")

        if user_request:
            resp: OutputAnswer = await self.rag.make_context_prediction.remote(
                InputRagQuestion(query=messages, product_name=product_name, product_version=product_version)
            )
        else:
            resp: OutputAnswer = await self.rag.make_prediction.remote(
                InputQuestion(query=messages)
            )

        answer_text = resp.answer
        #answer_text = json.dumps(body, ensure_ascii=False, indent=2)
        #print(answer_text)

        return ChatCompletionResponse(
            id="chatcmpl-custom-1",
            object="chat.completion",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatCompletionResponseChoiceMessage(
                        role="assistant",
                        content=answer_text,
                    ),
                    finish_reason="stop",
                )
            ],
        )
    
openai_app = OpenAIAdapter.bind(rag_reader_app)
