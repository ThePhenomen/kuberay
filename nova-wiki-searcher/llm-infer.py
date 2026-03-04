import os
import json
from typing import List, Dict, Any

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from ray import serve
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

MODEL_NAME = os.getenv(
    "READER_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
NOVA_COLLECTION_NAME = os.getenv(
    "NOVA_COLLECTION_NAME", 
    "NovaWikiDocs"
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
        "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
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
        "content": """You are Qwen, created by Alibaba Cloud. 
You are a helpful assistant. 
You are ready to answer every question you receive.
You will receive promts from OpenWebUI to generate context for users.
Answer very precisely only to provided questions""",
    },
    {
        "role": "user",
        "content": """Question: {question}""",
    },
]

class InputRagQuestion(BaseModel):
    query: str
    nova_version: str

class InputQuestion(BaseModel):
    query: str

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
    ray_actor_options={"num_cpus": 8},
)
class RAGReader:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16
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
            max_new_tokens=500,
        )

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
        
        self.nova_collection = self.weavite_connection.collections.use(NOVA_COLLECTION_NAME)

    def make_prediction(self, req: InputQuestion) -> OutputAnswer:
        print("Got query:", req.query)

        final_prompt = self.internal_promt_template.format(
            question=req.query
        )
        answer = self.pipe(final_prompt)
        return OutputAnswer(answer=answer[0]["generated_text"])

    def make_context_prediction(self, req: InputRagQuestion) -> OutputAnswer:
        print("Got query:", req.query)

        docs = self.nova_collection.query.hybrid(
            query=req.query,
            limit=3,
            filters=Filter.by_property("version").equal(req.nova_version),
        )

        if not docs.objects:
            return OutputAnswer(answer="Не нашёл релевантной документации для этого запроса.")

        texts = [obj.properties["page_content"] for obj in docs.objects]
        links = [obj.properties["source"] for obj in docs.objects]
        sources = "\n".join(links)
        context = "\n\n---\n\n".join(texts)
        final_prompt = self.internal_rag_promt_template.format(
            question=req.query, context=context
        )

        llm_answer = self.pipe(final_prompt)[0]["generated_text"]
        answer = llm_answer + f"\nИсточники:\n{sources}"
        return OutputAnswer(answer=answer)

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
        print(request.json())
        body: Dict[str, Any] = await request.json()
        print("Тело", body)

        model = body.get("model", "wiki-searcher")
        messages = body.get("messages", [])
        nova_version = str(body.get("nova_version", "latest"))
        user_request = body.get("user_request", False)

        user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), {}
        )
        query = user_msg.get("content", "")

        print(f"[OpenAIAdapter] model={model}, nova_version={nova_version}, query={query!r}")

        if user_request:
            resp: OutputAnswer = await self.rag.make_context_prediction.remote(
                InputRagQuestion(query=query, nova_version=nova_version)
            )
        else:
            resp: OutputAnswer = await self.rag.make_prediction.remote(
                InputQuestion(query=query)
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
