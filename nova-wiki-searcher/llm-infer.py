from ray import serve
from transformers import pipeline
from fastapi import FastAPI, HTTPException, Request
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from langchain_core.documents import Document as LangchainDocument
import requests
import torch


READER_MODEL_NAME = os.getenv(
    "READER_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
SEARCH_EMBEDDER_URL = os.getenv(
    "SEARCH_EMBEDDER_URL",
    "http://127.0.0.1:8000/search",
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
        "content": """You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You are ready to answer every question you receive.""",
    },
    {
        "role": "user",
        "content": """Question: {question}""",
    },
]


app = FastAPI()


class InputQuestion(BaseModel):
    query: str


class OutputAnswer(BaseModel):
    answer: str


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 8, "num_gpus": 1},
)
@serve.ingress(app)
class RAGReader:

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            READER_MODEL_NAME, torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
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

    @app.post("/question", response_model=OutputAnswer)
    def make_prediction(self, req: InputQuestion) -> OutputAnswer:
        print("Got query:", req.query)

        final_promt = self.internal_promt_template.format(
            question=req.query
        )

        answer = self.pipe(final_promt)
        return OutputAnswer(answer=answer[0]["generated_text"])

    @app.post("/wiki-question", response_model=OutputAnswer)
    async def make_context_prediction(self, request: Request) -> OutputAnswer:
        """
        Обрабатываем тело запроса от OpenWebUI целиком, чтобы вытащить metadata.nova_version.
        Ожидается формат body, который ты привёл в примере.
        """
        body = await request.json()

        # 1. Извлекаем вопрос и версию
        # Если ты по-прежнему хочешь использовать InputQuestion, можно взять query из messages
        messages = body.get("messages", [])
        # берём последний user-message
        user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), {}
        )
        query = user_msg.get("content", "")

        metadata = body.get("metadata", {}) or {}
        nova_version = metadata.get("nova_version", "latest")

        print(f"Got query: {query}")
        print(f"nova_version from metadata: {nova_version}")

        return OutputAnswer(answer=body)


rag_reader_app = RAGReader.bind()
