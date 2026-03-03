# app.py
import os
import json
from typing import List, Dict, Any

import requests
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from ray import serve
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


READER_MODEL_NAME = os.getenv(
    "READER_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
SEARCH_EMBEDDER_URL = os.getenv(
    "SEARCH_EMBEDDER_URL",
    "http://wiki-searcher.nova-wiki.svc.cluster.local:8000/search",
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


# ----- внутренние модели -----

class InputQuestion(BaseModel):
    query: str


class OutputAnswer(BaseModel):
    answer: str


# ----- OpenAI-совместимые модели -----

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


# Один общий FastAPI для всего приложения
app = FastAPI()


# ----- RAGReader deployment -----

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 8, "num_gpus": 1},
)
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

    def make_prediction(self, req: InputQuestion) -> OutputAnswer:
        print("Got query:", req.query)

        final_prompt = self.internal_promt_template.format(
            question=req.query
        )
        answer = self.pipe(final_prompt)
        return OutputAnswer(answer=answer[0]["generated_text"])

    def make_context_prediction(self, req: InputQuestion) -> OutputAnswer:
        print("Got query:", req.query)

        url = SEARCH_EMBEDDER_URL
        payload = {"query": req.query}
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Embedder returned {response.status_code}: {response.text}",
            )

        retrieved_docs = [d["page_content"] for d in response.json()["retrieved_docs"]]
        context = "\nExtracted documents:\n"
        context += "".join(
            [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs)]
        )
        final_prompt = self.internal_rag_promt_template.format(
            question=req.query, context=context
        )

        answer = self.pipe(final_prompt)
        return OutputAnswer(answer=answer[0]["generated_text"])


rag_reader_app = RAGReader.bind()


# ----- OpenAI-совместимый адаптер -----

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
                    "id": "qwen-wiki",
                    "object": "model",
                    "owned_by": "custom",
                }
            ],
        }

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(self, request: Request):
        print(request.json())
        body: Dict[str, Any] = await request.json()
        print("Тело", body)

        model = body.get("model", "qwen-wiki")
        messages = body.get("messages", [])
        nova_version = body.get("nova_version", "latest")

        # последний user message
        user_msg = next(
            (m for m in reversed(messages) if m.get("role") == "user"), {}
        )
        query = user_msg.get("content", "")

        print(f"[OpenAIAdapter] model={model}, nova_version={nova_version}, query={query!r}")

        # if model == "qwen-wiki" or "wiki" in model:
        #     # вызываем wiki-question через Ray Serve handle
        #     resp: OutputAnswer = await self.rag.make_context_prediction.remote(
        #         InputQuestion(query=query)
        #     )
        # else:
        #     resp: OutputAnswer = await self.rag.make_prediction.remote(
        #         InputQuestion(query=query)
        #     )

        #answer_text = resp.answer
        answer_text = json.dumps(body, ensure_ascii=False, indent=2)
        print(answer_text)

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
