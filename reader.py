from ray import serve
from transformers import pipeline
from fastapi import FastAPI, HTTPException
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from langchain_core.documents import Document as LangchainDocument
import requests

READER_MODEL_NAME = os.getenv(
    "READER_MODEL_NAME",
    "Qwen/Qwen2.5-3B-Instruct",
)
SEARCH_EMBDDER_URL = os.getenv(
    "SEARCH_EMBDDER_URL",
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
    ray_actor_options={"num_cpus": 4, "num_gpus": 1},
)
@serve.ingress(app)
class RAGReader:

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)
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

        self.internal_rag_promt_template = self.tokenizer.apply_chat_template(prompt_in_chat_format_for_rag, tokenize=False, add_generation_prompt=True)
        self.internal_promt_template = self.tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True)

    @app.post("/question", response_model=OutputAnswer)
    def make_prediction(self, req: InputQuestion) -> OutputAnswer:
        print("Got query:", req.query)
        
        final_promt = self.internal_promt_template.format(
            question=req.query
        )

        answer = self.pipe(final_promt)
        return OutputAnswer(answer=answer[0]["generated_text"])

    @app.post("/wiki-question", response_model=OutputAnswer)
    def make_context_prediction(self, req: InputQuestion) -> OutputAnswer:
        print("Got query:", req.query)
        
        url = SEARCH_EMBDDER_URL
        payload = {
            "query": f"{req.query}"
        }
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
        final_promt = self.internal_promt_template.format(
            question=req.query, context=context
        )

        answer = self.pipe(final_promt)
        return OutputAnswer(answer=answer[0]["generated_text"])

rag_reader_app = RAGReader.bind()
