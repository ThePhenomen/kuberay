from ray import serve
from transformers import pipeline
from fastapi import FastAPI
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from peft import PeftModel
import pyarrow.fs as pa_fs

MODEL_NAME = os.getenv(
    "READER_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
S3_CHECKPOINT_ADDRESS = os.getenv("S3_CHECKPOINT_ADDRESS")

checkpoint_path = "/mnt/ray/lora"

system_wiki_promt = [
    {
        "role": "system",
        "content": """You are Nova Documentation Assistant, trained exclusively on Nova Container Platform (NCP) documentation. 
You were fine-tuned on documentation of Nova Container Platform (https://wiki.orionsoft.ru/nova/latest/).
If you receive a question about Nova (or NCP, Nova Container Platform) respond only using your information from training.
If the answer about Nova cannot be deduced from you information, do not come up with an answer.
If question is NOT about Nova/NCP. do not give an answer.
Use TECHNICAL language, include YAML/config examples when relevant""",
    },
    {
        "role": "user",
        "content": """Question: {question}""",
    },
]

def download_checkpoint_from_s3():
    fs = pa_fs.S3FileSystem(
        endpoint_override="https://minio.nova-minio.svc",
        anonymous=True,
        tls_ca_file_path="/etc/ssl/certs/ca-certificates.crt",
    )
    pa_fs.copy_files(S3_CHECKPOINT_ADDRESS, checkpoint_path, fs)

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
class WIKIHelper:

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.model = PeftModel.from_pretrained(self.model, os.path.join(checkpoint_path, "checkpoint"))

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

        self.internal_promt_template = self.tokenizer.apply_chat_template(system_wiki_promt, tokenize=False, add_generation_prompt=True)

    @app.post("/question", response_model=OutputAnswer)
    def make_prediction(self, req: InputQuestion) -> OutputAnswer:
        print("Got query:", req.query)
        
        final_promt = self.internal_promt_template.format(
            question=req.query
        )

        answer = self.pipe(final_promt)
        return OutputAnswer(answer=answer[0]["generated_text"])

wiki_helper_app = WIKIHelper.bind()
