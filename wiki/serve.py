from ray import serve
from transformers import pipeline
from fastapi import FastAPI
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from peft import PeftModelForCausalLM
import pyarrow.fs as pa_fs

MODEL_NAME = os.getenv(
    "READER_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
)
S3_CHECKPOINT_ADDRESS = os.getenv("S3_CHECKPOINT_ADDRESS")

checkpoint_path = "/home/ray/local_checkpoints/lora"

system_wiki_promt = [
    {
        "role": "system",
        "content": (
            "You are Nova Documentation Assistant trained ONLY on Nova Container "
            "Platform (NCP) docs (https://wiki.orionsoft.ru/nova/latest/). "
            "If the question is not strictly about Nova/NCP, answer exactly: "
            "\"Я могу отвечать только по документации Nova Container Platform.\" "
            "If you are not 100% sure the answer is in docs, say: "
            "\"По документации Nova у меня нет точного ответа на этот вопрос.\" "
            "Do NOT invent commands, components or installation steps."
        ),
    },
    {
        "role": "user",
        "content": "Вопрос по документации Nova: {question}",
    },
]

def download_checkpoint_from_s3():
    fs = pa_fs.S3FileSystem(
        endpoint_override="https://minio.nova-minio.svc",
        anonymous=True,
        tls_ca_file_path="/etc/ssl/certs/ca-certificates.crt",
    )
    print("Downloading remote checkpoint folder {S3_CHECKPOINT_ADDRESS} local...")
    pa_fs.copy_files(
        source=S3_CHECKPOINT_ADDRESS, 
        destination=checkpoint_path, 
        source_filesystem=fs
    )
    print("Checkpoint available at {checkpoint_path}")


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
        download_checkpoint_from_s3()
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        self.model = PeftModelForCausalLM.from_pretrained(self.model, os.path.join(checkpoint_path, "checkpoint"))
        self.model = self.model.merge_and_unload()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.pipe = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            do_sample=False,
            temperature=0.0,
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
