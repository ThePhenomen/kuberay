# file: rag_embedder_service.py
import re
import os
from typing import List

from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel

from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_postgres import PGVector
from transformers import AutoTokenizer
from langchain_core.documents import Document as LangchainDocument
from typing import Optional, List
from sentence_transformers import SentenceTransformer
import torch


PG_CONN = os.getenv("PG_CONN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "intfloat/multilingual-e5-large-instruct",
)

_env_chunk_size = os.getenv("CHUNK_SIZE")
if _env_chunk_size is not None:
    CHUNK_SIZE = int(_env_chunk_size)
else:
    CHUNK_SIZE = SentenceTransformer(EMBEDDING_MODEL_NAME).max_seq_length

_env_docs_to_retrieve = os.getenv("CHUNK_SIZE")
if _env_docs_to_retrieve is not None:
    RETRIEVED_DOCS = int(_env_docs_to_retrieve)
else:
    RETRIEVED_DOCS = 3

def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

def split_documents(
    docs_to_parse: List[LangchainDocument],
    tokenizer: AutoTokenizer,
    chunk_size: Optional[int] = CHUNK_SIZE,
) -> List[LangchainDocument]:
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
    )

    docs_processed = []
    for doc in docs_to_parse:
        docs_processed += text_splitter.split_documents([doc])

    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

app = FastAPI()

class EmbedRequest(BaseModel):
    url: str

class EmbedResponse(BaseModel):
    url: str
    added_chunks: int

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    retrieved_docs: List[LangchainDocument]

@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 4, "num_gpus": 1},
)
@serve.ingress(app)
class RAGEmbedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

        st_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            model_kwargs={"torch_dtype": torch.float16},
        )

        self.embeddings = HuggingFaceEmbeddings(
            model = st_model,
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=COLLECTION_NAME,
            connection=PG_CONN,
            use_jsonb=True,
        )

        self.retrieved_docs = RETRIEVED_DOCS

    @app.post("/embed-url", response_model=EmbedResponse)
    async def embed_url(self, req: EmbedRequest) -> EmbedResponse:
        print("URL to embed:", req.url)

        loader = RecursiveUrlLoader(req.url, extractor=bs4_extractor)
        docs = loader.load()

        if not docs:
            return EmbedResponse(url=req.url, added_chunks=0)

        for d in docs:
            d.metadata = {"source": d.metadata.get("source")}
        
        print("Pages to parse:", len(docs))

        docs_processed = split_documents(
            docs_to_parse=docs,
            tokenizer=self.tokenizer
        )

        self.vector_store.add_documents(docs_processed)

        return EmbedResponse(url=req.url, added_chunks=len(docs_processed))
    
    @app.post("/search", response_model=SearchResponse)
    async def search(self, req: SearchRequest) -> SearchResponse:
        print("Starting retrieval for query:", req.query)

        retrieved_docs = self.vector_store.similarity_search(query=req.query, k=self.retrieved_docs)

        return  SearchResponse(retrieved_docs = retrieved_docs)

rag_embedder_app = RAGEmbedder.bind()
