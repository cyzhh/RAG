"""\
DEPRECATED WARNING:
langchain_community.vectorstores.elasticsearch is deprecated since langchain-community==0.0.27.
"""


from dotenv import dotenv_values
from elasticsearch import Elasticsearch
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from modules.database_api import Database, FileSystemBlobLoader
from modules.json_parser import JsonParser

config = dotenv_values("./.env")
es = Elasticsearch(
    config["ES_HOST"],
    http_auth=(
        (
            config["ES_USER"],
            config["ES_SECRET"],
        )
    ),
)

embedding = HuggingFaceEmbeddings(model_kwargs={"device": "cuda:0"})
es_store = ElasticsearchStore(
    index_name="chem_papers.en.240229-10k", embedding=embedding, es_connection=es
)
content_func = lambda x: [i["paragraph"] for i in x["paragraphs"]]
metadata_func = lambda x: [
    {"title": x["title"], "author": x["authors"], "paragraph_id": i["paragraph_idx"]}
    for i in x["paragraphs"]
]
path = "./data/240229-10k/"
fs_blob_loader = FileSystemBlobLoader(path=path, glob="**/*.json")
json_parser = JsonParser(content_func, metadata_func)
db = Database(
    path=path, blob_parser=json_parser, vectorstore=es_store
)
db.store()
