from dotenv import dotenv_values, load_dotenv
from elasticsearch import Elasticsearch
from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from .search_utils import SEARCH_ENGINES, web_search
from typing import Any, Dict, List, Optional


config = dotenv_values()
es = Elasticsearch(
            config["ES_HOST"],
            http_auth=(
                (
                    config["ES_USER"],
                    config["ES_SECRET"],
                )
            ),
        )


class ESQueryRetriever(BaseRetriever):
    index = "chem_papers.en.*"
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, index):
        super().__init__()
        self.index = index

    def _es_search(self, query) -> Dict:
        results = es.search(
            index=self.index,
            body={"query": {"match": {"metadata.title": query}}}
        )
        return results

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results_list = self._es_search(
            query=query,
        )["hits"]["hits"]
        docs = [Document(page_content=i["_source"]["text"], metadata=i["_source"]["metadata"]) for i in results_list]
        return docs
