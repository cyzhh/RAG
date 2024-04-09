from langchain_core.callbacks.base import Callbacks
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from .search_utils import SEARCH_ENGINES, web_search
from typing import Any, Dict, List, Optional


class WebSearchRetriever(BaseRetriever):
    k: int = 3
    search_engine: SEARCH_ENGINES = "duckduckgo"

    class Config:
        arbitrary_types_allowed = True

    def _dict2doc(self, search_result: dict):
        doc = Document(
            page_content=(
                search_result.get("title", "") + "\n" + search_result.get("body", "")
            ),
            metadata={
                "source": search_result.get("title", ""),
                "href": search_result.get("href", ""),
            },
        )
        return doc

    def get_relevant_documents(self, query: str) -> List[Document]:
        results_list = web_search(
            search=query,
            search_engine=self.search_engine,
            max_results=self.k,
        )
        docs = [self._dict2doc(r) for r in results_list]
        return docs


class BaiduWebSearchRetriever(WebSearchRetriever):
    def __init__(self):
        super().__init__()
        self.search_engine = "baidu"


class DuckduckgoWebSearchRetriever(WebSearchRetriever):
    def __init__(self):
        super().__init__()
        self.search_engine = "duckduckgo"


class WikipediaWebSearchRetriever(WebSearchRetriever):
    def __init__(self):
        super().__init__()
        self.search_engine = "wikipedia"
