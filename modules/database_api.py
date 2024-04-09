"""\
DEPRECATED WARNING:
langchain_community.vectorstores.elasticsearch is deprecated since langchain-community==0.0.27.
"""


from collections.abc import Iterable, Iterator
from elasticsearch import Elasticsearch
from langchain_community.document_loaders.blob_loaders import BlobLoader
from langchain_community.document_loaders.blob_loaders.file_system import (
    FileSystemBlobLoader,
)
from langchain_community.document_loaders.blob_loaders.schema import Blob
from langchain_community.document_loaders.parsers.txt import TextParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import VectorStore
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from tqdm import tqdm


class Database:
    def __init__(
        self,
        path: str,
        blob_loader: BlobLoader | None = None,
        blob_parser=None,
        text_splitter=None,
        embeddings=None,
        vectorstore: VectorStore | None = None,
        retriever=None,
    ) -> None:
        self.path = path
        self.blob_loader = (
            blob_loader if blob_loader else FileSystemBlobLoader(path=path)
        )
        self.blob_parser = blob_parser if blob_parser else TextParser()
        self.text_splitter = (
            text_splitter if text_splitter else RecursiveCharacterTextSplitter()
        )
        self.embeddings = (
            embeddings
            if embeddings
            else HuggingFaceEmbeddings(model_kwargs={"device": "cuda"})
        )
        if vectorstore:
            self.vectorstore = vectorstore
        else:
            es_connection = Elasticsearch("http://localhost:9200")
            self.vectorstore = ElasticsearchStore(
                embedding=self.embeddings,
                index_name="langchain",
                es_connection=es_connection,
            )
        self.retriever = retriever if retriever else self.vectorstore.as_retriever()

    def _file2blobs(self) -> Iterator[Blob]:
        for blob in self.blob_loader.yield_blobs():
            yield blob

    def _blob2document(self, blob: Blob) -> Iterator[Document]:
        yield from self.blob_parser.lazy_parse(blob)

    def _documents2chunks(self, documents: Iterable[Document]) -> list[Document]:
        return self.text_splitter.split_documents(documents)

    def _chunks2embeddings(self, chunks: Iterable[Document]) -> list[list[float]]:
        """Use _chunks2store() instead if possible. This method gets metadata lost."""
        return self.embeddings.embed_documents(
            list(map(lambda x: x.page_content, chunks))
        )

    def _chunks2store(self, chunks: list[Document]) -> None:
        self.vectorstore.add_documents(chunks)

    def store(self, path: str = "") -> None:
        if path != "":
            # TODO better way to init blob_loader?
            self.path = path
            self.blob_loader = FileSystemBlobLoader(path=path)
        for blob in tqdm(self._file2blobs()):
            for document in self._blob2document(blob):
                chunks = self._documents2chunks([document])
                self._chunks2store(chunks)

    def retrieve(self, query: str) -> list[Document]:
        return self.vectorstore.similarity_search(query)

    def retriever_as_tool(self) -> Tool:
        return create_retriever_tool(
            self.retriever,
            name=self.retriever.name or "Vectorstore Retriever",
            description="Tool created from retriever",
        )
