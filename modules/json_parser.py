from collections.abc import Callable, Iterator
from langchain_community.document_loaders import Blob
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_core.documents import Document
import json


class JsonParser(BaseBlobParser):
    """
    A custom implementation of json parser.
    """

    def __init__(
        self,
        content_func: Callable[[dict | list], str | list[str]],
        metadata_func: Callable[[dict | list], dict | list[dict]],
    ) -> None:
        super().__init__()
        self.content_func = content_func
        self.metadata_func = metadata_func

    def _align_documents(
        self, contents: str | list[str], metadatas: dict | list[dict]
    ) -> Iterator[Document]:
        if type(contents) == str:
            if type(metadatas) == dict:
                yield Document(page_content=contents, metadata=metadatas)
            else:
                raise TypeError(
                    f"metadatas must be dict not {type(metadatas)} if contents is of type str"
                )
        elif type(metadatas) == dict:
            for content in contents:
                yield Document(page_content=content, metadata=metadatas)
        elif len(contents) == len(metadatas):
            for i in range(len(contents)):
                yield Document(page_content=contents[i], metadata=metadatas[i])
        else:
            raise ValueError(
                f"len(contents) and len(metadatas) not aligned: {len(contents)}, {len(metadatas)}"
            )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """
        **Warning: This method reads the entire blob into memory.**

        This is not true *lazy* parse. This method name is used for
        compatibility with other workflows.
        """
        if blob.path:
            with open(blob.path, "r") as f:
                data = json.load(f)
                contents = self.content_func(data)
                metadatas = self.metadata_func(data)
                yield from self._align_documents(contents, metadatas)
