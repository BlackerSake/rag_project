
import json
import hashlib

from data.kb_config import KBConfig
from data.stores import StoreManager

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

class DocumentManager:
    def __init__(self, config: KBConfig , stores: StoreManager, cache):
        self.config = config
        self.stores = stores
        self.cache = cache
    def _generate_doc_id(self, doc: Document) -> str:
        """生成文档ID"""
        payload = doc.page_content + json.dumps(doc.metadata, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    def add_documents(self, file_paths):
        """添加文档到知识库"""
        self.stores.ensure_connected()

        documents = []
        for file_path in file_paths:
            loader = TextLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["doc_id"] = self._generate_doc_id(doc)
            documents.extend(docs)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        for doc in split_docs:
            if "doc_id" not in doc.metadata:
                doc.metadata["doc_id"] = self._generate_doc_id(doc)

        self.stores.vector_store.add_documents(split_docs)
        self.stores.elasticsearch_store.add_documents(split_docs)

        self.cache.clear()

        return len(split_docs)

    def add_faq(self, faq_items):
        """添加FAQ到知识库"""
        self.stores.ensure_connected()

        documents = []
        for ques, anws in faq_items.items():
            content = f"问题: {ques}\n答案: {anws}"
            doc = Document(
                page_content=content,
                metadata={"type": "faq", "question": ques}
            )
            doc.metadata["doc_id"] = self._generate_doc_id(doc)
            documents.append(doc)

        self.stores.vector_store.add_documents(documents)
        self.stores.elasticsearch_store.add_documents(documents)

        self.cache.clear()

        return len(documents)
