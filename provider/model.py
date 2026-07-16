from ragas.llms.base import InstructorBaseRagasLLM
from ragas.embeddings.base import BaseRagasEmbeddings, BaseRagasEmbedding


class RagasAsyncClient:
    """A clean domain object container holding the modern Ragas v0.4+ native models."""

    def __init__(self, llm: InstructorBaseRagasLLM, embeddings: BaseRagasEmbeddings | BaseRagasEmbedding):
        self.llm = llm
        self.embeddings = embeddings
