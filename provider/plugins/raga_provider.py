from ragas.metrics.collections import AnswerRelevancy, BleuScore, ContextPrecision, Faithfulness, RougeScore
from ragas.embeddings import BaseRagasEmbedding
from ragas.llms import InstructorBaseRagasLLM
from typing import Self

from provider.protocol import BaseBuilder
from provider.builder_registry import BuilderRegistry


@BuilderRegistry().register_ragas_metric("bleu_score")
class BleuScoreBuilder(BaseBuilder[BleuScore]):
    registered_name = "bleu_score"

    def build(self) -> BleuScore:
        return BleuScore()


@BuilderRegistry().register_ragas_metric("context_precision")
class ContextPrecisionBuilder(BaseBuilder[ContextPrecision]):
    registered_name = "context_precision"

    def __init__(self) -> None:
        self.llm = None

    def with_llm(self, llm: InstructorBaseRagasLLM) -> Self:
        self.llm = llm
        return self

    def build(self) -> ContextPrecision:
        if not self.llm:
            raise ValueError("ContextPrecision requires an InstructorBaseRagasLLM instance")
        return ContextPrecision(llm=self.llm)


@BuilderRegistry().register_ragas_metric("answer_relevancy")
class AnswerRelevancyBuilder(BaseBuilder[AnswerRelevancy]):
    registered_name = "answer_relevancy"

    def __init__(self) -> None:
        self.llm = None
        self.embeddings = None
        self.strictness = 3

    def with_llm(self, llm: InstructorBaseRagasLLM) -> Self:
        self.llm = llm
        return self

    def with_embedding(self, embeddings: BaseRagasEmbedding) -> Self:
        self.embeddings = embeddings
        return self

    def with_strictness(self, strictness: int) -> Self:
        self.strictness = strictness
        return self

    def build(self) -> AnswerRelevancy:
        if not self.llm:
            raise ValueError("AnswerRelevancy requires an InstructorBaseRagasLLM instance")
        return AnswerRelevancy(llm=self.llm, embeddings=self.embeddings, strictness=self.strictness)


@BuilderRegistry().register_ragas_metric("faithfulness")
class FaithfulnessBuilder(BaseBuilder[Faithfulness]):
    registered_name = "faithfulness"

    def __init__(self) -> None:
        self.llm = None

    def with_llm(self, llm: InstructorBaseRagasLLM) -> Self:
        self.llm = llm
        return self

    def build(self) -> Faithfulness:
        if not self.llm:
            raise ValueError("Faithfulness requires an InstructorBaseRagasLLM instance")
        return Faithfulness(llm=self.llm)


@BuilderRegistry().register_ragas_metric("rouge_score")
class RougeScoreBuilder(BaseBuilder[RougeScore]):
    registered_name = "rouge_score"

    def __init__(self) -> None:
        self.rouge_type = "rougeL"
        self.mode = "fmeasure"

    def with_rouge_type(self, rouge_type: str) -> Self:
        self.rouge_type = rouge_type
        return self

    def with_mode(self, mode: str) -> Self:
        self.mode = mode
        return self

    def build(self) -> RougeScore:
        return RougeScore(rouge_type=self.rouge_type, mode=self.mode)
