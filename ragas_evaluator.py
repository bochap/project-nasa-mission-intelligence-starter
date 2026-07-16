import asyncio
import inspect
import pandas as pd
from ragas.dataset import Dataset
from typing import Dict, List

from provider.plugins.openai_provider import RagasAsyncOpenAIBuilder
from provider.plugins.raga_provider import (
    AnswerRelevancyBuilder,
    BleuScoreBuilder,
    ContextPrecisionBuilder,
    FaithfulnessBuilder,
    RougeScoreBuilder,
)
from bootstrap import configure_ragas, configure_ragas_metric

# RAGAS imports
try:
    from ragas import experiment

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(openai_key: str, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # Initialize async evaluator clients from our dependency injection factory shortcuts
    client = (
        configure_ragas(RagasAsyncOpenAIBuilder.registered_name, RagasAsyncOpenAIBuilder)
        .with_api_key(openai_key)
        .build()
    )
    evaluator_llm = client.llm
    evaluator_embeddings = client.embeddings
    # Filter out empty contexts
    contexts = [ctx.strip() for ctx in contexts if ctx and ctx.strip()]

    # Note: Modern Ragas evaluations look for 'user_input' and 'response' rather than 'question'/'answer'
    evaluation_data = {
        "user_input": [question],
        "contexts": [contexts],
        "response": [answer],
        "reference": [answer],
        "retrieved_contexts": [contexts],
    }

    # Convert data into a lightweight, isolated internal Ragas in-memory evaluation table
    dataset = Dataset.from_pandas(pd.DataFrame(evaluation_data), name="evaluation_run", backend="inmemory")

    # Map out and configure target metrics using our application's registration ecosystem
    metrics = [
        configure_ragas_metric(BleuScoreBuilder.registered_name, BleuScoreBuilder).build(),
        configure_ragas_metric(ContextPrecisionBuilder.registered_name, ContextPrecisionBuilder)
        .with_llm(llm=evaluator_llm)
        .build(),
        configure_ragas_metric(AnswerRelevancyBuilder.registered_name, AnswerRelevancyBuilder)
        .with_llm(llm=evaluator_llm)
        .with_embedding(embeddings=evaluator_embeddings)
        .build(),
        configure_ragas_metric(FaithfulnessBuilder.registered_name, FaithfulnessBuilder)
        .with_llm(llm=evaluator_llm)
        .build(),
        configure_ragas_metric(RougeScoreBuilder.registered_name, RougeScoreBuilder).build(),
    ]

    # Define the experiment using the modern @experiment decorator
    @experiment()
    async def quality_experiment(row):
        parameters = {
            "user_input": row.get("user_input"),
            "contexts": row.get("contexts"),  # Maps retrieved_contexts to contexts
            "response": row.get("response"),
            "reference": row.get("reference"),
            "retrieved_contexts": row.get("retrieved_contexts"),  # Maps retrieved_contexts to contexts
        }

        # Gathering metrics dynamically via inspect loops saves hardcoding distinct param configurations
        tasks = []
        for metric in metrics:
            signature_parameters = inspect.signature(metric.ascore).parameters
            allowed_parameters = {key: value for key, value in parameters.items() if key in signature_parameters}

            # Crucial: Use metric.ascore() instead of .score() inside async loops to prevent context stalling
            tasks.append(metric.ascore(**allowed_parameters))

        # Fire off evaluation threads concurrently to optimize scoring latency
        scores = await asyncio.gather(*tasks, return_exceptions=True)
        evaluated_metrics = {}
        for metric, score in zip(metrics, scores):
            # Capture the calculated float values safely if execution completes successfully
            evaluated_metrics[metric.name] = score.value if score is not None else 0.0

        # Merge metrics back directly into the row dict container
        return {**row, **evaluated_metrics}

    # Evaluate the response using the metrics via the experiment runner
    try:
        experiment_results = asyncio.run(quality_experiment.arun(dataset))
        scores_df = experiment_results.to_pandas()

        # Compile the evaluation results mapped directly as a dictionary
        results = {}
        metric_names = [metric.name for metric in metrics]

        for name in metric_names:
            if name in scores_df.columns:
                val = scores_df[name].iloc[0]
                results[name] = float(val) if pd.notna(val) else 0.0
            else:
                results[name] = 0.0

        return results

    except Exception as e:
        return {"error": f"Experiment evaluation failed: {str(e)}"}
