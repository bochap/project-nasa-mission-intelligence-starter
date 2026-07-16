#!/usr/bin/env python3
"""
NASA RAG Chat System - Multi-Mission Evaluation Dataset Runner

This script automates the validation workflow by executing a mixed suite of test cases
covering Apollo 11, Apollo 13, and Challenger. It queries the vector database using metadata
filters, generates factual responses, and calculates performance scores.
"""

import argparse
import json
import logging
from pathlib import Path
import os
import sys

from dotenv import load_dotenv
load_dotenv()


# 1. Path correction (grabs project root folder where bootstrap.py lives)
sys.path.insert(0, str(Path(__file__).resolve().parent))


# 2. Async and environment setups
import nest_asyncio
nest_asyncio.apply()


import llm_client
import rag_client
import ragas_evaluator

logger = logging.getLogger("rag_system.evaluation_runner")


def calculate_metrics_summary(results):
    """
    Computes summary statistics comparing Filtered vs. Unfiltered retrieval modes.
    
    Calculates overall averages, absolute differences, and percentage changes
    for each RAGAS metric, as well as mission-specific overall averages.
    
    Args:
        results (list): A list of dictionaries, where each dict contains test case info
            and the scores for "Filtered" and "Unfiltered" modes.
            
    Returns:
        tuple: A tuple containing:
            - metrics (list): List of metric names.
            - metric_labels (dict): Mapping of metric names to display labels.
            - overall_summary (dict): Overall averages and differences for each metric.
            - missions (list): Sorted list of unique mission names.
            - mission_summary (dict): Mission-wise averages and differences.
    """
    # Define the quality evaluation metrics to aggregate
    metrics = ["faithfulness", "answer_relevancy", "rouge_score", "bleu_score", "context_precision"]
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "rouge_score": "Rouge-L F1",
        "bleu_score": "Bleu-4 Precision",
        "context_precision": "Context Precision"
    }

    # Initialize aggregators for all metrics overall
    overall_filtered = {m: [] for m in metrics}
    overall_unfiltered = {m: [] for m in metrics}

    # Extract all distinct missions present in the test results
    missions = sorted(list(set(r["test_case"]["mission"] for r in results)))
    mission_filtered = {mission: {m: [] for m in metrics} for mission in missions}
    mission_unfiltered = {mission: {m: [] for m in metrics} for mission in missions}

    # Populate raw scores from results, skipping test cases with evaluation errors
    for r in results:
        mission = r["test_case"]["mission"]
        filtered_scores = r["modes"]["Filtered"]["scores"]
        unfiltered_scores = r["modes"]["Unfiltered"]["scores"]

        for m in metrics:
            if "error" not in filtered_scores:
                val_f = filtered_scores.get(m, 0.0)
                overall_filtered[m].append(val_f)
                mission_filtered[mission][m].append(val_f)
            if "error" not in unfiltered_scores:
                val_u = unfiltered_scores.get(m, 0.0)
                overall_unfiltered[m].append(val_u)
                mission_unfiltered[mission][m].append(val_u)

    # Helper function to compute the arithmetic mean
    def get_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # Compute overall metric averages and compare filtered vs unfiltered
    overall_summary = {}
    for m in metrics:
        avg_f = get_avg(overall_filtered[m])
        avg_u = get_avg(overall_unfiltered[m])
        diff = avg_f - avg_u
        rel = (diff / avg_u * 100) if avg_u > 0 else 0.0
        overall_summary[m] = {"filtered": avg_f, "unfiltered": avg_u, "diff": diff, "rel": rel}

    # Compute mission-specific averages aggregated across all metrics
    mission_summary = {}
    for mission in missions:
        all_f = []
        all_u = []
        for m in metrics:
            all_f.extend(mission_filtered[mission][m])
            all_u.extend(mission_unfiltered[mission][m])
        avg_f = get_avg(all_f)
        avg_u = get_avg(all_u)
        diff = avg_f - avg_u
        rel = (diff / avg_u * 100) if avg_u > 0 else 0.0
        mission_summary[mission] = {"filtered": avg_f, "unfiltered": avg_u, "diff": diff, "rel": rel}

    return metrics, metric_labels, overall_summary, missions, mission_summary


def print_comparison_summary(results):
    """
    Prints a formatted comparison table of Filtered vs. Unfiltered modes to standard output.
    
    Shows both the overall metric-by-metric comparison and the mission-wise comparison.
    
    Args:
        results (list): A list of evaluation results dictionaries containing scores.
    """
    metrics, metric_labels, overall, missions, mission_wise = calculate_metrics_summary(results)

    print("\n" + "=" * 80)
    print("                    RAG Filter Comparison Summary")
    print("=" * 80)

    # Display overall metrics table
    print("\n1. Overall Metrics Comparison")
    print("-" * 80)
    print(f"{'Metric':<20} | {'Filtered':<10} | {'Unfiltered':<10} | {'Diff':<8} | {'% Change':<8}")
    print("-" * 80)
    for m in metrics:
        data = overall[m]
        sign = "+" if data['diff'] >= 0 else ""
        rel_str = f"{sign}{data['rel']:.1f}%" if data['unfiltered'] > 0 else "N/A"
        print(f"{metric_labels[m]:<20} | {data['filtered']:^10.3f} | {data['unfiltered']:^10.3f} | {data['diff']:^+8.3f} | {rel_str:^8}")
    print("-" * 80)

    # Display mission-wise metrics table
    print("\n2. Mission-Wise Comparison (Average Across All Metrics)")
    print("-" * 80)
    print(f"{'Mission':<20} | {'Filtered':<10} | {'Unfiltered':<10} | {'Diff':<8} | {'% Change':<8}")
    print("-" * 80)
    for mission in missions:
        data = mission_wise[mission]
        sign = "+" if data['diff'] >= 0 else ""
        rel_str = f"{sign}{data['rel']:.1f}%" if data['unfiltered'] > 0 else "N/A"
        print(f"{mission:<20} | {data['filtered']:^10.3f} | {data['unfiltered']:^10.3f} | {data['diff']:^+8.3f} | {rel_str:^8}")
    print("-" * 80)
    print("=" * 80 + "\n")


def save_comparison_markdown(results, output_path: str):
    """
    Generates and saves a detailed Markdown report comparing Filtered and Unfiltered modes.
    
    The report contains summary tables for overall metric comparisons, mission-wise averages,
    and a detailed run-by-run table for every test case.
    
    Args:
        results (list): A list of evaluation results dictionaries containing scores.
        output_path (str): File system path where the Markdown report will be saved.
    """
    metrics, metric_labels, overall, missions, mission_wise = calculate_metrics_summary(results)

    md = []
    md.append("# NASA RAG Filter Comparison Report\n")
    md.append("This report compares the performance of the RAG system with and without mission-specific metadata filters.\n")

    # Construct the Markdown table for overall metrics
    md.append("## 1. Overall Metrics Comparison\n")
    md.append("| Metric | Filtered Avg | Unfiltered Avg | Absolute Difference | Relative Change |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for m in metrics:
        data = overall[m]
        sign = "+" if data['diff'] >= 0 else ""
        rel_str = f"{sign}{data['rel']:.1f}%" if data['unfiltered'] > 0 else "N/A"
        md.append(f"| {metric_labels[m]} | {data['filtered']:.3f} | {data['unfiltered']:.3f} | {sign}{data['diff']:.3f} | {rel_str} |")

    # Construct the Markdown table for mission-wise metrics
    md.append("\n## 2. Mission-Wise Comparison (Average Across All Metrics)\n")
    md.append("| Mission | Filtered Avg | Unfiltered Avg | Absolute Difference | Relative Change |")
    md.append("| :--- | :---: | :---: | :---: | :---: |")
    for mission in missions:
        data = mission_wise[mission]
        sign = "+" if data['diff'] >= 0 else ""
        rel_str = f"{sign}{data['rel']:.1f}%" if data['unfiltered'] > 0 else "N/A"
        md.append(f"| {mission} | {data['filtered']:.3f} | {data['unfiltered']:.3f} | {sign}{data['diff']:.3f} | {rel_str} |")

    # Construct the detailed test case results Markdown table
    md.append("\n## 3. Detailed Results per Test Case\n")
    md.append("| ID | Mission | Question | Mode | Avg Distance | Faithfulness | Relevancy | Rouge-L | Bleu-4 | Context Precision |")
    md.append("| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        tc = r["test_case"]
        for mode_name in ["Filtered", "Unfiltered"]:
            mode_data = r["modes"][mode_name]
            sc = mode_data["scores"]
            md.append(f"| {tc['id']} | {tc['mission']} | {tc['question']} | {mode_name} | {mode_data['avg_distance']:.4f} | {sc.get('faithfulness', 0.0):.3f} | {sc.get('answer_relevancy', 0.0):.3f} | {sc.get('rouge_score', 0.0):.3f} | {sc.get('bleu_score', 0.0):.3f} | {sc.get('context_precision', 0.0):.3f} |")

    # Ensure parent directory exists, then write report file
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(md))
        logger.info(f"Comparison report saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save comparison report: {str(e)}")


def run_evaluation(openai_key: str, chroma_dir: str, collection_name: str, evaluation_suite: list, n_results: int = 3, output_report: str = "evaluation_comparison.md"):
    """
    Executes the comparative evaluation suite.
    
    For each test case, retrieves context documents, generates answers using the LLM,
    and calculates RAGAS metrics in two modes:
      1. Filtered: Uses mission-specific metadata filters during document retrieval.
      2. Unfiltered: Does not use any metadata filter.
      
    Args:
        openai_key (str): The OpenAI API Key for response generation and evaluation.
        chroma_dir (str): Path to the persistent ChromaDB database.
        collection_name (str): Chroma collection name.
        evaluation_suite (list): List of test cases, each containing ID, mission, question, and expectation.
        n_results (int): Number of documents to retrieve for context.
        output_report (str): Output path for the comparison Markdown report.
    """
    logger.info("Connecting to ChromaDB database...")
    collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name, openai_key)
    if success != "success":
        logger.error(f"Failed to initialize RAG system: {error}")
        sys.exit(1)
    logger.info(f"Running evaluation on {len(evaluation_suite)} test cases...")

    print("=" * 80)
    print("NASA Multi-Mission RAG Evaluation - Comparative Report")
    print("=" * 80)

    comparison_results = []

    # Iterate through each test case in the suite
    for test in evaluation_suite:
        print("\n" + "=" * 80)
        print(f"Test Case {test['id']} - Mission: {test['mission']} (Category: {test.get('category', 'unknown')}, Metric: {test['test_metric']})")
        print(f"Question: {test['question']}")
        print("=" * 80)

        # Define the two retrieval modes: Filtered vs. Unfiltered
        test_modes = [
            {"name": "Filtered", "filter": test['mission_filter'], "is_filtered": True},
            {"name": "Unfiltered", "filter": None, "is_filtered": False}
        ]

        test_run_data = {
            "test_case": test,
            "modes": {}
        }

        # Evaluate the performance in both modes
        for mode in test_modes:
            mode_name = mode["name"]
            print(f"\n--- Running: {mode_name} ---")

            # Retrieve context documents from the vector database
            docs_result = rag_client.retrieve_documents(
                collection=collection,
                query=test['question'],
                n_results=n_results,
                mission_filter=mode["filter"]
            )

            context_str = ""
            contexts_list = []
            distance_scores = []

            # Format the retrieved documents if results are found
            if docs_result and docs_result.get("documents") and len(docs_result["documents"][0]) > 0:
                context_str = rag_client.format_context(docs_result["documents"][0], docs_result["metadatas"][0])
                contexts_list = docs_result["documents"][0]
                if "distances" in docs_result and docs_result["distances"]:
                    distance_scores = docs_result["distances"][0]

            # Assess relevance based on vector distance scores (lower is closer/more relevant)
            avg_distance = sum(distance_scores) / len(distance_scores) if distance_scores else 2.0
            relevance_status = "NOT_RELEVANT ✗"
            if avg_distance < 0.3:
                relevance_status = "HIGHLY_RELEVANT ✓✓"
            elif avg_distance < 0.5:
                relevance_status = "RELEVANT ✓"
            elif avg_distance < 0.7:
                relevance_status = "SOMEWHAT_RELEVANT ◐"

            print(f"-> Distance Assessment: {relevance_status} (Avg Distance: {avg_distance:.4f})")

            # Generate the answer using OpenAI API with the retrieved context
            print("-> Generating response...")
            generated_answer = llm_client.generate_response(
                api_key=openai_key,
                user_message=test['question'],
                context=context_str,
                conversation_history=[]
            )
            print(f"-> Generated Answer:\n   \"{generated_answer.strip()}\"")
            print(f"-> Expected Answer:\n   \"{test['expected'].strip()}\"")

            # Calculate evaluation metrics using the RAGAS framework
            print("-> Calculating RAGAS metrics...")
            scores = ragas_evaluator.evaluate_response_quality(
                openai_key=openai_key,
                question=test['question'],
                answer=generated_answer,
                contexts=contexts_list if contexts_list else ["No documents found context background reference."]
            )

            # Handle failures gracefully by defaulting scores to 0.0
            if "error" in scores:
                print(f"   ✗ RAGAS Evaluation Failed: {scores['error']}")
                scores = {
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "rouge_score": 0.0,
                    "bleu_score": 0.0,
                    "context_precision": 0.0,
                    "error": scores["error"]
                }
            else:
                print(f"   ✓ Faithfulness:         {scores.get('faithfulness', 0.0):.3f}")
                print(f"   ✓ Answer Relevancy:     {scores.get('answer_relevancy', 0.0):.3f}")
                print(f"   ✓ Rouge-L F1:           {scores.get('rouge_score', 0.0):.3f}")
                print(f"   ✓ Bleu-4 Precision:     {scores.get('bleu_score', 0.0):.3f}")
                print(f"   ✓ Context Precision:    {scores.get('context_precision', 0.0):.3f}")

            # Store results for comparison summary
            test_run_data["modes"][mode_name] = {
                "answer": generated_answer,
                "avg_distance": avg_distance,
                "scores": scores
            }

        comparison_results.append(test_run_data)
        print("-" * 80)

    # Print overall comparison summary to CLI
    print_comparison_summary(comparison_results)

    # Save detailed comparison report in Markdown format
    if output_report:
        save_comparison_markdown(comparison_results, output_report)


def main():
    """
    Main entry point for the evaluation script.
    
    Parses command-line arguments, configures system logging, loads the test
    suite JSON file containing evaluation questions, and initiates the run.
    """
    env_level = os.getenv("NASA_LOG_LEVEL", "WARNING").upper()
    parser = argparse.ArgumentParser(description="Run Mixed NASA RAG Validation Dataset Suite with Filter Comparison")
    parser.add_argument(
        "--log-level", 
        default=env_level, 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: WARNING)"
    )
    parser.add_argument("--openai-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="Active persistent database directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text", help="Chroma collection name")
    parser.add_argument("--n-results", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--test-questions", default="test_questions.json", help="Path to test questions JSON file")
    parser.add_argument("--output-report", default="evaluation_comparison.md", help="Path to save the comparison report")

    args = parser.parse_args()

    # Validate presence of OpenAI API Key
    if not args.openai_key:
        logger.error("API Key missing. Provide via OPENAI_API_KEY env var or --openai-key.")
        sys.exit(1)

    # Load test questions file
    if not os.path.exists(args.test_questions):
        logger.error(f"Test questions file not found: {args.test_questions}")
        sys.exit(1)
    try:
        with open(args.test_questions, "r", encoding="utf-8") as f:
            evaluation_suite = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load test questions file: {e}")
        sys.exit(1)

    # Configure logging level and format
    numeric_level = getattr(logging, args.log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Execute evaluation suite
    run_evaluation(
        openai_key=args.openai_key,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        evaluation_suite=evaluation_suite,
        n_results=args.n_results,
        output_report=args.output_report
    )


if __name__ == "__main__":
    main()