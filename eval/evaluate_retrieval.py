"""
Evaluate CodeSense's retrieval quality against a labeled question set.

For each {question, expected_file} pair, this hits the /retrieve endpoint
(retrieval only — no LLM call, so Ollama doesn't need to be running) and
checks whether the expected file shows up in the top-k results, and at
what rank. Reports Recall@k and Mean Reciprocal Rank (MRR).

Usage:
    # 1. Start the backend (docker compose up, or uvicorn locally)
    # 2. Ingest the repo you want to evaluate against, e.g. index CodeSense
    #    itself for the bundled demo dataset:
    #       curl -X POST localhost:8000/ingest \\
    #            -H "Content-Type: application/json" \\
    #            -d '{"repo_path": "/repos/codesense", "collection_name": "codesense-self"}'
    # 3. Run the evaluation:
    python eval/evaluate_retrieval.py --collection codesense-self

Exits with a non-zero status if Recall@k falls below --min-recall, so this
can be wired into CI as a regression gate on retrieval quality.
"""
import argparse
import json
import os
import sys

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def retrieve(collection_name: str, question: str, top_k: int, token: str = None) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = requests.post(
        f"{API_URL}/retrieve",
        json={"collection_name": collection_name, "question": question, "history": [], "top_k": top_k},
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["sources"]


def evaluate(dataset: list[dict], collection_name: str, top_k: int, token: str = None) -> tuple[list, float, float]:
    results = []
    reciprocal_ranks = []
    hits = 0

    for item in dataset:
        question = item["question"]
        expected_file = item["expected_file"]
        expected_function = item.get("expected_function")

        sources = retrieve(collection_name, question, top_k, token)

        rank = None
        function_match = False
        for i, s in enumerate(sources, start=1):
            if expected_file in s["file_path"]:
                if rank is None:
                    rank = i
                if expected_function and s.get("function_name") == expected_function:
                    function_match = True

        hit = rank is not None
        hits += int(hit)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        results.append({
            "question": question,
            "expected_file": expected_file,
            "expected_function": expected_function,
            "hit": hit,
            "rank": rank,
            "function_match": function_match,
        })

    recall_at_k = hits / len(dataset) if dataset else 0.0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return results, recall_at_k, mrr


def print_report(results: list[dict], recall_at_k: float, mrr: float, top_k: int) -> None:
    print(f"\n{'Q#':<4}{'Hit':<6}{'Rank':<6}{'Func✓':<7}Question")
    print("-" * 95)
    for i, r in enumerate(results, start=1):
        hit_mark = "PASS" if r["hit"] else "FAIL"
        rank_str = str(r["rank"]) if r["rank"] else "-"
        func_mark = "yes" if r["function_match"] else ("-" if r["hit"] else "")
        q_short = (r["question"][:65] + "...") if len(r["question"]) > 65 else r["question"]
        print(f"{i:<4}{hit_mark:<6}{rank_str:<6}{func_mark:<7}{q_short}")
    print("-" * 95)
    print(f"Recall@{top_k}: {recall_at_k:.1%}   MRR: {mrr:.3f}   ({len(results)} questions)\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CodeSense retrieval quality.")
    parser.add_argument("--dataset", default=os.path.join(os.path.dirname(__file__), "dataset.json"),
                         help="Path to a JSON file of {question, expected_file, expected_function?} items")
    parser.add_argument("--collection", required=True,
                         help="Name of the already-ingested collection to test against")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--token", default=None, help="Bearer token, if AUTH_REQUIRED=true")
    parser.add_argument("--output", default=None, help="Optional path to save detailed results as JSON")
    parser.add_argument("--min-recall", type=float, default=0.5,
                         help="Exit non-zero if Recall@k falls below this (useful for CI)")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    results, recall_at_k, mrr = evaluate(dataset, args.collection, args.top_k, args.token)
    print_report(results, recall_at_k, mrr, args.top_k)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"recall_at_k": recall_at_k, "mrr": mrr, "results": results}, f, indent=2)
        print(f"Saved detailed results to {args.output}")

    if recall_at_k < args.min_recall:
        print(f"Recall@{args.top_k} ({recall_at_k:.1%}) is below the --min-recall threshold "
              f"({args.min_recall:.1%}).")
        sys.exit(1)


if __name__ == "__main__":
    main()
