import argparse
import json

from rag import retrieve


def recall_at_k(dataset, k=5):
    hits = 0
    results = []
    for item in dataset:
        retrieved = retrieve(item["question"], k)
        expected = item["expected_phrase"].lower()
        hit = any(expected in result["text"].lower() for result in retrieved)
        hits += int(hit)
        results.append({"question": item["question"], "hit": hit})
    return hits / len(dataset) if dataset else 0.0, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="JSON evaluation dataset")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    dataset = json.load(open(args.dataset, encoding="utf-8"))
    score, results = recall_at_k(dataset, args.k)
    print(f"Recall@{args.k}: {score:.1%} ({sum(r['hit'] for r in results)}/{len(results)})")
    for result in results:
        print(("PASS" if result["hit"] else "MISS"), "-", result["question"])
