from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CASES = {
    "matrix": "the matrix movie",
    "insidious": "insidious 2 netflix",
    "avatar": "avatar 2",
    "top_gun": "top gun maverick",
    "wakanda": "black panther wakanda forever",
}


def format_metadata(metadata: str, width: int = 140) -> str:
    compact = " ".join(metadata.split())
    return textwrap.shorten(compact, width=width, placeholder="...")


def run_case(name: str, query: str, k: int) -> None:
    from apps.demo.network import recommend

    result = recommend(query, k=k)
    print(f"=== {name} ===")
    print(f"Query: {query}")
    for rank, candidate in enumerate(result["recommendations"], start=1):
        title = candidate["title"]
        score = candidate["score"]
        metadata = format_metadata(candidate["metadata"])
        print(f"{rank:02d}. {title}  score={score:.4f}")
        print(f"    {metadata}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the same retrieval path used by the FastAPI demo against preset or custom queries."
    )
    parser.add_argument("--case", action="append", choices=sorted(DEFAULT_CASES.keys()))
    parser.add_argument("--query", type=str, help="Custom query to evaluate.")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if args.query:
        run_case("custom", args.query, args.k)
        return

    selected_cases = args.case or list(DEFAULT_CASES.keys())
    for case_name in selected_cases:
        run_case(case_name, DEFAULT_CASES[case_name], args.k)


if __name__ == "__main__":
    main()
