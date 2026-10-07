"""python -m evals [--model] [--set goals|questions|documents] [--out report.md]: score interpretation and retrieval."""

import argparse
import sys
from pathlib import Path

from core.llm import get_provider, use_provider

from .runner import datasets, report, run_agents, run_documents, run_goals, run_questions


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate goal and question interpretation.")
    parser.add_argument("--model", action="store_true", help="Use the configured language model (AIDS_LLM_PROVIDER).")
    parser.add_argument("--out", type=Path, help="Write the Markdown report to this file.")
    parser.add_argument("--set", choices=["all", "goals", "questions", "documents", "agents"], default="all")
    args = parser.parse_args()

    provider = get_provider(refresh=True) if args.model else None
    if args.model and provider is None:
        sys.stderr.write("No language model available: set AIDS_LLM_PROVIDER=ollama or azure.\n")
        return 1
    use_provider(None)

    results = []
    if args.set in ("all", "goals", "questions"):
        bundles = datasets()
        if args.set in ("all", "goals"):
            results += run_goals(bundles, provider)
        if args.set in ("all", "questions"):
            results += run_questions(bundles, provider)
    if args.set == "agents":
        if provider is None:
            sys.stderr.write("The agent set needs --model.\n")
            return 1
        results += run_agents(datasets(), provider)
    if args.set in ("all", "documents"):
        if provider is not None:
            use_provider(provider)
        results += run_documents(use_model=provider is not None)
        use_provider(None)
    mode = f"rules + {provider.describe()}" if provider else "rules only"
    text = report(results, mode)
    sys.stdout.write(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
