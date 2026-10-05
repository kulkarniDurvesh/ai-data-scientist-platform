"""python -m evals [--model] [--out report.md]: score goal and question interpretation."""

import argparse
import sys
from pathlib import Path

from core.llm import get_provider, use_provider

from .runner import datasets, report, run_goals, run_questions


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate goal and question interpretation.")
    parser.add_argument("--model", action="store_true", help="Use the configured language model (AIDS_LLM_PROVIDER).")
    parser.add_argument("--out", type=Path, help="Write the Markdown report to this file.")
    args = parser.parse_args()

    provider = get_provider(refresh=True) if args.model else None
    if args.model and provider is None:
        sys.stderr.write("No language model available: set AIDS_LLM_PROVIDER=ollama or azure.\n")
        return 1
    use_provider(None)

    bundles = datasets()
    results = run_goals(bundles, provider) + run_questions(bundles, provider)
    mode = f"rules + {provider.describe()}" if provider else "rules only"
    text = report(results, mode)
    sys.stdout.write(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
