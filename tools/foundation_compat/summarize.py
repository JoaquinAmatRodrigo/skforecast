"""
Summarize the results of the foundation backends compatibility check.

Reads the JSON files written by `run.py` and prints a Markdown report with one
row per adapter. An environment without a JSON file (job cancelled
or crashed before writing it) is reported as "no result".

Usage:
    python tools/foundation_compat/summarize.py results

With `--github-output <file>`, it also writes `failures` (number of failed
environments) and `fingerprint` (a hash of what failed, to tell whether the
failures changed since the previous run) in the format of `$GITHUB_OUTPUT`.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path

COMPAT_DIR = Path(__file__).resolve().parent
LEVELS = ("contract", "smoke")
SYMBOLS = {
    "passed": "✓",
    "failed": "✗",
    "skipped": "skipped",
    "not run": "not run",
}
HEADER = (
    "| Adapter | Backend | Installed | Latest on PyPI | torch "
    "| Install | Contract | Smoke | Detail |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def failed_checks(result: dict | None) -> list[str]:
    """
    Return what failed in the result of an environment: `'no result'`,
    `'install'`, `'held back'` or the name of a level. Empty when it passed.
    """

    if result is None:
        return ["no result"]
    if result["install"]["status"] == "failed":
        return ["install"]

    failed = ["held back"] if result["held_back"] else []
    failed.extend(
        level for level in LEVELS if result["levels"][level]["status"] == "failed"
    )

    return failed


def detail(result: dict) -> str:
    """
    Return the explanation shown in the last column for an environment.
    """

    if result["install"]["status"] == "failed":
        return result["install"]["error"]

    parts = []
    if result["held_back"]:
        parts.append("The resolver did not install the latest version.")
    for level in LEVELS:
        errors = result["levels"][level]["errors"]
        if errors:
            more = f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""
            parts.append(f"{level}: {errors[0]}{more}")

    return " ".join(parts).replace("|", "\\|").replace("\n", " ")


def row(adapter: str, result: dict | None, run_url: str) -> str:
    """
    Return the Markdown row of an environment.
    """

    if result is None:
        link = f"see the [jobs of the run]({run_url})" if run_url else ""
        cells = [adapter, "", "", "", "", "no result", "", "", link]
        return "| " + " | ".join(cells) + " |"

    versions = result["versions"]
    install = SYMBOLS[result["install"]["status"]]
    if result["held_back"]:
        install = "held back"
    elif result["overrides"] and result["install"]["status"] == "passed":
        install += f" (override {', '.join(result['overrides'])})"

    cells = [
        adapter,
        result["backend"],
        versions.get(result["backend"]) or "",
        result["latest_on_pypi"] or "",
        versions.get("torch") or "",
        install,
        SYMBOLS[result["levels"]["contract"]["status"]],
        SYMBOLS[result["levels"]["smoke"]["status"]],
        detail(result),
    ]

    return "| " + " | ".join(cells) + " |"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the foundation backends compatibility results."
    )
    parser.add_argument("results_dir", help="Directory with the JSON results.")
    parser.add_argument(
        "--github-output", help="File where `failures` and `fingerprint` are written."
    )
    args = parser.parse_args()

    with open(COMPAT_DIR / "adapters.toml", "rb") as f:
        adapters = list(tomllib.load(f))

    results = {}
    for file in sorted(Path(args.results_dir).rglob("*.json")):
        result = json.loads(file.read_text())
        results[result["adapter"]] = result

    run_url = ""
    if os.environ.get("GITHUB_RUN_ID"):
        run_url = (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        )

    rows = []
    failures = []
    for adapter in adapters:
        result = results.get(adapter)
        rows.append(row(adapter, result, run_url))
        failures.extend(f"{adapter}: {check}" for check in failed_checks(result))

    n_failed = len({failure.split(":")[0] for failure in failures})
    print("## Foundation backends compatibility\n")
    if failures:
        print(f"**{n_failed} of {len(rows)} environments failed.**\n")
    else:
        print(f"All {len(rows)} environments passed.\n")
    print(HEADER)
    print("\n".join(rows))

    if args.github_output:
        fingerprint = hashlib.sha256("\n".join(failures).encode()).hexdigest()[:16]
        with open(args.github_output, "a") as f:
            f.write(f"failures={n_failed}\n")
            f.write(f"fingerprint={fingerprint}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
