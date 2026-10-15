"""
Scope of the verify skill: list the changed files and decide what to run.

Collects the files changed on the branch (committed since the merge base with
the release branch `X.Y.x`, plus uncommitted and untracked files), classifies
them and prints the ruff, pytest and check commands that cover them, plus notes
that need a decision. Prints `nothing to verify` when only out-of-scope files
(dev/, benchmarks/, images/...) changed. Standard library only.

Usage:
    python .claude/skills/verify/affected.py                 # current branch
    python .claude/skills/verify/affected.py --files a.py b  # explicit files
    python .claude/skills/verify/affected.py --json
"""

import argparse
import json
import os
import re
import subprocess
import sys

OUT_OF_SCOPE_PREFIXES = ("dev/", "benchmarks/", "images/")
OUT_OF_SCOPE_FILES = {
    "CITATION.cff", "CNAME", "CODE_OF_CONDUCT.md", "CONTRIBUTING.md",
    "CONTRIBUTOR_LICENSE_AGREEMENT.md", "LICENSE", "SECURITY.md",
    "changelog.md", "funding.json", ".gitignore",
}
AI_CONTEXT_PREFIXES = ("tools/ai/", "skills/", ".claude-plugin/")
AI_CONTEXT_FILES = {
    "llms.txt", "llms-full.txt", "context7.json", "AGENTS.md", "CLAUDE.md",
    ".github/copilot-instructions.md",
}
# Modules used by many forecasters: their own tests are not enough.
SHARED_PREFIXES = (
    "skforecast/utils/", "skforecast/base/", "skforecast/preprocessing/",
    "skforecast/model_selection/_utils.py",
)
# Modules whose tests do not follow `tests/tests_<module>/` or `tests/test_<module>.py`.
EXCEPTIONS = {
    "skforecast/foundation/_adapter_base.py": "skforecast/foundation/tests/tests_foundation_models",
    "skforecast/foundation/_adapters.py": "skforecast/foundation/tests/tests_foundation_models",
    "skforecast/foundation/_foundation_model.py": "skforecast/foundation/tests/tests_foundation_models",
    "skforecast/foundation/_model_info.py": "skforecast/foundation/tests/tests_foundation_models",
    "skforecast/foundation/_utils.py": "skforecast/foundation/tests/tests_foundation_models",
    "skforecast/stats/seasonal/_differencing.py": "skforecast/stats/seasonal/tests/test_seasonal.py",
    "skforecast/stats/seasonal/_seasonal_strength.py": "skforecast/stats/seasonal/tests/test_seasonal.py",
}
HARNESS_TESTS = ".claude/hooks .claude/skills"

CHECKS = {
    "ai_context": "python tools/ai/generate_ai_context_files.py --check",
    "harness": f"python -m pytest {HARNESS_TESTS} -q -p no:cacheprovider",
    "docs": "python tools/docs/hooks/page_meta.py",
    "foundation_compat": "python -m pytest tools/foundation_compat/tests -q",
}
NOTES = {
    "deps": (
        "pyproject.toml changed: if dependencies changed, propose a run with the "
        "minimum versions (separate environment) and say whether the full suite "
        "is needed; the user decides"
    ),
    "ci": ".github/ changed: workflows do not run locally, review them by reading",
    "other": "files with no automatic check: review them by reading",
}


def git(args, root):
    result = subprocess.run(
        ["git"] + args, cwd=root, capture_output=True, text=True
    )
    return result.stdout if result.returncode == 0 else None


def release_branch(root):
    """Return `X.Y.x` from `__version__` in skforecast/__init__.py, or None."""
    try:
        with open(os.path.join(root, "skforecast", "__init__.py"), encoding="utf-8") as f:
            match = re.search(r"__version__\s*=\s*['\"](\d+)\.(\d+)", f.read())
    except OSError:
        return None
    return f"{match.group(1)}.{match.group(2)}.x" if match else None


def changed_files(root):
    """Return (base description, sorted changed files) for the working tree."""
    branch = release_branch(root)
    base, base_desc = "HEAD", "HEAD"
    for ref in ([branch, f"origin/{branch}"] if branch else []):
        merge_base = git(["merge-base", "HEAD", ref], root)
        if merge_base:
            base, base_desc = merge_base.strip(), f"merge base with {ref}"
            break
    files = set((git(["diff", "--name-only", base], root) or "").split())
    files |= set((git(["ls-files", "--others", "--exclude-standard"], root) or "").split())
    return base_desc, sorted(files)


def classify(path):
    if path.startswith(OUT_OF_SCOPE_PREFIXES) or path in OUT_OF_SCOPE_FILES:
        return "out_of_scope"
    if path in AI_CONTEXT_FILES or path.startswith(AI_CONTEXT_PREFIXES):
        return "ai_context"
    if re.fullmatch(r"docs/llms.*\.txt", path):
        return "ai_context"
    if path.startswith("skforecast/"):
        return "tests" if "/tests/" in path else "package"
    if path.startswith("tests/"):
        return "tests"
    if path.startswith("docs/") and path.endswith(".ipynb"):
        return "notebooks"
    if path.startswith("docs/") or path in ("mkdocs.yml", "README.md"):
        return "docs"
    if path.startswith(".claude/"):
        return "harness"
    if path.startswith("tools/docs/hooks/"):
        return "docs_hooks"
    if path.startswith("tools/foundation_compat/"):
        return "foundation_compat"
    if path == "pyproject.toml":
        return "deps"
    if path.startswith(".github/"):
        return "ci"
    return "other"


def tests_for_module(path, root):
    """
    Return (test path or None, flag) for a package file. The flag is "" when the
    tests were found by convention or EXCEPTIONS, "unmapped" when the whole test
    folder of the package is used and "none" when there is nothing to run.
    """
    def exists(rel_path):
        return os.path.exists(os.path.join(root, rel_path))

    if path in EXCEPTIONS:
        return EXCEPTIONS[path], ""
    pkg, filename = os.path.split(path)
    if filename == "__init__.py":
        if path == "skforecast/__init__.py":
            return "tests/test_skforecast_version.py", ""
        return None, "none"  # Exports: covered by the AI context check.
    if not filename.endswith(".py"):
        return None, "none"
    name = filename[:-3].lstrip("_")
    for candidate in (f"{pkg}/tests/tests_{name}", f"{pkg}/tests/test_{name}.py"):
        if exists(candidate):
            return candidate, ""
    if exists(f"{pkg}/tests"):
        return f"{pkg}/tests", "unmapped"
    return None, "none"


def tests_for_test_file(path, root):
    """A test file runs itself; a fixture or conftest runs its folder."""
    if not os.path.exists(os.path.join(root, path)):
        return None  # Deleted.
    filename = os.path.basename(path)
    if re.fullmatch(r"test_.*\.py", filename):
        return path
    return os.path.dirname(path)


def deduplicate(paths):
    """Drop paths inside another selected folder; keep the order."""
    unique = list(dict.fromkeys(paths))
    return [
        p for p in unique
        if not any(p != q and p.startswith(q.rstrip("/") + "/") for q in unique)
    ]


def build_plan(files, root):
    by_category = {}
    for path in files:
        by_category.setdefault(classify(path), []).append(path)

    in_scope = [p for p in files if classify(p) != "out_of_scope"]
    lint = [
        p for p in in_scope
        if p.endswith(".py") and os.path.exists(os.path.join(root, p))
    ]

    pytest_paths, notes = [], []
    for path in by_category.get("package", []):
        test_path, flag = tests_for_module(path, root)
        if test_path:
            pytest_paths.append(test_path)
        if flag == "unmapped":
            notes.append(f"{path}: no tests by name, running the whole {test_path}")
        elif flag == "none" and path.endswith(".py") and not path.endswith("__init__.py"):
            notes.append(f"{path}: no tests found")
        if path.startswith(SHARED_PREFIXES):
            notes.append(
                f"{path} is shared code: add the test folders of the callers "
                "whose behavior changed and list the packages not run"
            )
    for path in by_category.get("tests", []):
        test_path = tests_for_test_file(path, root)
        if test_path:
            pytest_paths.append(test_path)

    checks = []
    package_init_changed = any(
        p.endswith("__init__.py") for p in by_category.get("package", [])
    )
    if "ai_context" in by_category or package_init_changed:
        checks.append(CHECKS["ai_context"])
    if "harness" in by_category:
        checks.append(CHECKS["harness"])
    if {"docs", "notebooks", "docs_hooks"} & by_category.keys():
        checks.append(CHECKS["docs"])
    if "notebooks" in by_category:
        # The runner takes paths relative to docs/.
        notebooks = " ".join(p[len("docs/"):] for p in by_category["notebooks"])
        checks.append(
            f"python tools/docs/execute_notebooks/execute_notebooks.py --check {notebooks}"
        )
        notes.append(
            "notebooks changed: if their code changed, re-execute each one with "
            "`python tools/docs/execute_notebooks/execute_notebooks.py <path "
            "relative to docs/>`"
        )
    if "foundation_compat" in by_category:
        checks.append(CHECKS["foundation_compat"])
    for category in ("deps", "ci", "other"):
        if category in by_category:
            notes.append(f"{NOTES[category]}: {', '.join(by_category[category])}")

    user_facing = {"package", "docs", "notebooks"} & by_category.keys()
    if user_facing and "docs/releases/releases.md" not in files:
        notes.append(
            "no change in docs/releases/releases.md: if the change is user-facing, "
            "follow the release-note skill"
        )

    return {
        "files": by_category,
        "lint": lint,
        "pytest": deduplicate(pytest_paths),
        "checks": checks,
        "notes": notes,
        "nothing_to_verify": not in_scope,
    }


def format_plan(plan, base_desc):
    n_files = sum(len(v) for v in plan["files"].values())
    lines = [f"Changed files: {n_files} ({base_desc})"]
    for category, paths in plan["files"].items():
        lines.append(f"  {category}: {', '.join(paths)}")
    if plan["nothing_to_verify"]:
        lines.append("nothing to verify")
        return "\n".join(lines)
    if plan["lint"]:
        lines.append(f"lint: ruff check {' '.join(plan['lint'])}")
    if plan["pytest"]:
        lines.append(f"pytest: python -m pytest {' '.join(plan['pytest'])} -q")
    if plan["checks"]:
        lines.append("checks:")
        lines += [f"  {c}" for c in plan["checks"]]
    if plan["notes"]:
        lines.append("notes:")
        lines += [f"  - {n}" for n in plan["notes"]]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--files", nargs="+", help="changed files, relative to the root")
    parser.add_argument("--root", help="repository root (default: git top level)")
    parser.add_argument("--json", action="store_true", help="print JSON")
    args = parser.parse_args(argv)

    root = args.root or (git(["rev-parse", "--show-toplevel"], os.getcwd()) or "").strip()
    if not root:
        print("Not inside a git repository.", file=sys.stderr)
        return 1
    if args.files:
        base_desc, files = "files given", sorted(set(args.files))
    else:
        base_desc, files = changed_files(root)

    plan = build_plan(files, root)
    print(json.dumps(plan, indent=2) if args.json else format_plan(plan, base_desc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
