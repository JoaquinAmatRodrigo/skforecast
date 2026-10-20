"""
Hook (SessionStart, and PreToolUse on Bash and GitHub MCP write tools): branch
names follow `<type>/<topic>` (CLAUDE.md, "Git workflow"), locally and in cloud
sessions.

Claude Code creates branches with names the repository cannot configure:
`claude/<name>` in cloud sessions (created from `main`), `worktree-<name>` for
`claude --worktree`, and the prefix of the desktop app setting. This hook:

- At session start, says so when the current branch does not follow the rule,
  and names the release branch to create the work branch from.
- Before a Bash command, blocks `git checkout -b/-B/--orphan`, `git switch
  -c/-C/--orphan`, `git branch <name>` (also `-m`, `-c`), `git worktree add -b`,
  `git push` (the branches it updates, the current one when no refspec is
  given) and `gh pr create --head` with a branch name that does not follow it.
- Before a GitHub MCP tool that creates or writes a branch, or opens a pull
  request, checks its `branch` or `head`.

`main` and the release branches (`X.Y.x`) are accepted too. Deletions, tags
and `--all`/`--mirror` pushes are not checked. Exit code 2 blocks the call and
sends the reason back to Claude. Standard library only (Python 3.9+).
"""

import json
import os
import re
import shlex
import subprocess
import sys

BRANCH_TYPES = ("feature", "fix", "docs", "chore", "perf", "refactor", "test", "ci")
BRANCH_NAME = re.compile(
    r"^(" + "|".join(BRANCH_TYPES) + r")/[a-z0-9][a-z0-9._-]*$"
    r"|^main$|^\d+\.\d+\.x$"
)
RELEASE_REF = re.compile(r"(?:^|/)(\d+)\.(\d+)\.x$")
SEGMENT_SEPARATOR = re.compile(r"\n|&&|\|\||;|\|")
ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# A redirection (`2>&1`, `>log`, `> log`, `<input`), which is not an argument.
REDIRECTION = re.compile(r"^(?:\d+|&)?(?:>>?|<<?|>&|<&)(.*)$")
GIT_OPTIONS_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}
PUSH_OPTIONS_WITH_VALUE = {"--repo", "-o", "--push-option", "--receive-pack", "--exec"}
BRANCH_NOT_CREATING = {
    "-d", "-D", "--delete", "-l", "--list", "-a", "--all", "-r", "--remotes",
    "-v", "-vv", "--verbose", "--show-current", "--contains", "--no-contains",
    "--merged", "--no-merged", "--points-at", "-u", "--set-upstream-to",
    "--unset-upstream", "--edit-description",
}
BRANCH_RENAMING = {"-m", "-M", "--move", "-c", "-C", "--copy"}
CREATE_OPTIONS = {
    "checkout": {"-b", "-B", "--orphan"},
    "switch": {"-c", "-C", "--create", "--force-create", "--orphan"},
}
# GitHub MCP tools and the argument that names a branch.
MCP_BRANCH_ARGUMENT = {
    "create_branch": "branch",
    "push_files": "branch",
    "create_or_update_file": "branch",
    "create_pull_request": "head",
}
MCP_TOOL = re.compile(r"^mcp__.*github.*__(\w+)$")
TIMEOUT = 15
RULE = (
    "Branch names follow `<type>/<topic>` (CLAUDE.md, Git workflow), with type "
    f"one of {', '.join(BRANCH_TYPES)} and a short lowercase topic, e.g. "
    "`fix/multiseries-folds-without-series`, created from the release branch "
    "`origin/{release}`."
)


def segments(command):
    """
    Split `command` into simple commands and return the words of each, with
    leading env assignments and redirections removed. A segment that is not
    valid shell (for example a line inside a heredoc) is skipped.
    """
    result = []
    for segment in SEGMENT_SEPARATOR.split(command):
        try:
            words = shlex.split(segment.strip().lstrip("({ "))
        except ValueError:
            continue
        while words and ENV_ASSIGNMENT.match(words[0]):
            words.pop(0)
        arguments = []
        skip_target = False
        for word in words:
            if skip_target:
                skip_target = False
            elif match := REDIRECTION.match(word):
                skip_target = match[1] == ""
            else:
                arguments.append(word)
        if arguments:
            result.append(arguments)
    return result


def git_subcommand(words):
    """
    For a `git ...` simple command, return `(cwd option, subcommand, args)`,
    where the cwd option is the value of `-C` (or None). Otherwise None.
    """
    if os.path.basename(words[0]) != "git":
        return None
    directory = None
    i = 1
    while i < len(words) and words[i].startswith("-"):
        if words[i] == "-C" and i + 1 < len(words):
            directory = words[i + 1]
        i += 2 if words[i] in GIT_OPTIONS_WITH_VALUE else 1
    if i >= len(words):
        return None
    return directory, words[i], words[i + 1:]


def option_value(args, options):
    """Values of `options` in the forms `-b name`, `--create name`, `--create=name`."""
    values = []
    for i, arg in enumerate(args):
        if arg in options and i + 1 < len(args):
            values.append(args[i + 1])
        elif "=" in arg and arg.split("=", 1)[0] in options:
            values.append(arg.split("=", 1)[1])
    return values


def git(cwd, *args):
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True,
            timeout=TIMEOUT, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def current_branch(cwd):
    branch = git(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    return branch if branch and branch != "HEAD" else None


def pushed_branches(args, cwd):
    """Branches that a `git push` updates on the remote (deletions and tags left out)."""
    if any(a in ("-d", "--delete", "--tags", "--all", "--mirror") for a in args):
        return []
    positional = []
    i = 0
    while i < len(args):
        if args[i] in PUSH_OPTIONS_WITH_VALUE:
            i += 2
            continue
        if not args[i].startswith("-"):
            positional.append(args[i])
        i += 1
    refspecs = positional[1:]
    if not refspecs:
        branch = current_branch(cwd)
        return [branch] if branch else []
    branches = []
    for refspec in refspecs:
        source, _, destination = refspec.lstrip("+").partition(":")
        if refspec.lstrip("+").startswith(":"):
            continue  # `:branch` deletes it.
        name = destination or source
        if name == "HEAD":
            name = current_branch(cwd) or name
        if name.startswith("refs/tags/") or (
            not name.startswith("refs/heads/")
            and git(cwd, "rev-parse", "--verify", "--quiet", f"refs/tags/{name}")
        ):
            continue
        branches.append(name.removeprefix("refs/heads/"))
    return branches


def created_branches(subcommand, args):
    """Branches that a `git checkout`, `switch`, `branch` or `worktree add` creates."""
    if subcommand in CREATE_OPTIONS:
        return option_value(args, CREATE_OPTIONS[subcommand])
    if subcommand == "worktree" and args[:1] == ["add"]:
        return option_value(args[1:], {"-b", "-B"})
    if subcommand == "branch":
        if any(a in BRANCH_NOT_CREATING or a.startswith("--format") for a in args):
            return []
        positional = [a for a in args if not a.startswith("-")]
        if any(a in BRANCH_RENAMING for a in args):
            return positional[-1:]
        return positional[:1]
    return []


def bash_branches(command, cwd):
    """Branch names that a Bash command creates or pushes."""
    branches = []
    for words in segments(command):
        if os.path.basename(words[0]) == "gh" and words[1:3] == ["pr", "create"]:
            heads = option_value(words[3:], {"-H", "--head"})
            branches += [head.split(":")[-1] for head in heads]
            continue
        parsed = git_subcommand(words)
        if parsed is None:
            continue
        directory, subcommand, args = parsed
        git_cwd = os.path.join(cwd, directory) if directory else cwd
        if subcommand == "push":
            branches += pushed_branches(args, git_cwd)
        else:
            branches += created_branches(subcommand, args)
    return branches


def latest_release_branch(cwd):
    """
    Release branch (`X.Y.x`) with the highest version, from the refs of
    `origin` in the checkout or, if there are none, from GitHub. `__version__`
    is not used because the checkout may be a stale `main`.
    """
    refs = git(cwd, "for-each-ref", "--format=%(refname)", "refs/remotes/origin/")
    versions = [m.groups() for ref in refs.splitlines() if (m := RELEASE_REF.search(ref))]
    if not versions:
        heads = git(cwd, "ls-remote", "--heads", "origin", "refs/heads/*.x")
        versions = [
            m.groups() for line in heads.splitlines()
            if (m := RELEASE_REF.search(line.split("\t")[-1]))
        ]
    if not versions:
        return "X.Y.x"
    major, minor = max((int(a), int(b)) for a, b in versions)
    return f"{major}.{minor}.x"


def session_start(cwd):
    branch = current_branch(cwd)
    if branch is None or BRANCH_NAME.match(branch):
        return 0
    release = latest_release_branch(cwd)
    print(
        f"The current branch '{branch}' does not follow the branch naming rule, "
        f"do not build on it or push it. Before the first change, create the "
        f"work branch: `git fetch origin {release} && git checkout -b "
        f"<type>/<topic> origin/{release}`. " + RULE.format(release=release)
    )
    return 0


def main():
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0

    cwd = payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or "."
    if payload.get("hook_event_name") == "SessionStart":
        return session_start(cwd)

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    if tool_name == "Bash":
        branches = bash_branches(tool_input.get("command", ""), cwd)
    elif (match := MCP_TOOL.match(tool_name)) and match[1] in MCP_BRANCH_ARGUMENT:
        branch = tool_input.get(MCP_BRANCH_ARGUMENT[match[1]]) or ""
        branches = [branch.split(":")[-1]] if branch else []
    else:
        branches = []

    invalid = [b for b in branches if not BRANCH_NAME.match(b)]
    if invalid:
        release = latest_release_branch(cwd)
        sys.stderr.write(
            f"Blocked: branch {', '.join(repr(b) for b in invalid)} does not "
            f"follow the branch naming rule. " + RULE.format(release=release)
            + " To keep the work of a branch with another name, create the new "
            f"branch from `origin/{release}` and cherry-pick its commits (or "
            "`git branch -m <type>/<topic>` if it already starts from "
            f"`origin/{release}`).\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
