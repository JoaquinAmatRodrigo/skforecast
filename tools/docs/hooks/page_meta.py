"""
MkDocs hook that sets the search engine title and description of each page.

mkdocs-jupyter ignores front matter in notebooks, and reading it from the
notebook metadata would mean editing every notebook. Instead, the texts live in
`page_meta.yml`, next to this file, keyed by the path of the page relative to
`docs/`. For each page listed there, the hook fills `page.meta`:

- `title`: the `<title>` tag (Material appends " - <site_name>") and the
  Open Graph title. The short title of `nav` in `mkdocs.yml` still names the
  page in the sidebar, the breadcrumbs and the footer links.
- `description`: the meta description, also used by the Open Graph and Twitter
  tags of `docs/overrides/main.html`. Without it, a page gets `site_description`.

Front matter of a Markdown page wins over the YAML.

Registered in `mkdocs.yml` as:

    hooks:
      - tools/docs/hooks/page_meta.py

Notes for maintainers:

- The YAML is read in `on_config`, so editing it during `mkdocs serve` triggers a
  rebuild with the new texts (`watch: [tools/docs/hooks]` in `mkdocs.yml`).
  Editing *this file* has no effect until the server is restarted.
- A page of `nav` without a description, or an entry of the YAML that matches no
  page, is reported as a warning: the site is still built, and
  `mkdocs build --strict` fails.
"""

import logging
from pathlib import Path

import yaml
from mkdocs.exceptions import PluginError

logger = logging.getLogger("mkdocs.hooks.page_meta")

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "page_meta.yml"
KEYS = {"title", "description"}

# Populated by on_config: {path relative to docs/: {"title": ..., "description": ...}}
PAGE_META = None


def on_config(config):
    global PAGE_META
    data = yaml.safe_load(DATA_FILE.read_text(encoding="utf-8")) or {}
    for path, entry in data.items():
        if not isinstance(entry, dict) or not set(entry) <= KEYS:
            raise PluginError(
                f"page_meta: entry '{path}' of {DATA_FILE.name} must be a "
                f"mapping with keys among {sorted(KEYS)}"
            )
    PAGE_META = data
    return config


def on_files(files, config):
    # The documentation pages, `.md` and notebooks, are known at this point
    paths = {file.src_uri for file in files.documentation_pages()}
    for path in PAGE_META.keys() - paths:
        logger.warning(f"page_meta: '{path}' in {DATA_FILE.name} matches no page")
    return files


def on_page_markdown(markdown, page, config, files):
    # Runs after `page.read_source`, which resets `page.meta` to the front
    # matter. mkdocs-jupyter renders notebooks later but calls this hook too.
    for key, value in PAGE_META.get(page.file.src_uri, {}).items():
        page.meta.setdefault(key, value.strip())
    if not page.meta.get("description"):
        logger.warning(
            f"page_meta: '{page.file.src_uri}' has no description in "
            f"{DATA_FILE.name}"
        )
    return markdown
