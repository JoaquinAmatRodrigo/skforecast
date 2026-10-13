"""
MkDocs hook that renders the user guides table of contents from `nav`.

`docs/user_guides/table-of-contents.md` contains a single
`<!-- USER-GUIDES-TOC -->` marker. This hook replaces it with one card per
section of the `User Guides` entry of `nav` in `mkdocs.yml`, so the page cannot
drift from the sidebar: adding a guide to `nav` adds it to the page.
`user_guides_toc_texts.yml`, sitting next to this file, holds the texts `nav`
cannot say: one sentence per section and, optionally, a longer title for a guide.

Registered in `mkdocs.yml` as:

    hooks:
      - tools/docs/hooks/user_guides_toc.py

Notes for maintainers:

- The YAML is read in `on_config`, so editing it during `mkdocs serve` triggers a
  rebuild with the new content (`watch: [tools/docs/hooks]` in `mkdocs.yml`).
  Editing *this file* has no effect until the server is restarted.
- Rendering happens in `on_page_markdown`, so the links are rewritten and
  validated by MkDocs. Each card title is a `##` heading with its own anchor,
  but `on_page_content` empties `page.toc`: with `toc.integrate` the sidebar
  would list the sections twice, as headings of the page and as navigation
  (`hide: toc` in the page does not remove the integrated list).
- A section of `nav` without a sentence, or an entry of the YAML that no longer
  matches `nav`, is reported as a warning: the page is still built, and
  `mkdocs build --strict` fails.
- The cards are styled by the "User Guides Table of Contents" block of
  `docs/stylesheets/extra.css`.
"""

import logging
from pathlib import Path

import yaml
from mkdocs.exceptions import PluginError
from mkdocs.structure.toc import get_toc

logger = logging.getLogger("mkdocs.hooks.user_guides_toc")

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "user_guides_toc_texts.yml"
MARKER = "<!-- USER-GUIDES-TOC -->"
NAV_ENTRY = "User Guides"
TOC_PAGE = "user_guides/table-of-contents.md"

# Populated by on_config: list of (section title, sentence, [(title, path)]).
SECTIONS = None


def _nav_sections(nav):
    """
    Return the sections of the `User Guides` entry of `nav` as a list of
    (section title, [(guide title, path)]). A page that is not inside a section
    is a section with a single guide. The table of contents page is skipped.
    """
    entries = next(
        (item[NAV_ENTRY] for item in nav or [] if NAV_ENTRY in item), None
    )
    if not isinstance(entries, list):
        raise PluginError(
            f"user_guides_toc: mkdocs.yml has no '{NAV_ENTRY}' section in nav"
        )

    sections = []
    for entry in entries:
        (title, content), = entry.items()
        if content == TOC_PAGE:
            continue
        if isinstance(content, str):
            sections.append((title, [(title, content)]))
            continue
        guides = []
        for guide in content:
            (guide_title, path), = guide.items()
            if not isinstance(path, str):
                raise PluginError(
                    f"user_guides_toc: '{title}' > '{guide_title}' is a nested "
                    f"section, which the table of contents does not support"
                )
            guides.append((guide_title, path))
        sections.append((title, guides))

    return sections


def on_config(config):
    global SECTIONS

    try:
        data = yaml.safe_load(DATA_FILE.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise PluginError(f"user_guides_toc: {DATA_FILE} not found") from exc
    except yaml.YAMLError as exc:
        raise PluginError(
            f"user_guides_toc: {DATA_FILE} is not valid YAML: {exc}"
        ) from exc

    sentences = data.get("sections") or {}
    titles = data.get("titles") or {}
    nav_sections = _nav_sections(config["nav"])
    nav_titles = {title for title, _ in nav_sections}
    nav_paths = {path for _, guides in nav_sections for _, path in guides}

    for title in nav_titles - set(sentences):
        logger.warning(
            "user_guides_toc: section '%s' of nav has no sentence in %s",
            title,
            DATA_FILE.name,
        )
    for title in set(sentences) - nav_titles:
        logger.warning(
            "user_guides_toc: %s has a sentence for '%s', which is not a "
            "section of '%s' in nav",
            DATA_FILE.name,
            title,
            NAV_ENTRY,
        )
    for path in set(titles) - nav_paths:
        logger.warning(
            "user_guides_toc: %s has a title for '%s', which is not a guide of "
            "'%s' in nav",
            DATA_FILE.name,
            path,
            NAV_ENTRY,
        )

    SECTIONS = [
        (
            title,
            sentences.get(title),
            [(titles.get(path, guide_title), path) for guide_title, path in guides],
        )
        for title, guides in nav_sections
    ]

    return config


def _render():
    """
    Return the Markdown of the card grid: one list item per section with its
    title, its sentence and the links to its guides.
    """
    lines = ['<div class="grid cards ug-grid" markdown>', ""]
    for title, sentence, guides in SECTIONS:
        lines += [f"-   ## {title}", ""]
        if sentence:
            lines += [f"    {sentence}", ""]
        for guide_title, path in guides:
            # nav paths are relative to docs/, one level above this page
            lines.append(f"    - [{guide_title}](../{path})")
        lines.append("")
    lines.append("</div>")

    return "\n".join(lines)


def on_page_markdown(markdown, *, page, config, files):
    if page.file.src_uri != TOC_PAGE:
        return markdown
    if MARKER not in markdown:
        raise PluginError(
            f"user_guides_toc: {TOC_PAGE} does not contain the {MARKER} marker"
        )

    return markdown.replace(MARKER, _render())


def on_page_content(html, *, page, config, files):
    if page.file.src_uri == TOC_PAGE:
        page.toc = get_toc([])

    return html
