"""
MkDocs hook that renders the Examples pages from a single YAML source.

The three pages under `docs/examples/` (English, Spanish, Chinese) each contain a
single `<!-- EXAMPLES-GRID -->` marker. This hook replaces that marker with a
language switcher and one Material card grid per section, all built from
`examples.yml` sitting next to this file.

Every page lists every example. An example that is not translated into the
language of the page is shown with its English card and an "In English" tag.

Registered in `mkdocs.yml` as:

    hooks:
      - tools/docs/hooks/examples_grid.py

Notes for maintainers:

- The YAML is read in `on_config`, so editing it during `mkdocs serve` triggers a
  rebuild with the new content (`watch: [tools/docs/hooks]` in `mkdocs.yml` makes
  the server notice the change). Editing *this file* has no effect until the
  server is restarted, because MkDocs caches hook modules for the life of the
  process.
- Rendering happens in `on_page_markdown`, before `page.render()`, so the
  generated `##` headings reach `page.toc` (needed by the `toc.integrate`
  feature) and the language switcher's `.md` links get rewritten and validated.
- Any problem in the YAML raises `PluginError` from `on_config`, which aborts the
  build with a single clean message before any page is rendered.
"""

import html
from pathlib import Path

import yaml
from mkdocs.exceptions import PluginError

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "examples.yml"
MARKER = "<!-- EXAMPLES-GRID -->"
FALLBACK_LANG = "en"

# Populated by on_config.
DATA = None


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def _require(condition, message):
    if not condition:
        raise PluginError(f"tools/docs/hooks/examples.yml: {message}")


def _validate(data):
    """Validate the YAML structure, raising PluginError on the first problem."""
    _require(isinstance(data, dict), "file is empty or not a mapping")
    for key in ("levels", "languages", "ui", "sections", "examples"):
        _require(data.get(key), f"top-level key '{key}' is missing or empty")

    levels = data["levels"]
    languages = data["languages"]
    _require(FALLBACK_LANG in languages, f"languages.{FALLBACK_LANG} is missing")

    for code, cfg in languages.items():
        _require(cfg.get("page"), f"languages.{code} is missing 'page'")
        _require(cfg.get("label"), f"languages.{code} is missing 'label'")
        ui = data["ui"].get(code) or {}
        missing = sorted(set(levels) - set(ui.get("levels") or {}))
        _require(not missing, f"ui.{code}.levels is missing entries for {missing}")
        if code != FALLBACK_LANG:
            _require(ui.get("in_english"), f"ui.{code} is missing 'in_english'")

    section_ids = []
    featured = set()
    for section in data["sections"]:
        sid = section.get("id")
        _require(sid, "a section is missing 'id'")
        _require(sid not in section_ids, f"duplicate section id '{sid}'")
        section_ids.append(sid)
        if section.get("featured"):
            featured.add(sid)
        _require(section.get("icon"), f"section '{sid}' is missing 'icon'")
        for code in languages:
            _require(section.get(code), f"section '{sid}' is missing '{code}'")

    seen_ids = set()
    for example in data["examples"]:
        eid = example.get("id")
        _require(eid, "an example is missing 'id'")
        _require(eid not in seen_ids, f"duplicate example id '{eid}'")
        seen_ids.add(eid)
        _require(
            example.get("section") in section_ids,
            f"example '{eid}' has unknown section '{example.get('section')}'",
        )
        if example["section"] not in featured or "level" in example:
            _require(
                example.get("level") in levels,
                f"example '{eid}' has unknown level '{example.get('level')}'",
            )
        _require(
            (example.get(FALLBACK_LANG) or {}).get("url"),
            f"example '{eid}' is missing {FALLBACK_LANG}.url",
        )
        for code in languages:
            block = example.get(code)
            if block is None and code != FALLBACK_LANG:
                continue
            for key in ("title", "summary"):
                _require(
                    (block or {}).get(key),
                    f"example '{eid}' is missing {code}.{key}",
                )

    return data


def on_config(config):
    """Read and validate examples.yml once per build (and per serve rebuild)."""
    global DATA

    try:
        raw = yaml.safe_load(DATA_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PluginError(f"examples_grid: {DATA_FILE} not found") from exc
    except yaml.YAMLError as exc:
        raise PluginError(
            f"examples_grid: {DATA_FILE} is not valid YAML: {exc}"
        ) from exc

    DATA = _validate(raw)
    return config


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _render_language_switcher(lang):
    """A row of links to the same page in the other languages."""
    parts = []
    for code, cfg in DATA["languages"].items():
        classes = ".ex-lang"
        if code == lang:
            classes += " .ex-lang--active aria-current=page"
        parts.append(f"[{cfg['label']}]({cfg['page']}){{ {classes} }}")
    return "\n".join(parts) + "\n{ .ex-langs }"


def _render_card(lang, example):
    """
    One Material card, as plain HTML.

    The title and summary come from the block of the page language, or from the
    English block (marked with `lang="en"`) when the example is not translated.
    The "In English" tag is added whenever the link goes to the English article,
    which also covers a translated block without its own `url`.
    """
    ui = DATA["ui"][lang]
    translated = lang in example
    block = example[lang] if translated else example[FALLBACK_LANG]
    url = block.get("url") or example[FALLBACK_LANG]["url"]
    text_lang = "" if translated else f' lang="{FALLBACK_LANG}"'

    tags = []
    level = example.get("level")
    if level:
        tags.append(
            f'<span class="ex-tag ex-tag--level ex-tag--{level}">'
            f"{html.escape(ui['levels'][level])}</span>"
        )
    if lang != FALLBACK_LANG and not (translated and block.get("url")):
        tags.append(f'<span class="ex-tag">{html.escape(ui["in_english"])}</span>')

    return "\n".join(
        [
            '<li class="ex-card">',
            f'<p class="ex-card__title"{text_lang}>'
            f'<a href="{html.escape(url, quote=True)}">'
            f"{html.escape(block['title'])}</a></p>",
            f'<p class="ex-card__summary"{text_lang}>'
            f"{html.escape(block['summary'])}</p>",
            f'<p class="ex-card__tags">{"".join(tags)}</p>',
            "</li>",
        ]
    )


def _render_section(lang, section, examples):
    """
    A heading with the section icon, followed by the card grid.

    The grid has no blank lines, so Markdown passes it through as one raw HTML
    block. A `featured` section gets a modifier class that highlights its cards.
    """
    grid_class = "grid cards ex-grid"
    if section.get("featured"):
        grid_class += " ex-grid--featured"
    cards = "\n".join(_render_card(lang, example) for example in examples)

    return "\n".join(
        [
            f"## :{section['icon']}:{{ .ex-section__icon }} {section[lang]} "
            f"{{ #{section['id']} }}",
            "",
            f'<div class="{grid_class}">',
            "<ul>",
            cards,
            "</ul>",
            "</div>",
        ]
    )


def _render_page(lang):
    parts = [_render_language_switcher(lang)]

    for section in DATA["sections"]:
        in_section = [
            example
            for example in DATA["examples"]
            if example["section"] == section["id"]
        ]
        if in_section:
            parts.append(_render_section(lang, section, in_section))

    return "\n\n".join(parts)


def on_page_markdown(markdown, *, page, config, files):
    """Replace the marker on the three examples pages."""
    src_path = page.file.src_uri
    for lang, cfg in DATA["languages"].items():
        if src_path != f"examples/{cfg['page']}":
            continue
        if MARKER not in markdown:
            raise PluginError(
                f"examples_grid: {src_path} does not contain the {MARKER} marker"
            )
        return markdown.replace(MARKER, _render_page(lang))
    return markdown
