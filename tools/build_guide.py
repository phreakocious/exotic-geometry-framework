#!/usr/bin/env python3
"""Build static geometry guide pages from markdown content + atlas JSON.

Reads hand-written geometry descriptions from docs/geometry-guide.md
and merges them with auto-generated metric data from the atlas JSON. Outputs
static HTML pages matching the atlas aesthetic.

Run after each atlas rebuild:
    python tools/build_guide.py

Output:
    atlas/guide/index.html          — master index
    atlas/guide/<slug>.html         — one page per geometry
"""

import json
import re
import sys
import textwrap
from html import escape
from pathlib import Path

ATLAS_JSON = Path("atlas/structure_atlas_data.json")
GUIDE_MD = Path("docs/geometry-guide.md")
OUT_DIR = Path("atlas/guide")
ATLAS_BASE = ".."  # relative path from guide/ to atlas/
SITE_BASE = "https://nullphase.net/sa/guide"





# ---------------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------------

def parse_geometry_sections(md_text: str) -> dict:
    """Parse markdown into geometry sections.

    Returns dict: geometry_name -> {
        'intro': str,        # "What it measures" paragraph
        'metrics': str,      # Full metrics section (markdown)
        'lights_up': str,    # "When it lights up" paragraph
        'tier': str,         # 'tier1', 'tier2', or 'tier3'
    }
    """
    geometries = {}
    current_tier = "tier1"

    # Split on ## headers (geometry names)
    parts = re.split(r'^## ', md_text, flags=re.MULTILINE)

    for part in parts[1:]:  # skip preamble
        lines = part.strip().split('\n')
        name = lines[0].strip()

        # Check for tier header
        if name.startswith("Tier 2") or name.startswith("# Tier 2"):
            current_tier = "tier2"
            continue
        if name.startswith("Tier 3") or name.startswith("# Tier 3"):
            current_tier = "tier3"
            continue
        if name.startswith("Tier ") or name.startswith("# Tier"):
            continue

        body = '\n'.join(lines[1:]).strip()

        # Extract sections
        intro = ""
        metrics_text = ""
        lights_up = ""

        # "What it measures" — everything before ### Metrics
        m = re.search(r'^### Metrics\s*$', body, re.MULTILINE)
        if m:
            intro = body[:m.start()].strip()
            rest = body[m.end():].strip()
        else:
            intro = body
            rest = ""

        # "When it lights up" — everything after ### When it lights up
        m2 = re.search(r'^### When it lights up\s*$', rest, re.MULTILINE)
        if m2:
            metrics_text = rest[:m2.start()].strip()
            lights_up = rest[m2.end():].strip()
        else:
            metrics_text = rest

        geometries[name] = {
            'intro': intro,
            'metrics': metrics_text,
            'lights_up': lights_up,
            'tier': current_tier,
        }

    return geometries


def md_inline_to_html(text: str) -> str:
    """Convert basic markdown inline formatting to HTML."""
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    # Inline code
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    # mdash
    text = text.replace(' — ', ' &mdash; ')
    text = text.replace('—', '&mdash;')
    # Links to atlas sources: wrap "Name (score)" patterns
    return text


def md_metrics_to_html(text: str) -> str:
    """Convert the metrics section markdown to HTML blocks."""
    if not text:
        return ""

    blocks = []
    # Split on bold metric names at start of line
    parts = re.split(r'^(\*\*\w[\w_]*\*\*)', text, flags=re.MULTILINE)

    i = 0
    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue
        if part.startswith('**') and part.endswith('**'):
            metric_name = part[2:-2]
            desc = parts[i + 1].strip() if i + 1 < len(parts) else ""
            # Process the description: split into sentences for readability
            desc_html = md_inline_to_html(escape(desc))
            # Remove leading " — " if present
            desc_html = re.sub(r'^[\s]*&mdash;\s*', '', desc_html)
            blocks.append(
                f'<div class="metric-block">'
                f'<h4 class="metric-name">{escape(metric_name)}</h4>'
                f'<p class="metric-desc">{desc_html}</p>'
                f'</div>'
            )
            i += 2
        else:
            i += 1

    return '\n'.join(blocks)


def md_paragraph_to_html(text: str) -> str:
    """Convert markdown paragraph text to HTML paragraphs."""
    if not text:
        return ""
    # Remove the "What it measures:" bold prefix if present
    text = re.sub(r'^\*\*What it measures:\*\*\s*', '', text)
    # Remove markdown horizontal rules
    text = re.sub(r'^---+\s*$', '', text, flags=re.MULTILINE)
    paragraphs = text.split('\n\n')
    html_parts = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        html_parts.append(f'<p>{md_inline_to_html(escape(p))}</p>')
    return '\n'.join(html_parts)


# ---------------------------------------------------------------------------
# Atlas data helpers
# ---------------------------------------------------------------------------

DEGENERATE_SOURCES = {'Constant 0x00', 'Constant 0xFF'}


def _collect_ranked(ordered_indices, n, col, sources, degen_indices):
    """Collect n ranked sources, deduplicating degenerate constants.

    When two degenerate sources share the same metric value, only the first
    encountered is kept.  Both appear only when their values differ — that
    difference itself is informative.
    """
    import numpy as np
    result = []
    seen_degen_val = None
    for pos in ordered_indices:
        if len(result) >= n:
            break
        if np.isnan(col[pos]):
            continue
        if pos in degen_indices:
            val = float(col[pos])
            if seen_degen_val is not None and val == seen_degen_val:
                continue  # redundant constant — skip
            seen_degen_val = val
        result.append((sources[pos]['name'], sources[pos]['domain'], float(col[pos])))
    return result


def get_metric_extremes(data: dict, geo_name: str, n: int = 5) -> list:
    """Get top/bottom sources for each metric of a geometry.

    Returns list of dicts: {
        'metric': str,
        'top': [(name, domain, value), ...],
        'bottom': [(name, domain, value), ...],
    }
    """
    import numpy as np

    metric_names = data['metric_names']
    profiles = np.array(data['profiles'], dtype=float)
    sources = data['sources']

    prefix = f'{geo_name}:'
    indices = [(i, name.split(':')[1]) for i, name in enumerate(metric_names)
               if name.startswith(prefix)]

    degen_indices = {i for i, s in enumerate(sources)
                     if s['name'] in DEGENERATE_SOURCES}

    results = []
    for idx, metric in indices:
        col = profiles[:, idx]
        order = np.argsort(col)

        top = _collect_ranked(order[::-1], n, col, sources, degen_indices)
        bottom = _collect_ranked(order, n, col, sources, degen_indices)

        results.append({
            'metric': metric,
            'top': top,
            'bottom': bottom,
        })

    return results


def get_geometry_catalog_entry(data: dict, geo_name: str) -> dict:
    """Find geometry catalog entry by name."""
    for g in data.get('geometry_catalog', []):
        if g['name'] == geo_name:
            return g
    return {}


def slugify(name: str) -> str:
    """Convert geometry name to URL-friendly slug."""
    import unicodedata
    # Decompose unicode, strip combining marks (ö → o, etc.)
    s = unicodedata.normalize('NFKD', name)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    # Replace special math chars
    s = s.replace('²', '2').replace('ℝ', 'r').replace('ℙ', 'p')
    s = re.sub(r'[^a-z0-9]+', '-', s)
    s = s.strip('-')
    return s


def source_link(name: str) -> str:
    """Generate a deep link to the atlas viewer for a source."""
    encoded = name.replace(' ', '%20')
    return f'{ATLAS_BASE}/index.html?source={encoded}'


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

def css() -> str:
    return textwrap.dedent("""\
    :root {
        --bg: #0a0015;
        --bg-card: rgba(10,5,25,0.55);
        --text: #ddd;
        --text-muted: #888;
        --text-dim: #8b8bab;
        --accent: #00ffb4;
        --accent-hover: #7bffce;
        --accent-math: #7b8fff;
        --accent-magenta: #ff6eff;
        --accent-blue: #00b4ff;
        --border: rgba(0,255,180,0.12);
        --border-hover: rgba(0,255,180,0.3);
    }

    *, *::before, *::after { box-sizing: border-box; }

    html {
        scroll-behavior: smooth;
    }

    body {
        margin: 0;
        padding: 0;
        background: var(--bg);
        color: var(--text);
        font-family: 'Exo 2', 'Inter', system-ui, sans-serif;
        font-size: 15px;
        line-height: 1.65;
        -webkit-font-smoothing: antialiased;
    }

    .container {
        max-width: 860px;
        margin: 0 auto;
        padding: 40px 24px 80px;
    }

    /* Navigation */
    .nav {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 32px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .nav a {
        color: var(--accent);
        text-decoration: none;
        transition: color 0.2s;
    }
    .nav a:hover { color: var(--accent-hover); }
    .nav .sep { color: var(--text-dim); }
    .nav .current { color: var(--text-muted); }

    /* Page title */
    h1 {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin: 0 0 8px;
        background: linear-gradient(135deg, var(--accent), var(--accent-math));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle {
        font-size: 13px;
        color: var(--text-dim);
        margin-bottom: 32px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Glass cards */
    .card {
        background: var(--bg-card);
        padding: 24px;
        border-radius: 8px;
        border: 1px solid var(--border);
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }

    .card h2 {
        font-size: 0.75rem;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-dim);
        margin: 0 0 14px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
        font-family: 'JetBrains Mono', monospace;
    }

    .card p {
        margin: 0 0 12px;
        color: #ccc;
    }
    .card p:last-child { margin-bottom: 0; }

    /* Metric blocks */
    .metric-block {
        margin-bottom: 18px;
        padding: 14px 16px;
        background: rgba(123,143,255,0.03);
        border: 1px solid rgba(123,143,255,0.08);
        border-radius: 6px;
    }
    .metric-block:last-child { margin-bottom: 0; }

    .metric-name {
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 500;
        color: var(--accent);
        margin: 0 0 6px;
        letter-spacing: 0.02em;
    }
    .metric-name::before {
        content: '›';
        color: var(--accent-math);
        margin-right: 7px;
        font-weight: 300;
    }
    .metric-desc {
        margin: 0;
        font-size: 14px;
        color: #bbb;
        line-height: 1.7;
    }

    /* Inline code */
    code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88em;
        color: var(--accent-magenta);
        background: rgba(255,110,255,0.08);
        padding: 1px 5px;
        border-radius: 3px;
    }

    /* Metric extreme tables — 2-col grid */
    .extremes-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
    }
    .extremes-grid .extremes-table:last-child:nth-child(odd) {
        grid-column: 1 / -1;
        max-width: 50%;
    }
    .extremes-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(123,143,255,0.02);
        border: 1px solid rgba(123,143,255,0.06);
        border-radius: 5px;
        overflow: hidden;
    }
    .extremes-table caption {
        text-align: left;
        font-size: 11px;
        color: var(--accent);
        padding: 8px 10px 5px;
        font-weight: 500;
    }
    .extremes-table th {
        text-align: left;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-dim);
        padding: 3px 10px;
        border-bottom: 1px solid var(--border);
        font-weight: 400;
    }
    .extremes-table td {
        padding: 2px 10px;
        color: #aaa;
        border-bottom: 1px solid rgba(255,255,255,0.02);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .extremes-table td.val {
        color: var(--accent);
        text-align: right;
        font-weight: 500;
    }
    .extremes-table td.domain {
        color: var(--text-dim);
        font-size: 10px;
    }
    .extremes-table a {
        color: #ccc;
        text-decoration: none;
        transition: color 0.2s;
    }
    .extremes-table a:hover {
        color: var(--accent);
        text-decoration: underline;
    }
    .extremes-table .separator td {
        padding: 1px 10px;
        border: none;
        color: var(--text-dim);
        font-size: 9px;
    }
    @media (max-width: 600px) {
        .extremes-grid {
            grid-template-columns: 1fr;
        }
        .extremes-grid .extremes-table:last-child:nth-child(odd) {
            max-width: 100%;
        }
    }

    /* Properties badge row */
    .props {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 24px;
    }
    .prop-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        padding: 3px 10px;
        border-radius: 12px;
        border: 1px solid var(--border);
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .prop-badge.highlight {
        border-color: var(--accent);
        color: var(--accent);
        background: rgba(0,255,180,0.06);
    }

    /* Index page */
    .geo-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 14px;
    }
    .geo-card {
        background: var(--bg-card);
        padding: 18px;
        border-radius: 8px;
        border: 1px solid var(--border);
        backdrop-filter: blur(10px);
        transition: border-color 0.2s, transform 0.15s;
        text-decoration: none;
        display: block;
    }
    .geo-card:hover {
        border-color: var(--border-hover);
        transform: translateY(-2px);
    }
    .geo-card h3 {
        font-size: 15px;
        font-weight: 600;
        color: var(--text);
        margin: 0 0 6px;
    }
    .geo-card .detects {
        font-size: 12px;
        color: var(--text-muted);
        margin: 0 0 8px;
    }
    .geo-card .badge-row {
        display: flex;
        gap: 6px;
        flex-wrap: wrap;
    }
    .geo-card .badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 9px;
        padding: 2px 7px;
        border-radius: 10px;
        border: 1px solid rgba(123,143,255,0.2);
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .geo-card .badge.ordinal {
        border-color: var(--accent);
        color: var(--accent);
    }

    /* Section headers on index */
    .tier-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-dim);
        margin: 40px 0 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
        font-weight: 400;
    }
    .tier-header:first-of-type { margin-top: 0; }

    /* Atlas link button */
    .atlas-link {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--accent);
        text-decoration: none;
        padding: 6px 14px;
        border: 1px solid var(--accent);
        border-radius: 4px;
        transition: all 0.2s;
        margin-top: 8px;
    }
    .atlas-link:hover {
        background: rgba(0,255,180,0.1);
        color: var(--accent-hover);
    }

    /* Footer */
    .footer {
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid var(--border);
        font-size: 11px;
        color: var(--text-dim);
        font-family: 'JetBrains Mono', monospace;
    }
    .footer a {
        color: var(--accent);
        text-decoration: none;
    }
    .footer a:hover { color: var(--accent-hover); }

    /* Strong in descriptions */
    strong { color: #eee; font-weight: 600; }

    /* Responsive */
    @media (max-width: 600px) {
        .container { padding: 24px 16px 60px; }
        h1 { font-size: 1.4rem; }
        .geo-grid { grid-template-columns: 1fr; }
    }
    """)


def head_html(title: str, slug: str = "") -> str:
    page_url = f'{SITE_BASE}/{slug}.html' if slug else f'{SITE_BASE}/'
    og_desc = (f'Geometry guide: {escape(title)}. '
               f'What it measures, how it works, atlas rankings.')
    return (
        f'<!DOCTYPE html>\n'
        f'<html lang="en">\n'
        f'<head>\n'
        f'<meta charset="UTF-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f'<title>{escape(title)} \u2014 Structure Atlas</title>\n'
        f'<meta name="description" content="Exotic geometry guide: {escape(title)}. '
        f'Part of the Structure Atlas \u2014 211 data sources analyzed through 200 '
        f'metrics across 54 exotic geometries.">\n'
        f'<link rel="canonical" href="{page_url}" />\n'
        f'<meta property="og:type" content="article" />\n'
        f'<meta property="og:url" content="{page_url}" />\n'
        f'<meta property="og:title" content="{escape(title)} \u2014 Structure Atlas" />\n'
        f'<meta property="og:description" content="{og_desc}" />\n'
        f'<meta property="og:image" content="{SITE_BASE}/og-image.png" />\n'
        f'<meta property="og:image:width" content="1200" />\n'
        f'<meta property="og:image:height" content="630" />\n'
        f'<meta name="twitter:card" content="summary_large_image" />\n'
        f'<meta name="twitter:title" content="{escape(title)} \u2014 Structure Atlas" />\n'
        f'<meta name="twitter:description" content="{og_desc}" />\n'
        f'<meta name="twitter:image" content="{SITE_BASE}/og-image.png" />\n'
        f'<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        f'<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        f'<link href="https://fonts.googleapis.com/css2?family=Exo+2:ital,wght@0,300;'
        f'0,400;0,600;0,700;0,800;1,300&family=JetBrains+Mono:wght@300;400;500'
        f'&display=swap" rel="stylesheet">\n'
        f'<style>\n{css()}</style>\n'
        f'<script src="/edgeviz.js?id=NP_CLIENT_DC0AF9577AD33806"></script>\n'
        f'</head>\n'
    )


def nav_html(crumbs: list) -> str:
    """Generate breadcrumb nav. crumbs = [(label, href), ...], last is current."""
    parts = []
    for i, (label, href) in enumerate(crumbs):
        if i == len(crumbs) - 1:
            parts.append(f'<span class="current">{escape(label)}</span>')
        else:
            parts.append(f'<a href="{href}">{escape(label)}</a>')
            parts.append('<span class="sep">/</span>')
    return f'<nav class="nav">{"".join(parts)}</nav>'


def extremes_html(extremes: list) -> str:
    """Generate metric extreme tables."""
    if not extremes:
        return ""

    tables = []
    for ext in extremes:
        rows = []
        # Top values
        for name, domain, val in ext['top'][:3]:
            link = source_link(name)
            rows.append(
                f'<tr>'
                f'<td><a href="{link}">{escape(name)}</a></td>'
                f'<td class="domain">{escape(domain)}</td>'
                f'<td class="val">{val:.4f}</td>'
                f'</tr>'
            )
        # Separator
        rows.append(
            '<tr class="separator"><td colspan="3">···</td></tr>'
        )
        # Bottom values (nonzero ones are more interesting)
        for name, domain, val in ext['bottom'][:3]:
            link = source_link(name)
            rows.append(
                f'<tr>'
                f'<td><a href="{link}">{escape(name)}</a></td>'
                f'<td class="domain">{escape(domain)}</td>'
                f'<td class="val">{val:.4f}</td>'
                f'</tr>'
            )

        tables.append(
            f'<table class="extremes-table">'
            f'<caption>{escape(ext["metric"])}</caption>'
            f'<thead><tr><th style="width:55%">Source</th><th style="width:25%">Domain</th><th style="width:20%">Value</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody>'
            f'</table>'
        )

    return '<div class="extremes-grid">\n' + '\n'.join(tables) + '\n</div>'


def footer_html(n_sources: int, n_metrics: int, n_geos: int) -> str:
    return (
        f'<div class="footer">'
        f'<p>Structure Atlas &mdash; {n_sources} sources, {n_metrics} metrics, '
        f'{n_geos} geometries &mdash; '
        f'<a href="{ATLAS_BASE}/index.html">Open Atlas</a> &middot; '
        f'<a href="https://github.com/phreakocious/exotic-geometry-framework">GitHub</a></p>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------

def build_geometry_page(geo_name: str, content: dict, catalog: dict,
                        extremes: list, n_sources: int, n_metrics: int,
                        n_geos: int, all_geos: list) -> str:
    """Build a single geometry page."""
    slug = slugify(geo_name)
    view = catalog.get('view', '')
    detects = catalog.get('detects', '')
    enc_inv = catalog.get('encoding_invariant', False)
    metrics_list = catalog.get('metrics', [])
    dim = catalog.get('dimension', '')

    # Property badges
    badges = []
    if view:
        badges.append(f'<span class="prop-badge">{escape(str(view))}</span>')
    if enc_inv:
        badges.append('<span class="prop-badge highlight">encoding-invariant</span>')
    if dim:
        badges.append(f'<span class="prop-badge">dim {escape(str(dim))}</span>')
    badges.append(f'<span class="prop-badge">{len(metrics_list)} metrics</span>')

    # Prev/next navigation
    idx = next((i for i, g in enumerate(all_geos) if g == geo_name), -1)
    prev_link = ""
    next_link = ""
    if idx > 0:
        prev_name = all_geos[idx - 1]
        prev_link = f'<a href="{slugify(prev_name)}.html">&larr; {escape(prev_name)}</a>'
    if idx < len(all_geos) - 1:
        next_name = all_geos[idx + 1]
        next_link = f'<a href="{slugify(next_name)}.html">{escape(next_name)} &rarr;</a>'

    page_nav = ""
    if prev_link or next_link:
        page_nav = (
            f'<div style="display:flex;justify-content:space-between;'
            f'font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'margin-top:32px">'
            f'<span>{prev_link}</span><span>{next_link}</span></div>'
        )

    html = head_html(geo_name, slug)
    html += '<body>\n<div class="container">\n'

    # Nav
    html += nav_html([
        ('Atlas', f'{ATLAS_BASE}/index.html'),
        ('Guide', 'index.html'),
        (geo_name, '#'),
    ])

    # Title + badges
    html += f'<h1>{escape(geo_name)}</h1>\n'
    if detects:
        html += f'<div class="subtitle">{escape(detects)}</div>\n'
    if badges:
        html += f'<div class="props">{"".join(badges)}</div>\n'

    # What it measures
    if content.get('intro'):
        html += '<div class="card">\n'
        html += '<h2>What It Measures</h2>\n'
        html += md_paragraph_to_html(content['intro'])
        html += '\n</div>\n'

    # Metrics (hand-written descriptions)
    if content.get('metrics'):
        html += '<div class="card">\n'
        html += '<h2>Metrics</h2>\n'
        html += md_metrics_to_html(content['metrics'])
        html += '\n</div>\n'

    # Metric extremes (auto-generated from atlas data)
    if extremes:
        html += '<div class="card">\n'
        html += '<h2>Atlas Rankings</h2>\n'
        html += extremes_html(extremes)
        html += '</div>\n'

    # When it lights up
    if content.get('lights_up'):
        html += '<div class="card">\n'
        html += '<h2>When It Lights Up</h2>\n'
        html += md_paragraph_to_html(content['lights_up'])
        html += '\n</div>\n'

    # Atlas link
    html += (
        f'<a class="atlas-link" href="{ATLAS_BASE}/index.html">'
        f'Open in Atlas</a>\n'
    )

    html += page_nav
    html += footer_html(n_sources, n_metrics, n_geos)
    html += '\n</div>\n</body>\n</html>\n'
    return html


def build_index_page(geometries: dict, catalog_lookup: dict,
                     n_sources: int, n_metrics: int, n_geos: int,
                     all_catalog: list) -> str:
    """Build the master index page."""
    html = head_html("Geometry Guide")
    html += '<body>\n<div class="container">\n'

    html += nav_html([
        ('Atlas', f'{ATLAS_BASE}/index.html'),
        ('Guide', '#'),
    ])

    html += '<h1>Geometry Guide</h1>\n'
    html += (
        f'<div class="subtitle">{n_geos} exotic geometries &mdash; '
        f'{n_sources} data sources &mdash; {n_metrics} metrics</div>\n'
    )

    # Intro
    html += '<div class="card">\n'
    html += (
        '<p>Each geometry is a question asked of data. The answers are gradients '
        'from low to high, with specific atlas sources as landmarks. '
        'These pages explain what each geometry detects through contrast '
        'and atlas examples, not mathematical definitions.</p>\n'
    )
    html += '</div>\n'

    # Group by tier: Tier 1 (encoding-invariant), Tier 2 (high-discrimination),
    # then all remaining from catalog
    tier1 = [(name, c) for name, c in geometries.items() if c['tier'] == 'tier1']
    tier2 = [(name, c) for name, c in geometries.items() if c['tier'] == 'tier2']
    tier3 = [(name, c) for name, c in geometries.items() if c['tier'] == 'tier3']

    # Get names of geometries we have content for
    written = set(geometries.keys())

    # Remaining geometries from catalog (no content yet)
    remaining = []
    for g in all_catalog:
        if g['name'] not in written:
            remaining.append(g)

    def geo_cards(items, from_content=True) -> str:
        cards = []
        for item in items:
            if from_content:
                name, content = item
                cat = catalog_lookup.get(name, {})
            else:
                cat = item
                name = cat['name']

            slug = slugify(name)
            view = cat.get('view', '')
            detects = cat.get('detects', '')
            enc_inv = cat.get('encoding_invariant', False)
            has_page = name in written

            badge_html = ""
            badges = []
            if view:
                badges.append(f'<span class="badge">{escape(view)}</span>')
            if enc_inv:
                badges.append('<span class="badge ordinal">ordinal</span>')
            if badges:
                badge_html = f'<div class="badge-row">{"".join(badges)}</div>'

            if has_page:
                cards.append(
                    f'<a class="geo-card" href="{slug}.html">'
                    f'<h3>{escape(name)}</h3>'
                    f'<p class="detects">{escape(detects)}</p>'
                    f'{badge_html}'
                    f'</a>'
                )
            else:
                cards.append(
                    f'<div class="geo-card" style="opacity:0.5;cursor:default">'
                    f'<h3>{escape(name)}</h3>'
                    f'<p class="detects">{escape(detects)}</p>'
                    f'{badge_html}'
                    f'</div>'
                )

        return '\n'.join(cards)

    if tier1:
        html += '<div class="tier-header">Tier 1 &mdash; Encoding-Invariant</div>\n'
        html += f'<div class="geo-grid">\n{geo_cards(tier1)}\n</div>\n'

    if tier2:
        html += '<div class="tier-header">Tier 2 &mdash; High Discrimination</div>\n'
        html += f'<div class="geo-grid">\n{geo_cards(tier2)}\n</div>\n'

    if tier3:
        html += '<div class="tier-header">Tier 3 &mdash; Remaining Geometries</div>\n'
        html += f'<div class="geo-grid">\n{geo_cards(tier3)}\n</div>\n'

    if remaining:
        html += '<div class="tier-header">Uncategorized</div>\n'
        html += f'<div class="geo-grid">\n{geo_cards(remaining, from_content=False)}\n</div>\n'

    html += footer_html(n_sources, n_metrics, n_geos)
    html += '\n</div>\n</body>\n</html>\n'
    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not ATLAS_JSON.exists():
        print(f"Error: {ATLAS_JSON} not found", file=sys.stderr)
        sys.exit(1)
    if not GUIDE_MD.exists():
        print(f"Error: {GUIDE_MD} not found", file=sys.stderr)
        sys.exit(1)

    # Load data
    data = json.loads(ATLAS_JSON.read_text())
    md_text = GUIDE_MD.read_text()

    n_sources = len(data['sources'])
    n_metrics = data.get('n_metrics', len(data.get('metric_names', [])))
    n_geos = data.get('n_geometries', len(data.get('geometry_catalog', [])))
    all_catalog = data.get('geometry_catalog', [])

    # Parse markdown
    geometries = parse_geometry_sections(md_text)

    # Build catalog lookup
    catalog_lookup = {g['name']: g for g in all_catalog}

    # Ordered list of all geometries with content (for prev/next nav)
    all_geo_names = list(geometries.keys())

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build geometry pages
    for geo_name, content in geometries.items():
        catalog = catalog_lookup.get(geo_name, {})
        extremes = get_metric_extremes(data, geo_name)
        slug = slugify(geo_name)

        page_html = build_geometry_page(
            geo_name, content, catalog, extremes,
            n_sources, n_metrics, n_geos, all_geo_names,
        )
        out_path = OUT_DIR / f'{slug}.html'
        out_path.write_text(page_html)
        print(f"  {out_path}")

    # Build index
    index_html = build_index_page(
        geometries, catalog_lookup,
        n_sources, n_metrics, n_geos, all_catalog,
    )
    index_path = OUT_DIR / 'index.html'
    index_path.write_text(index_html)
    print(f"  {index_path}")

    print(f"\nGuide built: {len(geometries)} geometry pages + index")
    print(f"  {n_sources} sources, {n_metrics} metrics, {n_geos} geometries")


if __name__ == "__main__":
    main()
