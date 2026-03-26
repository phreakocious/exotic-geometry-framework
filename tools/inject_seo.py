#!/usr/bin/env python3
"""Inject crawlable SEO content into atlas/index.html from the atlas JSON.

Generates a hidden-but-crawlable <div> containing all source names,
descriptions, domain groupings, and cross-domain neighbor connections.
Also updates the og: and twitter: meta tag counts.

Run after each atlas rebuild:
    python tools/inject_seo.py
"""

import json
import re
from pathlib import Path

ATLAS_JSON = Path("atlas/structure_atlas_data.json")
ATLAS_HTML = Path("atlas/index.html")

# Marker comments for the injected block
START_MARKER = "<!-- SEO:BEGIN -->"
END_MARKER = "<!-- SEO:END -->"


def build_seo_block(data: dict) -> str:
    """Build crawlable HTML content from atlas JSON."""
    sources = data["sources"]
    n_sources = len(sources)
    n_metrics = data.get("n_metrics", "?")
    n_geos = data.get("n_geometries", len(data.get("geometry_catalog", [])))
    views = data.get("views", {})
    clusters = data.get("clusters", {})

    # Group sources by domain
    by_domain = {}
    for s in sources:
        by_domain.setdefault(s["domain"], []).append(s)

    lines = [
        START_MARKER,
        '<div id="seo-content" style="position:absolute;left:-9999px;width:1px;height:1px;overflow:hidden" aria-hidden="true">',
        f"<h1>Structure Atlas — Exotic Geometry Framework</h1>",
        f"<p>{n_sources} data sources from {len(by_domain)} domains, analyzed through "
        f"{n_metrics} metrics across {n_geos} exotic geometries.</p>",
        "",
    ]

    # Domain sections
    for domain in sorted(by_domain):
        domain_sources = by_domain[domain]
        lines.append(f"<h2>{domain.title()} ({len(domain_sources)} sources)</h2>")
        lines.append("<ul>")
        for s in sorted(domain_sources, key=lambda x: x["name"]):
            desc = s.get("description", "")
            neighbors = s.get("neighbors", [])
            cross = [n for n in neighbors if n.get("cross_domain")]
            li = f'<li><strong>{s["name"]}</strong>'
            if desc:
                li += f" — {desc}"
            if cross:
                nn_names = [n["name"] for n in cross[:3]]
                li += f' (structural neighbors: {", ".join(nn_names)})'
            li += "</li>"
            lines.append(li)
        lines.append("</ul>")

    # Views summary
    if views:
        lines.append("<h2>Geometric Views</h2>")
        lines.append("<ul>")
        for vname, vdata in sorted(views.items()):
            q = vdata.get("question", "")
            geos = vdata.get("geometries", [])
            if q:
                lines.append(f"<li><strong>{vname}</strong>: {q} ({len(geos)} geometries)</li>")
        lines.append("</ul>")

    # Geometry catalog
    geo_cat = data.get("geometry_catalog", [])
    if geo_cat:
        lines.append("<h2>Exotic Geometries</h2>")
        lines.append("<ul>")
        for g in sorted(geo_cat, key=lambda x: x.get("name", "")):
            name = g.get("name", "")
            desc = g.get("description", "")
            det = g.get("detects", "")
            li = f"<li><strong>{name}</strong>"
            if desc:
                li += f" — {desc}"
            if det:
                li += f" Detects: {det}."
            li += "</li>"
            lines.append(li)
        lines.append("</ul>")

    lines.append("</div>")
    lines.append(END_MARKER)
    return "\n".join(lines)


def update_meta_counts(html: str, n_sources: int, n_metrics: int) -> str:
    """Update hardcoded counts in og: and twitter: meta tags."""
    # Match patterns like "185 data sources" or "211 data sources"
    html = re.sub(
        r"\d+ data sources from \d+ domains, fingerprinted through \d+ exotic geometry metrics",
        f"{n_sources} data sources from 16 domains, fingerprinted through {n_metrics} exotic geometry metrics",
        html,
    )
    html = re.sub(
        r"\d+ data sources from \d+ domains, analyzed through \d+ exotic geometry metrics",
        f"{n_sources} data sources from 16 domains, analyzed through {n_metrics} exotic geometry metrics",
        html,
    )
    # JSON-LD
    html = re.sub(
        r"across \d+ sources from \d+ domains, computed through \d+ exotic geometry metrics",
        f"across {n_sources} sources from 16 domains, computed through {n_metrics} exotic geometry metrics",
        html,
    )
    return html


def main():
    data = json.loads(ATLAS_JSON.read_text())
    html = ATLAS_HTML.read_text()

    n_sources = len(data["sources"])
    n_metrics = data.get("n_metrics", 200)

    # Remove old SEO block if present
    if START_MARKER in html:
        html = re.sub(
            re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
            "",
            html,
            flags=re.DOTALL,
        )

    # Build and inject new SEO block after <body>
    seo_block = build_seo_block(data)
    html = html.replace("<body>", f"<body>\n{seo_block}", 1)

    # Update meta tag counts
    html = update_meta_counts(html, n_sources, n_metrics)

    ATLAS_HTML.write_text(html)
    print(f"Injected SEO content: {n_sources} sources, {n_metrics} metrics")
    print(f"Updated: {ATLAS_HTML}")


if __name__ == "__main__":
    main()
