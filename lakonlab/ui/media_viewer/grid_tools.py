# Copyright (c) 2025 Hansheng Chen

import html
import json
import os
import pathlib
from urllib.parse import quote


ASSET_DIR = pathlib.Path(__file__).parent
TEMPLATE = (ASSET_DIR / "viewer.html").read_text(encoding="utf-8")


def build_thumbnails(entries, n_cols):
    """return markup, sources list, captions list"""
    sources, caps, blocks = [], [], []
    for i, (d, src, cap) in enumerate(entries):
        if not (src.startswith("http://") or src.startswith("https://")):
            src = quote(src, safe="/%")
        esc_src = html.escape(src)
        sources.append(esc_src)
        caps.append(cap)

        ext = os.path.splitext(esc_src)[-1].lower()
        thumb = (f'<video src="{esc_src}" preload="metadata" muted></video>'
                 if ext == '.mp4' else f'<img src="{esc_src}" alt="thumb">')
        blocks.append(f'<div class="item" data-idx="{i}">{thumb}'
                      f'<textarea class="prompt" readonly>'
                      f'{html.escape(cap)}</textarea></div>')
    grid = (f'<div class="grid" style="grid-template-columns:repeat({n_cols},1fr);">\n'
            if n_cols else '<div class="grid">\n')
    return grid + '\n'.join(blocks) + '\n</div>', sources, caps


def grid_html(entries, n_cols=None, *, inline_assets=False):
    grid_markup, srcs, caps = build_thumbnails(entries, n_cols)
    blob = f'<script>window.GRID_DATA={json.dumps({"sources": srcs, "captions": caps})};</script>'
    page = TEMPLATE.replace("{{GRID_MARKUP}}", grid_markup + blob)
    if inline_assets:
        css = (ASSET_DIR / "viewer.css").read_text(encoding="utf-8")
        js = (ASSET_DIR / "viewer.js").read_text(encoding="utf-8")
        page = page.replace(
            '<link rel="stylesheet" href="viewer.css" />', f'<style>\n{css}\n</style>'
        ).replace(
            '<script src="viewer.js"></script>', f'<script>\n{js}\n</script>')
    return page


def write_html(html_path, entries, file_client):
    if not entries:
        return
    formatted = [(d, img, f'[{d}] {name}') for d, img, name in entries]
    file_client.put_text(grid_html(formatted, inline_assets=True), html_path)
