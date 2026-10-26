"""One-run PDF report: an HTML page printed to PDF by headless Chrome (handles Korean text and CSS layout).

    report.pdf next to the run's scene.json
"""

import base64
import html
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import markdown

from .config import ROOT

IMAGE_WIDTH = 1400  # px; photos are downscaled and JPEG-encoded so the PDF stays a few MB
CHROME_NAMES = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm; }
:root { --ink: #22303A; --muted: #6E7B83; --line: #D6DDE1; --accent: #2C5E9E; --soft: #F1F4F5; }
* { box-sizing: border-box; }
body { font-family: 'Instrument Sans', 'Noto Sans CJK KR', 'Noto Sans KR', sans-serif; color: var(--ink);
       font-size: 9.5pt; line-height: 1.5; margin: 0; }
h1 { font-size: 20pt; letter-spacing: -0.02em; margin: 0 0 2pt; }
h2 { font-size: 12pt; margin: 18pt 0 6pt; padding-bottom: 3pt; border-bottom: 1.5pt solid var(--ink); }
.sub { color: var(--muted); margin: 0 0 12pt; }
.facts { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6pt; margin-bottom: 4pt; }
.fact { background: var(--soft); border-radius: 4pt; padding: 6pt 8pt; }
.fact b { display: block; font-size: 13pt; font-variant-numeric: tabular-nums; }
.fact span { color: var(--muted); font-size: 8pt; }
figure { margin: 0 0 8pt; break-inside: avoid; }
figure img { width: 100%; border-radius: 3pt; display: block; }
figcaption { color: var(--muted); font-size: 8pt; margin-top: 2pt; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8pt; }
table { width: 100%; border-collapse: collapse; font-size: 8.5pt; font-variant-numeric: tabular-nums; }
th { text-align: left; color: var(--muted); font-weight: 600; border-bottom: 1pt solid var(--line); padding: 3pt 4pt; }
td { border-bottom: 0.5pt solid var(--line); padding: 3pt 4pt; vertical-align: top; }
tr { break-inside: avoid; }
td.num, th.num { text-align: right; }
.swatch { display: inline-block; width: 18pt; height: 10pt; border-radius: 2pt; border: 0.5pt solid var(--line);
          vertical-align: middle; }
.thinking { columns: 2; column-gap: 14pt; font-size: 8.5pt; }
.thinking p { margin: 0 0 5pt; }
.thinking strong { color: var(--accent); }
.note { color: var(--muted); font-size: 8pt; }
footer { margin-top: 16pt; color: var(--muted); font-size: 7.5pt; border-top: 0.5pt solid var(--line); padding-top: 4pt; }
"""


def _img(path: Path) -> str:
    img = cv2.imread(str(path))
    if img.shape[1] > IMAGE_WIDTH:
        img = cv2.resize(img, (IMAGE_WIDTH, round(img.shape[0] * IMAGE_WIDTH / img.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _fmt(v, digits: int | None = None) -> str:
    if v is None or v == "":
        return "-"
    if isinstance(v, float) and digits is not None:
        return f"{v:.{digits}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return html.escape(str(v))


def _table(headers: list[str], rows: list[list[str]], numeric: set[int] = frozenset()) -> str:
    head = "".join(f"<th class='{'num' if i in numeric else ''}'>{h}</th>" for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f"<td class='{'num' if i in numeric else ''}'>{c}</td>" for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _asset_cell(o: dict) -> str:
    """The object's search phrase and what it found, with the credit a CC-BY model needs."""
    if not o.get("asset_query"):
        return "-"
    m = o.get("asset_match") or {}
    if not m.get("path"):
        found = "없음 (도형)"
    elif m.get("source") == "objaverse":
        found = f"Objaverse: {_fmt(m.get('name'))} ({_fmt(m.get('license'))}, {_fmt(m.get('author'))})"
    else:
        found = f"Isaac Sim: {_fmt(m.get('text'))}"
    return f"{_fmt(o['asset_query'])}<br><span class='note'>→ {found}</span>"


def build_html(run_dir: Path) -> str:
    """The report page for one run folder (results/<scene>/<experiment>/<model folder>/<condition>/<time>/)."""
    run_dir = run_dir.resolve()
    run = json.loads((run_dir / "run.json").read_text())
    spec = json.loads((run_dir / "scene.json").read_text()) if (run_dir / "scene.json").exists() else None
    model_folder = "/".join(run_dir.parts[-4:-2])
    started = datetime.strptime(run_dir.name, "%Y%m%d_%H%M%S")

    facts = [
        (run.get("num_objects"), "찾은 물체"), (len(run.get("images", [])), "보낸 이미지"),
        (run.get("input_tokens"), "입력 토큰"), (run.get("output_tokens"), "출력 토큰"),
        (run.get("thinking_tokens"), "생각 토큰"), (run.get("thinking", "-"), "생각 깊이 설정"),
        (f"${run.get('cost_usd') or 0:.4f}", "비용"), (f"{run.get('latency_s') or 0:.1f} s", "응답 시간"),
    ]
    parts = [
        f"<h1>{html.escape(model_folder)}, {html.escape(run.get('condition', '-'))}, {html.escape(run.get('scene', '-'))}</h1>",
        f"<p class='sub'>{html.escape(run.get('model_id', '-'))}, {started:%Y-%m-%d %H:%M} 실행</p>",
        "<div class='facts'>" + "".join(f"<div class='fact'><b>{_fmt(v)}</b><span>{k}</span></div>" for v, k in facts)
        + "</div>",
    ]

    if (run_dir / "compare.png").exists():
        parts += ["<h2>사진과 재구성 비교</h2>",
                  f"<figure><img src='{_img(run_dir / 'compare.png')}'><figcaption>왼쪽 front 사진, 오른쪽 모델 출력으로 만든 "
                  "Isaac Sim 렌더 (모델이 추정한 카메라 위치)</figcaption></figure>"]
    else:
        parts.append("<p class='note'>Isaac Sim 렌더가 아직 없습니다. 앱이나 view --capture로 렌더하면 보고서에 들어갑니다.</p>")

    boxes = sorted(run_dir.glob("*_boxes.png"), key=lambda p: (not p.name.startswith("front"), p.name))
    if boxes:
        parts.append("<h2>모델이 표시한 물체 위치</h2><div class='grid2'>" + "".join(
            f"<figure><img src='{_img(p)}'><figcaption>{html.escape(p.stem.removesuffix('_boxes'))}</figcaption></figure>"
            for p in boxes) + "</div>")

    if spec is None:
        parts.append("<h2>모델 출력</h2><p>출력이 JSON이 아니어서 장면을 만들지 못했습니다.</p>")
    else:
        rows = []
        for o in spec.get("objects", []):
            size = o.get("size_mm") or {}
            rgb = [min(max(float(c), 0.0), 1.0) for c in (o.get("color_rgb") or [0.5, 0.5, 0.5])[:3]]
            hexc = "#{:02x}{:02x}{:02x}".format(*(round(c * 255) for c in rgb))
            rows.append([
                _fmt(o.get("id")), _fmt(o.get("identified_as")),
                _fmt(o.get("shape")) + (" (lying)" if o.get("lying") and o.get("shape") in ("cylinder", "capsule") else "")
                + (f", 부품 {len(o['parts'])}" if o.get("parts") else ""),
                "×".join(str(size.get(k, "?")) for k in ("length", "width", "height")),
                ", ".join(f"{float(v):.3f}" for v in o.get("position", [])),
                _fmt(o.get("yaw_deg"), 0), _fmt(o.get("mass_kg"), 3),
                f"<span class='swatch' style='background:{hexc}'></span>", _fmt(o.get("confidence"), 2),
                _asset_cell(o),
            ])
        parts += ["<h2>물체</h2>", _table(
            ["id", "무엇", "모양", "크기 mm (L×W×H)", "위치 m (x, y, z)", "yaw°", "질량 kg", "색", "확신도", "에셋 검색"],
            rows, numeric={5, 6, 8})]

        desk = (spec.get("environment") or {}).get("desk") or {}
        rel = spec.get("relations") or {}
        desk_html = "<p>테이블 없이 바닥에 놓인 장면입니다.</p>"
        if desk:
            size = desk.get("size") or {}
            desk_html = _table(["항목", "값"], [
                ["무엇", _fmt(desk.get("identified_as"))], ["윗면 모양", _fmt(desk.get("top_shape"))],
                ["크기 (m)", f"{_fmt(size.get('x'))} × {_fmt(size.get('y'))}"],
                ["높이 (m)", _fmt(desk.get("height_from_floor"))], ["다리", _fmt(len(desk.get("legs") or []))],
                ["텍스처", _fmt(desk.get("texture") or "없음")],
            ])
        rel_html = _table(["a", "관계", "b"], [[_fmt(i.get("a")), _fmt(i.get("rel")), _fmt(i.get("b"))]
                                              for i in rel.get("items") or []])
        focus = spec.get("focus_object") or {}
        if focus:
            parts.append(f"<h2>모델이 판단한 중심 물체</h2><p><b>{_fmt(focus.get('id'))}</b>: {_fmt(focus.get('reason'))}</p>")
        parts.append(f"<div class='grid2'><div><h2>테이블</h2>{desk_html}</div>"
                     f"<div><h2>위치 관계, 기준 {_fmt(rel.get('anchor'))}</h2>{rel_html}</div></div>")

    thinking = run_dir / "thinking_ko.md"
    if thinking.exists() and thinking.read_text().strip():
        parts += ["<h2>모델의 생각 (요약의 한국어 번역)</h2>",
                  f"<div class='thinking'>{markdown.markdown(thinking.read_text())}</div>"]

    parts.append(f"<footer>Glimpse2Sim 보고서, {html.escape(str(run_dir.relative_to(ROOT)))}, "
                 f"프롬프트 {html.escape(run.get('prompt_sha', '-'))}, 생성 {datetime.now():%Y-%m-%d %H:%M}</footer>")
    return (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{html.escape(model_folder)} report</title>"
            "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600;700&display=swap'>"
            f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>")


def _print_pdf(page_html: str, out: Path) -> Path:
    """Print an HTML page to `out` with headless Chrome."""
    chrome = next((shutil.which(n) for n in CHROME_NAMES if shutil.which(n)), None)
    if chrome is None:
        raise RuntimeError(f"no Chrome or Chromium found (tried {', '.join(CHROME_NAMES)})")
    out = out.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(page_html)
        subprocess.run(
            [chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={tmp}/profile",
             "--virtual-time-budget=8000", f"--print-to-pdf={out}", page.as_uri()],
            check=True, capture_output=True, timeout=120,
        )
    return out


def write_pdf(run_dir: Path) -> Path:
    """Render the report of one run to run_dir/report.pdf with headless Chrome. Return its path."""
    return _print_pdf(build_html(run_dir), run_dir / "report.pdf")


# --- One model, one scene, every level -------------------------------------------------------------------------------

MATCH_RADIUS_M = 0.10  # objects of two levels within this xy distance are taken to be the same object


def latest_runs(model_dir: Path) -> dict[str, Path]:
    """Level -> the latest run folder under results/<scene>/<experiment>/<model folder> that has a scene.json
    (L1, L2, L3 order)."""
    runs = {}
    for level_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        done = sorted(d for d in level_dir.glob("*/") if (d / "scene.json").exists())
        if done:
            runs[level_dir.name] = done[-1]
    return runs


def _match_objects(specs: dict[str, dict]) -> list[dict[str, dict | None]]:
    """Rows of the same physical object across levels, matched greedily by nearest xy position.

    The level that found the most objects is the reference; objects no reference object is near get rows of
    their own.
    """
    ref_level = max(specs, key=lambda k: (len(specs[k].get("objects", [])), k))
    rows = [{ref_level: o} for o in specs[ref_level].get("objects", [])]
    for level, spec in specs.items():
        if level == ref_level:
            continue
        free = list(spec.get("objects", []))
        pairs = sorted(
            ((((r[ref_level]["position"][0] - o["position"][0]) ** 2
               + (r[ref_level]["position"][1] - o["position"][1]) ** 2) ** 0.5, i, j)
             for i, r in enumerate(rows) if ref_level in r for j, o in enumerate(free)),
        )
        used_r, used_o = set(), set()
        for dist, i, j in pairs:
            if dist <= MATCH_RADIUS_M and i not in used_r and j not in used_o:
                rows[i][level] = free[j]
                used_r.add(i)
                used_o.add(j)
        rows += [{level: o} for j, o in enumerate(free) if j not in used_o]
    return rows


def build_summary_html(model_dir: Path) -> str:
    """Report comparing the latest L1, L2, L3 runs of one model on one scene in one experiment round
    (model_dir = results/<scene>/<experiment>/<model folder>)."""
    model_dir = model_dir.resolve()
    model_folder = "/".join(model_dir.parts[-2:])
    scene, experiment = model_dir.parts[-4:-2]
    runs = latest_runs(model_dir)
    if not runs:
        raise FileNotFoundError(f"no runs of {scene} under {model_dir}")
    levels = list(runs)
    meta = {k: json.loads((d / "run.json").read_text()) for k, d in runs.items()}
    specs = {k: json.loads((d / "scene.json").read_text()) for k, d in runs.items()}

    parts = [f"<h1>{html.escape(model_folder)}, {html.escape(scene)}, 실험 {html.escape(experiment)}</h1>",
             f"<p class='sub'>{html.escape(meta[levels[0]].get('model_id', '-'))}, 단계 {', '.join(levels)}의 최근 실행 비교</p>"]

    rows = []
    for k in levels:
        m, spec = meta[k], specs[k]
        focus = (spec.get("focus_object") or {}).get("id")
        rows.append([f"<b>{k}</b>", _fmt(len(m.get("images", []))), _fmt(m.get("num_objects")),
                     _fmt(m.get("input_tokens")), _fmt(m.get("output_tokens")), _fmt(m.get("thinking_tokens")),
                     f"${m.get('cost_usd') or 0:.4f}", f"{m.get('latency_s') or 0:.1f}", _fmt(focus),
                     _fmt((spec.get("relations") or {}).get("anchor"))])
    total_cost = sum(meta[k].get("cost_usd") or 0 for k in levels)
    total_time = sum(meta[k].get("latency_s") or 0 for k in levels)
    rows.append(["<b>합계</b>", "", "", "", "", "", f"<b>${total_cost:.4f}</b>", f"<b>{total_time:.1f}</b>", "", ""])
    parts += ["<h2>단계별 요약</h2>", _table(
        ["단계", "이미지", "물체", "입력 토큰", "출력 토큰", "생각 토큰", "비용", "시간 s", "중심 물체", "관계 기준"],
        rows, numeric={1, 2, 3, 4, 5, 6, 7})]

    desk_rows = []
    for k in levels:
        d = (specs[k].get("environment") or {}).get("desk") or {}
        size = d.get("size") or {}
        desk_rows.append([f"<b>{k}</b>", _fmt(d.get("identified_as")), _fmt(d.get("top_shape")),
                          f"{_fmt(size.get('x'))} × {_fmt(size.get('y'))}", _fmt(d.get("height_from_floor")),
                          _fmt(len(d.get("legs") or []))])
    parts += ["<h2>테이블</h2>", _table(["단계", "무엇", "윗면 모양", "크기 m (x × y)", "높이 m", "다리"], desk_rows)]

    figures = [(k, runs[k] / "compare.png") for k in levels if (runs[k] / "compare.png").exists()]
    if figures:
        parts.append("<h2>사진과 재구성 비교</h2>" + "".join(
            f"<figure><img src='{_img(path)}'><figcaption>{k}: 왼쪽 front 사진, 오른쪽 Isaac Sim 렌더</figcaption></figure>"
            for k, path in figures))
    missing = [k for k in levels if not (runs[k] / "compare.png").exists()]
    if missing:
        parts.append(f"<p class='note'>{', '.join(missing)}은 아직 렌더하지 않아 비교 그림이 없습니다.</p>")

    matched = _match_objects(specs)

    def name(row: dict) -> str:
        o = next(o for o in row.values() if o)
        return f"<b>{_fmt(o.get('id'))}</b><br><span class='note'>{_fmt(o.get('identified_as'))}</span>"

    def cell(row: dict, k: str, what: str) -> str:
        o = row.get(k)
        if not o:
            return "<span class='note'>없음</span>"
        if what == "size":
            size = o.get("size_mm") or {}
            return "×".join(str(size.get(a, "?")) for a in ("length", "width", "height"))
        if what == "pos":
            return ", ".join(f"{float(v):.3f}" for v in o.get("position", [])[:2])
        return _fmt(o.get("yaw_deg"), 0)

    parts.append("<h2>물체별 비교</h2><p class='note'>단계 사이의 같은 물체는 테이블 위 위치(x, y)가 "
                 f"{MATCH_RADIUS_M * 100:.0f} cm 안에 있는 것끼리 묶었습니다. 이름은 물체를 가장 많이 찾은 단계 기준입니다.</p>")
    for what, title in [("size", "크기 mm (L×W×H)"), ("pos", "위치 m (x, y)"), ("yaw", "yaw°")]:
        parts += [f"<h3>{title}</h3>", _table(["물체", *levels], [[name(r), *(cell(r, k, what) for k in levels)]
                                                               for r in matched], numeric=set(range(1, len(levels) + 1)))]

    parts.append(f"<footer>Glimpse2Sim 종합 보고서, {html.escape(str(model_dir.relative_to(ROOT)))}, "
                 + ", ".join(f"{k} {runs[k].name}" for k in levels) + f", 생성 {datetime.now():%Y-%m-%d %H:%M}</footer>")
    css = CSS + "h3 { font-size: 10pt; margin: 10pt 0 4pt; } figure { margin-bottom: 10pt; }"
    return (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{html.escape(model_folder)} {scene}</title>"
            "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600;700&display=swap'>"
            f"<style>{css}</style></head><body>{''.join(parts)}</body></html>")


def write_summary_pdf(model_dir: Path) -> Path:
    """results/<scene>/<experiment>/<family>/<variant>/summary.pdf comparing the model's levels on that scene."""
    return _print_pdf(build_summary_html(model_dir), model_dir / "summary.pdf")
