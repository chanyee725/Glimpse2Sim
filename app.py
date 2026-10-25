"""Glimpse2Sim UI: run scene generation and browse the results.

    uv run streamlit run app.py
"""

import html
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

import glimpse2sim.config
import glimpse2sim.generation.scene
import glimpse2sim.models
import glimpse2sim.report

# The file watcher is off (.streamlit/config.toml), so Streamlit would keep the first-imported version of these
# modules for the life of the server. Reload them on every rerun so code edits apply without a restart.
for _module in (glimpse2sim.config, glimpse2sim.models, glimpse2sim.generation.scene, glimpse2sim.report):
    importlib.reload(_module)

from glimpse2sim.config import ROOT, load_config  # noqa: E402
from glimpse2sim.generation.scene import RESULTS_DIR  # noqa: E402
from glimpse2sim.models import THINKING_LEVELS, model_names, thinking_setting  # noqa: E402
from glimpse2sim.report import latest_runs, write_pdf, write_summary_pdf  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
LOG_LINES = 300  # lines of the running command's output kept on screen
# Rough per-call tokens for a cost estimate when a model has no earlier run of that condition.
EST_TOKENS_PER_IMAGE, EST_PROMPT_TOKENS, EST_ANSWER_TOKENS = 1300, 6000, 3500
# Thinking tokens per call by depth, from the runs so far (Flash L3 at its default thought about 9k tokens);
# a provider default is taken as medium.
EST_THINKING_TOKENS = {"low": 1500, "medium": 4500, "high": 9000}
PROVIDERS = {"claude": "Claude", "gpt": "GPT", "gemini": "Gemini", "glm": "GLM"}
VARIANTS = {"er": "ER", "flashx": "FlashX"}  # display names that plain capitalizing gets wrong
VIEW_ORDER = ["front", "top", "left", "right"]  # front is the reference view; the rest follow it

st.set_page_config(page_title="Glimpse2Sim", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&display=swap');
:root {
  --ground: #1B2025; --surface: #232A30; --ink: #E4E9EC; --muted: #8E9BA4; --line: #333D45; --accent: #78A9E4;
  --depth: linear-gradient(90deg, #C0392B, #E9B93A 45%, #4FA3C7 75%, #23408E);
}
.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp h1, .stApp h2, .stApp h3,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-baseweb="tab"], [data-baseweb="select"] {
  font-family: 'Instrument Sans', sans-serif; }
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: var(--ground); color: var(--ink); }
.block-container { max-width: 1320px; padding-top: 2.2rem; }
h1, h2, h3 { color: var(--ink); letter-spacing: -0.01em; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--surface); border-color: var(--line) !important; border-radius: 12px; }
.stTabs [data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-weight: 600; color: var(--muted); padding: 0.4rem 0; }
.stTabs [aria-selected="true"] { color: var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent); }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background: var(--accent); color: #0F1A26; border: none; border-radius: 10px;
  font-weight: 700; padding: 0.7rem 0; }
.stButton > button[kind="primary"]:hover { background: #9BC0EC; color: #0F1A26; }
.stButton > button[kind="primary"]:disabled { background: var(--line); color: var(--muted); }
.stButton > button:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
.brand { display: flex; align-items: baseline; gap: 0.9rem; margin-bottom: 0.4rem; }
.brand b { font-size: 1.9rem; letter-spacing: -0.02em; }
.brand span { color: var(--muted); font-size: 1rem; }
.section { font-size: 1.05rem; font-weight: 700; margin: 0.2rem 0 0.6rem; }
.hint { color: var(--muted); font-size: 0.88rem; margin-top: -0.3rem; }
.frame { position: relative; margin-bottom: 0.2rem; }
.frame-name { font-weight: 600; font-size: 0.92rem; }
.frame-name.ref::after { content: "기준 시점"; margin-left: 0.5rem; font-weight: 500; font-size: 0.78rem;
  color: var(--accent); border: 1px solid var(--accent); border-radius: 999px; padding: 0 0.45rem; }
.provider { color: var(--muted); font-size: 0.82rem; margin: 0.5rem 0 -0.2rem; }
[data-testid="stCheckbox"] { margin-bottom: -0.55rem; }
[data-testid="stCheckbox"] label p { font-variant-numeric: tabular-nums; }
.chips { display: flex; flex-wrap: wrap; gap: 4px; margin: 0.05rem 0 0.7rem 1.75rem; }
.chips.inline { margin: 0; justify-content: flex-end; flex-wrap: nowrap; gap: 3px; }
.chip { width: 22px; height: 14px; border-radius: 3px; }
[data-testid="stPills"] button { font-weight: 600; }
.chip.rgb { background: #A99E90; border: 1px solid #C2B8AB; }
.chip.d { background: var(--depth); }
.summary { display: flex; justify-content: space-between; align-items: baseline; font-variant-numeric: tabular-nums;
  border-top: 1px solid var(--line); padding-top: 0.8rem; margin-top: 0.4rem; }
.summary b { font-size: 1.5rem; }
.summary span { color: var(--muted); font-size: 0.9rem; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""", unsafe_allow_html=True)


def captured_scenes() -> dict[str, Path]:
    """Scene name -> outputs/<camera>/<scene>, for every folder with at least one color frame."""
    return {p.name: p for p in sorted(OUTPUT_DIR.glob("*/*")) if any(p.glob("*_color.png"))}


def run_streaming(cmd: list[str], box) -> int:
    """Run a main.py command and show its output live in `box`. Return the exit code."""
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        box.code("\n".join(lines[-LOG_LINES:]), language=None)
    return proc.wait()


def all_runs() -> pd.DataFrame:
    """One row per run folder: results/<family>/<variant>/<condition>/<scene>/<time>/run.json."""
    rows = []
    for run_json in RESULTS_DIR.glob("*/*/*/*/*/run.json"):
        d = run_json.parent
        family, variant, condition, scene, time = d.relative_to(RESULTS_DIR).parts
        r = json.loads(run_json.read_text())
        rows.append({
            "time": time, "model": f"{family}/{variant}", "condition": condition, "scene": scene,
            "images": len(r.get("images", [])), "objects": r.get("num_objects"), "valid_json": r.get("valid_json"),
            "thinking": r.get("thinking", "-"), "in_tokens": r.get("input_tokens"),
            "out_tokens": r.get("output_tokens"), "thinking_tokens": r.get("thinking_tokens"),
            "cost_usd": round(r.get("cost_usd") or 0.0, 4), "latency_s": round(r.get("latency_s") or 0.0, 1),
            "rendered": (d / "compare.png").exists(), "prompt_sha": r.get("prompt_sha", "-"), "path": str(d),
        })
    cols = ["time", "model", "condition", "scene", "images", "objects", "valid_json", "rendered", "thinking",
            "in_tokens", "out_tokens", "thinking_tokens", "cost_usd", "latency_s", "prompt_sha", "path"]
    return pd.DataFrame(rows, columns=cols).sort_values("time", ascending=False, ignore_index=True)


def render_run(run_dir: Path, scene: str, box) -> int:
    """Build a run's scene.json in Isaac Sim and save sim_capture.png and compare.png next to it."""
    return run_streaming([sys.executable, "main.py", "view", scene, "--spec", str(run_dir / "scene.json"),
                          "--capture"], box)


def render_all(rows: pd.DataFrame, box) -> list[str]:
    """Render each run in turn, one Isaac Sim launch per run. Return the runs that failed."""
    failed = []
    progress = st.progress(0.0)
    for k, row in enumerate(rows.itertuples(), 1):
        progress.progress((k - 1) / len(rows), text=f"Isaac Sim 렌더 {k}/{len(rows)}: {row.model}, {row.condition}")
        if render_run(Path(row.path), row.scene, box) != 0:
            failed.append(f"{row.model} {row.condition} {row.time}")
    progress.progress(1.0, text=f"Isaac Sim 렌더 {len(rows)}개 완료")
    return failed


def images_sent(condition: dict, n_frames: int) -> tuple[int, int]:
    """(RGB images, depth images) a condition sends for a scene with n_frames views."""
    n = n_frames if condition["frames"] == "all" else len(condition["frames"])
    return n, n if condition.get("depth") else 0


def effective_depth(model: str, thinking: str | None) -> str:
    """The thinking depth a call will run at: the override, else the configured one (provider default = medium)."""
    depth = thinking or thinking_setting(model)
    return depth if depth in EST_THINKING_TOKENS else "medium"


def estimate_cost(model: str, cfg: dict, thinking: str | None, condition: str, n_images: int,
                  runs: pd.DataFrame) -> float:
    """Expected cost of one call.

    The mean of earlier runs of this model, condition, and depth when there are some; otherwise the condition's
    usual input tokens and answer length plus the thinking tokens of the chosen depth, at the model's prices.
    """
    folder = cfg["results"] + (f"-{thinking}" if thinking else "")
    same = runs[(runs["model"] == folder) & (runs["condition"] == condition)]
    if not same.empty and same["cost_usd"].sum() > 0:
        return float(same["cost_usd"].mean())
    cond = runs[runs["condition"] == condition]
    tokens_in = cond["in_tokens"].mean() if cond["in_tokens"].notna().any() else (
        EST_PROMPT_TOKENS + EST_TOKENS_PER_IMAGE * n_images)
    tokens_out = EST_ANSWER_TOKENS + EST_THINKING_TOKENS[effective_depth(model, thinking)]
    return (tokens_in * cfg["pricing"]["input"] + tokens_out * cfg["pricing"]["output"]) / 1e6


def objects_table(spec: dict) -> pd.DataFrame:
    rows = []
    for o in spec.get("objects", []):
        size = o.get("size_mm") or {}
        rgb = o.get("color_rgb") or [0.5, 0.5, 0.5]
        rows.append({
            "id": o.get("id"), "identified_as": o.get("identified_as"),
            "shape": str(o.get("shape")) + (" (lying)" if o.get("lying") and o.get("shape") in ("cylinder", "capsule") else ""),
            "parts": len(o.get("parts") or []),
            "size_mm (L×W×H)": "×".join(str(size.get(k, "?")) for k in ("length", "width", "height")),
            "position_m": ", ".join(f"{v:.3f}" for v in o.get("position", [])),
            "yaw_deg": o.get("yaw_deg"), "mass_kg": o.get("mass_kg"),
            "asset_query": o.get("asset_query"),
            "asset": ((o.get("asset_match") or {}).get("text") or "없음") if o.get("asset_query") else "-",
            "color": "#{:02x}{:02x}{:02x}".format(*(round(min(max(c, 0), 1) * 255) for c in rgb[:3])),
            "confidence": o.get("confidence"),
        })
    return pd.DataFrame(rows)


def pdf_actions(d: Path, run: pd.Series) -> None:
    """Top-right of a run: make the PDF report, then save it; a small link rebuilds it after a new render."""
    pdf = d / "report.pdf"
    if pdf.exists():
        st.download_button("PDF 보고서 저장", pdf.read_bytes(), type="primary", width="stretch", mime="application/pdf",
                           file_name=f"{run['model'].replace('/', '-')}_{run['condition']}_{run['scene']}_{run['time']}.pdf",
                           key=f"dl_{d}")
    make = st.button("보고서 다시 만들기" if pdf.exists() else "PDF 보고서 만들기", key=f"pdf_{d}",
                     type="tertiary" if pdf.exists() else "primary", width="stretch")
    if make:
        try:
            with st.spinner("보고서를 만드는 중입니다."):
                write_pdf(d)
            st.rerun()
        except Exception as e:  # e.g. no Chrome, or Chrome failed to print
            st.error(f"PDF를 만들지 못했습니다: {e}")


def show_run(run: pd.Series) -> None:
    d = Path(run["path"])
    head, actions = st.columns([5, 2], vertical_alignment="center")
    with head:
        st.markdown(f"### {run['model']}, {run['condition']}, {run['scene']}")
        st.markdown(f"<p class='hint'>{run['time']}</p>", unsafe_allow_html=True)
    with actions:
        pdf_actions(d, run)
    c = st.columns(7)
    c[0].metric("물체 수", run["objects"])
    c[1].metric("이미지", run["images"])
    c[2].metric("입력 토큰", f"{run['in_tokens']:,.0f}" if pd.notna(run["in_tokens"]) else "-")
    c[3].metric("출력 토큰", f"{run['out_tokens']:,.0f}" if pd.notna(run["out_tokens"]) else "-")
    c[4].metric("생각 토큰", f"{run['thinking_tokens']:,.0f}" if pd.notna(run["thinking_tokens"]) else "-",
                help=f"생각 깊이 설정: {run['thinking']}. 출력 토큰에 포함됩니다.")
    c[5].metric("비용", f"${run['cost_usd']:.4f}")
    c[6].metric("시간", f"{run['latency_s']} s")

    spec_path = d / "scene.json"
    if (d / "compare.png").exists():
        st.image(str(d / "compare.png"), caption="왼쪽 사진, 오른쪽 Isaac Sim 렌더")
    if spec_path.exists():
        label = "Isaac Sim에서 다시 렌더" if (d / "compare.png").exists() else "Isaac Sim에서 렌더"
        if st.button(label, key=f"cap_{d}"):
            with st.spinner("Isaac Sim을 띄워 렌더하는 중입니다. 1분 정도 걸립니다."):
                code = render_run(d, run["scene"], st.empty())
            if code == 0:
                st.rerun()
            st.error(f"렌더하지 못했습니다 (exit {code}). 위 로그를 확인하세요.")

    boxes = sorted(d.glob("*_boxes.png"))
    if boxes:
        st.markdown("<div class='section'>모델이 표시한 물체 위치</div>", unsafe_allow_html=True)
        cols = st.columns(min(len(boxes), 2))
        for i, p in enumerate(boxes):
            cols[i % len(cols)].image(str(p), caption=p.stem.removesuffix("_boxes"))

    if not spec_path.exists():
        st.warning("모델 출력이 JSON이 아니어서 장면을 만들 수 없습니다. 아래 원문을 확인하세요.")
        if (d / "response.txt").exists():
            st.code((d / "response.txt").read_text()[:5000])
        return
    spec = json.loads(spec_path.read_text())

    st.markdown("<div class='section'>물체</div>", unsafe_allow_html=True)
    table = objects_table(spec)
    if not table.empty:
        styled = table.style.map(lambda v: f"background-color: {v}; color: {v}", subset=["color"]).format(
            {"yaw_deg": "{:.0f}", "mass_kg": "{:.3f}", "confidence": "{:.2f}"}, na_rep="-")
        st.dataframe(styled, hide_index=True, width="stretch",
                     column_config={"color": st.column_config.TextColumn("color", width="small")})

    env = spec.get("environment") or {}
    desk = env.get("desk") or {}
    rel = spec.get("relations") or {}
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='section'>테이블</div>", unsafe_allow_html=True)
        if desk:
            size = desk.get("size") or {}
            facts = {
                "무엇": desk.get("identified_as") or "-", "윗면 모양": desk.get("top_shape") or "-",
                "크기 (m)": f"{size.get('x')} × {size.get('y')}", "높이 (m)": desk.get("height_from_floor"),
                "다리": len(desk.get("legs") or []), "텍스처": desk.get("texture") or "없음",
            }
            st.dataframe(pd.DataFrame({"항목": list(facts), "값": [str(v) for v in facts.values()]}),
                         hide_index=True, width="stretch")
        else:
            st.write("테이블 없이 바닥에 놓인 장면입니다.")
    with right:
        focus = spec.get("focus_object") or {}
        if focus:
            st.markdown(f"<div class='section'>모델이 판단한 중심 물체: {html.escape(str(focus.get('id', '-')))}</div>"
                        f"<p class='hint'>{html.escape(str(focus.get('reason', '')))}</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='section'>위치 관계, 기준 물체 {html.escape(str(rel.get('anchor', '-')))}</div>",
                    unsafe_allow_html=True)
        items = rel.get("items") or []
        if items:
            st.dataframe(pd.DataFrame(items), hide_index=True, width="stretch", height="content")

    for name, title in [("thinking_ko.md", "모델의 생각 (한국어 번역)"), ("thinking.log", "모델의 생각 (원문)"),
                        ("prompt.md", "보낸 프롬프트"), ("scene.json", "scene.json")]:
        if (d / name).exists():
            with st.expander(title):
                text = (d / name).read_text()
                st.json(json.loads(text)) if name.endswith(".json") else st.markdown(text)


# ---------------------------------------------------------------------------------------------------------------------

st.markdown("<div class='brand'><b>Glimpse2Sim</b><span>촬영한 장면을 모델에게 보여주고 시뮬레이션으로 다시 만듭니다</span></div>",
            unsafe_allow_html=True)
tab_run, tab_results = st.tabs(["실행", "결과"])
runs = all_runs()

with tab_run:
    scenes = captured_scenes()
    if not scenes:
        st.info("촬영한 장면이 없습니다. 먼저 `uv run main.py capture <장면 이름>`으로 찍어 주세요.")
    else:
        left, right = st.columns([1, 1], gap="small")

        with left:
            with st.container(border=True):
                st.markdown("<div class='section'>촬영한 장면</div>", unsafe_allow_html=True)
                scene = st.radio("장면", list(scenes), horizontal=True, label_visibility="collapsed")
                frames = sorted(scenes[scene].glob("*_color.png"), key=lambda p: (
                    VIEW_ORDER.index(n) if (n := p.name.removesuffix("_color.png")) in VIEW_ORDER else len(VIEW_ORDER), n))
                st.markdown(f"<p class='hint'>{len(frames)}개 시점, {scenes[scene].relative_to(ROOT)}</p>",
                            unsafe_allow_html=True)
                grid = st.columns(2)
                for i, p in enumerate(frames):
                    name = p.name.removesuffix("_color.png")
                    with grid[i % 2]:
                        ref = " ref" if name == "front" else ""
                        st.markdown(f"<div class='frame-name{ref}'>{html.escape(name)}</div>", unsafe_allow_html=True)
                        st.image(str(p), width="stretch")

        with right:
            models_cfg = load_config("models")["models"]
            conditions = load_config("conditions")
            with st.container(border=True):
                st.markdown("<div class='section'>모델</div>", unsafe_allow_html=True)

                def model_label(m: str) -> str:
                    family, variant = models_cfg[m]["results"].split("/")
                    return f"{PROVIDERS.get(family, family)} {VARIANTS.get(variant, variant.capitalize())}"

                chosen_models = st.pills("모델", model_names(), selection_mode="multi", default=["gemini-flash"],
                                         format_func=model_label, key="models", label_visibility="collapsed") or []
                if chosen_models:
                    lines = []
                    for m in chosen_models:
                        cfg, price = models_cfg[m], models_cfg[m]["pricing"]
                        note = "무료, 디버그용" if cfg.get("debug") else f"${price['input']:g} / ${price['output']:g}"
                        lines.append(f"<b>{html.escape(model_label(m))}</b> {html.escape(cfg['model_id'])}, {note}")
                    st.markdown("<p class='hint' style='margin:0.3rem 0 0'>" + "<br>".join(lines)
                                + "<br>가격은 입력 / 출력, 1M 토큰당 달러</p>", unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown("<div class='section'>단계</div>", unsafe_allow_html=True)
                chosen_levels = []
                for key, cond in conditions.items():
                    n_rgb, n_depth = images_sent(cond, len(frames))
                    view = "front 한 장" if cond["frames"] != "all" else f"모든 시점 {n_rgb}장"
                    desc = view + (" + depth" if n_depth else ", RGB만")
                    c_box, c_chips = st.columns([1, 1], vertical_alignment="center")
                    if c_box.checkbox(f"**{key}**&nbsp;&nbsp;:gray[{desc}]", value=True, key=f"level_{key}"):
                        chosen_levels.append(key)
                    chips = "".join("<span class='chip rgb'></span>" for _ in range(n_rgb)) + "".join(
                        "<span class='chip d'></span>" for _ in range(n_depth))
                    c_chips.markdown(f"<div class='chips inline' role='img' aria-label='RGB {n_rgb}장, depth {n_depth}장'>"
                                     f"{chips}</div>", unsafe_allow_html=True)

                r1, r2 = st.columns([2, 3])
                with r1:
                    repeat = st.segmented_control("반복 횟수", [1, 2, 3, 5], default=1, key="repeat",
                                                  help="같은 모델과 단계를 몇 번 실행할지 정합니다.") or 1
                with r2:
                    depth = st.segmented_control(
                        "생각 깊이", ["설정값", *THINKING_LEVELS], default="설정값", key="thinking",
                        help="설정값은 configs/models.yaml을 따릅니다. 다른 값을 고르면 결과가 "
                             "<모델>-<깊이> 폴더(예: gemini/argon-high)에 따로 저장됩니다. GLM에는 쓸 수 없습니다.",
                    ) or "설정값"
                thinking = None if depth == "설정값" else depth
                calls = len(chosen_models) * len(chosen_levels) * repeat
                cost = repeat * sum(
                    estimate_cost(m, models_cfg[m], thinking, c, sum(images_sent(conditions[c], len(frames))), runs)
                    for m in chosen_models for c in chosen_levels
                )
                st.markdown(f"<div class='summary'><b>{calls}회 호출</b><span>예상 비용 약 ${cost:.2f}</span></div>",
                            unsafe_allow_html=True)
                if chosen_models:
                    depths = sorted({effective_depth(m, thinking) for m in chosen_models},
                                    key=list(EST_THINKING_TOKENS).index)
                    st.markdown("<p class='hint' style='margin-top:0.4rem'>생각 깊이 " + ", ".join(
                        f"{d} (회당 생각 약 {EST_THINKING_TOKENS[d] // 1000}천 토큰)" for d in depths)
                        + " 기준. 같은 모델, 단계, 깊이로 돌린 기록이 있으면 그 평균을 씁니다.</p>",
                        unsafe_allow_html=True)
                render_after = st.checkbox("끝나면 Isaac Sim에서 렌더", value=True, key="render_after",
                                           help="새로 생긴 실행마다 Isaac Sim 창이 잠깐 열려 사진을 찍고 닫힙니다.")
                start = st.button("실행", type="primary", width="stretch", disabled=calls == 0)

        if calls == 0:
            st.caption("모델과 단계를 하나 이상 고르면 실행할 수 있습니다.")
        if start:
            st.markdown("<div class='section'>실행 기록</div>", unsafe_allow_html=True)
            box = st.empty()
            failed = []
            before = set(runs["path"])
            for m in chosen_models:
                code = run_streaming([sys.executable, "main.py", "generate", scene, "--model", m,
                                      "--condition", *chosen_levels, "--repeat", str(repeat),
                                      *(["--thinking", thinking] if thinking else [])], box)
                if code != 0:
                    failed.append(m)
            if failed:
                st.error(f"{', '.join(failed)} 실행이 중간에 멈췄습니다. 위 로그의 마지막 오류를 확인하세요.")
            new = all_runs()
            new = new[~new["path"].isin(before) & new["valid_json"].fillna(False).astype(bool)]
            if render_after and not new.empty:
                not_rendered = render_all(new, box)
                if not_rendered:
                    st.error(f"렌더하지 못한 실행: {', '.join(not_rendered)}. 위 로그를 확인하세요.")
            st.success(f"새 실행 {len(new)}개를 마쳤습니다. 결과 탭에서 확인하세요.")

with tab_results:
    if runs.empty:
        st.info("아직 실행한 결과가 없습니다. 실행 탭에서 모델과 단계를 골라 실행해 보세요.")
    else:
        with st.container(border=True):
            st.markdown("<div class='section'>모델별 종합 보고서</div><p class='hint'>한 모델의 L1~L3 최근 실행을 "
                        "한 장면 기준으로 모아 비교합니다.</p>", unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns([3, 2, 2, 2], vertical_alignment="bottom")
            sum_model = s1.selectbox("모델", sorted(runs["model"].unique()), key="sum_model")
            sum_scene = s2.selectbox("장면", sorted(runs[runs["model"] == sum_model]["scene"].unique()), key="sum_scene")
            sum_pdf = RESULTS_DIR / sum_model / f"{sum_scene}_summary.pdf"
            if s3.button("종합 보고서 만들기" if not sum_pdf.exists() else "다시 만들기", key="sum_make", width="stretch"):
                try:
                    with st.spinner("종합 보고서를 만드는 중입니다."):
                        write_summary_pdf(RESULTS_DIR / sum_model, sum_scene)
                except Exception as e:  # e.g. no Chrome, or no finished run
                    st.error(f"종합 보고서를 만들지 못했습니다: {e}")
            if sum_pdf.exists():
                s4.download_button("PDF 저장", sum_pdf.read_bytes(), type="primary", width="stretch",
                                   file_name=f"{sum_model.replace('/', '-')}_{sum_scene}_summary.pdf",
                                   mime="application/pdf", key="sum_dl")
            # What the report will compare: this model's latest run of each level, as rendered.
            level_runs = latest_runs(RESULTS_DIR / sum_model, sum_scene)
            cols = st.columns(max(len(level_runs), 1))
            for col, (level, run_dir) in zip(cols, level_runs.items()):
                shot = run_dir / "sim_capture.png"
                if shot.exists():
                    col.image(str(shot), caption=f"{level}, {run_dir.name}")
                else:
                    col.markdown(f"<p class='hint'>{level}, {run_dir.name}: 아직 렌더하지 않았습니다.</p>",
                                 unsafe_allow_html=True)

        st.markdown("<div class='section'>실행 목록</div><p class='hint'>표에서 거른 뒤 아래에서 실행 하나를 골라 "
                    "자세히 봅니다. 위 종합 보고서의 모델 선택과는 따로 움직입니다.</p>", unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        pick = lambda col, key, label: col.multiselect(label, sorted(runs[key].unique()), key=f"filter_{key}", placeholder="전체")  # noqa: E731
        sel = {"model": pick(f1, "model", "모델"), "condition": pick(f2, "condition", "단계"),
               "scene": pick(f3, "scene", "장면")}
        view = runs
        for col, values in sel.items():
            if values:
                view = view[view[col].isin(values)]
        st.dataframe(view.drop(columns=["path"]), hide_index=True, width="stretch")
        pending = view[~view["rendered"] & view["valid_json"].fillna(False).astype(bool)]
        if not pending.empty and st.button(f"렌더하지 않은 실행 {len(pending)}개를 Isaac Sim에서 렌더",
                                           key="render_pending"):
            not_rendered = render_all(pending, st.empty())
            if not_rendered:
                st.error(f"렌더하지 못한 실행: {', '.join(not_rendered)}. 위 로그를 확인하세요.")
            else:
                st.rerun()
        if not view.empty:
            labels = [f"{r.time}, {r.model}, {r.condition}, {r.scene}" for r in view.itertuples()]
            i = st.selectbox("자세히 볼 실행", range(len(view)), format_func=lambda k: labels[k])
            st.divider()
            show_run(view.iloc[i])
