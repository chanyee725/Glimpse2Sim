"""Glimpse2Sim UI: run scene generation and browse the results.

    uv run streamlit run app.py
"""

import html
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from glimpse2sim.config import ROOT, load_config
from glimpse2sim.generation.scene import RESULTS_DIR
from glimpse2sim.models import model_names

OUTPUT_DIR = ROOT / "outputs"
LOG_LINES = 300  # lines of the running command's output kept on screen
# Rough per-call tokens for a cost estimate when a model has no earlier run of that condition.
EST_TOKENS_PER_IMAGE, EST_PROMPT_TOKENS, EST_OUTPUT_TOKENS = 1300, 3000, 7000
PROVIDERS = {"claude": "Claude", "gpt": "GPT", "gemini": "Gemini", "glm": "GLM"}
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
.stButton > button[kind="primary"] { background: var(--accent); color: #0F1A26; border: none; border-radius: 10px;
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
.chip { width: 26px; height: 16px; border-radius: 3px; }
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
            "in_tokens": r.get("input_tokens"), "out_tokens": r.get("output_tokens"),
            "cost_usd": round(r.get("cost_usd") or 0.0, 4), "latency_s": round(r.get("latency_s") or 0.0, 1),
            "prompt_sha": r.get("prompt_sha", "-"), "path": str(d),
        })
    cols = ["time", "model", "condition", "scene", "images", "objects", "valid_json", "in_tokens", "out_tokens",
            "cost_usd", "latency_s", "prompt_sha", "path"]
    return pd.DataFrame(rows, columns=cols).sort_values("time", ascending=False, ignore_index=True)


def images_sent(condition: dict, n_frames: int) -> tuple[int, int]:
    """(RGB images, depth images) a condition sends for a scene with n_frames views."""
    n = n_frames if condition["frames"] == "all" else len(condition["frames"])
    return n, n if condition.get("depth") else 0


def estimate_cost(model: str, cfg: dict, condition: str, n_images: int, runs: pd.DataFrame) -> float:
    """Mean cost of this model's earlier runs of the condition, else a rough token estimate."""
    earlier = runs[(runs["model"] == cfg["results"]) & (runs["condition"] == condition)]
    if not earlier.empty and earlier["cost_usd"].sum() > 0:
        return float(earlier["cost_usd"].mean())
    tokens_in = EST_PROMPT_TOKENS + EST_TOKENS_PER_IMAGE * n_images
    return (tokens_in * cfg["pricing"]["input"] + EST_OUTPUT_TOKENS * cfg["pricing"]["output"]) / 1e6


def objects_table(spec: dict) -> pd.DataFrame:
    rows = []
    for o in spec.get("objects", []):
        size = o.get("size_mm") or {}
        rgb = o.get("color_rgb") or [0.5, 0.5, 0.5]
        rows.append({
            "id": o.get("id"), "identified_as": o.get("identified_as"),
            "shape": str(o.get("shape")) + (" (lying)" if o.get("lying") else ""),
            "parts": len(o.get("parts") or []),
            "size_mm (L×W×H)": "×".join(str(size.get(k, "?")) for k in ("length", "width", "height")),
            "position_m": ", ".join(f"{v:.3f}" for v in o.get("position", [])),
            "yaw_deg": o.get("yaw_deg"), "mass_kg": o.get("mass_kg"), "texture": o.get("texture"),
            "color": "#{:02x}{:02x}{:02x}".format(*(round(min(max(c, 0), 1) * 255) for c in rgb[:3])),
            "confidence": o.get("confidence"),
        })
    return pd.DataFrame(rows)


def show_run(run: pd.Series) -> None:
    d = Path(run["path"])
    st.markdown(f"### {run['model']}, {run['condition']}, {run['scene']}")
    st.markdown(f"<p class='hint'>{run['time']}</p>", unsafe_allow_html=True)
    c = st.columns(6)
    c[0].metric("물체 수", run["objects"])
    c[1].metric("이미지", run["images"])
    c[2].metric("입력 토큰", f"{run['in_tokens']:,}" if pd.notna(run["in_tokens"]) else "-")
    c[3].metric("출력 토큰", f"{run['out_tokens']:,}" if pd.notna(run["out_tokens"]) else "-")
    c[4].metric("비용", f"${run['cost_usd']:.4f}")
    c[5].metric("시간", f"{run['latency_s']} s")

    spec_path = d / "scene.json"
    if (d / "compare.png").exists():
        st.image(str(d / "compare.png"), caption="왼쪽 사진, 오른쪽 Isaac Sim 렌더")
    if spec_path.exists():
        label = "Isaac Sim에서 다시 렌더" if (d / "compare.png").exists() else "Isaac Sim에서 렌더"
        if st.button(label, key=f"cap_{d}"):
            with st.spinner("Isaac Sim을 띄워 렌더하는 중입니다. 1분 정도 걸립니다."):
                box = st.empty()
                code = run_streaming([sys.executable, "main.py", "view", run["scene"], "--spec", str(spec_path),
                                      "--capture"], box)
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
        left, right = st.columns([3, 2], gap="small")

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
                chosen_models = []
                for family in dict.fromkeys(models_cfg[m]["results"].split("/")[0] for m in model_names()):
                    st.markdown(f"<div class='provider'>{PROVIDERS.get(family, family)}</div>",
                                unsafe_allow_html=True)
                    for m in model_names():
                        cfg = models_cfg[m]
                        if not cfg["results"].startswith(family + "/"):
                            continue
                        price = cfg["pricing"]
                        note = "무료, 디버그용" if cfg.get("debug") else f"\\${price['input']:g} / \\${price['output']:g}"
                        label = f"**{cfg['results'].split('/')[1]}**&nbsp;&nbsp;:gray[{cfg['model_id']}, {note}]"
                        if st.checkbox(label, value=(m == "gemini-flash"), key=f"model_{m}"):
                            chosen_models.append(m)
                st.markdown("<p class='hint' style='margin-top:0.9rem'>가격은 입력 / 출력, 1M 토큰당 달러</p>",
                            unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown("<div class='section'>단계</div>", unsafe_allow_html=True)
                st.markdown("<p class='hint'>단계가 올라갈수록 모델이 보는 이미지가 늘어납니다.</p>",
                            unsafe_allow_html=True)
                chosen_levels = []
                for key, cond in conditions.items():
                    n_rgb, n_depth = images_sent(cond, len(frames))
                    view = "front 한 장" if cond["frames"] != "all" else f"모든 시점 {n_rgb}장"
                    desc = view + (" + depth" if n_depth else ", RGB만")
                    if st.checkbox(f"**{key}**&nbsp;&nbsp;:gray[{desc}]", value=True, key=f"level_{key}"):
                        chosen_levels.append(key)
                    chips = "".join("<span class='chip rgb'></span>" for _ in range(n_rgb)) + "".join(
                        "<span class='chip d'></span>" for _ in range(n_depth))
                    st.markdown(f"<div class='chips' role='img' aria-label='RGB {n_rgb}장, depth {n_depth}장'>"
                                f"{chips}</div>", unsafe_allow_html=True)

                repeat = st.segmented_control("반복 횟수", [1, 2, 3, 5], default=1, key="repeat",
                                              help="같은 모델과 단계를 몇 번 실행할지 정합니다.") or 1
                calls = len(chosen_models) * len(chosen_levels) * repeat
                cost = repeat * sum(
                    estimate_cost(m, models_cfg[m], c, sum(images_sent(conditions[c], len(frames))), runs)
                    for m in chosen_models for c in chosen_levels
                )
                st.markdown(f"<div class='summary'><b>{calls}회 호출</b><span>예상 비용 약 ${cost:.2f}</span></div>",
                            unsafe_allow_html=True)
                start = st.button("실행", type="primary", width="stretch", disabled=calls == 0)

        if calls == 0:
            st.caption("모델과 단계를 하나 이상 고르면 실행할 수 있습니다.")
        if start:
            st.markdown("<div class='section'>실행 기록</div>", unsafe_allow_html=True)
            box = st.empty()
            failed = []
            for m in chosen_models:
                code = run_streaming([sys.executable, "main.py", "generate", scene, "--model", m,
                                      "--condition", *chosen_levels, "--repeat", str(repeat)], box)
                if code != 0:
                    failed.append(m)
            if failed:
                st.error(f"{', '.join(failed)} 실행이 중간에 멈췄습니다. 위 로그의 마지막 오류를 확인하세요.")
            else:
                st.success(f"{calls}회 실행을 마쳤습니다. 결과 탭에서 확인하세요.")

with tab_results:
    if runs.empty:
        st.info("아직 실행한 결과가 없습니다. 실행 탭에서 모델과 단계를 골라 실행해 보세요.")
    else:
        f1, f2, f3 = st.columns(3)
        pick = lambda col, key, label: col.multiselect(label, sorted(runs[key].unique()), key=f"filter_{key}", placeholder="전체")  # noqa: E731
        sel = {"model": pick(f1, "model", "모델"), "condition": pick(f2, "condition", "단계"),
               "scene": pick(f3, "scene", "장면")}
        view = runs
        for col, values in sel.items():
            if values:
                view = view[view[col].isin(values)]
        st.dataframe(view.drop(columns=["path"]), hide_index=True, width="stretch")
        if not view.empty:
            labels = [f"{r.time}, {r.model}, {r.condition}, {r.scene}" for r in view.itertuples()]
            i = st.selectbox("자세히 볼 실행", range(len(view)), format_func=lambda k: labels[k])
            st.divider()
            show_run(view.iloc[i])
