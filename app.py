"""Glimpse2Sim UI: run scene generation and browse the results.

    uv run streamlit run app.py
"""

import html
import importlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
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
from glimpse2sim.generation.scene import RESULTS_DIR, current_experiment, experiment_dir, experiments  # noqa: E402
from glimpse2sim.models import THINKING_LEVELS, model_names, thinking_setting  # noqa: E402
from glimpse2sim.report import latest_runs, write_pdf, write_summary_pdf  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
LOG_LINES = 1000  # lines of the running job's output kept in the log panel
POLL_S = 1.0  # how often the job panel refreshes while a job runs
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
.job { display: flex; align-items: center; gap: 0.7rem; font-variant-numeric: tabular-nums; }
.job b { font-size: 1.02rem; }
.job span { color: var(--muted); font-size: 0.88rem; }
.dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.dot.running { background: var(--accent); animation: pulse 1.2s ease-in-out infinite; }
.dot.done { background: #6CC08B; } .dot.failed { background: #E06C5F; } .dot.stopped { background: var(--muted); }
@keyframes pulse { 50% { opacity: 0.3; } }
/* column-reverse keeps the panel scrolled to the newest line without any script */
.log { height: 340px; overflow-y: auto; display: flex; flex-direction: column-reverse; background: #12171B;
  border: 1px solid var(--line); border-radius: 10px; margin-top: 0.6rem; }
/* Streamlit renders the <pre> as its own stMarkdownPre block */
.log pre, .log [data-testid="stMarkdownPre"] { margin: 0; background: none; border: none; padding: 0.7rem 0.9rem; font: 0.8rem/1.45 ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #C9D2D8; white-space: pre-wrap; word-break: break-all; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""", unsafe_allow_html=True)


def captured_scenes() -> dict[str, Path]:
    """Scene name -> outputs/<camera>/<scene>, for every folder with at least one color frame."""
    return {p.name: p for p in sorted(OUTPUT_DIR.glob("*/*")) if any(p.glob("*_color.png"))}


class Job:
    """A sequence of main.py commands run in a background thread, so the page stays responsive and can stop it.

    The thread never calls Streamlit; the job panel reads `state`, `stage`, `log`, and `message` on each refresh.
    """

    def __init__(self, title: str, work):
        self.title, self.stage, self.message = title, "", ""
        self.state = "running"  # running, done, failed, stopped
        self.log: deque[str] = deque(maxlen=LOG_LINES)
        self.errors: list[str] = []
        self.started, self.ended = time.time(), None
        self.proc: subprocess.Popen | None = None
        self.stopping = threading.Event()
        self.shown_end = False  # the page has rerun once since the job ended
        threading.Thread(target=self._main, args=(work,), daemon=True).start()

    def _main(self, work) -> None:
        try:
            self.message = work(self) or ""
        except Exception as e:
            self.errors.append(str(e))
        if self.stopping.is_set():
            self.message = "중단했습니다. 이미 끝난 실행은 결과 탭에 남아 있습니다."
        self.state = "stopped" if self.stopping.is_set() else "failed" if self.errors else "done"
        self.ended = time.time()

    def run(self, cmd: list[str]) -> int:
        """Run one command, appending its output to the log. Return its exit code (-1 if stopped first)."""
        if self.stopping.is_set():
            return -1
        self.log.append("$ " + " ".join(cmd[1:]))
        # A new session gives the command its own process group, so stopping also ends Isaac Sim's children.
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     bufsize=1, start_new_session=True, env=os.environ | {"PYTHONUNBUFFERED": "1"})
        for line in self.proc.stdout:
            self.log.append(line.rstrip("\n"))
        return self.proc.wait()

    def stop(self) -> None:
        self.stopping.set()
        proc = self.proc
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            threading.Timer(5, lambda: proc.poll() is None and os.killpg(proc.pid, signal.SIGKILL)).start()

    @property
    def running(self) -> bool:
        return self.state == "running"


@st.cache_resource
def _job_slot() -> dict:
    """The current job, kept across reruns and page reloads (one job at a time for the whole server)."""
    return {"job": None}


def current_job() -> Job | None:
    return _job_slot()["job"]


def busy() -> bool:
    return bool(current_job() and current_job().running)


def start_job(title: str, work) -> None:
    if not busy():
        _job_slot()["job"] = Job(title, work)
        st.rerun()


@st.fragment(run_every=POLL_S)
def job_panel() -> None:
    """Status, a stop button while running, and the log in a fixed-height panel that follows the newest line."""
    job = current_job()
    if job is None:
        return
    if not job.running and not job.shown_end:
        job.shown_end = True
        st.rerun(scope="app")  # re-enable the start buttons and pick up the new results
    elapsed = int((job.ended or time.time()) - job.started)
    word = {"running": "실행 중", "done": "완료", "failed": "오류", "stopped": "중단됨"}[job.state]
    detail = job.stage if job.running else (job.message or job.stage)
    with st.container(border=True):
        head, button = st.columns([6, 1], vertical_alignment="center")
        head.markdown(f"<div class='job'><span class='dot {job.state}'></span><b>{word}</b>"
                      f"<span>{html.escape(job.title)} · {html.escape(detail)} · {elapsed // 60}:{elapsed % 60:02d}"
                      f"</span></div>", unsafe_allow_html=True)
        if job.running:
            stopping = job.stopping.is_set()
            if button.button("중단하는 중" if stopping else "중단", key="job_stop", width="stretch", disabled=stopping):
                job.stop()
                st.rerun(scope="fragment")
        elif button.button("닫기", key="job_close", type="tertiary", width="stretch"):
            _job_slot()["job"] = None
            st.rerun(scope="app")
        for e in job.errors:
            st.error(e)
        st.markdown(f"<div class='log'><pre>{html.escape(chr(10).join(job.log)) or '…'}</pre></div>",
                    unsafe_allow_html=True)


def all_runs() -> pd.DataFrame:
    """One row per run folder: results/<scene>/<experiment>/<family>/<variant>/<condition>/<time>/run.json."""
    rows = []
    for run_json in RESULTS_DIR.glob("*/*/*/*/*/*/run.json"):
        d = run_json.parent
        scene, experiment, family, variant, condition, time = d.relative_to(RESULTS_DIR).parts
        r = json.loads(run_json.read_text())
        rows.append({
            "experiment": experiment, "time": time, "model": f"{family}/{variant}", "condition": condition, "scene": scene,
            "images": len(r.get("images", [])), "objects": r.get("num_objects"), "valid_json": r.get("valid_json"),
            "thinking": r.get("thinking", "-"), "in_tokens": r.get("input_tokens"),
            "out_tokens": r.get("output_tokens"), "thinking_tokens": r.get("thinking_tokens"),
            "cost_usd": round(r.get("cost_usd") or 0.0, 4), "latency_s": round(r.get("latency_s") or 0.0, 1),
            "rendered": (d / "compare.png").exists(), "prompt_sha": r.get("prompt_sha", "-"), "path": str(d),
        })
    cols = ["experiment", "time", "model", "condition", "scene", "images", "objects", "valid_json", "rendered", "thinking",
            "in_tokens", "out_tokens", "thinking_tokens", "cost_usd", "latency_s", "prompt_sha", "path"]
    return pd.DataFrame(rows, columns=cols).sort_values("time", ascending=False, ignore_index=True)


def render_run(job: Job, run_dir: Path, scene: str) -> int:
    """Build a run's scene.json in Isaac Sim and save sim_capture.png and compare.png next to it."""
    return job.run([sys.executable, "main.py", "view", scene, "--spec", str(run_dir / "scene.json"), "--capture"])


def render_all(job: Job, rows: pd.DataFrame) -> int:
    """Render each run in turn, one Isaac Sim launch per run. Return how many were rendered."""
    done = 0
    for k, row in enumerate(rows.itertuples(), 1):
        job.stage = f"Isaac Sim 렌더 {k}/{len(rows)}: {row.model}, {row.condition}"
        code = render_run(job, Path(row.path), row.scene)
        if job.stopping.is_set():
            break
        if code != 0:
            job.errors.append(f"렌더하지 못했습니다: {row.model} {row.condition} {row.time} (exit {code})")
        else:
            done += 1
    return done


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
                           file_name=f"{run['scene']}_{run['experiment']}_{run['model'].replace('/', '-')}_{run['condition']}_{run['time']}.pdf",
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
        st.markdown(f"### {run['model']}, {run['condition']}, {run['scene']}, 실험 {run['experiment']}")
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
        if st.button(label, key=f"cap_{d}", disabled=busy()):
            def work(job: Job) -> str:
                job.stage = "Isaac Sim 렌더"
                code = render_run(job, d, run["scene"])
                if code > 0:
                    job.errors.append(f"렌더하지 못했습니다 (exit {code}).")
                return "렌더를 마쳤습니다."
            start_job(f"렌더: {run['model']}, {run['condition']}", work)

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
                start = st.button("실행 중…" if busy() else "실행", type="primary", width="stretch",
                                  disabled=calls == 0 or busy())
                st.markdown(f"<p class='hint' style='margin-top:0.4rem'>결과는 실험 {current_experiment()} "
                            f"(results/{html.escape(scene)}/{current_experiment()})에 쌓입니다. 프롬프트나 파이프라인을 바꾸면 "
                            "configs/experiment.yaml의 번호를 올리세요.</p>", unsafe_allow_html=True)

        if calls == 0:
            st.caption("모델과 단계를 하나 이상 고르면 실행할 수 있습니다.")
        if start:
            before = set(runs["path"])

            def work(job: Job) -> str:
                for k, m in enumerate(chosen_models, 1):
                    job.stage = f"생성 {k}/{len(chosen_models)}: {m}"
                    code = job.run([sys.executable, "main.py", "generate", scene, "--model", m,
                                    "--condition", *chosen_levels, "--repeat", str(repeat),
                                    *(["--thinking", thinking] if thinking else [])])
                    if job.stopping.is_set():
                        return ""
                    if code != 0:
                        job.errors.append(f"{m} 실행이 중간에 멈췄습니다 (exit {code}). 로그의 마지막 오류를 확인하세요.")
                new = all_runs()
                new = new[~new["path"].isin(before) & new["valid_json"].fillna(False).astype(bool)]
                if render_after and not new.empty:
                    render_all(job, new)
                return f"새 실행 {len(new)}개를 마쳤습니다. 결과 탭에서 확인하세요."

            start_job(f"{scene}: {', '.join(chosen_models)} × {', '.join(chosen_levels)}", work)

with tab_results:
    # Results are kept per scene, then per experiment round, then per model: pick them in that order.
    notes = load_config("experiment").get("notes") or {}
    r1, r2, r3 = st.columns([2, 2, 3])
    res_scenes = sorted(runs["scene"].unique())
    res_scene = r1.selectbox("장면", res_scenes, key="res_scene") if res_scenes else None
    with_runs = experiments(res_scene) if res_scene else []
    rounds = sorted(set(with_runs) | {current_experiment()})
    # Open on the current round once it has runs, until then on the latest round that does.
    default = current_experiment() if current_experiment() in with_runs or not with_runs else with_runs[-1]
    exp = r2.selectbox("실험", rounds, index=rounds.index(default), key="res_experiment",
                       format_func=lambda e: f"{e} (현재)" if e == current_experiment() else e,
                       help="프롬프트나 파이프라인이 바뀔 때마다 번호를 올립니다 (configs/experiment.yaml).")
    runs = runs[(runs["scene"] == res_scene) & (runs["experiment"] == exp)]
    res_models = sorted(runs["model"].unique())
    res_model = r3.selectbox("모델", res_models, key="res_model") if res_models else None
    if notes.get(exp):
        st.markdown(f"<p class='hint'>실험 {html.escape(exp)}: {html.escape(str(notes[exp]))}</p>",
                    unsafe_allow_html=True)
    if res_model is None:
        st.info(f"{res_scene or '이 장면'}의 실험 {exp}에는 아직 실행한 결과가 없습니다. "
                "실행 탭에서 모델과 단계를 골라 실행해 보세요.")
    else:
        runs = runs[runs["model"] == res_model]
        model_dir = experiment_dir(res_scene, exp) / res_model
        with st.container(border=True):
            st.markdown("<div class='section'>종합 보고서</div><p class='hint'>이 모델의 L1~L3 최근 실행을 모아 "
                        "비교합니다.</p>", unsafe_allow_html=True)
            sum_pdf = model_dir / "summary.pdf"
            s1, s2, _ = st.columns([2, 2, 5])
            if s1.button("종합 보고서 만들기" if not sum_pdf.exists() else "다시 만들기", key="sum_make", width="stretch"):
                try:
                    with st.spinner("종합 보고서를 만드는 중입니다."):
                        write_summary_pdf(model_dir)
                except Exception as e:  # e.g. no Chrome, or no finished run
                    st.error(f"종합 보고서를 만들지 못했습니다: {e}")
            if sum_pdf.exists():
                s2.download_button("PDF 저장", sum_pdf.read_bytes(), type="primary", width="stretch",
                                   file_name=f"{res_scene}_{exp}_{res_model.replace('/', '-')}_summary.pdf",
                                   mime="application/pdf", key="sum_dl")
            # What the report will compare: this model's latest run of each level, as rendered.
            level_runs = latest_runs(model_dir)
            cols = st.columns(max(len(level_runs), 1))
            for col, (level, run_dir) in zip(cols, level_runs.items()):
                shot = run_dir / "sim_capture.png"
                if shot.exists():
                    col.image(str(shot), caption=f"{level}, {run_dir.name}")
                else:
                    col.markdown(f"<p class='hint'>{level}, {run_dir.name}: 아직 렌더하지 않았습니다.</p>",
                                 unsafe_allow_html=True)

        st.markdown("<div class='section'>실행 목록</div>", unsafe_allow_html=True)
        levels = st.pills("단계", sorted(runs["condition"].unique()), selection_mode="multi", key="res_levels",
                          help="고르지 않으면 모든 단계를 보여줍니다.")
        view = runs[runs["condition"].isin(levels)] if levels else runs
        st.dataframe(view.drop(columns=["path", "experiment", "scene", "model"]), hide_index=True, width="stretch")
        pending = view[~view["rendered"] & view["valid_json"].fillna(False).astype(bool)]
        if not pending.empty and st.button(f"렌더하지 않은 실행 {len(pending)}개를 Isaac Sim에서 렌더",
                                           key="render_pending", disabled=busy()):
            start_job(f"렌더 {len(pending)}개", lambda job: f"{render_all(job, pending)}개를 렌더했습니다.")
        if not view.empty:
            labels = [f"{r.condition}, {r.time}" for r in view.itertuples()]
            i = st.selectbox("자세히 볼 실행", range(len(view)), format_func=lambda k: labels[k])
            st.divider()
            show_run(view.iloc[i])

# The running (or last) job, below both tabs so it stays in view whichever tab is open.
if current_job():
    st.markdown("<div class='section' style='margin-top:1.2rem'>실행 기록</div>", unsafe_allow_html=True)
job_panel()
