"""Find a ready-made 3D model for an object from the model's own description of it.

Two free libraries, collected by fixed rules (no hand-picking):
- Isaac Sim: every USD file under the object folders of the Isaac Sim 6.2 asset server, described by its path.
- Objaverse 1.0: its 1,156 LVIS categories (e.g. "mouse (computer equipment)") with the synonyms LVIS lists for
  them ("cellular telephone": cellphone, mobile phone, smart phone); a matched category is narrowed to the
  downloadable CC-BY or CC0 objects whose name or tags name the kind, the SHAPE_SHORTLIST closest to the query are
  downloaded, and the one whose bounding-box proportions best fit the object's own size_mm is used (this drops
  models that come with a base plate or a scene around them).

Each entry is embedded once; a model's `asset_query` is embedded the same way, and the TOP_K nearest entries are
the candidates. The embedding only retrieves: which candidate (if any) is the object is decided by the scene model
itself (glimpse2sim/generation/asset_selection.py), so no similarity threshold or word rule picks the asset.

    uv run main.py assets-index      # build assets/index/ once
"""

import json
import re
import urllib.parse
import urllib.request
from datetime import date

import numpy as np

from .config import ROOT

SERVER = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
ASSET_ROOT = "Assets/Isaac/6.2"
FOLDERS = ["Isaac/Props/", "NVIDIA/Assets/ArchVis/", "NVIDIA/Assets/DigitalTwin/Assets/"]
# Parts of other files, not objects of their own: hidden folders and material, texture and geometry sub-files.
SKIP = re.compile(r"/(\.[^/]*|[Mm]aterials?|[Tt]extures?|SubUSDs|Instances|Geometry|geo|Payloads?|configuration|"
                  r"Collisions?)/", re.I)
INDEX_DIR = ROOT / "assets" / "index"
EMBED_MODEL = "gemini-embedding-2"
EMBED_DIMS = 768
BATCH = 100
# LVIS v1 category list with synonyms (the same categories as Objaverse's LVIS annotations).
LVIS_CATEGORIES_URL = ("https://raw.githubusercontent.com/facebookresearch/detectron2/main/"
                       "detectron2/data/datasets/lvis_v1_categories.py")
TOP_K = 20  # candidates shown to the model per object
OBJAVERSE_LICENSES = {"by", "cc0"}  # CC-BY (credit the author) and CC0; non-commercial and no-derivatives are left out
OBJAVERSE_MAX_FACES = 300_000  # heavier scans slow the simulator down for no visible gain at desk scale
OBJAVERSE_SHORTLIST = 150  # objects of a category considered (the annotations of each are downloaded)
SHAPE_SHORTLIST = 10  # of those, the closest by name are downloaded and compared by proportions
# Words that describe how an object looks rather than what it is; they never make two objects the same kind.
NOT_KINDS = set("""a an and the of with without for in on to by
    black white gray grey red green blue yellow orange purple pink brown beige silver gold golden dark light clear
    small large big tiny huge long short tall thick thin round square flat wide narrow mini
    plastic metal metallic wooden wood glass ceramic paper rubber steel aluminum cloth leather
    new old modern simple plain empty full open closed""".split())
# File-name words that are not part of what the object is.
NOT_NAMES = {"instanceable", "physics", "collision", "axis", "aligned", "lod", "low", "high", "poly", "var", "variant",
             "base", "asset", "model", "mesh", "prop", "nvd", "pr", "ml", "copy", "free", "lowpoly"}


def _keys(prefix: str):
    token = None
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"} | ({"continuation-token": token} if token else {})
        page = urllib.request.urlopen(f"{SERVER}/?{urllib.parse.urlencode(query)}", timeout=60).read().decode()
        yield from re.findall(r"<Key>([^<]+)</Key>", page)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", page)
        if not m:
            return
        token = m.group(1)


def describe(path: str) -> str:
    """Words for an asset path: 'NVIDIA/Assets/ArchVis/Residential/Decor/Books/Book_01.usd' -> 'book (residential
    decor books)'."""
    def words(s: str) -> str:
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)  # CamelCase
        s = re.sub(r"[_\-.]+|\d+", " ", s)
        return " ".join(w.lower() for w in s.split() if len(w) > 1 and w.lower() not in {"sm", "usd", "nvidia", "isaac"})
    folders = path.split("/")[:-1]
    context = [f for f in folders if f not in {"Isaac", "NVIDIA", "Assets", "Props", "ArchVis", "DigitalTwin"}]
    name = words(path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    return f"{name or words(folders[-1])} ({words(' '.join(context))})"


def describe_lvis(category: str) -> str:
    """'mouse_(computer_equipment)' -> 'mouse (computer equipment)', 'cellular_telephone' -> 'cellular telephone'."""
    m = re.match(r"(.*?)_\((.*)\)$", category)
    name, qualifier = (m.group(1), m.group(2)) if m else (category, "")
    name, qualifier = name.replace("_", " "), qualifier.replace("_", " ")
    return f"{name} ({qualifier})" if qualifier else name


def lvis_synonyms() -> dict[str, list[str]]:
    """LVIS category -> its synonyms, from the LVIS v1 category list."""
    import ast

    source = urllib.request.urlopen(LVIS_CATEGORIES_URL, timeout=60).read().decode()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "LVIS_CATEGORIES")
    return {c["name"]: c["synonyms"] for c in ast.literal_eval(node.value)}


def _embed(client, texts: list[str], task: str) -> np.ndarray:
    from google.genai import types

    out = []
    for i in range(0, len(texts), BATCH):
        # One Content per text: a plain list of strings is embedded as a single multi-part input.
        batch = [types.Content(parts=[types.Part(text=t)]) for t in texts[i:i + BATCH]]
        r = client.models.embed_content(model=EMBED_MODEL, contents=batch,
                                        config=types.EmbedContentConfig(task_type=task, output_dimensionality=EMBED_DIMS))
        if len(r.embeddings) != len(batch):
            raise RuntimeError(f"{EMBED_MODEL} returned {len(r.embeddings)} embeddings for {len(batch)} texts")
        out += [e.values for e in r.embeddings]
    v = np.asarray(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_index() -> int:
    """List both libraries, embed every entry, and save assets/index/. Return the number of entries."""
    import objaverse
    from google import genai

    paths = [k.removeprefix(ASSET_ROOT + "/") for f in FOLDERS for k in _keys(f"{ASSET_ROOT}/{f}")
             if k.endswith((".usd", ".usda", ".usdc", ".usdz"))]
    entries = [{"source": "isaac", "path": p, "text": (t := describe(p)), "names": [t.split(" (")[0]]}
                for p in sorted(p for p in paths if not SKIP.search("/" + p))]
    synonyms = lvis_synonyms()
    for c in sorted(objaverse.load_lvis_annotations()):
        names = list(dict.fromkeys(describe_lvis(n).split(" (")[0] for n in synonyms.get(c, [c])))
        text = describe_lvis(c) + (f"; also {', '.join(names[1:])}" if len(names) > 1 else "")
        entries.append({"source": "objaverse", "lvis": c, "text": text, "names": names})
    vectors = _embed(genai.Client(), [e["text"] for e in entries], "RETRIEVAL_DOCUMENT")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", vectors.astype(np.float16))
    (INDEX_DIR / "index.json").write_text(json.dumps({
        "built": str(date.today()), "isaac": {"server": f"{SERVER}/{ASSET_ROOT}", "folders": FOLDERS},
        "objaverse": {"version": "1.0", "categories": "LVIS", "licenses": sorted(OBJAVERSE_LICENSES)},
        "model": EMBED_MODEL, "dims": EMBED_DIMS, "assets": entries,
    }, indent=1))
    return len(entries)


def _singular(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us")) else w


def kind_words(name: str) -> set[str]:
    """Ways a name can state its kind: its head noun, and its last two words written as one ("smart phone" ->
    {"phone", "smartphone"})."""
    words = [w for w in re.findall(r"[a-z]+", name.lower()) if w not in NOT_KINDS and w not in NOT_NAMES]
    kinds = {_singular(words[-1])} if words else set()
    if len(words) >= 2:
        kinds.add(_singular(words[-2] + words[-1]))
    return kinds


def head_noun(text: str) -> str | None:
    """The last word naming the kind of thing, singular: "pair of scissors with orange handles" -> "scissors",
    "black wireless computer mouse" -> "mouse", "Book_01" -> "book"."""
    text = re.split(r"\b(?:with|without|in|on|for|from|that|which)\b", text.lower().replace("'s", ""))[0]
    words = [w for w in re.findall(r"[a-z]+", text) if len(w) > 1 and w not in NOT_KINDS and w not in NOT_NAMES]
    return _singular(words[-1]) if words else None


def _proportion_misfit(glb: str, size: np.ndarray | None) -> float:
    """How far a model's bounding-box proportions are from `size`: the spread of log(model / size) over the sorted
    extents (0 = same shape at some scale). Axis order is ignored, since the model is turned to fit later.
    inf if the file cannot be read; 0 if there is no size to compare with."""
    if size is None:
        return 0.0
    try:
        import trimesh

        ext = np.sort(np.ptp(trimesh.load(glb, force="scene").bounds, axis=0))
    except Exception:
        return float("inf")
    return float(np.ptp(np.log(np.maximum(ext, 1e-6) / np.maximum(np.sort(size), 1e-6))))


class AssetLibrary:
    """The saved index, loaded once; `match` finds an asset for each query."""

    def __init__(self):
        meta = json.loads((INDEX_DIR / "index.json").read_text())
        self.assets = meta["assets"]
        self.vectors = np.load(INDEX_DIR / "embeddings.npy").astype(np.float32)
        from google import genai

        self.client = genai.Client()

    def _pick_objaverse(self, entry: dict, query: str, query_vec: np.ndarray, size: np.ndarray | None) -> dict | None:
        """A downloadable CC-BY/CC0 object of an LVIS category whose name or tags name its kind, downloaded as GLB:
        of the SHAPE_SHORTLIST closest to the query by name, the one whose proportions best fit `size` (the object's
        extents, any order). None if the category has no such object (LVIS labels include some unrelated objects,
        e.g. a plate in "marker")."""
        import objaverse

        uids = sorted(objaverse.load_lvis_annotations()[entry["lvis"]])[:OBJAVERSE_SHORTLIST]
        ann = objaverse.load_annotations(uids)
        kinds = set().union(*(kind_words(n) for n in entry["names"])) | {head_noun(query)}

        def says_kind(a: dict) -> bool:
            text = " ".join([a.get("name", ""), *(t["name"] for t in a.get("tags", []))]).lower()
            words = [_singular(w) for w in re.findall(r"[a-z]+", text)]
            return bool(kinds & (set(words) | {a + b for a, b in zip(words, words[1:])}))

        usable = [u for u in uids if u in ann and ann[u].get("license") in OBJAVERSE_LICENSES
                  and (ann[u].get("faceCount") or 0) <= OBJAVERSE_MAX_FACES and says_kind(ann[u])]
        if not usable:
            return None
        texts = [" ".join([ann[u]["name"], *(t["name"] for t in ann[u].get("tags", [])[:6])]) for u in usable]
        by_name = np.argsort(_embed(self.client, texts, "RETRIEVAL_DOCUMENT") @ query_vec)[::-1][:SHAPE_SHORTLIST]
        shortlist = [usable[i] for i in by_name]
        glbs = objaverse.load_objects(shortlist)
        fits = {u: _proportion_misfit(glbs[u], size) for u in shortlist if u in glbs}
        best = min(fits, key=lambda u: (fits[u], shortlist.index(u)))
        a = ann[best]
        return {"uid": best, "name": a["name"], "license": a["license"], "author": (a.get("user") or {}).get("username"),
                "url": a.get("viewerUrl"), "path": glbs[best],
                "proportion_misfit": None if np.isinf(fits[best]) else round(float(fits[best]), 3)}

    def candidates(self, queries: list[str]) -> tuple[np.ndarray, list[list[dict]]]:
        """The query vectors, and for each query its TOP_K nearest entries, nearest first, each with its `score`."""
        q = _embed(self.client, queries, "RETRIEVAL_QUERY")
        scores = q @ self.vectors.T
        return q, [[self.assets[i] | {"score": round(float(row[i]), 3)} for i in np.argsort(row)[::-1][:TOP_K]]
                   for row in scores]

    def fetch(self, entry: dict, query: str, query_vec: np.ndarray, size: np.ndarray | None) -> dict | None:
        """Where the chosen entry's model comes from (`source`, `text`, `path`, and for Objaverse the object's uid,
        name, license and author), or None if an Objaverse category has no usable object."""
        found = ({"path": entry["path"]} if entry["source"] == "isaac"
                 else self._pick_objaverse(entry, query, query_vec, size))
        if not found:
            return None
        return ({"source": entry["source"], "text": entry["text"], "score": entry["score"]}
                | ({"lvis": entry["lvis"]} if entry["source"] == "objaverse" else {}) | found)
