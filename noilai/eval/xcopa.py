"""XCOPA (Vietnamese) for the re-encoding arms (E3).

Items come from data/external/xcopa_{test,val}_vi.jsonl (premise, choice1, choice2, question
in {cause, effect}, label in {0, 1}). `load_xcopa` turns them into harness items with
task "XCOPA", `render_xcopa` renders the Vietnamese COPA framing (answer "1" or "2" on the
"Đáp án:" line) and `completion_pair` gives the context and the two candidate continuations
for log-probability scoring on open models. Meaning-preserving arms re-encode the whole
message; strip arms re-encode the premise and the two alternatives (the item text) only,
as in noilai.eval.prompts.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

from ..vi import reencode as R
from ..vi import unicode as U
from .prompts import STRIP_ARMS, XCOPA_FILE, _fill, _tidy

XCOPA_TASK = "XCOPA"
XCOPA_PARAPHRASES = ("p0", "p1", "p2")


@lru_cache(maxsize=None)
def load_xcopa_templates(path: Path = XCOPA_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_xcopa_file(path: Path) -> bool:
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                d = json.loads(ln)
                return "premise" in d and "choice1" in d
    return False


def load_xcopa(path: Path, split: Optional[str] = None) -> list[dict]:
    split = split or ("val" if "val" in Path(path).name else "test")
    items = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            d = json.loads(ln)
            items.append({
                "item_id": f"XCOPA-{split}-{int(d['idx']):04d}",
                "task": XCOPA_TASK, "variant": "-",
                "premise": U.nfc(d["premise"]), "choice1": U.nfc(d["choice1"]), "choice2": U.nfc(d["choice2"]),
                "question": d["question"], "label": int(d["label"]), "gold": str(int(d["label"]) + 1),
                "input": U.nfc(d["premise"]), "split": split, "in_core": True, "source": "xcopa",
                "base_pair_id": f"xcopa-{split}-{int(d['idx'])}", "strata": {"question": d["question"],
                                                                             "changed": d.get("changed")},
            })
    return items


def render_xcopa(item: dict, paraphrase: str = "p0", arm: str = "nfc", templates: Optional[dict] = None,
                 system: Optional[str] = None) -> list[dict]:
    templates = templates or load_xcopa_templates()
    if arm not in R.ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    strip = arm in STRIP_ARMS
    premise, c1, c2 = item["premise"], item["choice1"], item["choice2"]
    if strip:
        premise, c1, c2 = (R.reencode(x, arm) for x in (premise, c1, c2))
    content = _tidy(_fill(templates["vi"][paraphrase], {
        "relation": templates["relation_vi"][item["question"]],
        "premise": premise, "choice1": c1, "choice2": c2,
    }))
    if not strip:
        content = R.reencode(content, arm)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:] if s else s


def completion_pair(item: dict, arm: str = "nfc", templates: Optional[dict] = None) -> tuple[str, list[str]]:
    """(context, [continuation_1, continuation_2]) for log-probability scoring: the premise
    without its final period, the connective, then each alternative with its first letter
    lower-cased. Continuations start with a space. The arm is applied to context and
    continuations alike (strip arms included: the whole completion is item text)."""
    templates = templates or load_xcopa_templates()
    premise = item["premise"].rstrip(" .")
    context = _fill(templates["completion_vi"][item["question"]], {"premise_no_period": premise})
    conts = [" " + _lower_first(item["choice1"]), " " + _lower_first(item["choice2"])]
    return R.reencode(context, arm), [R.reencode(c, arm) for c in conts]
