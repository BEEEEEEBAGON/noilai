"""Prompt rendering: YAML templates -> chat messages, few-shot demos, re-encoding arms.

`render(item, task, variant, paraphrase, shots, arm, input_format, ...)` returns a list of
chat messages (`[{"role": "user", "content": ...}]`; a system message only when the model
config asks for one). Everything the model sees is in the user message, so that chat
templates without a system role (Gemma) get the same text as the others.

Re-encoding arms
----------------
The meaning-preserving arms (nfc, nfd, win1258, placement_old, placement_new) are applied
to the ENTIRE user message: instruction, demonstrations and item. Reasons: (1) the
intervention is then uniform, so that the token-level change is the only difference between
arms and the item does not stand out as the one oddly encoded span in an otherwise normal
message, which would itself be a cue; (2) the demonstration answers are re-encoded with the
rest, so the model is shown answers in the encoding it is reading; (3) these arms preserve
the text's meaning, so the instruction stays intelligible.
The strip arms (strip_tones, strip_all) are applied to the ITEM TEXT only (the item's input
and candidate, and the inputs/candidates of the demonstrations, never the demonstration
answers or the instruction). They destroy information; stripping the instruction would
confound "the model cannot recover the syllable structure of a stripped item" with "the
model cannot read a stripped instruction", and the answer-line marker must survive intact
for extraction. Demonstration answers keep their diacritics because the model is still asked
to produce fully spelled Vietnamese.

Demonstrations
--------------
Three fixed base pairs (prompts/demos.yaml) are expanded by the rule engine into three
demonstrations per task x variant. T2 demonstrations are the same for every item (one
pair per variant listed in `t2_variants`), because the T2 prompt withholds the variant and
per-variant demonstrations would reveal it. `demo_overlap` reports every syllable of the
demonstrations (and of the examples inside the instruction text) that occurs in an item
file; the runner refuses such a file unless told otherwise.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import yaml

from ..gen import variants as V
from ..vi import reencode as R
from ..vi import unicode as U
from ..vi.syllable import Inventory, Syllable, spell, try_parse

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "prompts"
TEMPLATE_FILE = PROMPTS_DIR / "noilai.yaml"
DEMOS_FILE = PROMPTS_DIR / "demos.yaml"
XCOPA_FILE = PROMPTS_DIR / "xcopa.yaml"

TASKS = ("T1", "T2", "T3")
# item tasks rendered with another task's templates: attested examples are transformation items
TEMPLATE_TASK = {"attested": "T1"}
PARAPHRASES = ("p0", "p1", "p2")
LANGUAGES = ("vi", "en")
INSTRUCTIONS = ("explained", "name_only")
INPUT_FORMATS = ("raw", "components", "spaced")
STRIP_ARMS = ("strip_tones", "strip_all")
MAX_SHOTS = 3

_PLACEHOLDER = re.compile(r"\{([a-z_0-9]+)\}")


# ------------------------------------------------------------------ loading
@cache
def load_templates(path: Path = TEMPLATE_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def load_demo_spec(path: Path = DEMOS_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def prompt_file_hashes(paths: Iterable[Path] = (TEMPLATE_FILE, DEMOS_FILE, XCOPA_FILE)) -> dict[str, str]:
    """sha256 of every prompt file plus a combined hash (`_all`)."""
    out: dict[str, str] = {}
    h_all = hashlib.sha256()
    for p in sorted(paths, key=lambda q: q.name):
        b = Path(p).read_bytes()
        out[Path(p).name] = hashlib.sha256(b).hexdigest()
        h_all.update(Path(p).name.encode("utf-8") + b"\0" + b)
    out["_all"] = h_all.hexdigest()
    return out


def _fill(template: str, mapping: dict) -> str:
    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key not in mapping:
            raise KeyError(f"template placeholder {{{key}}} has no value")
        return str(mapping[key])

    return _PLACEHOLDER.sub(repl, template)


def _tidy(text: str) -> str:
    """Strip trailing spaces, collapse runs of blank lines (left by empty blocks), strip ends."""
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        if ln == "" and out and out[-1] == "":
            continue
        out.append(ln)
    return "\n".join(out).strip("\n")


# ------------------------------------------------------------------ demos
def _syl(text: str) -> Syllable:
    p = try_parse(text, strict=True)
    if p is None:
        raise ValueError(f"demo syllable {text!r} does not parse")
    return p.syllable


def _text(pair: tuple[Syllable, Syllable]) -> str:
    return f"{spell(pair[0])} {spell(pair[1])}"


def _twin(variant: str, a: Syllable, b: Syllable, out: tuple[Syllable, Syllable], inv: Inventory) -> str:
    """An other_variant twin for a T3 "no" demonstration; tone perturbation as fallback."""
    order = list(V.VARIANTS)
    start = order.index(variant)
    for k in range(1, len(order)):
        w = order[(start + k) % len(order)]
        o = V.apply(w, a, b)
        if o != out and o != (a, b) and all(inv.is_legal(s, "onset_rime") for s in o):
            return _text(o)
    for t in range(6):
        cand = (out[0].with_tone(t), out[1])
        if cand != out and cand != (a, b) and inv.is_legal(cand[0], "onset_rime"):
            return _text(cand)
    raise ValueError(f"no twin for demo pair {_text((a, b))} under {variant}")


def build_demos(spec: dict | None = None, inventory: Inventory | None = None) -> dict[tuple[str, str], list[dict]]:
    """Expand the fixed base pairs into demonstrations for every (task, variant).

    Returns {(task, variant): [demo, ...]} with demo = {input, answer, candidate, label,
    variant, base_pair}. Raises if a pair yields an identity or an illegal output under any
    variant, so that a bad pair in demos.yaml fails loudly instead of teaching a wrong rule.
    """
    spec = spec or load_demo_spec()
    if inventory is None:
        from ..vi import lexicon as L
        inventory = L.load_inventory()
    pairs = [tuple(_syl(s) for s in entry["syllables"]) for entry in spec["pairs"]]
    if len(pairs) < MAX_SHOTS:
        raise ValueError(f"need at least {MAX_SHOTS} demo pairs, got {len(pairs)}")
    # YAML reads a bare yes/no as a boolean; accept both spellings
    t3_pattern = [("yes" if x is True else "no" if x is False else str(x)) for x in spec.get("t3_pattern", ["yes", "no", "yes"])]
    if any(x not in ("yes", "no") for x in t3_pattern):
        raise ValueError(f"t3_pattern must contain only yes/no, got {t3_pattern}")
    t2_variants = list(spec.get("t2_variants", ["V1", "V2", "V3"]))
    demos: dict[tuple[str, str], list[dict]] = {}
    outputs: dict[tuple[int, str], tuple[Syllable, Syllable]] = {}
    for i, (a, b) in enumerate(pairs):
        for v in V.VARIANTS:
            if V.is_identity(v, a, b):
                raise ValueError(f"demo pair {_text((a, b))} is an identity under {v}")
            o = V.apply(v, a, b)
            if not all(inventory.is_legal(s, "onset_rime") for s in o):
                raise ValueError(f"demo pair {_text((a, b))} gives an illegal output under {v}: {_text(o)}")
            outputs[(i, v)] = o
    for v in V.VARIANTS:
        demos[("T1", v)] = [
            {"input": _text(pairs[i]), "answer": _text(outputs[(i, v)]), "candidate": None, "label": None,
             "variant": v, "base_pair": _text(pairs[i])}
            for i in range(len(pairs))
        ]
        t3 = []
        for i in range(len(pairs)):
            a, b = pairs[i]
            out = outputs[(i, v)]
            label = t3_pattern[i % len(t3_pattern)]
            cand = _text(out) if label == "yes" else _twin(v, a, b, out, inventory)
            t3.append({"input": _text(pairs[i]), "answer": "Có" if label == "yes" else "Không",
                       "candidate": cand, "label": label, "variant": v, "base_pair": _text(pairs[i])})
        demos[("T3", v)] = t3
    t2 = []
    for i in range(len(pairs)):
        v = t2_variants[i % len(t2_variants)]
        t2.append({"input": _text(outputs[(i, v)]), "answer": _text(pairs[i]), "candidate": None, "label": None,
                   "variant": v, "base_pair": _text(pairs[i])})
    for v in V.VARIANTS:
        demos[("T2", v)] = t2
    return demos


@cache
def default_demos() -> dict[tuple[str, str], list[dict]]:
    return build_demos()


def demo_syllables(demos: dict, templates: dict | None = None) -> set[str]:
    """Every syllable shown in a demonstration (inputs, answers, candidates) plus the
    example syllables used inside the instruction text."""
    out: set[str] = set()
    for lst in demos.values():
        for d in lst:
            for key in ("input", "answer", "candidate"):
                val = d.get(key)
                if not val or val in ("Có", "Không"):
                    continue
                out.update(U.nfc(val).lower().split())
    templates = templates or load_templates()
    out.update(U.nfc(s).lower() for s in templates.get("example_syllables", []))
    return out


def item_texts(item: dict) -> list[str]:
    """Every Vietnamese phrase an item carries (input, gold outputs, candidates)."""
    texts = [item["input"]]
    task = item.get("task")
    if task == "T1":
        texts.extend(item["gold"])
    elif task == "T2":
        texts.extend(g["output"] for g in item["gold"])
    elif task == "T3":
        texts.extend(x for x in (item.get("candidate"), item.get("correct_output")) if x)
    elif task == "attested":
        texts.extend(item.get("gold", []))
    return texts


def demo_overlap(demos: dict, items: Iterable[dict], templates: dict | None = None) -> dict[str, list[str]]:
    """{syllable: [item_id, ...]} for every demo/example syllable that occurs in the items."""
    sylls = demo_syllables(demos, templates)
    hits: dict[str, set[str]] = {}
    for it in items:
        for t in item_texts(it):
            for w in U.nfc(t).lower().split():
                if w in sylls:
                    hits.setdefault(w, set()).add(it.get("item_id", "?"))
    return {k: sorted(v) for k, v in sorted(hits.items())}


# ------------------------------------------------------------------ input formats
def _rime_display(onset_sp: str, rime_sp: str) -> str:
    if onset_sp == "gi" and (rime_sp == "" or rime_sp[0] not in U.VOWELS_NFC_SET or rime_sp[0] == "ê"):
        return "i" + rime_sp      # contracted gi: gì = gi + i, gìn = gi + in
    return rime_sp


def format_input(phrase: str, input_format: str, language: str = "vi", templates: dict | None = None) -> str:
    """Render a two-syllable phrase in the requested input format."""
    if input_format == "raw":
        return phrase
    templates = templates or load_templates()
    spec = templates["input_formats"][input_format]
    words = phrase.split()
    if input_format == "spaced":
        return " / ".join(" ".join(U.letters(w)) for w in words)
    if input_format == "components":
        parts = []
        for w in words:
            p = try_parse(w, strict=False)
            if p is None:
                parts.append(f"[{w}]")
                continue
            onset_sp = "q" if p.onset_spelling == "qu" else p.onset_spelling
            onset = (_fill(spec[f"onset_label_{language}"], {"onset": onset_sp}) if onset_sp
                     else spec[f"zero_onset_{language}"])
            rime = _fill(spec[f"rime_label_{language}"], {"rime": _rime_display(p.onset_spelling, p.rime_spelling)})
            tone = _fill(spec[f"tone_label_{language}"], {"tone": U.TONE_NAMES_VI[p.syllable.tone]})
            parts.append(f"[{onset}, {rime}, {tone}]")
        return " ".join(parts)
    raise ValueError(f"unknown input_format {input_format!r}; choose from {INPUT_FORMATS}")


# ------------------------------------------------------------------ rendering
def _variant_steps(variant: str, instruction: str, language: str, templates: dict) -> str:
    if instruction == "name_only":
        return ""
    blocks = templates["variant_blocks"][language]
    steps = templates["variants"][variant][f"steps_{language}"]
    return blocks["steps_header"] + "\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))


def _variants_overview(instruction: str, language: str, templates: dict) -> str:
    blocks = templates["variant_blocks"][language]
    lines = []
    for v in V.VARIANTS:
        spec = templates["variants"][v]
        name = spec[f"name_{language}"]
        if instruction == "name_only":
            lines.append(_fill(blocks["overview_name_only"], {"name": name}))
        else:
            # the distinctive steps (2 and 3); step 1 and step 4 are the same for every variant
            steps = " ".join(spec[f"steps_{language}"][1:3])
            lines.append(_fill(blocks["overview_explained"], {"name": name, "steps": steps}))
    return "\n".join(lines)


def _demo_block(task: str, variant: str, shots: int, language: str, input_format: str, item_arm: str | None,
                templates: dict, demos: dict) -> str:
    if shots <= 0:
        return ""
    if shots > MAX_SHOTS:
        raise ValueError(f"at most {MAX_SHOTS} fixed demonstrations exist; shots={shots}")
    fmt = templates["demo_format"][language]
    rendered = []
    for d in demos[(task, variant)][:shots]:
        inp = d["input"]
        cand = d.get("candidate")
        if item_arm is not None:
            inp = R.reencode(inp, item_arm)
            cand = R.reencode(cand, item_arm) if cand else cand
        rendered.append(_fill(fmt[task], {
            "input": format_input(inp, input_format, language, templates),
            "candidate": format_input(cand, input_format, language, templates) if cand else "",
            "answer": d["answer"],
        }))
    return fmt["header"] + "\n" + "\n\n".join(rendered) + "\n"


def prompt_id(paraphrase: str = "p0", shots: int = 3, instruction: str = "explained", input_format: str = "raw",
              language: str = "vi") -> str:
    return f"{language}-{paraphrase}-s{shots}-{instruction}-{input_format}"


def render(item: dict, task: str | None = None, variant: str | None = None, paraphrase: str = "p0",
           shots: int = 3, arm: str = "nfc", input_format: str = "raw", instruction: str = "explained",
           language: str = "vi", templates: dict | None = None, demos: dict | None = None,
           system: str | None = None) -> list[dict]:
    """Render one item into chat messages.

    task/variant default to the item's own; for T2 the variant selects nothing in the text
    (it is withheld) and only picks the demonstration set, which is the same for every
    variant. `arm` is a noilai.vi.reencode arm; see the module docstring for how it is
    applied. Raises if the item's canary string would appear in the prompt.
    """
    templates = templates or load_templates()
    demos = demos or default_demos()
    task = TEMPLATE_TASK.get(task or item["task"], task or item["task"])
    variant = variant or item["variant"]
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    if language not in LANGUAGES or instruction not in INSTRUCTIONS or input_format not in INPUT_FORMATS:
        raise ValueError(f"bad rendering options: language={language!r} instruction={instruction!r} "
                         f"input_format={input_format!r}")
    if arm not in R.ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {R.ARMS}")
    strip = arm in STRIP_ARMS
    if strip and input_format != "raw":
        raise ValueError("strip arms are defined on the raw input format only")
    item_arm = arm if strip else None

    lang_templates = templates["tasks"][task][language]
    if paraphrase not in lang_templates:
        raise ValueError(f"no paraphrase {paraphrase!r} for {task}/{language}; have {sorted(lang_templates)}")
    concepts = templates["concepts"][language]
    vspec = templates["variants"][variant]
    inp = item["input"]
    cand = item.get("candidate") if task == "T3" else None
    if task == "T3" and not cand:
        raise ValueError("T3 items need a candidate")
    if strip:
        inp = R.reencode(inp, arm)
        cand = R.reencode(cand, arm) if cand else cand
    ans_key = {"T1": "phrase", "T2": "original", "T3": "yesno"}[task]
    mapping = {
        "definition": concepts["definition"],
        "structure": concepts["structure"],
        "spelling": concepts["spelling"],
        "four_kinds_intro": concepts["four_kinds_intro"],
        "variant_name": vspec[f"name_{language}"],
        "variant_steps": _variant_steps(variant, instruction, language, templates),
        "variants_overview": _variants_overview(instruction, language, templates),
        "input_format_note": templates["input_formats"][input_format][f"note_{language}"],
        "demos": _demo_block(task, variant, shots, language, input_format, item_arm, templates, demos),
        "input": format_input(inp, input_format, language, templates),
        "candidate": format_input(cand, input_format, language, templates) if cand else "",
        "answer_instruction": templates["answer_instruction"][language][ans_key],
    }
    content = _tidy(_fill(lang_templates[paraphrase], mapping))
    if not strip:
        content = R.reencode(content, arm)
    canary = item.get("canary")
    if canary and canary in content:
        raise RuntimeError(f"canary of {item.get('item_id')} would be sent in the prompt")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": content})
    return messages


def messages_text(messages: list[dict]) -> str:
    """The rendered string that `prompt_hash` hashes: role-tagged, newline-joined."""
    return "\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)


def prompt_hash(messages: list[dict]) -> str:
    return hashlib.sha256(messages_text(messages).encode("utf-8")).hexdigest()


def answer_marker(templates: dict | None = None) -> str:
    return (templates or load_templates())["answer_marker"]


def describe(templates: dict | None = None) -> dict:
    """Which cells exist: tasks x languages x paraphrases."""
    templates = templates or load_templates()
    return {t: {lang: sorted(ps) for lang, ps in templates["tasks"][t].items()} for t in TASKS}


def dumps_messages(messages: list[dict]) -> str:
    return json.dumps(messages, ensure_ascii=False)
