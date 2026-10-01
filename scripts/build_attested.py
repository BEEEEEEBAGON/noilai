#!/usr/bin/env python
"""Turn data/attested_seed.tsv into the release file attested.jsonl (design 4.7).

Seed columns: input, output, variant (V1-V6), positions ("0-1", "1-2", "0-2", optionally
"reversed"), exactness (exact | approx(merger: <O3 name>) | approx(substitution: X→Y)),
region_tag (N/C/S or empty), gloss_input, gloss_output, note, confidence, vulgar, source,
verified_by.

For every row the rule engine is applied to the declared positions; the release row
records the attested output, the engine's output, whether they agree after
canonicalization, every position pair and order that reproduces the attested form (for
the record), the parsed structures, and the flags that decide where the row may be used:
only `exact` two-syllable rows enter T1/T3-style scoring and H6; every row enters T2 with
the attested gold; vulgar rows stay in the gated split. Rows whose input or output does
not parse are kept with `parse_ok = false` so that nothing is silently dropped.

Exactness is ENGINE-DERIVED (design 4.7(b), 12.16): `exact` iff the rule output
canonicalizes to the attested output; otherwise the seed's declared `approx(...)` tag is
kept and `approx_reachable` records whether the attested form is reachable from the rule
output by exactly the named change (design 3.7: an O3 merger, or one X→Y substitution in
one syllable). The seed's tag is kept as `declared_exactness`; a row declared `exact` that
the engine does not reproduce is reported (and fails the build under --strict), never
labelled exact.

Text fields (`input`, `attested_output`, `rule_output`, `gold`) are stored in the release
placement style (STYLE = old, design 12.5) like every other release file.

    python scripts/build_attested.py --out data/release/v0.3/attested.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from noilai.gen import variants as V
from noilai.gen.generate import STYLE, syl_dict
from noilai.vi import lexicon as L
from noilai.vi import unicode as U
from noilai.vi.reencode import canonical_text, convert_placement
from noilai.vi.syllable import Syllable, replace, spell, try_parse

POSITION_PAIRS_3 = ((0, 2), (1, 2), (0, 1))

# Design 2.1 O3 merger table: (component, the values a merger identifies). A merger name in an
# `approx(merger: ...)` tag lists two or more of the values separated by '/', '=' or '→'
# (e.g. 'hỏi/ngã', 'n/ng', 'd/gi', 'tr/ch'); several mergers are joined with '+'.
MERGERS: tuple[tuple[str, frozenset], ...] = (
    ("onset", frozenset({"d", "gi", "r"})),        # everywhere d = gi; North d/gi = r
    ("onset", frozenset({"ch", "tr"})),            # North
    ("onset", frozenset({"s", "x"})),              # North
    ("onset", frozenset({"v", "d", "gi"})),        # South/Centre v -> d/gi
    ("coda", frozenset({"n", "ng"})),              # South/Centre
    ("coda", frozenset({"t", "c"})),               # South/Centre
    ("tone", frozenset({3, 4})),                   # South/Centre hỏi = ngã
)
_TONE_BY_NAME = {n: i for i, n in enumerate(U.TONE_NAMES_VI)} | {n: i for i, n in enumerate(U.TONE_NAMES)}
_EXACTNESS_RE = re.compile(r"^approx\((merger|substitution):\s*(.+)\)$")


def stored(text: str) -> str:
    """A seed string as stored in the release: NFC, lower-case, release placement style."""
    return convert_placement(U.nfc(text).strip().lower(), STYLE)


def apply_at(sylls, variant: str, i: int, j: int, reverse: bool):
    x, y = V.apply(variant, sylls[i], sylls[j])
    o = list(sylls)
    if reverse:
        o[i], o[j] = y, x
    else:
        o[i], o[j] = x, y
    return tuple(o)


def parse_positions(spec: str, n: int) -> tuple[int, int, bool]:
    spec = (spec or "").strip()
    reverse = "reversed" in spec
    core = spec.replace("reversed", "").strip()
    if core:
        i, j = (int(x) for x in core.split("-"))
    else:
        i, j = (0, 1) if n == 2 else (0, 2)
    return i, j, reverse


def _resolve_merger(name: str) -> tuple[str, frozenset] | None:
    parts = [p.strip() for p in re.split(r"\s*[/=→]\s*", name.strip()) if p.strip()]
    if len(parts) < 2:
        return None
    if all(p in _TONE_BY_NAME for p in parts):
        vals: set = {_TONE_BY_NAME[p] for p in parts}
    else:
        vals = set(parts)
    for comp, allowed in MERGERS:
        if vals <= allowed:
            return comp, frozenset(vals)
    return None


def _component(s: Syllable, comp: str):
    return getattr(s, comp)


def reachable_by_named_change(rule: list[Syllable], att: list[Syllable], exactness: str) -> bool | None:
    """Design 3.7: is the attested form reachable from the rule output by exactly the named
    change? None for `exact` rows and for tags that name no change (`approx(unspecified)`)."""
    m = _EXACTNESS_RE.match(exactness.strip())
    if m is None or len(rule) != len(att):
        return None if not exactness.startswith("approx") else False
    kind, spec = m.group(1), m.group(2).strip()
    if kind == "merger":
        mergers = [_resolve_merger(n) for n in spec.split("+")]
        if not mergers or any(mg is None for mg in mergers):
            return False
        used = [False] * len(mergers)
        for r, a in zip(rule, att):
            for comp in ("onset", "glide", "nucleus", "coda", "tone"):
                rv, av = _component(r, comp), _component(a, comp)
                if rv == av:
                    continue
                hit = [k for k, (mc, vals) in enumerate(mergers) if mc == comp and rv in vals and av in vals]
                if not hit:
                    return False                      # a difference no named merger explains
                for k in hit:
                    used[k] = True
        return all(used) and rule != att              # every named merger applied, something changed
    # substitution X→Y in exactly one syllable (tone names or letters of the toneless spelling)
    xy = [p.strip() for p in re.split(r"\s*(?:→|->)\s*", spec)]
    if len(xy) != 2:
        return False
    x, y = xy
    diff_positions = [i for i, (r, a) in enumerate(zip(rule, att)) if r != a]
    if len(diff_positions) != 1:
        return False
    r, a = rule[diff_positions[0]], att[diff_positions[0]]
    if x in _TONE_BY_NAME and y in _TONE_BY_NAME:
        return replace(r, tone=_TONE_BY_NAME[y]) == a and r.tone == _TONE_BY_NAME[x]
    if r.tone != a.tone:
        return False
    rt, at_ = U.strip_tones(spell(r, "new")), U.strip_tones(spell(a, "new"))
    if rt.count(x) != 1:
        return False
    cand = rt.replace(x, y, 1)
    p = try_parse(cand, strict=False)                 # the toneless candidate must be a syllable
    return p is not None and replace(p.syllable, tone=a.tone) == a and cand == at_


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-file", type=Path, default=ROOT / "data" / "attested_seed.tsv")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when a row declared exact is not reproduced by the engine or an approx row is not reachable by its named change")
    args = ap.parse_args(argv)
    inv = L.load_inventory()
    with open(args.seed_file, encoding="utf-8") as f:
        rows = [{k: (v or "") for k, v in r.items() if k} for r in csv.DictReader(f, delimiter="\t")]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_match = 0
    problems: list[str] = []
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            inp = [try_parse(w, strict=False) for w in r["input"].split()]
            out = [try_parse(w, strict=False) for w in r["output"].split()]
            parse_ok = all(inp) and all(out) and len(inp) == len(out) and len(inp) in (2, 3)
            rule_out = None
            rule_sylls: list[Syllable] | None = None
            rule_matches = None
            matches = []
            pi = pj = None
            reverse = False
            if parse_ok:
                sy = [p.syllable for p in inp]
                pi, pj, reverse = parse_positions(r.get("positions", ""), len(sy))
                rule_sylls = list(apply_at(sy, r["variant"], pi, pj, reverse))
                rule_out = L.emit_phrase(rule_sylls, STYLE, inv)          # already in the release style
                target = canonical_text(r["output"])
                rule_matches = canonical_text(rule_out) == target
                pairs = POSITION_PAIRS_3 if len(sy) == 3 else ((0, 1),)
                for (a, b) in pairs:
                    for v in V.ALL_VARIANTS:
                        for rev in (False, True):
                            if canonical_text(L.emit_phrase(apply_at(sy, v, a, b, rev), STYLE, inv)) == target:
                                matches.append(f"{v}@{a}-{b}{' reversed' if rev else ''}")
                n_ok += 1
                n_match += bool(rule_matches)
            # exactness is engine-derived; the seed's tag is kept as declared_exactness
            declared = r.get("exactness", "").strip()
            exact = bool(rule_matches)
            if rule_matches:
                exactness = "exact"
            elif declared.startswith("approx"):
                exactness = declared
            else:
                exactness = "approx(unspecified)"
            approx_reachable = None
            if parse_ok and not rule_matches and rule_sylls is not None:
                approx_reachable = reachable_by_named_change(rule_sylls, [p.syllable for p in out], exactness)
            if declared == "exact" and parse_ok and not rule_matches:        # one problem per row
                problems.append(f"row {i+1} {r['input']!r} -> {r['output']!r} declared exact but the rule gives {rule_out!r}")
            elif approx_reachable is False:
                problems.append(f"row {i+1} {r['input']!r} -> {r['output']!r} tagged {exactness} is not reachable from {rule_out!r} by that change")
            in_stored, out_stored = stored(r["input"]), stored(r["output"])
            gold: dict[str, str] = {canonical_text(out_stored): out_stored}
            if rule_out:
                gold.setdefault(canonical_text(rule_out), rule_out)
            row = {
                "item_id": f"ATT-{i+1:04d}", "task": "attested", "declared_variant": r["variant"],
                "variant": (sorted(m.split("@")[0] for m in matches if "reversed" not in m)
                            or sorted(m.split("@")[0] for m in matches) or [r["variant"]])[0],
                "variant_labels": sorted({m.split("@")[0] for m in matches}),
                "declared_matches_engine": bool(matches) and any(m.startswith(r["variant"] + "@") for m in matches),
                "positions": f"{pi}-{pj}{' reversed' if reverse else ''}" if pi is not None else None,
                "input": in_stored, "attested_output": out_stored,
                "rule_output": rule_out, "rule_matches_attested": rule_matches,
                "reproducing_labels": matches,
                "exactness": exactness, "declared_exactness": declared or None, "exact": exact,
                "approx_reachable": approx_reachable,
                "eligible_h6": exact and len(inp) == 2,
                "gold": sorted(gold.values()),
                "input_syllables": [syl_dict(p.syllable, w) for p, w in zip(inp, in_stored.split())] if all(inp) else None,
                "n_syllables": len(inp),
                "region_tag": r.get("region_tag", "") or None,
                "gloss_input": r["gloss_input"], "gloss_output": r["gloss_output"], "note": r["note"],
                "confidence": r["confidence"], "vulgar": r["vulgar"].strip().lower() == "yes",
                "source": r.get("source", ""), "verified_by": r["verified_by"].strip() or None, "parse_ok": parse_ok,
                "base_pair_id": f"att-{canonical_text(r['input']).replace(' ', '_')}",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for p in problems:
        print(f"[WARN] {p}")
    print(f"{len(rows)} seed rows, {n_ok} parse, {n_match} where the rule reproduces the attested output, "
          f"{len(problems)} exactness problems -> {args.out}")
    return 1 if (args.strict and problems) else 0


if __name__ == "__main__":
    sys.exit(main())
