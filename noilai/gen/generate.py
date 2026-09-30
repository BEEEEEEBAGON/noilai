"""Build the NóiLái benchmark: base pairs -> T1/T2/T3 items -> splits -> files.

Item schema (one JSON object per line; see docs/DATA_FORMAT.md):
  item_id        'T1-V1-000123'          stable within a build
  task           'T1' | 'T2' | 'T3'
  variant        'V1'..'V4'              the variant asked (T1), the variant that produced the
                                         input (T2, withheld from the prompt), or the variant of
                                         the question (T3)
  input          'mèo cái'               NFC, lowercase, new-style placement
  input_syllables  [structure of each input syllable]
  gold           T1: ['mài kéo'];  T2: [{'variant','reversed','output'}, ...];  T3: 'yes'|'no'
  candidate      T3 only: the phrase to judge
  twin_type      T3 'no' items: 'other_variant'|'onset'|'rime'|'tone'|'spelling'
  pair_item_id   T3: the item_id of the yes/no counterpart with the same input
  base_pair_id   cluster id for the clustered bootstrap (same underlying pair -> same id)
  source         'lexicon' (real two-syllable word) | 'pseudo' (sampled syllables)
  strata         dict of item-level covariates (lexicality, glide, zero onset, spelling
                 triggers, tone pair, stop coda, syllable frequencies)
  split          'dev' | 'test'
  in_core        bool (balanced 125-per-cell subset of test for API models)
  canary         present on every test-split item

Everything is deterministic given (seed, resources). The generator never emits a
syllable that fails Inventory.is_legal(level='onset_rime') and never emits an
item whose output equals its input.
"""
from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..vi import lexicon as L
from ..vi import unicode as U
from ..vi.syllable import CANONICAL_ONSETS, Inventory, Syllable, replace, spell, try_parse
from . import variants as V

TASKS = ("T1", "T2", "T3")
TWIN_TYPES = ("other_variant", "onset", "rime", "tone", "spelling")
LEGALITY = "onset_rime"


@dataclass(frozen=True)
class BasePair:
    base_pair_id: str
    a: Syllable
    b: Syllable
    source: str                 # 'lexicon' | 'pseudo'
    freq_a: int
    freq_b: int

    @property
    def text(self) -> str:
        return f"{spell(self.a)} {spell(self.b)}"


def syl_dict(s: Syllable) -> dict:
    return {"onset": s.onset, "glide": s.glide, "nucleus": s.nucleus, "coda": s.coda,
            "tone": s.tone, "spelled": spell(s)}


def spelling_triggers(inp: tuple[Syllable, Syllable], out: tuple[Syllable, Syllable]) -> list[str]:
    """Spelling-rule applications visible in the output relative to the input, e.g. 'c>k'."""
    from ..vi.syllable import _onset_spelling  # local import: private helper
    trig = []
    for si, so in zip(inp, out):
        a, b = _onset_spelling(si), _onset_spelling(so)
        if a != b and si.onset == so.onset:
            trig.append(f"{a}>{b}")
    for so in out:
        sp = _onset_spelling(so)
        if sp in ("k", "gh", "ngh", "qu"):
            trig.append(f"uses:{sp}")
    return trig


def strata_for(inp: tuple[Syllable, Syllable], out: tuple[Syllable, Syllable], bp: BasePair,
               lex_pairs: set[tuple[str, str]], freqs: Counter) -> dict:
    o_text = (spell(out[0]), spell(out[1]))
    return {
        "input_lexical": bp.source == "lexicon",
        "output_lexical": o_text in lex_pairs or (o_text[1], o_text[0]) in lex_pairs,
        "output_syllables_attested": [freqs.get(o_text[0], 0) > 0, freqs.get(o_text[1], 0) > 0],
        "has_glide": any(s.glide for s in inp),
        "has_zero_onset": any(s.onset == "" for s in inp),
        "has_stop_coda": any(s.coda in ("p", "t", "c", "ch") for s in inp),
        "spelling_triggers": spelling_triggers(inp, out),
        "tone_pair": f"{U.TONE_NAMES[inp[0].tone]}-{U.TONE_NAMES[inp[1].tone]}",
        "same_tone": inp[0].tone == inp[1].tone,
        "same_rime": inp[0].rime == inp[1].rime,
        "input_freq": [bp.freq_a, bp.freq_b],
        "output_freq": [freqs.get(o_text[0], 0), freqs.get(o_text[1], 0)],
    }


class Generator:
    def __init__(self, seed: int = 20261004, inventory: Optional[Inventory] = None,
                 words: Optional[list[str]] = None, exclude_qu: bool = True):
        """exclude_qu: leave out base pairs containing a qu- syllable. The written 'qu' hides
        the glide inside the onset letters; school grammar swaps 'qu' as an onset while
        phonology (and this engine) treats the glide as part of the rime, and the two give
        different outputs (quả hồng -> cổng hoà vs quồng hả). Until native validators settle
        the convention, qu- pairs are generated only when explicitly requested and kept as
        their own stratum (see docs/DESIGN_DECISIONS.md)."""
        self.seed = seed
        self.exclude_qu = exclude_qu
        self.rng = random.Random(seed)
        self.inv = inventory or L.load_inventory()
        if words is None:
            self.freqs = L.syllable_frequencies()          # cached word-list index
            self.lex_pairs: set[tuple[str, str]] = L.lexical_pairs()
        else:
            self.freqs = L.syllable_frequencies(words)
            self.lex_pairs = set()
            for a, b in L.two_syllable_words(words):
                ca, cb = try_parse(a, strict=False), try_parse(b, strict=False)
                if ca and cb:
                    self.lex_pairs.add((spell(ca.syllable), spell(cb.syllable)))
        # attested syllables usable for pseudo pairs and twins (standard spellings, real tones)
        self.attested = sorted({s for s in self.inv.structures if self.inv.is_legal(s, "attested")},
                               key=lambda s: (s.onset, s.glide, s.nucleus, s.coda, s.tone))

    # ------------------------------------------------------------- base pairs
    def base_pairs(self, n_lexicon: int, n_pseudo: int) -> list[BasePair]:
        pairs: list[BasePair] = []
        lex = sorted(self.lex_pairs)
        self.rng.shuffle(lex)
        self.n_lexical_dropped = 0
        for a, b in lex:
            if sum(1 for p in pairs if p.source == "lexicon") >= n_lexicon:
                break
            sa, sb = try_parse(a).syllable, try_parse(b).syllable
            if self.exclude_qu and any(self._is_qu(x) for x in (sa, sb)):
                continue
            # both syllables must be attested in the inventory (guards against word-list typos,
            # loanwords and non-standard spellings; also guarantees the pair is a T2 reading)
            if not (self.inv.is_legal(sa, "attested") and self.inv.is_legal(sb, "attested")) or sa == sb:
                self.n_lexical_dropped += 1
                continue
            pairs.append(BasePair(self._bp_id("lexicon", a, b), sa, sb, "lexicon",
                                  self.freqs.get(a, 0), self.freqs.get(b, 0)))
        seen = set(self.lex_pairs)
        while sum(1 for p in pairs if p.source == "pseudo") < n_pseudo:
            sa, sb = self.rng.choice(self.attested), self.rng.choice(self.attested)
            key = (spell(sa), spell(sb))
            if sa == sb or key in seen:
                continue
            if self.exclude_qu and any(self._is_qu(x) for x in (sa, sb)):
                continue
            seen.add(key)
            pairs.append(BasePair(self._bp_id("pseudo", *key), sa, sb, "pseudo",
                                  self.freqs.get(key[0], 0), self.freqs.get(key[1], 0)))
        return pairs

    @staticmethod
    def _is_qu(s: Syllable) -> bool:
        return s.onset == "c" and s.glide

    @staticmethod
    def _bp_id(source: str, a: str, b: str) -> str:
        h = hashlib.sha1(f"{source}|{a} {b}".encode("utf-8")).hexdigest()[:10]
        return f"bp-{h}"

    # ------------------------------------------------------------------ T1
    def t1(self, bp: BasePair, variant: str) -> Optional[dict]:
        inp = (bp.a, bp.b)
        if V.is_identity(variant, *inp):
            return None
        out = V.apply(variant, *inp)
        if not all(self.inv.is_legal(s, LEGALITY) for s in out):
            return None
        gold = f"{spell(out[0])} {spell(out[1])}"
        return {
            "task": "T1", "variant": variant, "input": bp.text,
            "input_syllables": [syl_dict(bp.a), syl_dict(bp.b)],
            "gold": [gold], "gold_syllables": [syl_dict(out[0]), syl_dict(out[1])],
            "base_pair_id": bp.base_pair_id, "source": bp.source,
            "strata": strata_for(inp, out, bp, self.lex_pairs, self.freqs),
        }

    # ------------------------------------------------------------------ T2
    def t2(self, bp: BasePair, variant: str) -> Optional[dict]:
        """Decoding: the input is the nói lái form, the gold is every lexical reading."""
        if bp.source != "lexicon":
            return None
        t1 = self.t1(bp, variant)
        if t1 is None:
            return None
        x, y = V.apply(variant, bp.a, bp.b)
        golds = self.lexical_readings(x, y)
        assert any(g["output"] == bp.text for g in golds), "the base pair must be a reading"
        return {
            "task": "T2", "variant": variant, "input": f"{spell(x)} {spell(y)}",
            "input_syllables": [syl_dict(x), syl_dict(y)],
            "gold": golds, "base_pair_id": bp.base_pair_id, "source": bp.source,
            "strata": {**strata_for((x, y), (bp.a, bp.b), bp, self.lex_pairs, self.freqs),
                       "input_lexical": (spell(x), spell(y)) in self.lex_pairs,
                       "n_readings": len(golds)},
        }

    def lexical_readings(self, x: Syllable, y: Syllable) -> list[dict]:
        golds = []
        for v in V.VARIANTS:
            o = V.apply(v, x, y)
            if o == (x, y) or not all(self.inv.is_legal(s, LEGALITY) for s in o):
                continue
            t = (spell(o[0]), spell(o[1]))
            if t in self.lex_pairs:
                golds.append({"variant": v, "reversed": False, "output": f"{t[0]} {t[1]}"})
            if (t[1], t[0]) in self.lex_pairs:
                golds.append({"variant": v, "reversed": True, "output": f"{t[1]} {t[0]}"})
        return golds

    # ------------------------------------------------------------------ T3
    def t3_pair(self, t1_item: dict, twin_type: Optional[str] = None) -> Optional[tuple[dict, dict]]:
        """A yes item (candidate = gold) and a no item (candidate = twin) with the same input."""
        inp = tuple(self._syl(d) for d in t1_item["input_syllables"])
        out = tuple(self._syl(d) for d in t1_item["gold_syllables"])
        variant = t1_item["variant"]
        options = twin_type and [twin_type] or list(TWIN_TYPES)
        self.rng.shuffle(options)
        for tt in options:
            cand = self._twin(tt, variant, inp, out)
            if cand is not None:
                base = {k: t1_item[k] for k in ("variant", "input", "input_syllables", "base_pair_id", "source", "strata")}
                yes = {**base, "task": "T3", "gold": "yes", "candidate": t1_item["gold"][0], "twin_type": None,
                       "correct_output": t1_item["gold"][0]}
                no = {**base, "task": "T3", "gold": "no", "candidate": cand, "twin_type": tt,
                      "correct_output": t1_item["gold"][0]}
                return yes, no
        return None

    @staticmethod
    def _syl(d: dict) -> Syllable:
        return Syllable(onset=d["onset"], glide=d["glide"], nucleus=d["nucleus"], coda=d["coda"], tone=d["tone"])

    def _twin(self, tt: str, variant: str, inp, out) -> Optional[str]:
        gold = f"{spell(out[0])} {spell(out[1])}"
        if tt == "other_variant":
            others = [v for v in V.VARIANTS if v != variant]
            self.rng.shuffle(others)
            for v in others:
                o = V.apply(v, *inp)
                if o == inp or o == out or not all(self.inv.is_legal(s, LEGALITY) for s in o):
                    continue
                return f"{spell(o[0])} {spell(o[1])}"
            return None
        if tt == "spelling":
            return self._misspell(out)
        # component twins: perturb one syllable, keep the result legal and different from gold
        for _ in range(60):
            i = self.rng.randrange(2)
            s = out[i]
            if tt == "onset":
                new = replace(s, onset=self.rng.choice([o for o in CANONICAL_ONSETS if o != s.onset]))
            elif tt == "rime":
                r = self.rng.choice(self.attested)
                new = replace(s, glide=r.glide, nucleus=r.nucleus, coda=r.coda)
            elif tt == "tone":
                new = replace(s, tone=self.rng.choice([t for t in range(6) if t != s.tone]))
            else:
                raise ValueError(tt)
            if new == s or not self.inv.is_legal(new, LEGALITY) or (self.exclude_qu and self._is_qu(new)):
                continue
            cand = list(out)
            cand[i] = new
            text = f"{spell(cand[0])} {spell(cand[1])}"
            if text != gold and text != f"{spell(inp[0])} {spell(inp[1])}":
                return text
        return None

    @staticmethod
    def _misspell(out) -> Optional[str]:
        """Violate one c/k, g/gh or ng/ngh rule in one syllable of the gold."""
        from ..vi.syllable import _onset_spelling
        words = [spell(s) for s in out]
        for i, s in enumerate(out):
            sp = _onset_spelling(s)
            swap = {"k": "c", "c": "k", "gh": "g", "g": "gh", "ngh": "ng", "ng": "ngh"}.get(sp)
            if swap is None or sp == "qu":
                continue
            if s.onset not in ("c", "g", "ng"):
                continue
            w = words[i]
            assert w.startswith(sp)
            bad = swap + w[len(sp):]
            if try_parse(bad, strict=True) is not None:      # would be a valid spelling: not a misspelling
                continue
            cand = list(words)
            cand[i] = bad
            return " ".join(cand)
        return None

    # ------------------------------------------------------------- assembly
    def build(self, n_lexicon: int = 1500, n_pseudo: int = 1000, per_cell_t1: int = 1000,
              per_cell_t2: int = 500, per_cell_t3: int = 500, dev_frac: float = 0.2,
              core_per_cell: int = 125, canary: Optional[str] = None) -> dict:
        canary = canary or f"NOILAI-CANARY-{uuid.UUID(int=self.rng.getrandbits(128))}"
        pairs = self.base_pairs(n_lexicon, n_pseudo)
        # split by base pair
        ids = [p.base_pair_id for p in pairs]
        self.rng.shuffle(ids)
        n_dev = int(round(dev_frac * len(ids)))
        dev_ids = set(ids[:n_dev])
        split_of = {i: ("dev" if i in dev_ids else "test") for i in ids}

        cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for bp in pairs:
            for v in V.VARIANTS:
                it = self.t1(bp, v)
                if it:
                    cells[("T1", v)].append(it)
                it2 = self.t2(bp, v)
                if it2:
                    cells[("T2", v)].append(it2)
        # balance T1 cells by lexicality (half lexicon, half pseudo when possible), then cap
        items: list[dict] = []
        t1_selected: dict[str, list[dict]] = {}
        for v in V.VARIANTS:
            sel = self._balanced_sample(cells[("T1", v)], per_cell_t1, key=lambda it: it["source"])
            t1_selected[v] = sel
            items.extend(sel)
            sel2 = self._balanced_sample(cells[("T2", v)], per_cell_t2, key=lambda it: it["strata"]["input_lexical"])
            items.extend(sel2)
        # T3 from T1 items not used... use T1 items (same base pairs, so clustering still holds)
        for v in V.VARIANTS:
            pool = list(t1_selected[v])
            self.rng.shuffle(pool)
            n_pairs = 0
            for it in pool:
                if n_pairs >= per_cell_t3:
                    break
                tt = TWIN_TYPES[n_pairs % len(TWIN_TYPES)]
                pr = self.t3_pair(it, tt) or self.t3_pair(it)
                if pr is None:
                    continue
                items.extend(pr)
                n_pairs += 1
        # ids, splits, core
        counters: Counter = Counter()
        for it in items:
            it["split"] = split_of[it["base_pair_id"]]
        for it in items:
            k = (it["task"], it["variant"])
            counters[k] += 1
            it["item_id"] = f"{it['task']}-{it['variant']}-{counters[k]:06d}"
        # T3 counterparts
        by_key: dict[tuple, list[dict]] = defaultdict(list)
        for it in items:
            if it["task"] == "T3":
                by_key[(it["variant"], it["input"], it["correct_output"])].append(it)
        for grp in by_key.values():
            if len(grp) == 2:
                grp[0]["pair_item_id"], grp[1]["pair_item_id"] = grp[1]["item_id"], grp[0]["item_id"]
        # core: balanced per (task, variant) cell from test; T3 core keeps yes/no pairs together
        for it in items:
            it["in_core"] = False
        for t in ("T1", "T2"):
            for v in V.VARIANTS:
                cand = [it for it in items if it["task"] == t and it["variant"] == v and it["split"] == "test"]
                for it in self._balanced_sample(cand, core_per_cell, key=lambda it: it["source"]):
                    it["in_core"] = True
        for v in V.VARIANTS:
            yes = [it for it in items if it["task"] == "T3" and it["variant"] == v and it["split"] == "test"
                   and it["gold"] == "yes" and it.get("pair_item_id")]
            self.rng.shuffle(yes)
            by_id = {it["item_id"]: it for it in items}
            for it in yes[: core_per_cell // 2]:
                it["in_core"] = True
                by_id[it["pair_item_id"]]["in_core"] = True
        for it in items:
            if it["split"] == "test":
                it["canary"] = canary
        items.sort(key=lambda it: it["item_id"])
        return {"items": items, "canary": canary, "n_base_pairs": len(pairs),
                "cell_pool_sizes": {f"{k[0]}-{k[1]}": len(v) for k, v in sorted(cells.items())}}

    def _balanced_sample(self, pool: list[dict], n: int, key) -> list[dict]:
        groups: dict = defaultdict(list)
        for it in pool:
            groups[key(it)].append(it)
        for g in groups.values():
            self.rng.shuffle(g)
        out: list[dict] = []
        keys = sorted(groups, key=str)
        i = 0
        while len(out) < n and any(groups[k] for k in keys):
            k = keys[i % len(keys)]
            if groups[k]:
                out.append(groups[k].pop())
            i += 1
        return out


# ------------------------------------------------------------------ I/O
def resource_hashes() -> dict[str, str]:
    out = {}
    for name in ("vi-DauMoi.dic", "vi-DauCu.dic", "Viet74K.txt"):
        p = L.EXTERNAL / name
        if p.exists():
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def write_release(build: dict, out_dir: Path, manifest_extra: Optional[dict] = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = build["items"]
    files = {
        "dev": out_dir / "noilai_dev.jsonl",
        "test": out_dir / "noilai_test.jsonl",
        "core": out_dir / "noilai_core.jsonl",
    }
    for name, path in files.items():
        sel = [it for it in items if (it["in_core"] if name == "core" else it["split"] == name)]
        with open(path, "w", encoding="utf-8") as f:
            for it in sel:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    counts = Counter((it["task"], it["variant"], it["split"]) for it in items)
    manifest = {
        "n_items": len(items),
        "n_base_pairs": build["n_base_pairs"],
        "counts": {f"{t}-{v}-{s}": c for (t, v, s), c in sorted(counts.items())},
        "core_counts": dict(Counter(f"{it['task']}-{it['variant']}" for it in items if it["in_core"])),
        "cell_pool_sizes": build["cell_pool_sizes"],
        "canary": build["canary"],
        "resource_sha256": resource_hashes(),
        "files": {k: str(v.name) for k, v in files.items()},
    }
    manifest.update(manifest_extra or {})
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def load_items(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]
