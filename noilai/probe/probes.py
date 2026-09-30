"""Layer-wise linear probes with control tasks (Hewitt & Liang 2019; design 9.1–9.2).

For each hidden index, a multinomial logistic regression predicts a syllable feature (tone,
onset, rime, ...) from the residual stream at the probed position. The split is NESTED by
SYLLABLE IDENTITY 60/10/30 (`nested_group_split`, constants.PROBE_SPLIT_FRACS): train,
a dev fold that tunes the L2 strength once at one mid layer (`tune_C`, grid
constants.PROBE_L2_GRID) and a test fold of unseen syllables, so a probe cannot succeed by
memorizing whole syllables. A second split held out by RIME (`holdout_groups`) is reported
beside it. The control task assigns every syllable TYPE a random label drawn from the
empirical label distribution, with its own `control_seed` (design 9.2: 5 split seeds × 3
control-label seeds, paired); selectivity = accuracy(real) − accuracy(control) must exceed
constants.PROBE_SELECTIVITY_MIN (condition (i)). Condition (ii), `excess_over_structural`:
probe accuracy minus the structural baseline (the same probe on one-hot token ids of the
span plus the coda class) with a syllable-clustered bootstrap CI that must exclude 0.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from .. import constants


@dataclass
class LayerResult:
    layer: int
    feature: str
    acc: float
    control_acc: float
    selectivity: float
    majority: float
    n_train: int
    n_test: int
    n_classes: int
    split_seed: int = 0
    control_seed: int = 0
    C: float = 1.0
    split_key: str = "syllable"
    n_dev: int = 0


def group_split(groups: Sequence, test_frac: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two-way syllable-disjoint split (kept for callers that need only train/test)."""
    g = np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, round(test_frac * len(uniq)))
    test_groups = set(uniq[:n_test].tolist())
    test_mask = np.array([x in test_groups for x in g])
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def nested_group_split(groups: Sequence, fracs: Sequence[float] = constants.PROBE_SPLIT_FRACS,
                       seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(train, dev, test) index arrays over shuffled unique groups (design 9.2: 60/10/30 by
    syllable identity; the dev fold tunes the L2 strength). Every fold gets >= 1 group."""
    if len(fracs) != 3 or abs(sum(fracs) - 1.0) > 1e-9:
        raise ValueError("fracs must be three shares summing to 1")
    g = np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = max(1, round(fracs[2] * n))
    n_dev = max(1, round(fracs[1] * n))
    if n_test + n_dev >= n:
        raise ValueError(f"too few groups ({n}) for a nested split")
    test_g = set(uniq[:n_test].tolist())
    dev_g = set(uniq[n_test:n_test + n_dev].tolist())
    fold = np.array([2 if x in test_g else (1 if x in dev_g else 0) for x in g])
    return np.where(fold == 0)[0], np.where(fold == 1)[0], np.where(fold == 2)[0]


def control_labels(groups: Sequence, labels: Sequence, seed: int = 0) -> np.ndarray:
    """Random label per group (syllable type), sampled from the label distribution."""
    g = np.asarray(groups)
    y = np.asarray(labels)
    rng = np.random.default_rng(seed)
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    mapping = {grp: rng.choice(values, p=probs) for grp in np.unique(g)}
    return np.array([mapping[x] for x in g])


def _fit_predict(Xtr, ytr, Xte, C: float = 1.0, max_iter: int = 2000, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(ytr)) < 2:
        return np.full(len(Xte), ytr[0])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter, random_state=seed))
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def tune_C(X: np.ndarray, labels: Sequence, groups: Sequence, seed: int = 0,
           grid: Sequence[float] = constants.PROBE_L2_GRID, fracs: Sequence[float] = constants.PROBE_SPLIT_FRACS) -> dict:
    """Choose the inverse L2 strength once (design 9.2): fit on the train fold, score on the
    dev fold of the nested split at ONE hidden index, freeze the best C (ties -> the
    strongest regularization). Returns {'C', 'dev_acc': {C: acc}, 'split_seed'}."""
    y = np.asarray(labels)
    tr, dev, _ = nested_group_split(groups, fracs, seed)
    scores = {}
    for c in grid:
        pred = _fit_predict(X[tr], y[tr], X[dev], C=c, seed=seed)
        scores[float(c)] = float(np.mean(pred == y[dev]))
    best = min(scores, key=lambda c: (-scores[c], c))
    return {"C": best, "dev_acc": scores, "split_seed": seed, "n_dev": len(dev)}


def run_layer_probes(H: np.ndarray, labels: Sequence, groups: Sequence, feature: str, layers: Sequence[int] | None = None,
                     seed: int = 0, C: float = 1.0, control_seed: int | None = None, holdout_groups: Sequence | None = None,
                     fracs: Sequence[float] = constants.PROBE_SPLIT_FRACS, split_key: str = "syllable") -> list[LayerResult]:
    """H: [n, L+1, d]. One LayerResult per hidden index. `seed` fixes the nested split (test
    fold = unseen groups), `control_seed` the control labels (default: = seed), `holdout_groups`
    an alternative split key (e.g. the rime; design 9.2's second split) while the control task
    stays per syllable type (`groups`). Rows with a NaN representation (a `mark` position with
    no bare-mark token) are dropped before splitting."""
    y = np.asarray(labels)
    g = np.asarray(groups)
    ok = ~np.isnan(H[:, 0, 0])
    if not ok.all():
        H, y, g = H[ok], y[ok], g[ok]
        holdout_groups = np.asarray(holdout_groups)[ok] if holdout_groups is not None else None
    split_on = np.asarray(holdout_groups) if holdout_groups is not None else g
    tr, dev, te = nested_group_split(split_on, fracs, seed)
    cs = seed if control_seed is None else control_seed
    yc = control_labels(g, y, cs)
    majority = float(np.mean(y[te] == _mode(y[tr])))
    results = []
    layers = layers if layers is not None else range(H.shape[1])
    for L in layers:
        X = H[:, L, :]
        pred = _fit_predict(X[tr], y[tr], X[te], C=C, seed=seed)
        acc = float(np.mean(pred == y[te]))
        predc = _fit_predict(X[tr], yc[tr], X[te], C=C, seed=seed)
        acc_c = float(np.mean(predc == yc[te]))
        results.append(LayerResult(layer=int(L), feature=feature, acc=acc, control_acc=acc_c, selectivity=acc - acc_c,
                                   majority=majority, n_train=len(tr), n_test=len(te), n_classes=len(np.unique(y)),
                                   split_seed=seed, control_seed=cs, C=C, split_key=split_key, n_dev=len(dev)))
    return results


def _mode(a: np.ndarray):
    vals, counts = np.unique(a, return_counts=True)
    return vals[counts.argmax()]


def results_table(results: Sequence[LayerResult]):
    import pandas as pd
    return pd.DataFrame([asdict(r) for r in results])


def best_layer(results: Sequence[LayerResult]) -> LayerResult:
    return max(results, key=lambda r: r.selectivity)


def _structural_features(token_ids: Sequence[Sequence[int]], coda_class: Sequence) -> np.ndarray:
    from sklearn.feature_extraction import DictVectorizer

    feats = []
    for ids, coda in zip(token_ids, coda_class):
        d = {f"tok:{t}": 1.0 for t in ids}
        d[f"coda:{coda}"] = 1.0
        d["n_tokens"] = float(len(ids))
        feats.append(d)
    return DictVectorizer(sparse=False).fit_transform(feats)


def structural_baseline(token_ids: Sequence[Sequence[int]], coda_class: Sequence, labels: Sequence, groups: Sequence,
                        seed: int = 0, C: float = 1.0, fracs: Sequence[float] = constants.PROBE_SPLIT_FRACS) -> float:
    """Accuracy of the same probe fitted on one-hot token ids of the syllable's span plus the
    coda class (design 9.1, condition ii): what a probe can achieve from the token identity
    alone, without any representation. Same nested syllable-disjoint split as run_layer_probes."""
    X = _structural_features(token_ids, coda_class)
    y = np.asarray(labels)
    tr, _, te = nested_group_split(groups, fracs, seed)
    pred = _fit_predict(X[tr], y[tr], X[te], C=C, seed=seed)
    return float(np.mean(pred == y[te]))


def excess_over_structural(H_layer: np.ndarray, token_ids, coda_class, labels, groups, seed: int = 0, C: float = 1.0,
                           n_boot: int = constants.BOOTSTRAP_B, fracs: Sequence[float] = constants.PROBE_SPLIT_FRACS,
                           level: float = 0.95) -> dict:
    """Condition (ii) of design 9.1: probe accuracy at one hidden index minus the structural
    baseline on the same test fold, with a SYLLABLE-clustered bootstrap CI (resample the test
    fold's syllable types, recompute acc − baseline on the resampled rows; the two probes are
    fitted once on the train fold). Returns acc, structural_baseline, excess, lo, hi,
    n_test_groups and `excludes_zero`."""
    from ..stats.bootstrap import cluster_bootstrap

    y = np.asarray(labels)
    g = np.asarray(groups)
    tr, _, te = nested_group_split(g, fracs, seed)
    pred = _fit_predict(H_layer[tr], y[tr], H_layer[te], C=C, seed=seed)
    Xs = _structural_features(token_ids, coda_class)
    pred_s = _fit_predict(Xs[tr], y[tr], Xs[te], C=C, seed=seed)
    hit = (pred == y[te]).astype(float)
    hit_s = (pred_s == y[te]).astype(float)
    ci = cluster_bootstrap(hit - hit_s, g[te], n_boot=n_boot, seed=seed, level=level, small_cell_rule=False)
    return {"acc": float(hit.mean()), "structural_baseline": float(hit_s.mean()), "excess": float(ci.estimate),
            "lo": ci.lo, "hi": ci.hi, "n_test": len(te), "n_test_groups": int(ci.n_clusters), "ci_method": ci.method,
            "excludes_zero": bool(ci.lo > 0 or ci.hi < 0)}
