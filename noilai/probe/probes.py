"""Layer-wise linear probes with control tasks (Hewitt & Liang 2019).

For each layer, a multinomial logistic regression predicts a syllable feature (tone,
onset, rime, ...) from the residual stream at the probed position. Train and test are
split by SYLLABLE IDENTITY (GroupShuffleSplit on the syllable string), so a probe cannot
succeed by memorizing whole syllables and must read the feature from the representation
of unseen syllables. The control task assigns every syllable TYPE a random label drawn
from the empirical label distribution (fixed per seed); a probe that fits the control as
well as the real task is expressive enough to memorize, so we report selectivity =
accuracy(real) − accuracy(control) and require it to exceed a pre-registered margin.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np


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


def group_split(groups: Sequence, test_frac: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(test_frac * len(uniq))))
    test_groups = set(uniq[:n_test].tolist())
    test_mask = np.array([x in test_groups for x in g])
    return np.where(~test_mask)[0], np.where(test_mask)[0]


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


def run_layer_probes(H: np.ndarray, labels: Sequence, groups: Sequence, feature: str, layers: Optional[Sequence[int]] = None,
                     test_frac: float = 0.3, seed: int = 0, C: float = 1.0) -> list[LayerResult]:
    """H: [n, L+1, d]. Returns one LayerResult per layer."""
    y = np.asarray(labels)
    g = np.asarray(groups)
    tr, te = group_split(g, test_frac, seed)
    yc = control_labels(g, y, seed)
    majority = float(np.mean(y[te] == np.bincount(np.unique(y[tr], return_inverse=True)[1]).argmax() if False else
                             (y[te] == _mode(y[tr]))))
    results = []
    layers = layers if layers is not None else range(H.shape[1])
    for L in layers:
        X = H[:, L, :]
        pred = _fit_predict(X[tr], y[tr], X[te], C=C, seed=seed)
        acc = float(np.mean(pred == y[te]))
        predc = _fit_predict(X[tr], yc[tr], X[te], C=C, seed=seed)
        acc_c = float(np.mean(predc == yc[te]))
        results.append(LayerResult(layer=int(L), feature=feature, acc=acc, control_acc=acc_c, selectivity=acc - acc_c,
                                   majority=majority, n_train=len(tr), n_test=len(te), n_classes=int(len(np.unique(y)))))
    return results


def _mode(a: np.ndarray):
    vals, counts = np.unique(a, return_counts=True)
    return vals[counts.argmax()]


def results_table(results: Sequence[LayerResult]):
    import pandas as pd
    return pd.DataFrame([asdict(r) for r in results])


def best_layer(results: Sequence[LayerResult]) -> LayerResult:
    return max(results, key=lambda r: r.selectivity)
