"""Compatibility re-export. The estimands live in `noilai.stats.dose_response` (design 8.6:
Δtokens dose–response labelled associational, and the zero-dose contrast). No "share
mediated by token count" is computed anywhere (design 12.7); this module name is kept only
so that existing imports resolve.
"""
from __future__ import annotations

from .dose_response import MIN_STRATUM, Decomposition, decompose

__all__ = ["MIN_STRATUM", "Decomposition", "decompose"]
