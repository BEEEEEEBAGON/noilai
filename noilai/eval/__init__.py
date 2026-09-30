"""Evaluation harness: prompt rendering, model backends, answer extraction, scoring, runs.

    prompts.py   YAML templates -> chat messages; few-shot demos; re-encoding arms; prompt hash
    backends.py  Backend interface (generate / logprobs) for HF, vLLM, OpenAI-compatible,
                 Gemini, and the Echo/Scripted test backends; factory from a YAML entry
    extract.py   the "Đáp án:" line
    score.py     T1/T2/T3 scoring, error taxonomy, aggregation, scores.jsonl
    xcopa.py     XCOPA (vi) items, prompt and scoring for the re-encoding arms
    run.py       a run: requests, streaming outputs.jsonl, manifest.json, resume, API guards
"""
