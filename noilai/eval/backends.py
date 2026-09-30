"""Model backends behind one interface.

    Backend.generate(messages_batch, max_new_tokens, greedy) -> [{text, n_prompt_tokens, n_output_tokens}]
    Backend.logprobs(prompt_text, continuations) -> [sum log P(continuation | prompt), ...]
    Backend.chat_to_text(messages) -> the templated prompt string (for logprob prompts)
    Backend.info() -> what the manifest records (backend, version, model id, dtype, quant, flags)

Implementations
    HFBackend          transformers; chat template with enable_thinking=False when the template
                       knows the switch; dtype / bitsandbytes quantization / device options;
                       left-padded batching; seeds; log-probabilities by a forward pass
    VLLMBackend        vLLM (import guarded); greedy SamplingParams; prompt_logprobs for the
                       continuations; dtype=half, max_model_len, gpu_memory_utilization,
                       enforce_eager, tensor_parallel_size, quantization as config
    OpenAICompatBackend  the openai client with base_url (Groq, OpenRouter, a vLLM or llama.cpp
                       server); rate-limit retry with exponential backoff; no log-probabilities
    GeminiBackend      google-genai (import guarded); thinking budget as config; retry
    EchoBackend / ScriptedBackend   for tests: return the prompt / canned answers

`make_backend(entry)` builds one from a model entry of configs/models.yaml (see
`load_models_config` / `get_model_entry`); `llama_cpp` entries are served by a llama.cpp
server through the OpenAI-compatible backend (local, so not an API for the safety guard).
"""
from __future__ import annotations

import contextlib
import importlib
import os
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS_FILE = ROOT / "configs" / "models.yaml"
API_BACKENDS = ("openai_compat", "gemini")
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def _version(module: str) -> str | None:
    try:
        return getattr(importlib.import_module(module), "__version__", None)
    except Exception:  # noqa: BLE001
        return None


def backend_versions() -> dict[str, str | None]:
    return {m: _version(m) for m in ("torch", "transformers", "vllm", "openai", "google.genai", "tokenizers",
                                     "sentencepiece", "accelerate", "bitsandbytes")}


class BackendError(RuntimeError):
    pass


# ------------------------------------------------------------------ interface
class Backend:
    kind: str = "base"
    name: str = "base"
    is_api: bool = False
    supports_logprobs: bool = False
    model_id: str | None = None

    def generate(self, messages_batch: list[list[dict]], max_new_tokens: int = 64, greedy: bool = True) -> list[dict]:
        raise NotImplementedError

    def logprobs(self, prompt: str, continuations: list[str]) -> list[float]:
        raise NotImplementedError(f"{self.kind} backend has no log-probabilities")

    def chat_to_text(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        """Fallback plain rendering; local backends override with the tokenizer's template."""
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)

    def info(self) -> dict:
        return {"backend": self.kind, "backend_version": None, "model_id": self.model_id, "dtype": None,
                "quantization": None, "engine_flags": {}}

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ test backends
class EchoBackend(Backend):
    """Returns the user message verbatim (prompt inspection; canary checks)."""
    kind = "echo"

    def __init__(self, name: str = "echo"):
        self.name = name
        self.model_id = "echo"
        self.calls: list[list[list[dict]]] = []

    def generate(self, messages_batch, max_new_tokens=64, greedy=True):
        self.calls.append(messages_batch)
        out = []
        for msgs in messages_batch:
            text = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
            out.append({"text": text, "n_prompt_tokens": len(text.split()), "n_output_tokens": len(text.split())})
        return out


class ScriptedBackend(Backend):
    """Canned answers for tests.

    `script` may be a callable (user content -> completion or None), a dict {substring:
    completion} (the longest key found in the user content wins), or a list cycled in call
    order. `default` is returned when nothing matches. `logprob_fn(prompt, continuations)`
    enables `logprobs`; `is_api` marks the backend as hosted for the API-safety guard.
    Every call is recorded in `calls` and every prompt in `prompts`.
    """
    kind = "scripted"

    def __init__(self, script=None, default: str = "Đáp án: ?", logprob_fn: Callable | None = None,
                 is_api: bool = False, name: str = "scripted", model_id: str = "scripted"):
        self.script = script
        self.default = default
        self.logprob_fn = logprob_fn
        self.supports_logprobs = logprob_fn is not None
        self.is_api = is_api
        self.name = name
        self.model_id = model_id
        self.calls: list[list[list[dict]]] = []
        self.prompts: list[str] = []
        self.logprob_calls: list[tuple[str, list[str]]] = []
        self._i = 0

    def respond(self, content: str) -> str:
        s = self.script
        if callable(s):
            r = s(content)
            return self.default if r is None else r
        if isinstance(s, dict):
            keys = [k for k in s if k in content]
            if keys:
                return s[max(keys, key=len)]
            return self.default
        if isinstance(s, (list, tuple)) and s:
            r = s[self._i % len(s)]
            self._i += 1
            return r
        return self.default

    def generate(self, messages_batch, max_new_tokens=64, greedy=True):
        self.calls.append(messages_batch)
        out = []
        for msgs in messages_batch:
            content = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
            self.prompts.append(content)
            text = self.respond(content)
            out.append({"text": text, "n_prompt_tokens": len(content.split()), "n_output_tokens": len(text.split()),
                        "latency_s": 0.0})
        return out

    def logprobs(self, prompt, continuations):
        if self.logprob_fn is None:
            raise NotImplementedError("scripted backend without logprob_fn")
        self.logprob_calls.append((prompt, list(continuations)))
        return [float(x) for x in self.logprob_fn(prompt, list(continuations))]


# ------------------------------------------------------------------ Hugging Face
_DTYPES = {"float32": "float32", "fp32": "float32", "float16": "float16", "fp16": "float16", "half": "float16",
           "bfloat16": "bfloat16", "bf16": "bfloat16", "auto": "auto", None: "auto"}


class HFBackend(Backend):
    kind = "hf"
    supports_logprobs = True

    def __init__(self, model_id: str | None = None, revision: str | None = None, dtype: str | None = "auto",
                 quantization: dict | None = None, device: str = "auto", batch_size: int = 8, seed: int = 0,
                 trust_remote_code: bool = False, chat_template_kwargs: dict | None = None,
                 max_model_len: int | None = None, attn_implementation: str | None = None,
                 model=None, tokenizer=None, name: str | None = None):
        import torch

        self.torch = torch
        self.model_id = model_id or getattr(getattr(model, "config", None), "_name_or_path", None) or "in-memory"
        self.name = name or self.model_id
        self.revision = revision
        self.dtype = _DTYPES.get(dtype, dtype)
        self.quantization = quantization
        self.batch_size = batch_size
        self.seed = seed
        self.max_model_len = max_model_len
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.trust_remote_code = trust_remote_code
        self.device_requested = device
        self._seed_all(seed)
        if model is not None and tokenizer is not None:
            self.model, self.tokenizer = model.eval(), tokenizer
        else:
            self.model, self.tokenizer = self._load(attn_implementation)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.device = next(self.model.parameters()).device
        tmpl = getattr(self.tokenizer, "chat_template", None) or ""
        self.supports_enable_thinking = "enable_thinking" in tmpl
        if self.supports_enable_thinking and "enable_thinking" not in self.chat_template_kwargs:
            self.chat_template_kwargs["enable_thinking"] = False

    # ---- setup
    def _seed_all(self, seed: int) -> None:
        random.seed(seed)
        with contextlib.suppress(Exception):      # numpy is optional for the backend itself
            import numpy as np
            np.random.seed(seed % (2 ** 32))
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

    def _torch_dtype(self):
        if self.dtype == "auto":
            return "auto"
        return getattr(self.torch, self.dtype)

    def _quant_config(self):
        q = self.quantization
        if not q:
            return None
        method = str(q.get("method", "")).lower()
        if method in ("bitsandbytes", "bnb", "nf4", "int4", "int8"):
            from transformers import BitsAndBytesConfig

            bits = int(q.get("bits", 4))
            if bits == 4:
                return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=q.get("quant_type", "nf4"),
                                          bnb_4bit_compute_dtype=getattr(self.torch, q.get("compute_dtype", "float16")),
                                          bnb_4bit_use_double_quant=bool(q.get("double_quant", True)))
            return BitsAndBytesConfig(load_in_8bit=True)
        # awq / gptq checkpoints carry their own quantization config; load the checkpoint id
        return None

    def _load(self, attn_implementation):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        q = self.quantization or {}
        load_id = q.get("checkpoint") or self.model_id
        tok = AutoTokenizer.from_pretrained(load_id, revision=self.revision, trust_remote_code=self.trust_remote_code)
        kw = {"revision": self.revision, "trust_remote_code": self.trust_remote_code}
        qc = self._quant_config()
        if qc is not None:
            kw["quantization_config"] = qc
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        use_map = self.device_requested == "auto" and self.torch.cuda.is_available()
        if use_map:
            kw["device_map"] = "auto"
        try:
            model = AutoModelForCausalLM.from_pretrained(load_id, dtype=self._torch_dtype(), **kw)
        except TypeError:  # transformers < 5 spells it torch_dtype
            model = AutoModelForCausalLM.from_pretrained(load_id, torch_dtype=self._torch_dtype(), **kw)
        if not use_map and qc is None:
            dev = self.device_requested
            if dev == "auto":
                dev = "cuda" if self.torch.cuda.is_available() else "cpu"
            model = model.to(dev)
        return model.eval(), tok

    # ---- prompts
    def chat_to_text(self, messages, add_generation_prompt=True):
        tmpl = getattr(self.tokenizer, "chat_template", None)
        if not tmpl:
            return super().chat_to_text(messages, add_generation_prompt)
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                                      **self.chat_template_kwargs)
        except Exception as e:
            if any(m["role"] == "system" for m in messages):
                folded = [{"role": "user", "content": "\n\n".join(m["content"] for m in messages)}]
                return self.tokenizer.apply_chat_template(folded, tokenize=False, add_generation_prompt=add_generation_prompt,
                                                          **self.chat_template_kwargs)
            raise BackendError(f"chat template failed: {e}") from e

    def _add_special(self, text: str) -> bool:
        bos = getattr(self.tokenizer, "bos_token", None)
        return not (bos and text.startswith(bos))

    # ---- generation
    def generate(self, messages_batch, max_new_tokens=64, greedy=True):
        torch = self.torch
        results: list[dict] = []
        for start in range(0, len(messages_batch), self.batch_size):
            chunk = messages_batch[start: start + self.batch_size]
            texts = [self.chat_to_text(m) for m in chunk]
            add_special = self._add_special(texts[0])
            enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=add_special,
                                 truncation=self.max_model_len is not None,
                                 max_length=(self.max_model_len - max_new_tokens) if self.max_model_len else None)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            torch.manual_seed(self.seed)
            gen_kw = {"max_new_tokens": max_new_tokens, "do_sample": not greedy,
                      "pad_token_id": self.tokenizer.pad_token_id}
            if greedy:
                gen_kw.update(temperature=None, top_p=None, top_k=None)
            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kw)
            n_in = enc["input_ids"].shape[1]
            gen = out[:, n_in:]
            eos_ids = set()
            eid = self.model.generation_config.eos_token_id if hasattr(self.model, "generation_config") else None
            for x in (eid if isinstance(eid, (list, tuple)) else [eid]):
                if x is not None:
                    eos_ids.add(int(x))
            if self.tokenizer.eos_token_id is not None:
                eos_ids.add(int(self.tokenizer.eos_token_id))
            pad = self.tokenizer.pad_token_id
            for i in range(len(chunk)):
                ids = gen[i].tolist()
                n_out = 0
                for t in ids:
                    if t in eos_ids or t == pad:
                        break
                    n_out += 1
                text = self.tokenizer.decode(ids[:n_out], skip_special_tokens=True)
                results.append({"text": text, "n_prompt_tokens": int(enc["attention_mask"][i].sum().item()),
                                "n_output_tokens": n_out})
        return results

    # ---- log-probabilities
    def _cont_ids(self, prompt: str, cont: str, add_special: bool) -> tuple[list[int], list[int]]:
        p_ids = self.tokenizer(prompt, add_special_tokens=add_special)["input_ids"]
        f_ids = self.tokenizer(prompt + cont, add_special_tokens=add_special)["input_ids"]
        if f_ids[: len(p_ids)] == p_ids and len(f_ids) > len(p_ids):
            return p_ids, f_ids[len(p_ids):]
        # the boundary merged into one token: tokenize the continuation on its own
        c_ids = self.tokenizer(cont, add_special_tokens=False)["input_ids"]
        return p_ids, c_ids

    def logprobs(self, prompt, continuations):
        torch = self.torch
        add_special = self._add_special(prompt)
        seqs, spans = [], []
        for c in continuations:
            p_ids, c_ids = self._cont_ids(prompt, c, add_special)
            seqs.append(p_ids + c_ids)
            spans.append((len(p_ids), len(p_ids) + len(c_ids)))
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        L = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), L), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):          # right padding: positions stay aligned
            input_ids[i, : len(s)] = torch.tensor(s)
            attn[i, : len(s)] = 1
        input_ids, attn = input_ids.to(self.device), attn.to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attn).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        out = []
        for i, (a, b) in enumerate(spans):
            tot = 0.0
            for pos in range(a, b):
                tot += float(logp[i, pos - 1, input_ids[i, pos]].item())
            out.append(tot)
        return out

    def info(self):
        q = self.quantization
        return {"backend": "hf", "backend_version": _version("transformers"), "torch_version": _version("torch"),
                "model_id": self.model_id, "revision": self.revision,
                "dtype": str(next(self.model.parameters()).dtype).replace("torch.", ""),
                "dtype_requested": self.dtype, "quantization": q, "device": str(self.device),
                "engine_flags": {"batch_size": self.batch_size, "max_model_len": self.max_model_len,
                                 "chat_template_kwargs": self.chat_template_kwargs,
                                 "supports_enable_thinking": self.supports_enable_thinking,
                                 "trust_remote_code": self.trust_remote_code}, "seed": self.seed}


# ------------------------------------------------------------------ vLLM
class VLLMBackend(Backend):
    """vLLM offline engine. Import guarded: constructing it without vllm installed raises
    BackendError. Engine flags are configuration (configs/models.yaml)."""
    kind = "vllm"
    supports_logprobs = True

    def __init__(self, model_id: str, revision: str | None = None, dtype: str = "half",
                 quantization: dict | None = None, max_model_len: int = 2048, gpu_memory_utilization: float = 0.9,
                 enforce_eager: bool = False, tensor_parallel_size: int = 1, seed: int = 0,
                 trust_remote_code: bool = False, chat_template_kwargs: dict | None = None,
                 extra_engine_kwargs: dict | None = None, name: str | None = None):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise BackendError("vllm is not installed; `pip install vllm` on the GPU machine") from e
        self.SamplingParams = SamplingParams
        self.model_id = model_id
        self.name = name or model_id
        self.revision = revision
        self.dtype = {"float16": "half", "fp16": "half", "bfloat16": "bfloat16", "bf16": "bfloat16",
                      "float32": "float32", "fp32": "float32"}.get(dtype, dtype)
        self.quantization = quantization
        self.seed = seed
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        q = quantization or {}
        self.engine_flags = {
            "dtype": self.dtype, "max_model_len": max_model_len, "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager, "tensor_parallel_size": tensor_parallel_size,
            "quantization": q.get("method"), "seed": seed, "trust_remote_code": trust_remote_code,
            **(extra_engine_kwargs or {}),
        }
        load_id = q.get("checkpoint") or model_id
        kw = dict(self.engine_flags)
        kw.pop("seed", None)
        self.llm = LLM(model=load_id, revision=revision, seed=seed, **kw)   # [UNCERTAIN: verify] flag names on the pinned vLLM
        self.tokenizer = self.llm.get_tokenizer()
        tmpl = getattr(self.tokenizer, "chat_template", None) or ""
        self.supports_enable_thinking = "enable_thinking" in tmpl
        if self.supports_enable_thinking and "enable_thinking" not in self.chat_template_kwargs:
            self.chat_template_kwargs["enable_thinking"] = False

    def chat_to_text(self, messages, add_generation_prompt=True):
        if not getattr(self.tokenizer, "chat_template", None):
            return super().chat_to_text(messages, add_generation_prompt)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                                  **self.chat_template_kwargs)

    def generate(self, messages_batch, max_new_tokens=64, greedy=True):
        texts = [self.chat_to_text(m) for m in messages_batch]
        params = self.SamplingParams(temperature=0.0 if greedy else 1.0, max_tokens=max_new_tokens, seed=self.seed)
        outs = self.llm.generate(texts, params, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append({"text": c.text, "n_prompt_tokens": len(o.prompt_token_ids), "n_output_tokens": len(c.token_ids)})
        return res

    def logprobs(self, prompt, continuations):
        # prompt_logprobs=0 returns, for every prompt position, the log-probability of the
        # token actually present (a dict {token_id: Logprob}); the first position is None.
        # [UNCERTAIN: verify] the structure of RequestOutput.prompt_logprobs on the pinned vLLM.
        params = self.SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        p_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        out = []
        for c in continuations:
            full = prompt + c
            f_ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
            start = len(p_ids) if f_ids[: len(p_ids)] == p_ids else len(p_ids) - 1
            ro = self.llm.generate([full], params, use_tqdm=False)[0]
            tot = 0.0
            for pos in range(max(start, 1), len(ro.prompt_token_ids)):
                entry = ro.prompt_logprobs[pos]
                tid = ro.prompt_token_ids[pos]
                lp = entry[tid].logprob if entry and tid in entry else float("nan")
                tot += lp
            out.append(tot)
        return out

    def info(self):
        return {"backend": "vllm", "backend_version": _version("vllm"), "torch_version": _version("torch"),
                "model_id": self.model_id, "revision": self.revision, "dtype": self.dtype,
                "quantization": self.quantization, "engine_flags": {**self.engine_flags,
                                                                    "chat_template_kwargs": self.chat_template_kwargs,
                                                                    "supports_enable_thinking": self.supports_enable_thinking},
                "seed": self.seed}


# ------------------------------------------------------------------ retry helper
def _sleep_for(attempt: int, base: float, cap: float, retry_after: float | None = None) -> float:
    if retry_after is not None and retry_after > 0:
        return min(retry_after, cap)
    return min(cap, base * (2 ** attempt)) * (0.5 + random.random() / 2)


def _retry_after_seconds(exc) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if headers is None:
        return None
    val = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


class _Throttle:
    def __init__(self, requests_per_minute: float | None):
        self.interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self.last = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        now = time.monotonic()
        dt = self.last + self.interval - now
        if dt > 0:
            time.sleep(dt)
        self.last = time.monotonic()


# ------------------------------------------------------------------ OpenAI-compatible
class OpenAICompatBackend(Backend):
    """Groq, OpenRouter, a vLLM or llama.cpp server: anything speaking the chat-completions
    API. Retries 429 and 5xx with exponential backoff (Retry-After honored). No log-probs:
    chat APIs do not return them for arbitrary continuations."""
    kind = "openai_compat"

    def __init__(self, model_id: str, base_url: str | None = None, api_key: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", generation_kwargs: dict | None = None,
                 extra_body: dict | None = None, max_retries: int = 8, base_sleep: float = 2.0,
                 max_sleep: float = 120.0, requests_per_minute: float | None = None, timeout: float = 120.0,
                 seed: int | None = None, client=None, is_api: bool | None = None, name: str | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.model_id = model_id
        self.name = name or model_id
        self.base_url = base_url
        self.generation_kwargs = dict(generation_kwargs or {})
        self.extra_body = dict(extra_body or {})
        self.max_retries = max_retries
        self.base_sleep = base_sleep
        self.max_sleep = max_sleep
        self.seed = seed
        self.throttle = _Throttle(requests_per_minute)
        self.requests_per_minute = requests_per_minute
        self._sleep = sleep
        self.n_retries = 0
        host = urlparse(base_url).hostname if base_url else "api.openai.com"
        self.is_api = (host not in LOCAL_HOSTS) if is_api is None else is_api
        if client is None:
            from openai import OpenAI

            key = api_key or os.environ.get(api_key_env)
            if key is None and self.is_api:
                raise BackendError(f"no API key: set {api_key_env}")
            client = OpenAI(api_key=key or "EMPTY", base_url=base_url, timeout=timeout, max_retries=0)
        self.client = client
        self._token_param = "max_tokens"

    def _retryable(self, exc) -> bool:
        try:
            import openai
        except ImportError:  # pragma: no cover
            return False
        if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                            openai.InternalServerError)):
            return True
        status = getattr(exc, "status_code", None)
        return isinstance(exc, openai.APIStatusError) and status is not None and (status == 429 or status >= 500)

    def _call(self, messages: list[dict], max_new_tokens: int, greedy: bool):
        kw = {"model": self.model_id, "messages": messages, self._token_param: max_new_tokens}
        kw["temperature"] = 0.0 if greedy else self.generation_kwargs.get("temperature", 1.0)
        if self.seed is not None:
            kw["seed"] = self.seed
        extra = {k: v for k, v in self.generation_kwargs.items() if k != "temperature"}
        extra.update(self.extra_body)
        if extra:
            kw["extra_body"] = extra
        return self.client.chat.completions.create(**kw)

    def generate(self, messages_batch, max_new_tokens=64, greedy=True):
        results = []
        for messages in messages_batch:
            attempt = 0
            while True:
                self.throttle.wait()
                t0 = time.monotonic()
                try:
                    resp = self._call(messages, max_new_tokens, greedy)
                    break
                except Exception as e:
                    msg = str(e)
                    if self._token_param == "max_tokens" and "max_completion_tokens" in msg:
                        self._token_param = "max_completion_tokens"
                        continue
                    if not self._retryable(e) or attempt >= self.max_retries:
                        raise
                    self.n_retries += 1
                    self._sleep(_sleep_for(attempt, self.base_sleep, self.max_sleep, _retry_after_seconds(e)))
                    attempt += 1
            latency = time.monotonic() - t0
            choice = resp.choices[0]
            text = choice.message.content or ""
            usage = getattr(resp, "usage", None)
            results.append({"text": text,
                            "n_prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                            "n_output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                            "latency_s": latency, "finish_reason": getattr(choice, "finish_reason", None),
                            "reasoning": getattr(choice.message, "reasoning", None)
                            or getattr(choice.message, "reasoning_content", None)})
        return results

    def info(self):
        return {"backend": "openai_compat", "backend_version": _version("openai"), "model_id": self.model_id,
                "base_url": self.base_url, "dtype": None, "quantization": None,
                "engine_flags": {"generation_kwargs": self.generation_kwargs, "extra_body": self.extra_body,
                                 "requests_per_minute": self.requests_per_minute, "max_retries": self.max_retries},
                "seed": self.seed, "n_retries": self.n_retries}


# ------------------------------------------------------------------ Gemini
class GeminiBackend(Backend):
    """google-genai. `generation_kwargs.thinking_budget` (0 = off) goes to ThinkingConfig;
    retries 429 / 5xx with backoff. No log-probabilities."""
    kind = "gemini"
    is_api = True

    def __init__(self, model_id: str, api_key: str | None = None, api_key_env: str = "GEMINI_API_KEY",
                 generation_kwargs: dict | None = None, max_retries: int = 8, base_sleep: float = 4.0,
                 max_sleep: float = 120.0, requests_per_minute: float | None = None, seed: int | None = None,
                 client=None, name: str | None = None, sleep: Callable[[float], None] = time.sleep):
        self.model_id = model_id
        self.name = name or model_id
        self.generation_kwargs = dict(generation_kwargs or {})
        self.max_retries = max_retries
        self.base_sleep = base_sleep
        self.max_sleep = max_sleep
        self.seed = seed
        self.throttle = _Throttle(requests_per_minute)
        self.requests_per_minute = requests_per_minute
        self._sleep = sleep
        self.n_retries = 0
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise BackendError("google-genai is not installed") from e
        self.types = types
        self.errors = importlib.import_module("google.genai.errors")
        if client is None:
            key = api_key or os.environ.get(api_key_env)
            if key is None:
                raise BackendError(f"no API key: set {api_key_env}")
            client = genai.Client(api_key=key)
        self.client = client

    def _config(self, system: str | None, max_new_tokens: int, greedy: bool):
        t = self.types
        kw = {"temperature": 0.0 if greedy else self.generation_kwargs.get("temperature", 1.0),
              "max_output_tokens": max_new_tokens}
        if system:
            kw["system_instruction"] = system
        if self.seed is not None:
            kw["seed"] = self.seed
        budget = self.generation_kwargs.get("thinking_budget")
        if budget is not None:
            kw["thinking_config"] = t.ThinkingConfig(thinking_budget=int(budget))   # [UNCERTAIN: verify] budget semantics per model
        return t.GenerateContentConfig(**kw)

    def _retryable(self, exc) -> bool:
        if isinstance(exc, self.errors.ServerError):
            return True
        if isinstance(exc, self.errors.ClientError):
            return getattr(exc, "code", None) in (429, 408)
        return False

    def generate(self, messages_batch, max_new_tokens=64, greedy=True):
        results = []
        for messages in messages_batch:
            system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
            contents = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
            attempt = 0
            while True:
                self.throttle.wait()
                t0 = time.monotonic()
                try:
                    resp = self.client.models.generate_content(model=self.model_id, contents=contents,
                                                               config=self._config(system, max_new_tokens, greedy))
                    break
                except Exception as e:
                    if not self._retryable(e) or attempt >= self.max_retries:
                        raise
                    self.n_retries += 1
                    self._sleep(_sleep_for(attempt, self.base_sleep, self.max_sleep))
                    attempt += 1
            latency = time.monotonic() - t0
            try:
                text = resp.text or ""
            except Exception:  # noqa: BLE001  (blocked or empty candidates)
                text = ""
            um = getattr(resp, "usage_metadata", None)
            results.append({"text": text,
                            "n_prompt_tokens": getattr(um, "prompt_token_count", None) if um else None,
                            "n_output_tokens": getattr(um, "candidates_token_count", None) if um else None,
                            "n_thinking_tokens": getattr(um, "thoughts_token_count", None) if um else None,
                            "latency_s": latency})
        return results

    def info(self):
        return {"backend": "gemini", "backend_version": _version("google.genai"), "model_id": self.model_id,
                "dtype": None, "quantization": None,
                "engine_flags": {"generation_kwargs": self.generation_kwargs, "requests_per_minute": self.requests_per_minute,
                                 "max_retries": self.max_retries}, "seed": self.seed, "n_retries": self.n_retries}


# ------------------------------------------------------------------ factory
def load_models_config(path: Path = MODELS_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_entry(cfg: dict, name: str) -> dict:
    """The entry called `name`, merged with `defaults` and its provider's connection fields."""
    entries = cfg.get("models") or []
    match = [e for e in entries if e.get("name") == name]
    if not match:
        raise KeyError(f"no model entry named {name!r}; have {[e.get('name') for e in entries]}")
    entry = dict(match[0])
    for k, v in (cfg.get("defaults") or {}).items():
        entry.setdefault(k, v)
    prov = (cfg.get("providers") or {}).get(entry.get("provider") or "", {})
    for k in ("api_key_env", "base_url"):
        if entry.get(k) is None and prov.get(k) is not None:
            entry[k] = prov[k]
    if entry.get("backend") is None and prov.get("backend"):
        entry["backend"] = prov["backend"]
    hw = (cfg.get("hardware") or {}).get(entry.get("hardware") or "", {})
    entry.setdefault("tensor_parallel_size", hw.get("tensor_parallel_size", 1))
    return entry


def make_backend(entry: dict, backend: str | None = None, **overrides) -> Backend:
    """Build a Backend from a model entry (configs/models.yaml schema). `backend` overrides the
    entry's backend (e.g. the entry's `fallback_backend`); keyword overrides win over the
    entry's fields (device, batch_size, ...)."""
    e = {**entry, **overrides}
    kind = backend or e.get("backend")
    if kind is None:
        raise BackendError(f"model entry {e.get('name')!r} has no backend")
    seed = e.get("seed", 0)
    if kind == "hf":
        return HFBackend(model_id=e.get("hf_id") or e.get("model_id"), revision=e.get("revision"), dtype=e.get("dtype"),
                         quantization=e.get("quantization"), device=e.get("device", "auto"),
                         batch_size=int(e.get("batch_size", 8)), seed=seed,
                         trust_remote_code=bool(e.get("trust_remote_code", False)),
                         chat_template_kwargs=e.get("chat_template_kwargs"), max_model_len=e.get("max_model_len"),
                         attn_implementation=e.get("attn_implementation"), name=e.get("name"))
    if kind == "vllm":
        return VLLMBackend(model_id=e.get("hf_id") or e.get("model_id"), revision=e.get("revision"),
                           dtype=e.get("dtype") or "half", quantization=e.get("quantization"),
                           max_model_len=int(e.get("max_model_len") or 2048),
                           gpu_memory_utilization=float(e.get("gpu_memory_utilization", 0.9)),
                           enforce_eager=bool(e.get("enforce_eager", False)),
                           tensor_parallel_size=int(e.get("tensor_parallel_size", 1)), seed=seed,
                           trust_remote_code=bool(e.get("trust_remote_code", False)),
                           chat_template_kwargs=e.get("chat_template_kwargs"),
                           extra_engine_kwargs=e.get("engine_kwargs"), name=e.get("name"))
    if kind in ("openai_compat", "openai", "llama_cpp"):
        base_url = e.get("base_url")
        if kind == "llama_cpp":
            base_url = e.get("llama_server_url") or base_url or "http://127.0.0.1:8080/v1"
        return OpenAICompatBackend(model_id=e.get("provider_model_id") or e.get("model_id") or e.get("hf_id"),
                                   base_url=base_url, api_key=e.get("api_key"),
                                   api_key_env=e.get("api_key_env") or "OPENAI_API_KEY",
                                   generation_kwargs=e.get("generation_kwargs"), extra_body=e.get("extra_body"),
                                   max_retries=int(e.get("max_retries", 8)),
                                   requests_per_minute=e.get("requests_per_minute"), seed=e.get("api_seed"),
                                   client=e.get("client"), is_api=e.get("is_api"), name=e.get("name"))
    if kind == "gemini":
        return GeminiBackend(model_id=e.get("provider_model_id") or e.get("model_id"), api_key=e.get("api_key"),
                             api_key_env=e.get("api_key_env") or "GEMINI_API_KEY",
                             generation_kwargs=e.get("generation_kwargs"), max_retries=int(e.get("max_retries", 8)),
                             requests_per_minute=e.get("requests_per_minute"), seed=e.get("api_seed"),
                             client=e.get("client"), name=e.get("name"))
    if kind == "echo":
        return EchoBackend(name=e.get("name", "echo"))
    if kind == "scripted":
        return ScriptedBackend(script=e.get("script"), default=e.get("default", "Đáp án: ?"),
                               logprob_fn=e.get("logprob_fn"), is_api=bool(e.get("is_api", False)),
                               name=e.get("name", "scripted"))
    raise BackendError(f"unknown backend {kind!r}")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "model"
