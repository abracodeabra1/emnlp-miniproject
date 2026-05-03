"""
LLM Judge runner. All text judges use cloud APIs (no local HF weights).

  - prometheus  : Groq — openai/gpt-oss-20b
  - judgelm     : Groq — qwen/qwen3-32b
  - llama       : Groq — llama-3.3-70b-versatile (strong general judge)
  - nvidia      : NVIDIA NIM — OpenAI-compatible chat. CLI `--judges nvidia` expands to
                  one result row per model (default: minimax-m2.7 + kimi-k2-thinking).

Requires:
  - GROQ_API_KEY    for prometheus / judgelm / llama (https://console.groq.com/)
  - NVIDIA_API_KEY  for nvidia (https://build.nvidia.com/models)

Optional for Groq (client-side throttling; see https://console.groq.com/docs/rate-limits):
  - GROQ_MAX_REQUESTS_PER_MINUTE   (if unset, per-model RPM defaults slightly under Groq caps)
  - GROQ_MAX_TOKENS_PER_MINUTE     (if unset, per-model TPM defaults slightly under Groq caps)
  Set either to 0 to disable that limiter only.

Optional for nvidia:
  - NVIDIA_JUDGE_MODELS             (comma-separated NIM model ids; overrides defaults)
  - NVIDIA_JUDGE_MODEL              (single model id; used only if NVIDIA_JUDGE_MODELS unset)
  - NVIDIA_MAX_REQUESTS_PER_MINUTE  (default: 30; set 0 to disable client-side throttling)
  - NVIDIA_HTTP_READ_TIMEOUT        (seconds, default: 360 — NIM can hang without a read deadline)
  - NVIDIA_HTTP_CONNECT_TIMEOUT     (seconds, default: 30)
  - NVIDIA_MAX_TOKENS               (default: 1024; raise for long “thinking” outputs)
  - NVIDIA_DEBUG                    (set to 1 to log each HTTP call timing)
  - NVIDIA_API_BASE_URL             (default: https://integrate.api.nvidia.com/v1; no trailing slash)

Groq enforces RPM, RPD, TPM, TPD per model; client-side limits approximate RPM/TPM only.
On 429, the client backs off and honors `retry-after` when present. Keep batches modest
for RPD caps (see account limits on the console).

Usage:
  from src.judge import Judge
  judge = Judge("nvidia/minimaxai/minimax-m2.7")
  score = judge.direct_score(article, summary, dimension="coherence", variant="standard")
  winner = judge.pairwise(article, summary_a, summary_b, variant="cot")
  scores = judge.rubric(article, summary, variant="standard")
"""

import os
import re
import sys
import time
from collections import deque
from typing import Optional, Sequence

from src.prompts import (
    DIMENSIONS,
    build_direct,
    build_pairwise,
    build_rubric,
)

# GroqCloud production model IDs (OpenAI-compatible chat completions).
# Logical names map to Groq production IDs (not the Prometheus-2 / JudgeLM fine-tunes).
GROQ_JUDGE_MODELS = {
    "prometheus": "openai/gpt-oss-20b",
    "judgelm": "qwen/qwen3-32b",
    "llama": "llama-3.3-70b-versatile",
}

# JSONL `judge` keys for the two Groq specialist backbones (figures use API ids below).
FIGURE_GROQ_JUDGE_KEYS: tuple[str, ...] = ("prometheus", "judgelm")


def judge_figure_label(judge_key: str) -> str:
    """Axis / legend text for plots: Groq slots use API model id, not logical keys."""
    return GROQ_JUDGE_MODELS.get(judge_key, judge_key)


# Defaults slightly under https://console.groq.com/docs/rate-limits (Developer table;
# your org may differ — tune with GROQ_MAX_* env vars).
GROQ_MODEL_LIMITS: dict[str, dict[str, int]] = {
    "openai/gpt-oss-20b": {"rpm": 28, "tpm": 7600},  # 30 RPM, 8K TPM
    "qwen/qwen3-32b": {"rpm": 58, "tpm": 5600},  # 60 RPM, 6K TPM
    "llama-3.3-70b-versatile": {"rpm": 28, "tpm": 11000},  # 30 RPM, 12K TPM
}

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
NVIDIA_DEFAULT_BASE = "https://integrate.api.nvidia.com/v1"
# Default when --judges nvidia (matches build.nvidia.com model ids)
NVIDIA_DEFAULT_MODELS: tuple[str, ...] = (
    "minimaxai/minimax-m2.7",
    "moonshotai/kimi-k2-thinking",
)
_NVIDIA_PREFIX = "nvidia/"


class _SlidingWindowRPM:
    """Client-side cap on requests per rolling 60s window (single-threaded experiments)."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max(0, max_per_minute)
        self._times: deque[float] = deque()

    def acquire(self) -> None:
        if self.max_per_minute <= 0:
            return
        while True:
            now = time.monotonic()
            while self._times and self._times[0] < now - 60.0:
                self._times.popleft()
            if len(self._times) < self.max_per_minute:
                self._times.append(now)
                return
            wait = 60.0 - (now - self._times[0]) + 0.05
            time.sleep(max(wait, 0.05))


class _SlidingWindowTPM:
    """Rolling 60s sum of token usage; block until sum + estimate fits under TPM cap."""

    def __init__(self, max_tokens_per_minute: int):
        self.max_tpm = max(0, max_tokens_per_minute)
        self._events: deque[tuple[float, int]] = deque()

    def _purge_and_sum(self, now: float) -> int:
        cutoff = now - 60.0
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        return sum(w for _, w in self._events)

    def acquire(self, estimated_tokens: int) -> None:
        if self.max_tpm <= 0:
            return
        while True:
            now = time.monotonic()
            total = self._purge_and_sum(now)
            if total + estimated_tokens <= self.max_tpm:
                return
            wait = 60.0 - (now - self._events[0][0]) + 0.05
            time.sleep(max(wait, 0.05))

    def record(self, actual_tokens: int) -> None:
        if self.max_tpm <= 0:
            return
        self._events.append((time.monotonic(), max(0, actual_tokens)))


def _groq_default_limits(model_id: str) -> tuple[int, int]:
    row = GROQ_MODEL_LIMITS.get(model_id, {"rpm": 25, "tpm": 4000})
    return int(row["rpm"]), int(row["tpm"])


def _groq_effective_rpm(model_id: str) -> int:
    raw = os.environ.get("GROQ_MAX_REQUESTS_PER_MINUTE", "").strip()
    if raw:
        return max(0, int(raw))
    return _groq_default_limits(model_id)[0]


def _groq_effective_tpm(model_id: str) -> int:
    raw = os.environ.get("GROQ_MAX_TOKENS_PER_MINUTE", "").strip()
    if raw:
        return max(0, int(raw))
    return _groq_default_limits(model_id)[1]


_groq_limiter_pairs: dict[str, tuple[_SlidingWindowRPM, _SlidingWindowTPM]] = {}


def _groq_limiters_for(model_id: str) -> tuple[_SlidingWindowRPM, _SlidingWindowTPM]:
    if model_id not in _groq_limiter_pairs:
        rpm, tpm = _groq_effective_rpm(model_id), _groq_effective_tpm(model_id)
        _groq_limiter_pairs[model_id] = (_SlidingWindowRPM(rpm), _SlidingWindowTPM(tpm))
    return _groq_limiter_pairs[model_id]


def _groq_retry_sleep_seconds(err: Exception, attempt: int) -> float:
    """Prefer server `retry-after` on 429 (Groq sets it when rate-limited)."""
    try:
        from openai import APIStatusError

        if isinstance(err, APIStatusError) and err.response is not None:
            h = err.response.headers.get("retry-after")
            if h is not None:
                try:
                    return min(120.0, float(h))
                except ValueError:
                    pass
    except Exception:
        pass
    return min(60.0, 2.0**attempt)


_nvidia_rpm_limiter: Optional[_SlidingWindowRPM] = None
_nvidia_progress_note_printed = False


def _nvidia_limiter() -> _SlidingWindowRPM:
    global _nvidia_rpm_limiter
    if _nvidia_rpm_limiter is None:
        rpm = int(os.environ.get("NVIDIA_MAX_REQUESTS_PER_MINUTE", "30"))
        _nvidia_rpm_limiter = _SlidingWindowRPM(rpm)
    return _nvidia_rpm_limiter


def nvidia_api_model_ids() -> list[str]:
    """NIM `model` ids for each NVIDIA judge variant (order preserved)."""
    raw = os.environ.get("NVIDIA_JUDGE_MODELS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    single = os.environ.get("NVIDIA_JUDGE_MODEL", "").strip()
    if single:
        return [single]
    return list(NVIDIA_DEFAULT_MODELS)


def nvidia_result_judge_name(api_model_id: str) -> str:
    """Stable `judge` field in JSONL, e.g. nvidia/minimaxai/minimax-m2.7."""
    return f"{_NVIDIA_PREFIX.rstrip('/')}/{api_model_id}"


def expand_judge_names(judges: Sequence[str]) -> list[str]:
    """Turn CLI `nvidia` into one judge label per NIM model."""
    out: list[str] = []
    for j in judges:
        if j == "nvidia":
            for mid in nvidia_api_model_ids():
                out.append(nvidia_result_judge_name(mid))
        else:
            out.append(j)
    return out


def _extract_score_1_5(text: str) -> Optional[int]:
    """Parse a 1–5 integer score from model output."""
    # Look for "Score: N" pattern first
    m = re.search(r"[Ss]core[:\s]+([1-5])", text)
    if m:
        return int(m.group(1))
    # Fallback: last standalone digit 1–5
    digits = re.findall(r"\b([1-5])\b", text)
    if digits:
        return int(digits[-1])
    return None


def _extract_winner(text: str) -> Optional[str]:
    """Parse 'A' or 'B' winner from pairwise output."""
    m = re.search(r"[Ww]inner[:\s]+([AB])", text)
    if m:
        return m.group(1)
    # Last standalone A or B
    tokens = re.findall(r"\b([AB])\b", text)
    if tokens:
        return tokens[-1]
    return None


def _extract_rubric_scores(text: str) -> dict:
    """Parse multi-dimension rubric output into {dim: score} dict."""
    scores = {}
    for dim in DIMENSIONS:
        m = re.search(rf"{dim.capitalize()}[:\s]+([1-5])", text, re.IGNORECASE)
        if m:
            scores[dim] = int(m.group(1))
    return scores


class _GroqJudge:
    """Chat completions on Groq (OpenAI-compatible HTTP API)."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "The 'openai' package is required for Groq judges. "
                    "Install with: pip install openai"
                ) from e
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Create a key at https://console.groq.com/ "
                    "and export it before running judge experiments."
                )
            self._client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
        return self._client

    def generate(self, prompt: str) -> str:
        from openai import APIStatusError, RateLimitError

        client = self._get_client()
        rpm_lim, tpm_lim = _groq_limiters_for(self.model_id)
        max_out = 512
        est_tokens = max(1, len(prompt) // 4 + max_out + 128)

        last_err: Optional[Exception] = None
        for attempt in range(5):
            rpm_lim.acquire()
            tpm_lim.acquire(est_tokens)
            try:
                resp = client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_out,
                )
                msg = resp.choices[0].message
                usage = getattr(resp, "usage", None)
                if usage is not None and getattr(usage, "total_tokens", None) is not None:
                    tpm_lim.record(int(usage.total_tokens))
                else:
                    tpm_lim.record(est_tokens)
                return (msg.content or "").strip()
            except RateLimitError as e:
                last_err = e
            except APIStatusError as e:
                last_err = e
                if e.status_code not in (429, 503):
                    raise
            except Exception as e:
                last_err = e
                err_s = str(e).lower()
                if "429" not in err_s and "503" not in err_s and "rate" not in err_s:
                    raise

            time.sleep(_groq_retry_sleep_seconds(last_err, attempt))
        raise RuntimeError(f"Groq request failed after retries ({self.model_id}): {last_err}")


class _NvidiaJudge:
    """Chat completions on NVIDIA NIM (OpenAI-compatible). https://build.nvidia.com/models"""

    def __init__(self, model_id: str):
        self.model_id = model_id.strip()
        self._base_url = os.environ.get(
            "NVIDIA_API_BASE_URL", NVIDIA_DEFAULT_BASE
        ).strip().rstrip("/")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "The 'openai' package is required for the NVIDIA judge. "
                    "Install with: pip install openai"
                ) from e
            key = os.environ.get("NVIDIA_API_KEY")
            if not key:
                raise RuntimeError(
                    "NVIDIA_API_KEY is not set. Create a key at https://build.nvidia.com/models "
                    "and export it before running judge experiments with --judges nvidia."
                )
            import httpx

            read_s = float(os.environ.get("NVIDIA_HTTP_READ_TIMEOUT", "360"))
            conn_s = float(os.environ.get("NVIDIA_HTTP_CONNECT_TIMEOUT", "30"))
            timeout = httpx.Timeout(
                connect=conn_s, read=read_s, write=conn_s, pool=conn_s
            )
            # We implement our own backoff + RPM limiting; SDK retries would stack badly.
            self._client = OpenAI(
                api_key=key,
                base_url=self._base_url,
                timeout=timeout,
                max_retries=0,
            )
        return self._client

    def generate(self, prompt: str) -> str:
        from openai import APIStatusError, APITimeoutError, RateLimitError

        global _nvidia_progress_note_printed
        if not _nvidia_progress_note_printed:
            _nvidia_progress_note_printed = True
            print(
                "[nvidia] tqdm advances once per article pair; direct mode issues ~48 "
                "HTTP calls per pair (× variants), so 0/50 can sit for many minutes. "
                "NVIDIA_HTTP_READ_TIMEOUT (default 360s) bounds hung requests. "
                "Set NVIDIA_DEBUG=1 to log each call.",
                file=sys.stderr,
                flush=True,
            )

        client = self._get_client()
        limiter = _nvidia_limiter()
        last_err: Optional[Exception] = None
        for attempt in range(8):
            limiter.acquire()
            if os.environ.get("NVIDIA_DEBUG"):
                t_req = time.perf_counter()
                print(
                    f"[nvidia] POST model={self.model_id!r} attempt={attempt + 1} "
                    f"prompt_chars={len(prompt)}",
                    flush=True,
                )
            try:
                resp = client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=int(os.environ.get("NVIDIA_MAX_TOKENS", "1024")),
                )
                msg = resp.choices[0].message
                out = (msg.content or "").strip()
                if os.environ.get("NVIDIA_DEBUG"):
                    print(
                        f"[nvidia] OK model={self.model_id!r} "
                        f"{1000 * (time.perf_counter() - t_req):.0f}ms "
                        f"response_chars={len(out)}",
                        flush=True,
                    )
                return out
            except APITimeoutError as e:
                last_err = e
            except RateLimitError as e:
                last_err = e
            except APIStatusError as e:
                last_err = e
                if e.status_code not in (429, 503):
                    raise
            except Exception as e:
                last_err = e
                err_s = str(e).lower()
                if "429" not in err_s and "503" not in err_s and "rate" not in err_s:
                    if "timeout" in err_s or "timed out" in err_s:
                        pass
                    else:
                        raise
                if os.environ.get("NVIDIA_DEBUG"):
                    print(f"[nvidia] error: {e!r}", flush=True)

            wait = min(120.0, 3.0 * (2 ** attempt))
            time.sleep(wait)
        raise RuntimeError(
            f"NVIDIA NIM request failed after retries ({self.model_id}): {last_err}"
        )


class Judge:
    def __init__(self, judge_name: str):
        """
        judge_name: one of 'prometheus' (Groq openai/gpt-oss-20b), 'judgelm' (Groq qwen/qwen3-32b),
        'llama' (Groq llama-3.3-70b-versatile), or 'nvidia/<NIM model id>' (e.g. nvidia/minimaxai/minimax-m2.7).
        Use expand_judge_names() for CLI `nvidia`.
        """
        self.name = judge_name
        if judge_name.startswith(_NVIDIA_PREFIX):
            api_id = judge_name[len(_NVIDIA_PREFIX) :]
            if not api_id:
                raise ValueError(f"Invalid NVIDIA judge name: {judge_name!r}")
            self._backend = _NvidiaJudge(api_id)
        elif judge_name in GROQ_JUDGE_MODELS:
            self._backend = _GroqJudge(GROQ_JUDGE_MODELS[judge_name])
        else:
            raise ValueError(f"Unknown judge: {judge_name}")

    def _call(self, prompt: str) -> str:
        return self._backend.generate(prompt)

    def direct_score(
        self,
        article: str,
        summary: str,
        dimension: str,
        variant: str = "standard",
        reference: Optional[str] = None,
    ) -> dict:
        prompt = build_direct(article, summary, dimension, variant, reference)
        raw = self._call(prompt)
        score = _extract_score_1_5(raw)
        return {"score": score, "raw": raw, "judge": self.name,
                "dimension": dimension, "variant": variant, "mode": "direct"}

    def direct_score_all_dims(
        self,
        article: str,
        summary: str,
        variant: str = "standard",
        reference: Optional[str] = None,
    ) -> dict:
        results = {}
        for dim in DIMENSIONS:
            results[dim] = self.direct_score(article, summary, dim, variant, reference)
        return results

    def pairwise(
        self,
        article: str,
        summary_a: str,
        summary_b: str,
        variant: str = "standard",
        reference: Optional[str] = None,
    ) -> dict:
        prompt = build_pairwise(article, summary_a, summary_b, variant, reference)
        raw = self._call(prompt)
        winner = _extract_winner(raw)
        return {"winner": winner, "raw": raw, "judge": self.name,
                "variant": variant, "mode": "pairwise"}

    def rubric(
        self,
        article: str,
        summary: str,
        variant: str = "standard",
        reference: Optional[str] = None,
    ) -> dict:
        prompt = build_rubric(article, summary, variant, reference)
        raw = self._call(prompt)
        scores = _extract_rubric_scores(raw)
        return {"scores": scores, "raw": raw, "judge": self.name,
                "variant": variant, "mode": "rubric"}
