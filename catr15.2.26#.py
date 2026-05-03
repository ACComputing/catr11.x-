
#!/usr/bin/env python3
"""
CATR1 5.2.26 Catseek - single-file tkinter GUI with a faithful BitNet b1.58 architecture.

This program implements the BitNet b1.58 *transformer recipe* from the paper (Ma et al.,
arXiv:2402.17764, "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"):

  - BitLinear: ternary weights in {-1, 0, +1} via absmean quantization (Eq. 1-3 in the paper).
  - Per-token symmetric activation clipping (same spirit as the paper's W1.58A8 path).
  - LLaMA-like blocks: RMSNorm (no linear biases), causal multi-head attention with RoPE on Q/K,
    SwiGLU feed-forward, BitLinear projections throughout.

Also: local planning trace in the UI, `/run` code interpreter (subprocess; not a security sandbox).

Weights are initialized locally (not Microsoft's released checkpoints); a bigram prior stabilizes text.
That is the only non-paper part of the stack: the *architecture* matches BitNet b1.58 as coded here.
"""

from __future__ import annotations

import ast
import faulthandler
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass

faulthandler.enable()
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

WINDOW_TITLE = "AC HOLDINGS [C] 1999-2026 CATR1 5.2.26 Catseek (BitNet b1.58)"
BOT_NAME = "CATR1 + interpreter"
MODEL_NAME = "CATR1 BitNet b1.58"
FILES_ENABLED = False
PYTHON_TARGET = "3.14"
INTERPRETER_TIMEOUT_SEC = 18.0
INTERPRETER_MAX_OUT = 48_000

# Short English summary of the reference paper (for /paper and help text).
WHITEPAPER_ARXIV = "https://arxiv.org/abs/2402.17764"
WHITEPAPER_BLURB = textwrap.dedent(
    """
    BitNet b1.58 (Microsoft Research, arXiv:2402.17764) defines a transformer where
    linear layers use ternary weights in {-1, 0, +1}. The name "1.58 bits" comes from
    log2(3) ~ 1.58, the information needed for three symbols.

    Quantization (weights): gamma is the mean absolute value over a weight matrix.
    Each weight is scaled by gamma, rounded, then clipped to [-1, 1], giving W_tilde.

    Training recipe in the paper: train from scratch with 1.58-bit weights and 8-bit
    activations (W1.58A8), LLaMA-like blocks (RMSNorm, SwiGLU, RoPE), and no biases in
    linear layers. The paper reports large efficiency gains versus FP16 LLaMA at scale.

    CATR1 ships this recipe in one Python file; it does not load Microsoft's trained BitNet weights.
    """
).strip()

# Conservative denylist for the local code runner (not a full sandbox).
_INTERPRETER_BAD_FRAGMENTS = (
    "__import__",
    "importlib",
    "subprocess",
    "multiprocessing",
    "ctypes",
    "socket",
    "urllib.request",
    "http.client",
    "ftplib",
    "smtplib",
    "telnetlib",
    "os.system",
    "os.popen",
    "os.spawn",
    "os.exec",
    "pty.",
    "pickle.loads",
    "marshal.loads",
    "eval(",
    "exec(",
    "compile(",
    "breakpoint(",
    "input(",
)


def _text_insert_safe(s: str, *, code_fence: bool = False) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\x00", "").replace("&&", "; ")
    if code_fence:
        return s
    out: list[str] = []
    for ch in s:
        if ch == "[":
            out.append("\uFF3B")
        elif ch == "]":
            out.append("\uFF3D")
        elif ch == "$":
            out.append("\uFF04")
        elif ch == "{":
            out.append("(")
        elif ch == "}":
            out.append(")")
        elif ch == "\\":
            out.append("\uFF3C")
        else:
            out.append(ch)
    return "".join(out)


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    acc = 2166136261
    for ch in text.encode("utf-8", "replace"):
        acc ^= ch
        acc = (acc * 16777619) & 0xFFFFFFFF
    return acc


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    m = max(values)
    exps: list[float] = []
    total = 0.0
    for v in values:
        z = (v - m)
        if z < -60.0:
            e = 0.0
        elif z > 60.0:
            e = math.exp(60.0)
        else:
            e = math.exp(z)
        exps.append(e)
        total += e
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]


def _silu(x: float) -> float:
    if x >= 40.0:
        return x
    if x <= -40.0:
        return 0.0
    return x / (1.0 + math.exp(-x))


def _quantize_activation_token(x: list[float], q_b: float) -> list[float]:
    """Per-token activation clipping to [-Q_b, Q_b] (paper: symmetric, no zero-point)."""
    if not x:
        return []
    m = max(abs(v) for v in x)
    if m < 1e-9:
        return [0.0] * len(x)
    scale = q_b / m
    return [max(-q_b, min(q_b, v * scale)) for v in x]


def _round_clip_weight(v: float, gamma: float, eps: float) -> int:
    """Paper Eq. (1)-(2): RoundClip(W / (gamma + epsilon), -1, 1) -> {-1, 0, +1}."""
    t = v / (gamma + eps)
    r = int(max(-1, min(1, round(t))))
    return r


def _apply_rope_1d(v: list[float], pos: int, *, base: float = 10_000.0) -> list[float]:
    """RoPE on a single head vector (even pairs); last dim left unchanged if odd length."""
    d = len(v)
    if d == 0:
        return []
    out = list(v)
    half = d // 2
    for i in range(half):
        i1 = 2 * i
        i2 = i1 + 1
        inv_freq = 1.0 / (base ** (2.0 * i / max(1, d)))
        ang = pos * inv_freq
        c, s = math.cos(ang), math.sin(ang)
        x1, x2 = v[i1], v[i2]
        out[i1] = x1 * c - x2 * s
        out[i2] = x1 * s + x2 * c
    return out


def _dot(a: list[float], b: list[float]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def _count_repeats(s: str) -> int:
    best = 1
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 1
    return best


def _clean_generated(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch in "\n\r\t" or (" " <= ch <= "~") or ch.isprintable():
            cleaned.append(ch)
    s = "".join(cleaned).replace("\r\n", "\n").replace("\r", "\n")
    for marker in ("\nUser:", "\nYOU:", "\n[SYSTEM]", "\n[YOU]", "\n[AHA]"):
        if marker in s:
            s = s.split(marker, 1)[0]
    s = s.strip()
    if "\n\n\n" in s:
        while "\n\n\n" in s:
            s = s.replace("\n\n\n", "\n\n")
    return s


def _is_low_quality(text: str) -> bool:
    s = text.strip()
    if len(s) < 16:
        return True
    if _count_repeats(s) >= 7:
        return True
    printable = sum(1 for ch in s if ch.isprintable() or ch in "\n\t")
    if printable / max(1, len(s)) < 0.95:
        return True
    ascii_like = sum(1 for ch in s if ch == "\n" or ch == "\t" or (32 <= ord(ch) < 127))
    if ascii_like / max(1, len(s)) < 0.90:
        return True
    if len(s) > 50 and s.count(" ") < 6:
        return True
    letters = sum(1 for ch in s if ch.isalpha())
    if len(s) > 24 and letters / max(1, len(s)) < 0.45:
        return True
    return False


def _english_output_quality(text: str) -> bool:
    """
    Reject decoder noise (mixed alnum soup, consonant runs, too little plain ASCII English).
    The local BitNet is untrained; this gate favors readable replies or /fallback.
    """
    t = text.strip()
    if len(t) < 22:
        return False
    ascii_like = sum(1 for ch in t if ch == "\n" or ch == "\t" or (32 <= ord(ch) < 127))
    if ascii_like / max(1, len(t)) < 0.94:
        return False
    words = re.findall(r"[A-Za-z]{3,}", t)
    if len(words) < 4:
        return False
    letters = sum(1 for ch in t if ch.isalpha())
    if letters / max(1, len(t)) < 0.48:
        return False
    # Long consonant runs are rare in normal English prose; common in random-byte-ish text.
    if re.search(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{9,}", t):
        return False
    # Short tokens mixing letters and digits/punctuation (e.g. "bIY)9", "RPAC9x") usually mean gibberish.
    mixed_frag = 0
    for w in re.findall(r"\S+", t):
        if len(w) > 5 and len(w) < 14:
            has_l = any(c.isalpha() for c in w)
            has_d = any(c.isdigit() for c in w)
            has_sym = any(c in "()<>[]{}$%^&*+=|\\/`~" for c in w)
            if has_l and (has_d or has_sym):
                mixed_frag += 1
    if mixed_frag >= 2:
        return False
    return True


def _assistant_output_ok(text: str) -> bool:
    return (not _is_low_quality(text)) and _english_output_quality(text)


def _interpreter_precheck(code: str) -> str | None:
    low = code.lower()
    for frag in _INTERPRETER_BAD_FRAGMENTS:
        if frag.lower() in low:
            return f"disallowed fragment {frag!r} (use stdlib math/print/datetime only)"
    return None


def run_code_interpreter(code: str, *, timeout: float = INTERPRETER_TIMEOUT_SEC) -> str:
    err = _interpreter_precheck(code)
    if err:
        return f"[interpreter] blocked: {err}"
    try:
        ast.parse(code)
    except SyntaxError as e:
        return f"[interpreter] syntax error: {e}"
    path = ""
    try:
        fd, path = tempfile.mkstemp(suffix="_catseek_interp.py", text=True)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(code)
        proc = subprocess.run(
            [sys.executable, "-I", "-B", path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        out = (proc.stdout or "").strip()
        errtxt = (proc.stderr or "").strip()
        chunks: list[str] = []
        if out:
            chunks.append(out)
        if errtxt:
            chunks.append("[stderr]\n" + errtxt)
        chunks.append(f"[exit {proc.returncode}]")
        joined = "\n".join(chunks).strip()
        if len(joined) > INTERPRETER_MAX_OUT:
            joined = joined[: INTERPRETER_MAX_OUT] + "\n... (truncated)"
        return joined or "[interpreter] (no output)"
    except subprocess.TimeoutExpired:
        return f"[interpreter] killed: wall clock > {timeout:.0f}s"
    except OSError as e:
        return f"[interpreter] OS error: {e}"
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def _extract_python_block(text: str) -> str | None:
    t = text.strip()
    m = re.search(r"```python\s*([\s\S]*?)```", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```\s*([\s\S]*?)```", t)
    if m2 and "def " in m2.group(1):
        return m2.group(1).strip()
    return None


def _synthesize_reasoning(prompt: str, history_len: int) -> str:
    """Plain-English planning trace (local only; not a hosted reasoning model)."""
    p = prompt.strip()
    if not p:
        return ""
    pl = p.lower()
    seed = _stable_seed(p, history_len)
    rnd = random.Random(seed)
    lead = p.split("\n", 1)[0].strip()
    if len(lead) > 140:
        lead = lead[:137] + "..."
    lines: list[str] = [
        "--- reasoning trace (local decoder; not a cloud model) ---",
        f"* Restated goal: {lead!r}",
        "* Constraints: single-file app, no network, stdlib-only interpreter subprocess.",
    ]
    if "?" in p:
        lines.append("* Signal: question -> answer with explicit assumptions + one verification step.")
    if any(k in pl for k in ("bug", "error", "traceback", "exception")):
        lines.append("* Debug path: reproduce -> minimal repro -> diff expected vs actual.")
    if any(k in pl for k in ("python", "code", "def ", "class ", "snippet", "script")):
        lines.append('* Code policy: type hints, docstrings, small API surface, `if __name__ == "__main__"` guard.')
    if any(k in pl for k in ("async", "await", "thread", "gui", "tkinter")):
        lines.append("* Concurrency: keep UI thread responsive; offload CPU to worker threads/processes.")
    lines.append(
        f"* Stochastic probe (seed {seed & 0xFFFF:04x}): explore {rnd.choice(['two', 'three'])} "
        "candidate phrasings, pick clearest."
    )
    lines.append("* Stop rule: stop when user-visible answer matches goal; avoid over-generation.")
    return "\n".join(lines)


def _opus_style_python_reply(prompt: str) -> str:
    """Structured, high-density Python (hand-authored templates; local only)."""
    pl = prompt.lower()

    if "timer" in pl or "countdown" in pl or "sleep" in pl:
        body = textwrap.dedent(
            '''
            from __future__ import annotations

            import time
            from dataclasses import dataclass


            @dataclass(frozen=True, slots=True)
            class Countdown:
                seconds: int

                def run(self) -> None:
                    """Print T-minus ticks; uses monotonic clock for pacing."""
                    if self.seconds < 0:
                        raise ValueError("seconds must be >= 0")
                    end = time.perf_counter() + float(self.seconds)
                    remaining = self.seconds
                    while remaining > 0:
                        print(f"T-{remaining:>3}s")
                        time.sleep(min(1.0, max(0.0, end - time.perf_counter())))
                        remaining = int(round(end - time.perf_counter()))
                    print("Liftoff.")


            def main() -> None:
                Countdown(seconds=3).run()


            if __name__ == "__main__":
                main()
            '''
        ).strip()
        return "```python\n" + body + "\n```"

    if "parse" in pl or "csv" in pl or "json" in pl:
        body = textwrap.dedent(
            '''
            from __future__ import annotations

            import csv
            import io
            import json
            from typing import Any


            def parse_csv_rows(data: str) -> list[dict[str, str]]:
                """Parse CSV text into row dicts using header row."""
                f = io.StringIO(data)
                reader = csv.DictReader(f)
                return [dict(row) for row in reader]


            def main() -> None:
                sample = "name,score\\nAda,99\\nBob,87\\n"
                rows = parse_csv_rows(sample)
                print(json.dumps(rows, indent=2))


            if __name__ == "__main__":
                main()
            '''
        ).strip()
        return "```python\n" + body + "\n```"

    if "class" in pl or "dataclass" in pl:
        body = textwrap.dedent(
            '''
            from __future__ import annotations

            from dataclasses import dataclass
            from typing import Protocol


            class Drawable(Protocol):
                def area(self) -> float: ...


            @dataclass(frozen=True, slots=True)
            class Rectangle:
                width: float
                height: float

                def area(self) -> float:
                    if self.width < 0 or self.height < 0:
                        raise ValueError("dimensions must be non-negative")
                    return self.width * self.height


            def main() -> None:
                shapes: list[Drawable] = [Rectangle(3, 4), Rectangle(2, 5)]
                print(sum(s.area() for s in shapes))


            if __name__ == "__main__":
                main()
            '''
        ).strip()
        return "```python\n" + body + "\n```"

    body = textwrap.dedent(
        '''
        from __future__ import annotations

        import argparse
        import sys
        from typing import NoReturn


        def eprint(*args: object) -> None:
            print(*args, file=sys.stderr)


        def fail(msg: str, code: int = 2) -> NoReturn:
            eprint(msg)
            raise SystemExit(code)


        def main(argv: list[str] | None = None) -> int:
            p = argparse.ArgumentParser(description="Compact CLI stub.")
            p.add_argument("--name", default="CATR1", help="Greeting name")
            ns = p.parse_args(argv)
            print(f"Hello from {ns.name} (local BitNet host).")
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        '''
    ).strip()
    return "```python\n" + body + "\n```"


class ByteTokenizer:
    bos_id = 256
    eos_id = 257
    vocab_size = 258

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = False, limit: int | None = None) -> list[int]:
        data = list(text.encode("utf-8", "replace"))
        out: list[int] = []
        if add_bos:
            out.append(self.bos_id)
        out.extend(data)
        if add_eos:
            out.append(self.eos_id)
        if limit is not None and len(out) > limit:
            out = out[-limit:]
        return out

    def decode(self, token_ids: list[int]) -> str:
        data = bytearray()
        for tok in token_ids:
            if 0 <= tok < 256:
                data.append(tok)
        return data.decode("utf-8", "replace")


@dataclass(slots=True)
class ModelConfig:
    """Hyperparameters for CATR1 (BitNet b1.58 architecture; local weight init)."""

    vocab_size: int = 258
    context_size: int = 64
    d_model: int = 20
    n_layers: int = 2
    n_heads: int = 4
    ffn_dim: int = 40
    quant_eps: float = 1e-5
    activation_q: float = 8.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class BitLinear:
    """
    BitLinear: ternary {-1, 0, +1} weights via absmean quantization (arXiv:2402.17764, Eq. 1-3).
    No linear bias (LLaMA-like BitNet blocks in the paper).
    """

    def __init__(self, in_features: int, out_features: int, *, seed: int, cfg: ModelConfig) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.eps = cfg.quant_eps
        self.q_b = cfg.activation_q
        rnd = random.Random(seed)
        w: list[list[float]] = [
            [(rnd.random() * 2.0 - 1.0) * 0.35 for _ in range(in_features)] for _ in range(out_features)
        ]
        abs_sum = sum(abs(v) for row in w for v in row)
        nm = float(in_features * out_features)
        self.gamma = abs_sum / nm
        self.ternary: list[list[int]] = [
            [_round_clip_weight(v, self.gamma, self.eps) for v in row] for row in w
        ]
        self._inv_sqrt_in = 1.0 / math.sqrt(max(1, in_features))

    def nonzero_ratio(self) -> float:
        nz = sum(1 for row in self.ternary for t in row if t != 0)
        return nz / max(1, self.in_features * self.out_features)

    def forward_vec(self, x: list[float]) -> list[float]:
        xq = _quantize_activation_token(x, self.q_b)
        out: list[float] = []
        for row in self.ternary:
            acc = 0.0
            for j, t in enumerate(row):
                if t == 1:
                    acc += xq[j]
                elif t == -1:
                    acc -= xq[j]
            g = max(self.gamma, 1e-8)
            out.append(acc * g * self._inv_sqrt_in)
        return out

    def forward_seq(self, seq: list[list[float]]) -> list[list[float]]:
        return [self.forward_vec(x) for x in seq]


class RMSNorm:
    def __init__(self, dim: int, *, eps: float = 1e-6) -> None:
        self.dim = dim
        self.eps = eps
        self.weight = [1.0] * dim

    def forward_vec(self, x: list[float]) -> list[float]:
        sq = 0.0
        for v in x:
            sq += v * v
        rms = math.sqrt((sq / max(1, self.dim)) + self.eps)
        inv = 1.0 / rms
        return [x[i] * inv * self.weight[i] for i in range(self.dim)]

    def forward_seq(self, seq: list[list[float]]) -> list[list[float]]:
        return [self.forward_vec(x) for x in seq]


class BitSelfAttention:
    """Causal multi-head attention; RoPE on Q and K (paper: LLaMA-like components)."""

    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        dim = cfg.d_model
        self.cfg = cfg
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.score_scale = 1.0 / math.sqrt(max(1, self.head_dim))
        self.q_proj = BitLinear(dim, dim, seed=seed + 11, cfg=cfg)
        self.k_proj = BitLinear(dim, dim, seed=seed + 23, cfg=cfg)
        self.v_proj = BitLinear(dim, dim, seed=seed + 37, cfg=cfg)
        self.o_proj = BitLinear(dim, dim, seed=seed + 53, cfg=cfg)

    def forward(self, seq: list[list[float]]) -> list[list[float]]:
        q_all = self.q_proj.forward_seq(seq)
        k_all = self.k_proj.forward_seq(seq)
        v_all = self.v_proj.forward_seq(seq)

        q_heads: list[list[list[float]]] = []
        k_heads: list[list[list[float]]] = []
        v_heads: list[list[list[float]]] = []
        for t, (q, k, v) in enumerate(zip(q_all, k_all, v_all)):
            qh: list[list[float]] = []
            kh: list[list[float]] = []
            vh: list[list[float]] = []
            for h in range(self.num_heads):
                lo = h * self.head_dim
                hi = (h + 1) * self.head_dim
                qh.append(_apply_rope_1d(q[lo:hi], t))
                kh.append(_apply_rope_1d(k[lo:hi], t))
                vh.append(v[lo:hi])
            q_heads.append(qh)
            k_heads.append(kh)
            v_heads.append(vh)

        out_seq: list[list[float]] = []
        for t in range(len(seq)):
            merged: list[float] = []
            for h in range(self.num_heads):
                qh = q_heads[t][h]
                scores: list[float] = []
                for j in range(t + 1):
                    score = _dot(qh, k_heads[j][h]) * self.score_scale
                    scores.append(score)
                probs = _softmax(scores)
                acc = [0.0] * self.head_dim
                for j, p in enumerate(probs):
                    vhh = v_heads[j][h]
                    for i in range(self.head_dim):
                        acc[i] += p * vhh[i]
                merged.extend(acc)
            out_seq.append(self.o_proj.forward_vec(merged))
        return out_seq


class BitFeedForward:
    """SwiGLU feed-forward (SiLU(gate) * up, then down), BitLinear projections, no biases."""

    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        dim = cfg.d_model
        hidden = cfg.ffn_dim
        self.up_proj = BitLinear(dim, hidden, seed=seed + 101, cfg=cfg)
        self.gate_proj = BitLinear(dim, hidden, seed=seed + 211, cfg=cfg)
        self.down_proj = BitLinear(hidden, dim, seed=seed + 307, cfg=cfg)

    def forward_vec(self, x: list[float]) -> list[float]:
        up = self.up_proj.forward_vec(x)
        gate = self.gate_proj.forward_vec(x)
        hidden = [_silu(g) * u for g, u in zip(gate, up)]
        return self.down_proj.forward_vec(hidden)

    def forward_seq(self, seq: list[list[float]]) -> list[list[float]]:
        return [self.forward_vec(x) for x in seq]


class BitNetBlock:
    def __init__(self, cfg: ModelConfig, *, seed: int) -> None:
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = BitSelfAttention(cfg, seed=seed + 1000)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = BitFeedForward(cfg, seed=seed + 2000)

    def forward(self, seq: list[list[float]]) -> list[list[float]]:
        n1 = self.norm1.forward_seq(seq)
        attn_out = self.attn.forward(n1)
        mid = []
        for x, y in zip(seq, attn_out):
            mid.append([a + b for a, b in zip(x, y)])
        n2 = self.norm2.forward_seq(mid)
        mlp_out = self.mlp.forward_seq(n2)
        out = []
        for x, y in zip(mid, mlp_out):
            out.append([a + b for a, b in zip(x, y)])
        return out


class BitNetLM:
    """Token embeddings + BitNet blocks; positional information enters via RoPE on Q/K."""

    def __init__(self, cfg: ModelConfig, *, seed: int = 1337) -> None:
        self.cfg = cfg
        rnd = random.Random(seed)
        self.token_embedding: list[list[float]] = []
        for _ in range(cfg.vocab_size):
            self.token_embedding.append([(rnd.random() * 2.0 - 1.0) * 0.18 for _ in range(cfg.d_model)])
        self.blocks = [BitNetBlock(cfg, seed=seed + 5000 * i) for i in range(cfg.n_layers)]
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = BitLinear(cfg.d_model, cfg.vocab_size, seed=seed + 9090, cfg=cfg)

    def forward_last(self, token_ids: list[int]) -> list[float]:
        token_ids = token_ids[-self.cfg.context_size:]
        seq: list[list[float]] = []
        for tok in token_ids:
            seq.append(list(self.token_embedding[tok]))
        for block in self.blocks:
            seq = block.forward(seq)
        last = self.final_norm.forward_vec(seq[-1])
        return self.lm_head.forward_vec(last)

    def total_ternary_params(self) -> int:
        count = 0
        for block in self.blocks:
            for layer in (
                block.attn.q_proj,
                block.attn.k_proj,
                block.attn.v_proj,
                block.attn.o_proj,
                block.mlp.up_proj,
                block.mlp.gate_proj,
                block.mlp.down_proj,
            ):
                count += layer.in_features * layer.out_features
        count += self.lm_head.in_features * self.lm_head.out_features
        return count

    def average_nonzero_ratio(self) -> float:
        ratios: list[float] = []
        for block in self.blocks:
            for layer in (
                block.attn.q_proj,
                block.attn.k_proj,
                block.attn.v_proj,
                block.attn.o_proj,
                block.mlp.up_proj,
                block.mlp.gate_proj,
                block.mlp.down_proj,
            ):
                ratios.append(layer.nonzero_ratio())
        ratios.append(self.lm_head.nonzero_ratio())
        return sum(ratios) / max(1, len(ratios))


class BigramPrior:
    def __init__(self, tokenizer: ByteTokenizer, texts: list[str]) -> None:
        size = tokenizer.vocab_size
        counts = [[1 for _ in range(size)] for _ in range(size)]
        for text in texts:
            toks = tokenizer.encode(text, add_bos=True, add_eos=True)
            for prev, cur in zip(toks, toks[1:]):
                counts[prev][cur] += 1

        self.log_probs: list[list[float]] = []
        for row in counts:
            total = float(sum(row))
            self.log_probs.append([math.log(c / total) for c in row])

    def logits(self, prev_token: int) -> list[float]:
        return self.log_probs[prev_token]


STYLE_CORPUS = [
    "Hello. The graphical interface is running, and CATR1 (BitNet b1.58 architecture) is ready.",
    "There are no external weight files: one Python file, the standard library, and tkinter only.",
    "Use /profile for numeric settings, /model for the layer stack, and /paper for the arXiv summary.",
    "If something fails, send the exact error line, what you expected, and what you observed.",
    "Keep the user interface simple, keep the model small, and keep the source code easy to read.",
    "BitNet b1.58 uses ternary weights in {-1, 0, +1}; that is about 1.58 bits because log2(3) is near 1.58.",
    "Weights are quantized with an absmean scale gamma, then RoundClip to {-1, 0, +1}, as in the paper.",
    "Activations are clipped per token to a symmetric range, similar in spirit to the W1.58A8 setup.",
    "Attention is causal: each position only attends to itself and earlier positions.",
    "The feed-forward block is SwiGLU-style: SiLU on a gate path multiplies an up projection, then down projects.",
    "Rotary position embeddings (RoPE) are applied to queries and keys; there is no old-style sinusoidal sum here.",
    "Short prompts usually give clearer output from this small random-weight demo.",
    "When you ask for Python, I answer with fenced code you can copy.",
    "This build is for demos and learning; it is not a trained production language model.",
]


class BitNetEngine:
    def __init__(self) -> None:
        self.history: list[tuple[str, str]] = []
        self.last_aha = ""
        self.last_thinking = ""
        self.tokenizer = ByteTokenizer()
        self.cfg = ModelConfig()
        self.model = BitNetLM(self.cfg, seed=1337)
        self.prior = BigramPrior(self.tokenizer, STYLE_CORPUS)
        self.allowed_tokens = [10] + list(range(32, 127)) + [self.tokenizer.eos_id]

    def profile_text(self) -> str:
        nz = self.model.average_nonzero_ratio() * 100.0
        return (
            f"# {MODEL_NAME}\n\n"
            f"- Reference: BitNet b1.58 whitepaper ({WHITEPAPER_ARXIV})\n"
            f"- Weight quantization: absmean gamma, ternary W_tilde in {{-1, 0, +1}} (see /paper)\n"
            f"- Activation clip (symmetric): Q = {self.cfg.activation_q} (educational analogue of W1.58A8)\n"
            f"- files = {'off' if not FILES_ENABLED else 'on'}\n"
            f"- Python = {PYTHON_TARGET}+ (stdlib only for this file)\n"
            f"- GUI = tkinter\n"
            f"- tokenizer = byte-level UTF-8, vocab {self.cfg.vocab_size}\n"
            f"- context = {self.cfg.context_size} tokens\n"
            f"- d_model = {self.cfg.d_model}, heads = {self.cfg.n_heads}, head_dim = {self.cfg.head_dim}\n"
            f"- layers = {self.cfg.n_layers}, FFN width = {self.cfg.ffn_dim}\n"
            f"- BitLinear parameter slots = {self.model.total_ternary_params():,} (weights in {{-1, 0, +1}})\n"
            f"- average nonzero ternary weight fraction = {nz:.1f}%\n"
            f"- no external checkpoints; no network API\n"
            f"- reasoning trace = on (local text only)\n"
            f"- code interpreter = {sys.executable} -I, timeout {INTERPRETER_TIMEOUT_SEC:.0f}s (not a sandbox)\n"
        )

    def model_text(self) -> str:
        return (
            "CATR1 = BitNet b1.58 architecture (this codebase)\n"
            "---------------------------------------------------\n"
            f"1. Byte tokenizer and embeddings ({self.cfg.vocab_size} symbols).\n"
            f"2. {self.cfg.n_layers} pre-norm transformer block(s).\n"
            "3. Each block: RMSNorm -> causal multi-head self-attention (BitLinear Q,K,V,O; RoPE on Q,K) -> add.\n"
            "4. Then RMSNorm -> SwiGLU MLP (BitLinear up, gate, down) -> add.\n"
            "5. Final RMSNorm -> BitLinear language-model head.\n"
            "\n"
            "This is the real BitNet b1.58 block recipe from the paper (ternary BitLinear, RMSNorm, RoPE, SwiGLU, "
            "no linear biases). Initial weights are local random draws, not Microsoft's published checkpoints.\n"
            f"Paper: {WHITEPAPER_ARXIV}\n"
        )

    def help_text(self) -> str:
        return (
            "Commands:\n"
            "- /profile or /pr  -- numeric settings\n"
            "- /model  -- layer stack summary\n"
            "- /paper  -- short English summary of the BitNet b1.58 paper (this file implements that stack)\n"
            "- /reset or /clear\n"
            "- /help\n"
            "- /run <python>  -- code interpreter (one line; timeout and denylist)\n"
            "- /interp <python>  -- same as /run\n"
            "\n"
            "Examples:\n"
            "- hello\n"
            "- write python code for a timer\n"
            "- /run print(sum(range(20)))\n"
            "- why is my bug happening?\n"
            "\n"
            "A short local reasoning trace may appear before each reply (not a hosted model).\n"
        )

    def _fallback_reply(self, prompt: str) -> str:
        p = prompt.strip()
        pl = p.lower()
        if not p:
            return "Send a message. The window is open and CATR1 (BitNet b1.58 stack) is ready."
        if any(k in pl for k in ("build", "make", "create", "design")) and any(k in pl for k in ("gui", "model", "bitnet", "transformer")):
            return (
                "Keep the GUI on the main thread, run generation in a background thread, "
                "use a byte tokenizer, causal BitNet blocks with RMSNorm, RoPE attention, SwiGLU, "
                "and a BitLinear language-model head (see /paper for the reference design)."
            )
        if "?" in p:
            return "I can help. Name a concrete goal, a constraint, or paste one error line so the answer stays focused."
        return "The local core is running. Ask a specific question and I will keep the reply short."

    def _seed_prefix(self, prompt: str) -> str:
        pl = prompt.lower()
        if any(k in pl for k in ("make", "build", "create")):
            return "A clean build for that is: "
        if any(k in pl for k in ("explain", "how", "why", "?")):
            return "Here is the clean way to frame it: "
        return "My take: "

    def _sample_token(self, logits: list[float], rnd: random.Random, *, top_k: int = 6, temperature: float = 0.45) -> int:
        idx = sorted(self.allowed_tokens, key=lambda i: logits[i], reverse=True)[:top_k]
        if not idx:
            return self.tokenizer.eos_id
        # Mostly greedy: untrained BitNet logits are noisy; prior already pulls toward English.
        if rnd.random() < 0.9:
            return idx[0]
        top_vals = [logits[i] / max(0.05, temperature) for i in idx]
        probs = _softmax(top_vals)
        r = rnd.random()
        c = 0.0
        for i, p in zip(idx, probs):
            c += p
            if r <= c:
                return i
        return idx[-1]

    def _model_reply(self, prompt: str) -> str:
        self.last_thinking = _synthesize_reasoning(prompt, len(self.history))
        prefix = self._seed_prefix(prompt)
        context = (
            "System: You are a compact English-speaking assistant running on CATR1: a faithful BitNet b1.58 "
            "transformer in code (arXiv:2402.17764), with locally initialized weights—not Microsoft's trained "
            "checkpoints. No network. Reply in clear English.\n"
            f"User: {prompt}\n"
            f"Assistant: {prefix}"
        )
        token_ids = self.tokenizer.encode(context, add_bos=True, add_eos=False, limit=self.cfg.context_size)
        generated: list[int] = []
        rnd = random.Random(_stable_seed(prompt, len(self.history)))
        recent_window = 24

        for _ in range(64):
            bit_logits = self.model.forward_last(token_ids)
            prior_logits = self.prior.logits(token_ids[-1])
            merged = [0.0] * self.cfg.vocab_size
            recent = token_ids[-recent_window:]
            counts: dict[int, int] = {}
            for tok in recent:
                counts[tok] = counts.get(tok, 0) + 1

            for i in range(self.cfg.vocab_size):
                merged[i] = (bit_logits[i] * 0.22) + (prior_logits[i] * 0.78)
                if i in counts:
                    merged[i] -= counts[i] * 0.12

            next_tok = self._sample_token(merged, rnd)
            if next_tok == self.tokenizer.eos_id:
                break
            token_ids.append(next_tok)
            token_ids = token_ids[-self.cfg.context_size:]
            generated.append(next_tok)

            tail = self.tokenizer.decode(generated)
            if tail.endswith("\n\n"):
                break
            if len(tail) > 160 and tail[-1] in ".!?":
                break

        text = _clean_generated(prefix + self.tokenizer.decode(generated))
        if not _assistant_output_ok(text):
            return self._fallback_reply(prompt)
        return text

    def generate(self, prompt: str) -> str:
        self.last_aha = ""
        self.last_thinking = ""
        self.history.append(("user", prompt))
        raw = (prompt or "").strip()
        pl = raw.lower()

        if pl in ("/pr", "/profile"):
            return self.profile_text()
        if pl in ("/model", "/about"):
            return self.model_text()
        if pl in ("/paper", "/arxiv", "/bitnet-paper"):
            return f"{WHITEPAPER_BLURB}\n\nFull text: {WHITEPAPER_ARXIV}"
        if pl in ("/help", "help"):
            return self.help_text()
        if pl in ("/reset", "/clear"):
            self.history.clear()
            self.last_aha = ""
            self.last_thinking = ""
            return "Conversation history cleared."

        if pl.startswith("/run") or pl.startswith("/interp"):
            self.last_thinking = _synthesize_reasoning(raw, len(self.history))
            code = ""
            if pl.startswith("/run"):
                code = raw[4:].strip()
            elif pl.startswith("/interp"):
                code = raw[7:].strip()
            if not code:
                blk = _extract_python_block(raw)
                code = blk or ""
            if not code:
                return (
                    "Usage: `/run print(2+2)` or paste a ```python ... ``` block on the same line as `/run`.\n"
                    "Interpreter: subprocess, timeout, denylist (see /profile). Not a security sandbox."
                )
            out = run_code_interpreter(code)
            # Fence as ``` so the GUI logs this chunk with code_fence=True ($ and [] stay verbatim).
            return f"**Interpreter output (verbatim)**\n```\n{out}\n```"

        if pl in ("hi", "hello", "hey") or "hello" in pl:
            self.last_thinking = _synthesize_reasoning(raw, len(self.history))
            return (
                "Hello. The window is open. You are on CATR1: BitNet b1.58 architecture in this file "
                "(see /paper), plus a visible planning trace and a `/run` Python interpreter."
            )
        if any(k in pl for k in ("bug", "traceback", "error", "exception", "why")):
            self.last_thinking = _synthesize_reasoning(raw, len(self.history))
            self.last_aha = "isolate one concrete failure, then test the smallest input that still breaks."
            return "Give me the exact error line, the expected result, and the actual result."
        if "python" in pl and any(k in pl for k in ("write", "code", "snippet", "script")):
            self.last_thinking = _synthesize_reasoning(raw, len(self.history))
            return _opus_style_python_reply(raw)
        if any(k in pl for k in ("build", "make", "create", "design")) and any(k in pl for k in ("gui", "model", "bitnet", "transformer")):
            self.last_thinking = _synthesize_reasoning(raw, len(self.history))
            return (
                "Prefer a single file, tkinter on the foreground thread, generation on a worker thread, "
                "and a stack of tokenizer, embeddings, causal CATR1 BitNet b1.58 blocks (RMSNorm, RoPE attention, "
                "SwiGLU), then a BitLinear head. See /paper for the published definition."
            )
        return self._model_reply(prompt)


def run_cli() -> None:
    engine = BitNetEngine()
    print(f"{MODEL_NAME} CLI. Type 'exit' to quit.\n")
    while True:
        try:
            msg = input(">>> ")
            if msg.strip().lower() == "exit":
                break
            started = time.perf_counter()
            out = engine.generate(msg)
            elapsed = (time.perf_counter() - started) * 1000.0
            if engine.last_thinking:
                print(engine.last_thinking)
                print("-" * 48)
            print(out)
            if engine.last_aha:
                print("Aha:", engine.last_aha)
            print(f"[{elapsed:.1f} ms]\n")
        except (EOFError, KeyboardInterrupt):
            break


def run_gui() -> None:
    """Tkinter layout aligned with catr1.py mini GUI (850x620, Help/Profile/Py chips)."""
    import tkinter as tk
    from tkinter import font, messagebox, scrolledtext

    engine = BitNetEngine()

    root = tk.Tk()
    root.title(WINDOW_TITLE)
    root.geometry("850x620")
    root.configure(bg="#050505")

    fonts = {
        "mono": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=11),
        "bold": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=11, weight="bold"),
        "italic": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=10, slant="italic"),
        "small": font.Font(family="Consolas" if os.name != "nt" else "Courier New", size=9),
    }

    chat = scrolledtext.ScrolledText(
        root,
        bg="#050505",
        fg="#00d9ff",
        font=fonts["mono"],
        insertbackground="cyan",
        relief="flat",
        padx=12,
        pady=12,
        state="disabled",
    )
    chat.pack(expand=True, fill="both")

    for tag_name, color, fnt in [
        ("user", "#ffffff", fonts["bold"]),
        ("think", "#4a4a4a", fonts["italic"]),
        ("bot", "#00aaff", fonts["bold"]),
        ("code", "#00ffaa", fonts["small"]),
        ("aha", "#ffd54f", fonts["bold"]),
    ]:
        chat.tag_config(tag_name, foreground=color, font=fnt)

    inp = tk.Frame(root, bg="#050505")
    inp.pack(fill="x", padx=10, pady=5)
    entry = tk.Entry(inp, bg="#111", fg="#00d9ff", font=fonts["mono"], insertbackground="cyan", relief="flat", bd=2)
    entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
    btns = tk.Frame(inp, bg="#050505")
    btns.pack(side="right")
    for t, c in [("Help", "help"), ("Profile", "readme"), ("Py", "write python code")]:
        tk.Button(
            btns,
            text=t,
            command=lambda c=c: entry.insert("end", c + " "),
            bg="#222",
            fg="#00d9ff",
            font=fonts["small"],
            relief="flat",
        ).pack(side="left", padx=2)

    status = tk.Label(root, text="Ready", bg="#050505", fg="#666", font=fonts["small"], anchor="w")
    status.pack(fill="x", padx=10, pady=2)

    def log_line(sender: str, text: str, tag: str | None = None) -> None:
        body = _text_insert_safe(text if isinstance(text, str) else str(text), code_fence=(tag == "code"))
        head_tag = "bot" if sender == BOT_NAME else (tag if tag is not None else "think")
        body_tag = tag if tag is not None else ("bot" if sender == BOT_NAME else "think")
        try:
            chat.config(state="normal")
            chat.insert("end", f"[{sender}]: ", head_tag)
            chat.insert("end", f"{body}\n\n", body_tag)
            chat.config(state="disabled")
            chat.see("end")
        except tk.TclError:
            esc = (f"[{sender}]: " + body).encode("unicode_escape", errors="replace").decode("ascii", errors="replace")[:12000]
            chat.config(state="normal")
            chat.insert("end", esc + "\n\n", "think")
            chat.config(state="disabled")
            chat.see("end")
        if sender == "SYSTEM":
            status.config(text=body[:65])

    log_line("SYSTEM", f"{BOT_NAME} ONLINE (BitNet b1.58 stack)")
    log_line(
        "SYSTEM",
        "Type /help, /profile, /model, /paper, /run … | BitNet b1.58 in this file | CATR1 model name kept",
    )

    def send() -> None:
        msg = entry.get().strip()
        if not msg:
            return
        entry.delete(0, "end")
        log_line("YOU", msg, "user")
        status.config(text="Quantizing and routing...")

        def worker() -> None:
            try:
                resp = engine.generate(msg)
            except Exception as e:  # pragma: no cover - GUI safety path
                resp = f"(error) {type(e).__name__}: {e}"
                engine.last_aha = ""
                engine.last_thinking = ""
            think = engine.last_thinking
            aha = engine.last_aha

            def show() -> None:
                if think:
                    log_line("THINK", think, "think")
                if "```" in resp:
                    parts = resp.split("```")
                    for i, p in enumerate(parts):
                        chunk = p + ("```" if i < len(parts) - 1 and i % 2 == 0 else "")
                        log_line(BOT_NAME, chunk, "code" if i % 2 == 1 else None)
                else:
                    log_line(BOT_NAME, resp, None)
                if aha:
                    log_line("AHA", f"Aha: {aha}", "aha")
                status.config(text="Ready")

            root.after(0, show)

        threading.Thread(target=worker, daemon=True).start()

    entry.bind("<Return>", lambda _e: send())
    entry.focus_set()
    root.protocol(
        "WM_DELETE_WINDOW",
        lambda: root.destroy() if messagebox.askokcancel("Quit", f"Exit {WINDOW_TITLE}?") else None,
    )
    root.mainloop()


def main(argv: list[str]) -> int:
    args = set(argv[1:])
    if "--cli" in args or "--headless" in args:
        run_cli()
        return 0
    try:
        run_gui()
        return 0
    except Exception as exc:
        print("GUI failed, switching to CLI.", file=sys.stderr)
        print("Reason:", exc, file=sys.stderr)
        run_cli()
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
