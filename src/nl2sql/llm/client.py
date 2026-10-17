"""LLM 客户端 —— OpenAI Chat Completions 兼容层。

## 为什么自己写而不是用 openai SDK

1. 本机可用的服务商（火山方舟 ARK / CodeBuddy）都兼容 OpenAI 协议，但**只支持流式**。
   用 SDK 走 `stream=False` 会直接报 `Non-stream chat request is currently not supported`；
2. 需要**结构化输出重试**：服务商不一定支持 `response_format={"type":"json_object"}`，
   得自己做「提取 JSON → 校验 → 失败重试」；
3. 需要把**每次调用的 token 用量与耗时**记下来，写进实验报告。

因此这里做了一个最小实现：`httpx` + SSE 手写解析，只保留需要的功能。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx
import yaml


# ──────────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_s: float = 120.0
    max_retries: int = 3
    # 有些服务商的 thinking 模型要额外参数；留空表示不加
    extra_body: dict[str, Any] = field(default_factory=dict)
    # 是否读取系统代理环境变量。
    # 默认 False：本机 no_proxy 里有 `[::1]`/`<local>` 这类 httpx 解析不了的模式，
    # 且国内 API 直连更快。需要走代理时在 profile 里写 use_proxy: true。
    use_proxy: bool = False
    proxy_url: str = ""

    @classmethod
    def from_file(cls, path: str | Path, *, profile: str | None = None) -> "LLMConfig":
        """从 YAML 读配置。API key 支持 `${ENV_VAR}` 展开，便于 CI 里用 secrets。

        配置优先级：configs/local.yaml > configs/llm.yaml > 环境变量
        `local.yaml` 已被 .gitignore 排除，是唯一允许放明文 key 的地方。
        """
        root = Path(path)
        candidates = []
        if root.is_file():
            candidates.append(root)
        else:
            candidates += [root / "local.yaml", root / "llm.yaml"]
        merged: dict[str, Any] = {}
        for c in reversed([p for p in candidates if p.exists()]):
            data = yaml.safe_load(c.read_text(encoding="utf-8")) or {}
            merged = _deep_merge(merged, data)
        if not merged:
            raise FileNotFoundError(
                f"找不到 LLM 配置：{root}（需要 configs/llm.yaml 或 configs/local.yaml）"
            )

        name = profile or merged.get("default_profile") or "default"
        prof = (merged.get("profiles") or {}).get(name)
        if prof is None:
            raise KeyError(f"配置里没有 profile `{name}`，可选：{list((merged.get('profiles') or {}))}")

        api_key = _expand(prof.get("api_key") or "")
        if not api_key:
            env_name = prof.get("api_key_env")
            if env_name:
                api_key = os.environ.get(env_name, "")
        if not api_key:
            raise ValueError(
                f"profile `{name}` 缺少 api_key：在 configs/local.yaml 里写 api_key，"
                f"或设置环境变量 {prof.get('api_key_env')}"
            )
        return cls(
            base_url=prof["base_url"].rstrip("/"),
            model=prof["model"],
            api_key=api_key,
            temperature=float(prof.get("temperature", 0.0)),
            max_tokens=int(prof.get("max_tokens", 2048)),
            timeout_s=float(prof.get("timeout_s", 120.0)),
            extra_body=prof.get("extra_body") or {},
            use_proxy=bool(prof.get("use_proxy", False)),
            proxy_url=str(prof.get("proxy_url") or ""),
        )


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _expand(s: str) -> str:
    def repl(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")
    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, s or "")


# ──────────────────────────────────────────────────────────────────
# 调用结果
# ──────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: float = 0.0
    finish_reason: str = ""
    raw_chunks: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    pass


# ──────────────────────────────────────────────────────────────────
# 客户端
# ──────────────────────────────────────────────────────────────────

class LLMClient:
    """最小 OpenAI 兼容客户端，强制走流式。"""

    def __init__(self, cfg: LLMConfig, *, client: httpx.Client | None = None):
        self.cfg = cfg
        self._own_client = client is None
        proxy = cfg.proxy_url or None
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(cfg.timeout_s, connect=10.0),
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            trust_env=cfg.use_proxy,
            proxy=proxy,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "stream": True,
        }
        payload.update(self.cfg.extra_body or {})

        last_err: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                return self._chat_once(payload, on_token=on_token)
            except (httpx.TransportError, LLMError) as e:
                last_err = e
                if attempt < self.cfg.max_retries:
                    time.sleep(min(2 ** attempt, 8))
        raise LLMError(f"LLM 调用失败（重试 {self.cfg.max_retries} 次）：{last_err}")

    def _chat_once(
        self, payload: dict[str, Any], *, on_token: Callable[[str], None] | None = None
    ) -> LLMResponse:
        start = time.perf_counter()
        url = f"{self.cfg.base_url}/chat/completions"
        parts: list[str] = []
        usage: dict[str, Any] = {}
        finish = ""
        chunks = 0

        with self._client.stream("POST", url, json=payload) as resp:
            if resp.status_code >= 400:
                body = resp.read().decode("utf-8", "replace")[:600]
                raise LLMError(f"HTTP {resp.status_code}: {body}")
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                elif line.startswith("{"):
                    # 少数服务商直接返回裸 JSON 行
                    data = line.strip()
                else:
                    continue
                if data in ("[DONE]", "DONE"):
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                chunks += 1
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    # 只取 content：reasoning_content 属于模型内部思考，不应进入产物
                    piece = delta.get("content") or ""
                    if piece:
                        parts.append(piece)
                        if on_token:
                            on_token(piece)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    if not usage and ch.get("usage"):
                        usage = ch["usage"]

        text = "".join(parts)
        if not text.strip():
            raise LLMError("模型返回了空内容（可能是 max_tokens 太小或触发了内容过滤）")
        return LLMResponse(
            text=text,
            model=payload["model"],
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            elapsed_ms=(time.perf_counter() - start) * 1000,
            finish_reason=finish,
            raw_chunks=chunks,
        )

    def close(self) -> None:
        if self._own_client:
            self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ──────────────────────────────────────────────────────────────────
# 结构化输出：从自由文本里抠 JSON
# ──────────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict[str, Any]:
    """从模型输出里提取第一个 JSON 对象。

    现实情况：即使要求「只输出 JSON」，模型仍会加 ```json 围栏、
    加一句「好的，以下是结果：」、或者输出 `<|think|>` 残留。
    这里按「代码块 → 全文 → 括号配对扫描」三级降级提取。
    """
    s = text.strip()
    s = re.sub(r"<\|?think\|?>.*?<\|?/think\|?>", "", s, flags=re.IGNORECASE | re.DOTALL).strip()

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        return json.loads(m.group(1))

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # 括号配对扫描：从第一个 { 开始找匹配的 }
    start = s.find("{")
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(s[start:i + 1])
    raise ValueError(f"输出里找不到合法 JSON：{s[:200]}")
