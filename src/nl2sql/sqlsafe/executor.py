"""SQL 执行引擎 —— 只读连接 + 超时 + 行数上限 + 结果规模保护。

设计取舍：

- **不引入 `sqlite` 风格的 `PRAGMA query_only`**，DuckDB 用 `read_only=True` 打开连接即可；
- **超时用线程 + `interrupt()`**：DuckDB 的 Python 接口没有原生超时参数，
  在另一个线程里 `con.interrupt()` 是目前最可靠的软中断方式；
  它会把正在执行的查询打断并抛 `InterruptException`，连接仍可继续使用；
- **行数上限用 `fetchmany`**：不是先全拉回来再切片，避免大结果集打爆内存；
- **结果规模保护**：即使有 LIMIT，也可能被模型写成 `LIMIT 100000`，
  这里强制 `min(用户 LIMIT, 服务端上限)`。

这里也负责把结果**渲染成模型可读的表格**，供「结果解读」环节使用。
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb


@dataclass
class ExecutionResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: float = 0.0
    sql: str = ""
    error: str = ""
    error_kind: str = ""      # timeout / runtime / connection

    def to_dict(self, max_rows: int = 200) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "columns": self.columns,
            "rows": [
                [_jsonable(v) for v in r] for r in self.rows[:max_rows]
            ],
            "row_count": self.row_count,
            "truncated": self.truncated,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "error": self.error,
            "error_kind": self.error_kind,
        }


def _jsonable(v: Any) -> Any:
    """把 DuckDB 返回的 Decimal/date/datetime 转成 JSON 可序列化类型。"""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 6)
    try:
        import datetime as _dt
        import decimal as _dec
        if isinstance(v, _dec.Decimal):
            return float(v)
        if isinstance(v, (_dt.date, _dt.datetime)):
            return v.isoformat()
    except Exception:
        pass
    if isinstance(v, (list, dict, tuple)):
        return [_jsonable(x) for x in v] if isinstance(v, (list, tuple)) else v
    return v


class SQLRunner:
    """一个进程内复用的只读 DuckDB 连接。

    **线程安全说明**：DuckDB 的 Python 连接对象不是线程安全的。
    这里的做法是「每个 Runner 用一个连接 + 一把锁」，超时线程只调用
    `interrupt()`（这是 DuckDB 明确支持跨线程调用的方法）。
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        timeout_s: float = 8.0,
        max_rows: int = 500,
        memory_limit: str = "1GB",
        threads: int = 2,
    ):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"数据库不存在：{self.db_path}")
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self._lock = threading.Lock()
        # 关键：read_only=True + 限制内存/线程，防止查询把整机拖死
        self._con = duckdb.connect(str(self.db_path), read_only=True)
        for stmt in (
            f"SET memory_limit='{memory_limit}'",
            f"SET threads={threads}",
            "SET enable_progress_bar=false",
            # 关掉外部访问：这是护栏之外的第二道防线
            "SET enable_external_access=false",
        ):
            try:
                self._con.execute(stmt)
            except Exception:
                pass

    # ── 执行 ──────────────────────────────────────────────

    def execute(self, sql: str, *, max_rows: int | None = None) -> ExecutionResult:
        limit = max_rows or self.max_rows
        start = time.perf_counter()

        def _run() -> ExecutionResult:
            with self._lock:
                cur = self._con.cursor()
                try:
                    cur.execute(sql)
                    cols = [d[0] for d in (cur.description or [])]
                    # fetchmany 而不是 fetchall：大结果集不落内存
                    rows = cur.fetchmany(limit + 1)
                    truncated = len(rows) > limit
                    rows = rows[:limit]
                    # 截断时真实总行数未知，用 ">limit" 语义表示（渲染层会写成「至少 N 行」）
                    count = len(rows)
                    return ExecutionResult(
                        ok=True, columns=cols, rows=rows, row_count=count,
                        truncated=truncated, sql=sql,
                        elapsed_ms=(time.perf_counter() - start) * 1000,
                    )
                except Exception as e:  # noqa: BLE001 —— 需要把任何运行时错误回传给模型
                    msg = str(e).strip().splitlines()[0][:500] if str(e).strip() else type(e).__name__
                    is_timeout = "interrupt" in msg.lower() or "timeout" in msg.lower()
                    return ExecutionResult(
                        ok=False, sql=sql, error=msg,
                        error_kind="timeout" if is_timeout else "runtime",
                        elapsed_ms=(time.perf_counter() - start) * 1000,
                    )
                finally:
                    try:
                        cur.close()
                    except Exception:
                        pass

        result_box: list[ExecutionResult] = []

        def _target() -> None:
            result_box.append(_run())

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(self.timeout_s)
        if t.is_alive():
            # 软中断：DuckDB 会在下一个执行点抛出 InterruptException
            try:
                self._con.interrupt()
            except Exception:
                pass
            t.join(1.0)
            if result_box:
                r = result_box[0]
                if not r.ok:
                    r.error_kind = "timeout"
                    r.error = r.error or f"查询超过 {self.timeout_s}s 被中断"
                return r
            return ExecutionResult(
                ok=False, sql=sql, error_kind="timeout",
                error=f"查询超过 {self.timeout_s}s 未返回，已中断",
                elapsed_ms=(time.perf_counter() - start) * 1000,
            )
        return result_box[0]

    # ── 渲染 ──────────────────────────────────────────────

    def render_for_llm(self, result: ExecutionResult, max_rows: int = 30) -> str:
        """把结果渲染成 markdown 表格，喂给「结果解读」环节。

        刻意**截断到 30 行**：解读环节要的是「这份结果说明什么」，
        全量数据既浪费 token 又容易让模型开始逐行念数字。
        """
        if not result.ok:
            return f"查询失败：{result.error}"
        if not result.rows:
            return "查询返回 0 行（可能筛选条件过严，或该口径下确实没有数据）。"
        show = result.rows[:max_rows]
        header = "| " + " | ".join(result.columns) + " |"
        sep = "| " + " | ".join("---" for _ in result.columns) + " |"
        body = "\n".join(
            "| " + " | ".join(_fmt_cell(v) for v in row) + " |" for row in show
        )
        tail = ""
        if result.truncated:
            tail = (f"\n（结果已按上限截断：至少 {result.row_count} 行，"
                    f"此处只展示前 {len(show)} 行）")
        elif result.row_count > len(show):
            tail = f"\n（共 {result.row_count} 行，此处只展示前 {len(show)} 行）"
        return f"{header}\n{sep}\n{body}{tail}"

    def close(self) -> None:
        try:
            self._con.close()
        except Exception:
            pass


def _fmt_cell(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return "NULL"
        return f"{v:,.2f}" if abs(v) >= 1000 else f"{v:.4g}"
    try:
        import datetime as _dt
        import decimal as _dec
        if isinstance(v, _dec.Decimal):
            f = float(v)
            return f"{f:,.2f}"
        if isinstance(v, (_dt.date, _dt.datetime)):
            return v.isoformat()
    except Exception:
        pass
    return str(v)


# ──────────────────────────────────────────────────────────────────
# SQL 后处理：强制行数上限
# ──────────────────────────────────────────────────────────────────

def cap_limit(sql: str, max_rows: int) -> str:
    """把最外层的 LIMIT 收紧到 `max_rows`。

    模型经常写 `LIMIT 1000` 或 `LIMIT 10000`，`fetchmany` 虽然已经限制了内存，
    但排序/聚合的开销仍会发生。这里在文本层做一次加固（AST 层改写对复杂 CTE 更容易出错）。
    """
    s = sql.strip().rstrip(";")
    m = re.search(r"\bLIMIT\s+(\d+)\s*$", s, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        if n > max_rows:
            return re.sub(r"\bLIMIT\s+\d+\s*$", f"LIMIT {max_rows}", s, flags=re.IGNORECASE)
        return s
    # 没有 LIMIT（聚合查询）就原样返回
    return s
