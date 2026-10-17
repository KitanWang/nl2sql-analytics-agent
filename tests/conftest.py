"""测试夹具：确保 DuckDB 库存在。

## 为什么在这里自动建库

`data/warehouse.duckdb` **不入库**（它是从 Parquet 派生的、可用脚本重建的产物）。
很多测试需要真实执行 SQL。如果每个测试文件都写「库不存在就 skip」，
全新 clone 上 `pytest` 会静默跳过一大半测试——那比失败更糟，因为看起来是绿的。

正确做法：**缺库就自动构建一次**（3–6 秒，纯本地计算、不联网），
让「git clone → pytest」在干净环境里直接是绿的。

## 为什么要写这么多找路径的代码

这是踩过的坑：以 `pip install -e` 安装后，若机器上存在多个 checkout，
`__file__` 解析出的仓库根可能是**另一个 checkout**，于是 subprocess 去执行
一个不存在的脚本路径（报错形如 `can't open file '/tmp/scripts/build_warehouse.py'`）。
所以这里按优先级依次尝试，每一步都**验证目标文件真实存在**才采用。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

BUILD_REL = Path("scripts") / "build_warehouse.py"
DB_REL = Path("data") / "warehouse.duckdb"


def _candidate_roots() -> list[Path]:
    """按可靠性从高到低列出可能的仓库根。"""
    roots: list[Path] = []
    # 1) 已安装包的位置（editable 安装时 = <repo>/src/nl2sql）
    try:
        import nl2sql

        roots.append(Path(nl2sql.__file__).resolve().parents[2])
    except Exception:
        pass
    # 2) 当前工作目录（`pytest` 通常从仓库根启动）
    roots.append(Path.cwd().resolve())
    # 3) conftest 自身位置
    roots.append(Path(__file__).resolve().parents[1])
    # 4) 逐级向上找
    for parent in Path(__file__).resolve().parents:
        roots.append(parent)
    # 去重且保持顺序
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def find_repo_root() -> Path:
    for root in _candidate_roots():
        if (root / BUILD_REL).is_file():
            return root
    raise RuntimeError(
        "找不到 scripts/build_warehouse.py；候选根目录："
        + ", ".join(str(r) for r in _candidate_roots())
    )


def pytest_configure(config: pytest.Config) -> None:
    """会话开始时检查一次：库不存在就构建。"""
    root = find_repo_root()
    db = root / DB_REL
    if db.is_file():
        return
    script = root / BUILD_REL
    print(f"\n[conftest] 构建测试用数仓：{root}", file=sys.stderr)
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not db.is_file():
        pytest.exit(
            "自动构建数仓失败。请手动执行：\n"
            f"  cd {root} && {sys.executable} {script}\n"
            + (proc.stdout or "")[-1500:]
            + (proc.stderr or "")[-1500:],
            returncode=2,
        )
