"""合成一份「像真实业务」的 B2B 销售数据。

为什么要合成而不是下载公开数据集：

1. **公开数据集没有业务口径**。Titanic、NYC Taxi 这类数据集没法测「有效订单」「客单价」
   这类口径问题，而这恰好是 NL2SQL 真实失败的大头；
2. **需要可控的脏数据**。真实数据里的坑（金额与明细对不上、订单号重复、金额单位混用分/元、
   生日写进下单日期、客户名括号全半角不一致）是刻意注入的，否则清洗流水线形同虚设；
3. **评测集答案可精确计算**。合成数据 + 确定性随机种子 ⇒ 每条问题的标准答案可以
   用 pandas 独立算一遍，评测结果才可信。

脏数据的注入点都在 `_inject_*` 函数里，每处都写明了「现实中为什么会出现」。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────
# 基础字典
# ──────────────────────────────────────────────────────────────────

INDUSTRIES = ("汽车零部件", "电子制造", "机械加工", "化工", "食品饮料", "医疗器械", "建材", "新能源")
REGIONS = ("华东", "华北", "华南", "华中", "西南", "西北", "东北")
CITIES = {
    "华东": ("上海", "苏州", "杭州", "宁波", "无锡"),
    "华北": ("北京", "天津", "石家庄", "青岛"),
    "华南": ("深圳", "广州", "东莞", "佛山"),
    "华中": ("武汉", "长沙", "郑州"),
    "西南": ("成都", "重庆", "昆明"),
    "西北": ("西安", "兰州"),
    "东北": ("沈阳", "大连", "哈尔滨"),
}
CHANNELS = ("天猫", "京东", "抖音", "官网", "线下")
# 渠道权重：线下占比最高（B2B 直销），抖音是新增渠道
CHANNEL_WEIGHTS = (0.18, 0.15, 0.10, 0.22, 0.35)
TIERS = ("KA", "SMB", "长尾")
TIER_WEIGHTS = (0.12, 0.46, 0.42)
SALES_REPS = ("张伟", "李娜", "王强", "刘敏", "陈杰", "赵磊", "孙悦", "周涛")

CATEGORIES = {
    "传感器": ("光电传感器", "接近开关", "压力传感器", "温度传感器", "位移传感器"),
    "伺服电机": ("交流伺服电机", "步进电机", "伺服驱动器", "减速机"),
    "工业相机": ("面阵相机", "线扫相机", "3D 相机", "镜头"),
    "控制器": ("PLC 模块", "运动控制卡", "HMI 触摸屏", "工业网关"),
    "线缆": ("屏蔽电缆", "拖链电缆", "连接器", "光纤跳线"),
    "气动元件": ("气缸", "电磁阀", "气源处理件", "真空吸盘"),
}
BRANDS = ("Kistler", "Omron", "Sick", "Keyence", "Festo", "Delta", "Inovance", "Beckhoff")

REFUND_REASONS = ("质量问题", "发错货", "客户取消", "价格异议", "物流破损")
PAY_METHODS = ("银行转账", "支付宝", "微信支付", "承兑汇票")

# 客户名构成：地域 + 行业词 + 后缀，用于生成中文公司名
NAME_PREFIX = ("宏昌", "瑞泰", "华远", "金鹏", "恒信", "博实", "正泰", "联创",
               "嘉禾", "天工", "中航", "凯尔", "新锐", "东盛", "润泽", "恒通")
NAME_SUFFIX = ("精密五金有限公司", "智能装备有限公司", "自动化科技有限公司",
               "机电设备有限公司", "电子科技有限公司", "工业技术有限公司")


@dataclass(frozen=True)
class SynthConfig:
    seed: int = 20251018
    n_customers: int = 480
    n_products: int = 220
    n_orders: int = 9000
    start_date: date = date(2024, 1, 1)
    end_date: date = date(2025, 12, 31)
    items_min: int = 1
    items_max: int = 4


@dataclass
class Dataset:
    customers: pd.DataFrame
    products: pd.DataFrame
    orders: pd.DataFrame
    items: pd.DataFrame
    payments: pd.DataFrame
    refunds: pd.DataFrame
    calendar: pd.DataFrame


# ──────────────────────────────────────────────────────────────────
# 生成主流程
# ──────────────────────────────────────────────────────────────────

def generate(cfg: SynthConfig = SynthConfig()) -> Dataset:
    rng = random.Random(cfg.seed)
    npr = np.random.default_rng(cfg.seed)

    customers = _gen_customers(cfg, rng)
    products = _gen_products(cfg, rng)
    orders, items = _gen_orders(cfg, rng, npr, customers, products)
    payments = _gen_payments(orders, rng)
    refunds = _gen_refunds(orders, rng)
    calendar = _gen_calendar(cfg.start_date, cfg.end_date)

    return Dataset(
        customers=customers,
        products=products,
        orders=orders,
        items=items,
        payments=payments,
        refunds=refunds,
        calendar=calendar,
    )


def _gen_customers(cfg: SynthConfig, rng: random.Random) -> pd.DataFrame:
    rows = []
    for i in range(cfg.n_customers):
        region = rng.choices(REGIONS, weights=(0.30, 0.18, 0.18, 0.12, 0.10, 0.06, 0.06))[0]
        city = rng.choice(CITIES[region])
        tier = rng.choices(TIERS, weights=TIER_WEIGHTS)[0]
        industry = rng.choice(INDUSTRIES)
        signup = cfg.start_date - timedelta(days=rng.randint(30, 900))
        # KA 有更高授信
        base_credit = {"KA": 800_000, "SMB": 250_000, "长尾": 60_000}[tier]
        rows.append({
            "customer_id": f"C{i + 1:05d}",
            "customer_name": f"{city}{rng.choice(NAME_PREFIX)}{rng.choice(NAME_SUFFIX)}",
            "industry": industry,
            "region": region,
            "city": city,
            "tier": tier,
            "signup_date": signup,
            "credit_limit": round(base_credit * rng.uniform(0.6, 1.5), 2),
            "is_active": rng.random() > 0.08,
        })
    df = pd.DataFrame(rows)
    # 客户名去重：同城同前缀同后缀会撞名，真实系统会加分公司后缀
    seen: dict[str, int] = {}
    names = []
    for n in df["customer_name"]:
        seen[n] = seen.get(n, 0) + 1
        names.append(n if seen[n] == 1 else n.replace("有限公司", f"（第{seen[n]}分公司）"))
    df["customer_name"] = names
    return df


def _gen_products(cfg: SynthConfig, rng: random.Random) -> pd.DataFrame:
    rows = []
    pid = 0
    for category, subs in CATEGORIES.items():
        for _ in range(cfg.n_products // len(CATEGORIES)):
            pid += 1
            sub = rng.choice(subs)
            brand = rng.choice(BRANDS)
            # 价格按品类给不同的量级，这样「销售额 TOP 商品」不会是清一色便宜货
            scale = {
                "传感器": 380, "伺服电机": 2600, "工业相机": 4200,
                "控制器": 1800, "线缆": 120, "气动元件": 260,
            }[category]
            list_price = round(scale * rng.uniform(0.5, 2.4), 2)
            rows.append({
                "product_id": f"P{pid:05d}",
                "product_name": f"{brand} {sub} {rng.choice(['X', 'S', 'M', 'E'])}{rng.randint(100, 999)}",
                "category": category,
                "sub_category": sub,
                "brand": brand,
                "list_price": list_price,
                "cost": round(list_price * rng.uniform(0.42, 0.78), 2),
                "launch_date": cfg.start_date - timedelta(days=rng.randint(0, 700)),
                "is_on_sale": rng.random() > 0.12,
            })
    return pd.DataFrame(rows)


def _gen_orders(
    cfg: SynthConfig,
    rng: random.Random,
    npr: np.random.Generator,
    customers: pd.DataFrame,
    products: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_days = (cfg.end_date - cfg.start_date).days + 1
    # 日期上叠加「周末低谷 + 双十一/618 尖峰 + 年末冲量」的真实形态
    day_w = np.ones(n_days)
    for d in range(n_days):
        cur = cfg.start_date + timedelta(days=d)
        if cur.weekday() >= 5:
            day_w[d] *= 0.45
        if (cur.month, cur.day) in ((11, 11), (6, 18), (12, 12)):
            day_w[d] *= 6.0
        if cur.month == 12:
            day_w[d] *= 1.8
        if cur.month == 2:
            day_w[d] *= 0.7
    day_w = day_w / day_w.sum()
    day_idx = npr.choice(n_days, size=cfg.n_orders, p=day_w)

    # 客户下单频次服从长尾（KA 客户活跃，长尾客户偶尔下单）
    tier_w = customers["tier"].map({"KA": 8.0, "SMB": 3.0, "长尾": 1.0}).to_numpy()
    tier_w = tier_w / tier_w.sum()
    cust_idx = npr.choice(len(customers), size=cfg.n_orders, p=tier_w)

    order_rows, item_rows = [], []
    status_cycle = ("paid", "shipped", "completed")
    item_seq = 0
    for i in range(cfg.n_orders):
        c = customers.iloc[cust_idx[i]]
        odate = cfg.start_date + timedelta(days=int(day_idx[i]))
        # 状态随时间衰减：越老的订单越可能已完成，越新的越可能待付款
        age_days = (cfg.end_date - odate).days
        if age_days > 45:
            status = rng.choices(("completed", "refunded", "cancelled"),
                                 weights=(0.90, 0.04, 0.06))[0]
        elif age_days > 10:
            status = rng.choices(status_cycle + ("cancelled", "refunded"),
                                 weights=(0.10, 0.35, 0.45, 0.06, 0.04))[0]
        else:
            status = rng.choices(("pending", "paid", "shipped"),
                                 weights=(0.5, 0.3, 0.2))[0]
        # 收货大区：与客户注册大区一致的概率约 70%（现实里就是会不一致）
        ship_region = c["region"] if rng.random() < 0.70 else rng.choice(REGIONS)
        channel = rng.choices(CHANNELS, weights=CHANNEL_WEIGHTS)[0]
        order_id = f"SO{odate.strftime('%Y%m%d')}-{i + 1:04d}"

        n_items = rng.randint(cfg.items_min, cfg.items_max)
        # 订哪个品类有一点行业偏好，让「行业 × 品类」的分析有信号
        pref = {
            "汽车零部件": ("传感器", "伺服电机", "控制器"),
            "电子制造": ("传感器", "工业相机", "线缆"),
            "机械加工": ("伺服电机", "气动元件", "控制器"),
            "化工": ("传感器", "气动元件", "线缆"),
            "食品饮料": ("工业相机", "传感器", "气动元件"),
            "医疗器械": ("工业相机", "传感器", "线缆"),
            "建材": ("控制器", "线缆", "气动元件"),
            "新能源": ("传感器", "伺服电机", "工业相机"),
        }[c["industry"]]
        picked = set()
        for _ in range(n_items):
            for _try in range(20):
                p = products.iloc[rng.randrange(len(products))]
                if p["category"] in pref or rng.random() < 0.25:
                    break
            if p["product_id"] in picked:
                continue
            picked.add(p["product_id"])
            qty = rng.choices((1, 2, 3, 5, 8, 10, 20, 50, 100),
                              weights=(0.18, 0.18, 0.14, 0.12, 0.10, 0.10, 0.08, 0.06, 0.04))[0]
            is_gift = rng.random() < 0.03
            unit_price = 0.0 if is_gift else round(float(p["list_price"]) * rng.uniform(0.72, 0.98), 2)
            disc = 0.0 if is_gift else rng.choices((0.0, 0.02, 0.05, 0.08, 0.10),
                                                  weights=(0.55, 0.15, 0.15, 0.10, 0.05))[0]
            amount = round(qty * unit_price * (1 - disc), 2)
            item_seq += 1
            item_rows.append({
                "order_item_id": f"OI{item_seq:07d}",
                "order_id": order_id,
                "product_id": p["product_id"],
                "quantity": qty,
                "unit_price": unit_price,
                "discount_rate": disc,
                "item_amount": amount,
                "is_gift": is_gift,
            })
        ship_fee = rng.choices((0.0, 15.0, 25.0, 40.0), weights=(0.55, 0.15, 0.2, 0.1))[0]
        discount_amount = 0.0 if rng.random() < 0.7 else round(rng.uniform(20, 400), 2)
        order_rows.append({
            "order_id": order_id,
            "customer_id": c["customer_id"],
            "order_date": odate,
            "status": status,
            "channel": channel,
            "ship_region": ship_region,
            "sales_rep": rng.choice(SALES_REPS),
            "discount_amount": discount_amount,
            "shipping_fee": ship_fee,
            "order_amount_raw": None,   # 生成后再算
            "created_at": pd.Timestamp(odate) + pd.Timedelta(hours=rng.randint(9, 20),
                                                             minutes=rng.randint(0, 59)),
            "updated_at": pd.Timestamp(odate) + pd.Timedelta(days=rng.randint(0, 5),
                                                            hours=rng.randint(0, 23)),
        })

    items = pd.DataFrame(item_rows)
    orders = pd.DataFrame(order_rows)

    amount_by_order = items.groupby("order_id")["item_amount"].sum()
    orders["order_amount_raw"] = (
        orders["order_id"].map(amount_by_order)
        + orders["shipping_fee"] - orders["discount_amount"]
    ).round(2)
    return orders, items


def _gen_payments(orders: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    rows = []
    seq = 0
    for o in orders.itertuples(index=False):
        if o.status in ("pending", "cancelled"):
            continue
        seq += 1
        # 85% 一次付清；其余部分付款——故意制造「回款 ≠ 订单金额」，让「回款率」有区分度
        full = rng.random() < 0.85
        frac = 1.0 if full else round(rng.uniform(0.3, 0.7), 2)
        delay = rng.choices((0, 1, 3, 7, 15, 30), weights=(0.25, 0.2, 0.2, 0.15, 0.12, 0.08))[0]
        pay_status = "success"
        if o.status == "refunded":
            pay_status = "success" if rng.random() < 0.6 else "failed"
        elif rng.random() < 0.03:
            pay_status = "processing"
        rows.append({
            "payment_id": f"PAY{seq:07d}",
            "order_id": o.order_id,
            "pay_date": o.order_date + timedelta(days=delay),
            "amount": round(float(o.order_amount_raw) * frac, 2),
            "method": rng.choices(PAY_METHODS, weights=(0.45, 0.2, 0.15, 0.2))[0],
            "pay_status": pay_status,
        })
    return pd.DataFrame(rows)


def _gen_refunds(orders: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    rows = []
    seq = 0
    refunded = orders[orders["status"] == "refunded"]
    # 部分「已完成」订单也有售后小额部分退款（约 5%）
    completed = orders[orders["status"] == "completed"]
    partial = completed.sample(frac=0.05, random_state=7) if len(completed) else completed
    for o in pd.concat([refunded, partial]).itertuples(index=False):
        seq += 1
        full = o.status == "refunded"
        amount = float(o.order_amount_raw) if full else round(float(o.order_amount_raw) * rng.uniform(0.05, 0.3), 2)
        rows.append({
            "refund_id": f"RF{seq:06d}",
            "order_id": o.order_id,
            "refund_date": o.order_date + timedelta(days=rng.randint(3, 60)),
            "amount": amount,
            "reason": rng.choice(REFUND_REASONS),
            "status": "success" if rng.random() < 0.9 else "pending",
        })
    return pd.DataFrame(rows)


def _gen_calendar(start: date, end: date) -> pd.DataFrame:
    """日期维表。节假日只标了几个大节，足够支撑「工作日」类问题。"""
    holidays = {
        (1, 1), (5, 1), (5, 2), (5, 3), (10, 1), (10, 2), (10, 3),
        (10, 4), (10, 5), (10, 6), (10, 7), (1, 28), (1, 29), (1, 30),
    }
    rows = []
    d = start
    while d <= end:
        rows.append({
            "date_key": d,
            "year": d.year,
            "quarter": (d.month - 1) // 3 + 1,
            "month": d.month,
            "year_month": d.strftime("%Y-%m"),
            "week": d.isocalendar()[1],
            "is_workday": d.weekday() < 5 and (d.month, d.day) not in holidays,
        })
        d += timedelta(days=1)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────
# 脏数据注入（每处都对应现实里真实出现过的坑）
# ──────────────────────────────────────────────────────────────────

def inject_dirty(data: Dataset, seed: int = 7) -> Dataset:
    """把干净数据打成「脏」的，交给 pipeline 清洗。

    注入清单与真实来源：

    | # | 脏形态 | 现实中为什么会这样 |
    |---|---|---|
    | 1 | 订单号重复（同号两条，updated_at 不同） | 上游同步任务重跑，幂等没做对 |
    | 2 | 金额单位混用（部分回款是「分」） | 老系统接口单位是分，新系统是元，迁移期并存 |
    | 3 | 负数量 / 0 数量明细 | 红冲与冲销单据混进明细表 |
    | 4 | 客户名全半角括号、尾部空格不一致 | 多个来源系统各自维护客户档案 |
    | 5 | 下单日期晚于当前结束日（未来单） | 预售设置错年份 |
    | 6 | order_amount_raw 与明细汇总偏差 | 源系统用了旧价目表 |
    | 7 | 卖家自己下单（sales_rep 与客户联系人重名） | 内购测试单 |
    | 8 | 退款金额 > 订单金额 | 退款按含税价、订单按不含税价 |
    """
    rng = random.Random(seed)
    orders = data.orders.copy()
    items = data.items.copy()
    customers = data.customers.copy()
    payments = data.payments.copy()
    refunds = data.refunds.copy()

    # 1) 订单号重复
    dup = orders.sample(n=120, random_state=seed).copy()
    dup["status"] = "cancelled"
    dup["updated_at"] = dup["updated_at"] + pd.Timedelta(days=2)
    dup["order_amount_raw"] = (dup["order_amount_raw"] * 0.9).round(2)
    # 重复行改个 customer 也无所谓——清洗按 updated_at 取最新
    orders = pd.concat([orders, dup], ignore_index=True)

    # 2) 部分回款金额变成「分」
    idx = payments.sample(n=180, random_state=seed + 1).index
    payments.loc[idx, "amount"] = (payments.loc[idx, "amount"] * 100).round(2)

    # 3) 负数 / 0 数量明细（红冲）
    idx = items.sample(n=90, random_state=seed + 2).index
    items.loc[idx, "quantity"] = -items.loc[idx, "quantity"]
    items.loc[idx, "item_amount"] = -items.loc[idx, "item_amount"]

    # 4) 客户名全半角与空格
    idx = customers.sample(n=60, random_state=seed + 3).index
    customers.loc[idx, "customer_name"] = customers.loc[idx, "customer_name"].str.replace(
        "有限公司", "（有限公司）", regex=False
    )
    idx = customers.sample(n=40, random_state=seed + 4).index
    customers.loc[idx, "customer_name"] = customers.loc[idx, "customer_name"] + "  "

    # 5) 未来单：把 20 条的下单日期推后
    idx = orders.sample(n=20, random_state=seed + 5).index
    orders.loc[idx, "order_date"] = date(2026, 3, 15)

    # 6) order_amount_raw 偏差（模拟旧价目表）
    idx = orders.sample(n=400, random_state=seed + 6).index
    orders.loc[idx, "order_amount_raw"] = (
        orders.loc[idx, "order_amount_raw"] * 0.88
    ).round(2)

    # 7) 内购测试单：客户名里带「测试」「内部」
    idx = customers.sample(n=6, random_state=seed + 7).index
    customers.loc[idx, "customer_name"] = "内部测试_" + customers.loc[idx, "customer_name"]

    # 8) 退款金额超订单金额
    if len(refunds):
        idx = refunds.sample(n=min(25, len(refunds)), random_state=seed + 8).index
        refunds.loc[idx, "amount"] = (refunds.loc[idx, "amount"] * 1.6).round(2)

    return Dataset(
        customers=customers.reset_index(drop=True),
        products=data.products,
        orders=orders.reset_index(drop=True),
        items=items.reset_index(drop=True),
        payments=payments.reset_index(drop=True),
        refunds=refunds.reset_index(drop=True),
        calendar=data.calendar,
    )
