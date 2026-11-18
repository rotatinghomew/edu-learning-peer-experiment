"""生成同伴带动邀学试验的合成登记数据（固定种子，可复现）。

脚本只负责落盘业务登记记录，不参与任何分析口径：
- design.csv       登记的抽签设计：抽签单位（社区）、可分方案及概率、方案邀请比例
- persons.csv      试验花名册
- assignments.csv 抽签结果（随机分配 Z，社区共同受分配）
- notifications.csv 个人通知记录（实际接触 D，可能与 Z 不一致）
- attendance.csv   个人到课记录（结果 Y）
- edges.csv        关系边，pre_treatment=0 的课后新增边会被分析拒收
- budget.csv       下一轮按社区整体占用的邀请名额预算

所有记录均为合成数据，不含真实个人信息。
"""
import csv
import random
from pathlib import Path

SEED = 20261118
PERIOD = "2026-S3"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "trial"

# 社区（抽签单位）与人数
GROUPS = [("甲", 12), ("乙", 10), ("丙", 12), ("丁", 10), ("戊", 10), ("己", 12)]

# 登记的方案与概率。丙社区登记时就排除了高强度方案（概率为 0 即不出现在登记行中）
ARM_PROBS = {"control": 0.34, "low": 0.33, "high": 0.33}
ARM_LABELS = {"control": "对照", "low": "低强度邀请", "high": "高强度邀请"}
ARM_FRACTIONS = {"control": 0.0, "low": 0.3, "high": 0.6}
DESIGN_EXCLUSIONS = {"丙": {"high"}}

# 本轮实际抽签结果（社区共同受分配）
REALIZED_ARM = {
    "甲": "low", "乙": "high", "丙": "control",
    "丁": "low", "戊": "high", "己": "control",
}

# 到课生成参数（仅用于制造合成结果，分析侧并不知道这些参数）
BASE_RATE = 0.35
OWN_INVITE_LIFT = 0.18
PEER_SPILLOVER_LIFT = 0.12
JOINT_LIFT = 0.25  # 本人受邀且同伴也受邀的联合暴露，本轮刻意不实现该格子

BUDGET_INVITATIONS = 30


def invited_count(group_size, arm):
    return round(group_size * ARM_FRACTIONS[arm])


def main():
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    design_rows, persons, assignments, notifications, attendance, edges = [], [], [], [], [], []

    # ---- 登记设计与花名册、分配、通知 ----
    group_members = {}
    for group, size in GROUPS:
        # 丙社区登记时排除高强度方案；剩余方案概率在登记时重新归一化
        arms = {arm: prob for arm, prob in ARM_PROBS.items()
                if arm not in DESIGN_EXCLUSIONS.get(group, set())}
        total = sum(arms.values())
        ordered = list(arms.items())
        rounded = [round(prob / total, 4) for _, prob in ordered]
        rounded[-1] = round(1.0 - sum(rounded[:-1]), 4)  # 保证登记概率严格和为 1
        for (arm, _), prob in zip(ordered, rounded):
            design_rows.append({
                "period": PERIOD, "group": group, "arm": arm,
                "arm_label": ARM_LABELS[arm],
                "invite_flag": 0 if arm == "control" else 1,
                "probability": prob,
                "invite_fraction": ARM_FRACTIONS[arm],
            })
        members = [f"{group}{i:02d}" for i in range(1, size + 1)]
        group_members[group] = members
        arm = REALIZED_ARM[group]
        k = invited_count(size, arm)
        invited_members = members[:k]  # 组内前 k 人获得本人邀请（组共同分配决定 k）
        for pid in members:
            persons.append({"person_id": pid, "group": group, "eligible": 1})
            own = pid in invited_members
            assignments.append({
                "period": PERIOD, "person_id": pid, "group": group,
                "assigned_arm": arm, "assigned_invite": int(own),
            })
            notified = int(own)
            notifications.append({
                "period": PERIOD, "person_id": pid, "group": group,
                "notified": notified, "notified_at": "",
            })

    # 一处不依从：甲03 被分配在邀请方案但实际未收到通知（D != Z）
    for row in notifications:
        if row["person_id"] == "甲03":
            row["notified"] = 0

    # ---- 入组前关系边 ----
    # 组内：对照成员之间全连接；每位受邀者只连到本组对照锚点，受邀者之间不连边。
    # 这样本轮“本人受邀且同伴受邀”(E2) 格子刻意没有实现对象，但设计概率上仍可能出现。
    for group, members in group_members.items():
        arm = REALIZED_ARM[group]
        k = invited_count(len(members), arm)
        invited = set(members[:k])
        controls = members[k:]
        for i in range(len(controls)):
            for j in range(i + 1, len(controls)):
                edges.append((controls[i], controls[j], 1))
        if controls:
            anchor = controls[0]
            for pid in invited:
                edges.append((pid, anchor, 1))

    # 跨社区、但两端都在登记试验内的边（跨抽签边界，组级推断需显式说明）
    edges.append(("己01", "甲01", 1))
    edges.append(("己02", "戊02", 1))

    # 跨出登记设计边界的边：邻居住在未纳入抽签的外部社区
    edges.append(("丙01", "外X01", 1))

    # 课后新增关系：pre_treatment=0，分析侧必须拒收，不能用来回溯定义入组前同伴
    edges.append(("甲02", "甲04", 0))

    # ---- 到课结果 ----
    invited_set = {(r["person_id"]) for r in assignments if r["assigned_invite"] == 1}
    adjacency = {}
    for a, b, pre in edges:
        if pre != 1:
            continue
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    registered_groups = {g for g, _ in GROUPS}

    for row in assignments:
        pid, group = row["person_id"], row["group"]
        own = pid in invited_set
        peers = adjacency.get(pid, set())
        peer_groups = {p[:2] if p[0] in registered_groups else None for p in peers}
        # 注意：合成结果只依赖分配 Z 与入组前网络，不依赖实际通知 D
        peer_invited = any(
            p in invited_set for p in peers
            if p in invited_set or any(p.startswith(g) for g in registered_groups)
        )
        if own and peer_invited:
            p = BASE_RATE + JOINT_LIFT
        elif own:
            p = BASE_RATE + OWN_INVITE_LIFT
        elif peer_invited:
            p = BASE_RATE + PEER_SPILLOVER_LIFT
        else:
            p = BASE_RATE
        attendance.append({
            "period": PERIOD, "person_id": pid, "group": group,
            "attended": int(rng.random() < p),
        })

    _write("design.csv", design_rows)
    _write("persons.csv", persons)
    _write("assignments.csv", assignments)
    _write("notifications.csv", notifications)
    _write("attendance.csv", attendance)
    _write("edges.csv", [
        {"period": PERIOD, "from_person": a, "to_person": b, "pre_treatment": pre}
        for a, b, pre in edges
    ])
    _write("budget.csv", [{"period": PERIOD, "invitations_budget": BUDGET_INVITATIONS}])
    print(f"已生成 {len(design_rows)} 条设计登记、{len(persons)} 名花名册人员、"
          f"{len(edges)} 条关系边（其中 {sum(1 for e in edges if e[2] == 0)} 条课后边）、"
          f"预算 {BUDGET_INVITATIONS} 个邀请名额 -> {OUT_DIR}")


def _write(filename, rows):
    if not rows:
        return
    with (OUT_DIR / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
