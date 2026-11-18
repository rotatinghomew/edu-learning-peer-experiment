"""试验登记数据的读取与校验。

分析侧只承认登记事实：抽签设计、花名册、抽签分配 Z、实际通知 D、到课 Y、
入组前关系边、预算。课后新增关系（pre_treatment=0）一律拒收，不得用于
回溯定义入组前同伴；实际接触 D 与分配 Z 不一致时只做标记，分析口径仍以 Z 为准。
"""
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DesignRow:
    period: str
    group: str
    arm: str
    arm_label: str
    invite_flag: int
    probability: float
    invite_fraction: float


@dataclass(frozen=True)
class Assignment:
    period: str
    person_id: str
    group: str
    assigned_arm: str
    assigned_invite: int


@dataclass(frozen=True)
class Notification:
    period: str
    person_id: str
    group: str
    notified: int


@dataclass(frozen=True)
class Attendance:
    period: str
    person_id: str
    group: str
    attended: int


@dataclass(frozen=True)
class Edge:
    period: str
    from_person: str
    to_person: str
    pre_treatment: int


@dataclass(frozen=True)
class Budget:
    period: str
    invitations_budget: int


@dataclass(frozen=True)
class TrialData:
    design: list
    persons: list
    assignments: list
    notifications: list
    attendance: list
    edges: list
    budget: list

    @property
    def period(self):
        return self.design[0].period if self.design else ""


def _read(path, fields):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"空文件: {path}")
    missing = fields - set(rows[0])
    if missing:
        raise ValueError(f"{path.name} 缺少字段: {', '.join(sorted(missing))}")
    return rows


def audit_edges(directory):
    """在加载前审计关系边，返回课后新增边清单（这些边永远不进入暴露定义）。"""
    directory = Path(directory)
    rows = _read(directory / "edges.csv",
                 {"period", "from_person", "to_person", "pre_treatment"})
    rejected = [(r["from_person"], r["to_person"]) for r in rows
                if int(r["pre_treatment"]) != 1]
    return rejected


def load_trial(directory, strict_post_treatment=True):
    directory = Path(directory)
    design_rows = _read(directory / "design.csv",
                        {"period", "group", "arm", "arm_label", "invite_flag",
                         "probability", "invite_fraction"})
    design = [DesignRow(r["period"], r["group"], r["arm"], r["arm_label"],
                        int(r["invite_flag"]), float(r["probability"]),
                        float(r["invite_fraction"])) for r in design_rows]
    _validate_design(design)

    person_rows = _read(directory / "persons.csv", {"person_id", "group", "eligible"})
    persons = [(r["person_id"], r["group"]) for r in person_rows]
    person_ids = {pid for pid, _ in persons}

    assignment_rows = _read(directory / "assignments.csv",
                            {"period", "person_id", "group", "assigned_arm", "assigned_invite"})
    assignments = [Assignment(r["period"], r["person_id"], r["group"],
                              r["assigned_arm"], int(r["assigned_invite"]))
                   for r in assignment_rows]
    _validate_assignments(design, assignments)

    notification_rows = _read(directory / "notifications.csv",
                              {"period", "person_id", "group", "notified"})
    notifications = [Notification(r["period"], r["person_id"], r["group"], int(r["notified"]))
                     for r in notification_rows]

    attendance_rows = _read(directory / "attendance.csv",
                            {"period", "person_id", "group", "attended"})
    attendance = [Attendance(r["period"], r["person_id"], r["group"], int(r["attended"]))
                  for r in attendance_rows]

    edge_rows = _read(directory / "edges.csv",
                      {"period", "from_person", "to_person", "pre_treatment"})
    edges = []
    rejected_edges = []
    for r in edge_rows:
        edge = Edge(r["period"], r["from_person"], r["to_person"], int(r["pre_treatment"]))
        # 硬性规则：课后新增关系不能反过来定义入组前同伴
        if edge.pre_treatment != 1:
            if strict_post_treatment:
                raise ValueError(
                    f"拒收课后新增关系边 {edge.from_person} <-> {edge.to_person}："
                    "pre_treatment=0，不得用于定义入组前同伴暴露")
            rejected_edges.append((edge.from_person, edge.to_person))
            continue
        edges.append(edge)

    budget_rows = _read(directory / "budget.csv", {"period", "invitations_budget"})
    budget = [Budget(r["period"], int(r["invitations_budget"])) for r in budget_rows]

    _validate_roster_coverage(person_ids, assignments, notifications, attendance)
    return TrialData(design, persons, assignments, notifications, attendance, edges, budget)


def _validate_design(design):
    by_group = {}
    for row in design:
        by_group.setdefault(row.group, {})[row.arm] = row.probability
    for group, arms in by_group.items():
        total = sum(arms.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"社区 {group} 登记概率之和为 {total}，不等于 1")
    if not by_group:
        raise ValueError("抽签设计为空")


def _validate_assignments(design, assignments):
    legal = {(row.group, row.arm) for row in design}
    realized = {(row.group, row.assigned_arm) for row in assignments}
    illegal = realized - legal
    if illegal:
        raise ValueError(f"实际抽签结果落在登记设计之外: {sorted(illegal)}")
    arms_by_group = {}
    for row in assignments:
        arms_by_group.setdefault(row.group, set()).add(row.assigned_arm)
    for group, arms in arms_by_group.items():
        if len(arms) != 1:
            raise ValueError(f"社区 {group} 内部出现多个方案，违反社区共同受分配")


def _validate_roster_coverage(person_ids, assignments, notifications, attendance):
    for label, records, key in (
        ("分配记录", assignments, lambda r: r.person_id),
        ("通知记录", notifications, lambda r: r.person_id),
        ("到课记录", attendance, lambda r: r.person_id),
    ):
        ids = {key(r) for r in records}
        missing = person_ids - ids
        if missing:
            raise ValueError(f"{label}缺少花名册人员: {sorted(missing)}")
        extra = ids - person_ids
        if extra:
            raise ValueError(f"{label}出现花名册外人员: {sorted(extra)}")


def find_noncompliance(trial):
    """返回实际通知 D 与抽签分配 Z 不一致的人员（不依从只标记，不进分配口径）。"""
    z = {r.person_id: r.assigned_invite for r in trial.assignments}
    return [r.person_id for r in trial.notifications if r.notified != z[r.person_id]]
