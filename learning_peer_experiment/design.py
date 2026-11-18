"""登记抽签设计、入组前网络与暴露支持度。

口径纪律：
- 暴露只能由抽签分配 Z（含登记的方案邀请比例）与 pre_treatment=1 的关系边定义；
  实际通知 D、课后新增边都不进入暴露定义。
- 方案的组内邀请规则按登记的 invite_fraction 取组内前 k 人（花名册顺序），
  这是支持度计算所依赖的唯一附加假设，结果中会显式声明。
- 每条边分为：组内 / 跨抽签边界且两端在试验内 / 跨出登记设计边界。
- 支持度区分“设计概率支持”与“本轮实际出现”：设计为正但本轮为空的格子是证据缺口，
  不是效果为零。
"""
from dataclasses import dataclass, field
from itertools import product

WITHIN = "组内"
CROSS_INTERNAL = "跨社区-试验内"
CROSS_BOUNDARY = "跨出登记设计边界"


@dataclass(frozen=True)
class Cell:
    own_invited: int
    peer_invited: int

    @property
    def name(self):
        return f"E{self.own_invited * 2 + self.peer_invited}"


E0 = Cell(0, 0)   # 本人未受邀，同伴也无人受邀
E1 = Cell(0, 1)   # 本人未受邀，至少一名同伴受邀（同伴影响）
E2 = Cell(1, 0)   # 本人受邀，同伴无人受邀（纯本人邀请）
E3 = Cell(1, 1)   # 本人受邀且同伴也受邀（联合暴露）
ALL_CELLS = (E0, E1, E2, E3)


@dataclass
class EdgeAudit:
    classifications: list = field(default_factory=list)
    boundary_persons: set = field(default_factory=set)
    cross_internal_pairs: list = field(default_factory=list)


class DesignModel:
    def __init__(self, trial):
        self.trial = trial
        self.group_of = dict(trial.persons)
        self.registered_groups = sorted({g for _, g in trial.persons})
        self.members = {g: [pid for pid, gg in trial.persons if gg == g]
                        for g in self.registered_groups}
        self.arm_probs = {}
        self.fractions = {}
        for row in trial.design:
            self.arm_probs.setdefault(row.group, {})[row.arm] = row.probability
            self.fractions.setdefault(row.group, {})[row.arm] = row.invite_fraction
        self.realized_arm = {}
        for row in trial.assignments:
            self.realized_arm.setdefault(row.group, row.assigned_arm)
        self.adjacency, self.edge_audit = self._build_network(trial.edges)

    # ---- 网络 ----
    def _build_network(self, edges):
        adjacency = {pid: set() for pid in self.group_of}
        audit = EdgeAudit()
        for edge in edges:
            a, b = edge.from_person, edge.to_person
            ga, gb = self.group_of.get(a), self.group_of.get(b)
            if ga is not None and gb is not None:
                kind = WITHIN if ga == gb else CROSS_INTERNAL
                adjacency[a].add(b)
                adjacency[b].add(a)
                if kind == CROSS_INTERNAL:
                    audit.cross_internal_pairs.append((a, b))
            else:
                outside = b if ga is not None else a
                inside = a if ga is not None else b
                kind = CROSS_BOUNDARY
                audit.boundary_persons.add(inside)
                # 外部端点不在花名册，不进入可计算暴露的邻接表
                if inside in adjacency:
                    adjacency[inside].add(outside)
            audit.classifications.append({"from_person": a, "to_person": b, "class": kind})
        return adjacency, audit

    # ---- 邀请集合 ----
    def invited_set(self, draw):
        """给定一次抽签（社区->方案），返回被邀请人员集合。"""
        invited = set()
        for group, arm in draw.items():
            k = round(len(self.members[group]) * self.fractions[group][arm])
            invited.update(self.members[group][:k])
        return invited

    def person_cell(self, person_id, invited):
        own = int(person_id in invited)
        peers = self.adjacency.get(person_id, set())
        peer_invited = int(any(p in invited for p in peers))
        return Cell(own, peer_invited)

    def person_cell_internal_only(self, person_id, invited):
        """同伴只计试验登记内人员的版本，用于跨边界敏感性说明。"""
        own = int(person_id in invited)
        peers = {p for p in self.adjacency.get(person_id, set()) if p in self.group_of}
        peer_invited = int(any(p in invited for p in peers))
        return Cell(own, peer_invited)

    # ---- 设计分布 ----
    def design_draws(self):
        """枚举登记设计下所有抽签结果及其概率。"""
        groups = self.registered_groups
        arm_choices = [sorted(self.arm_probs[g]) for g in groups]
        for arms in product(*arm_choices):
            draw = dict(zip(groups, arms))
            prob = 1.0
            for g, arm in draw.items():
                prob *= self.arm_probs[g][arm]
            yield draw, prob

    def realized_draw(self):
        return dict(self.realized_arm)

    # ---- 支持度 ----
    def support_table(self):
        """返回每个暴露格子的设计期望人数、本轮实际人数及支持判定。"""
        person_ids = [pid for pid, _ in self.trial.persons]
        design_expected = {cell: 0.0 for cell in ALL_CELLS}
        design_person_support = {pid: {cell: 0.0 for cell in ALL_CELLS} for pid in person_ids}
        for draw, prob in self.design_draws():
            invited = self.invited_set(draw)
            for pid in person_ids:
                cell = self.person_cell(pid, invited)
                design_expected[cell] += prob
                design_person_support[pid][cell] += prob

        realized_invited = self.invited_set(self.realized_draw())
        realized_counts = {cell: 0 for cell in ALL_CELLS}
        realized_members = {cell: [] for cell in ALL_CELLS}
        for pid in person_ids:
            cell = self.person_cell(pid, realized_invited)
            realized_counts[cell] += 1
            realized_members[cell].append(pid)

        rows = []
        for cell in ALL_CELLS:
            rows.append({
                "cell": cell.name,
                "own_invited": cell.own_invited,
                "peer_invited": cell.peer_invited,
                "design_expected_n": round(design_expected[cell], 3),
                "design_supported": design_expected[cell] > 1e-12,
                "realized_n": realized_counts[cell],
                "realized_members": sorted(realized_members[cell]),
                "status": self._cell_status(cell, design_expected[cell], realized_counts[cell]),
            })
        return rows, design_person_support

    @staticmethod
    def _cell_status(cell, expected, realized):
        if expected <= 1e-12:
            return "设计无概率支持：任何抽签下都不会出现，不可估计"
        if realized == 0:
            return "设计支持但本轮未实现：证据缺口，不能解释为效果为零"
        return "设计支持且本轮有实现：可估计"

    def peer_contrast_eligibility(self):
        """同伴暴露对比（非受邀者 E1 vs E0）的可辨认性判定与分母。"""
        rows, person_support = self.support_table()
        by_name = {r["cell"]: r for r in rows}
        eligible_persons = []
        for pid, probs in person_support.items():
            if probs[E0] > 1e-12 and probs[E1] > 1e-12:
                eligible_persons.append(pid)
        result = {
            "contrast": "E1(同伴受邀) - E0(无人受邀)，限定本人未受邀",
            "design_support_both_cells": by_name["E0"]["design_supported"]
                                         and by_name["E1"]["design_supported"],
            "eligible_person_n": len(eligible_persons),
            "eligible_persons": sorted(eligible_persons),
            "realized_e0_n": by_name["E0"]["realized_n"],
            "realized_e1_n": by_name["E1"]["realized_n"],
            "identifiable": (by_name["E0"]["design_supported"]
                             and by_name["E1"]["design_supported"]
                             and by_name["E0"]["realized_n"] > 0
                             and by_name["E1"]["realized_n"] > 0),
            "boundary_persons": sorted(self.edge_audit.boundary_persons),
            "cross_internal_pairs": list(self.edge_audit.cross_internal_pairs),
        }
        result["caveats"] = self._contrast_caveats(result)
        return result

    def _contrast_caveats(self, result):
        caveats = []
        if result["cross_internal_pairs"]:
            caveats.append(
                "存在跨社区、两端均在试验内的边 "
                f"{result['cross_internal_pairs']}：暴露跨越抽签边界，"
                "社区级 ITT 仍是有效政策效果，但把同伴效果归因到单一社区的解释不成立")
        if result["boundary_persons"]:
            caveats.append(
                "存在跨出登记设计边界的边，涉及人员 "
                f"{result['boundary_persons']}：外部端点无抽签概率，"
                "其同伴暴露不可辨认，主分析仅用试验内邻接，相关人员的暴露状态需单独标注")
        caveats.append("组内邀请位置按登记比例取花名册前 k 人，支持度依赖该登记规则")
        return caveats
