"""下一轮邀请名额的社区级决策。

纪律：
- 名额按社区整体占用：决策单位是“社区 × 登记方案”，不产生任何个人预测分数，
  也不按分数拼名额。
- 效果证据只取按原分配 ITT 的社区等权方案效果（含组内同伴带动等全部政策后果），
  同伴暴露对比只作为机制解释，绝不叠加到方案收益上（避免把受邀者之外的收益重复计入）。
- 凡依赖“试验从未支持的暴露程度”的方案一律列入不可比清单，保留缺口、不做确定外推；
  登记设计之外的方案（概率为 0 的社区方案）只登记为未支持选项，不产生收益数字。
"""
import random
from itertools import product
from dataclasses import dataclass

from .design import ALL_CELLS


@dataclass
class PlanResult:
    plan: dict
    slots: int
    expected_gain: object = None
    ci95: object = None
    feasible: bool = True
    comparable: bool = True
    reasons: list = None
    unsupported_cells: list = None


class Planner:
    def __init__(self, model, trial, causal, bootstrap_reps=2000, seed=20261118):
        self.model = model
        self.trial = trial
        self.causal = causal
        self.bootstrap_reps = bootstrap_reps
        self.rng = random.Random(seed)
        self.budget = trial.budget[0].invitations_budget
        self._arm_evidence = None

    @property
    def base_arm(self):
        """登记设计中 invite_flag=0 的基准（对照）方案。"""
        controls = sorted({row.arm for row in self.trial.design
                           if row.invite_flag == 0})
        if controls:
            return controls[0]
        return sorted({row.arm for row in self.trial.design})[0]

    # ---- 方案级证据（只来自原分配 ITT）----
    def arm_evidence(self):
        if self._arm_evidence is not None:
            return self._arm_evidence
        rates = self.causal.itt_by_assignment()["community_rates"]
        by_arm = {}
        for row in rates:
            by_arm.setdefault(row["arm"], []).append(row)
        base = self.base_arm
        control_mean = sum(r["rate"] for r in by_arm[base]) / len(by_arm[base])
        evidence = {}
        for arm, rows in by_arm.items():
            arm_mean = sum(r["rate"] for r in rows) / len(rows)
            evidence[arm] = {
                "arm": arm,
                "realized_communities": [r["group"] for r in rows],
                "community_mean_rate": round(arm_mean, 4),
                "effect_vs_control_rate": round(arm_mean - control_mean, 4),
                "denominator": f"{len(rows)} 个社区等权（基准 {base} {len(by_arm[base])} 个）",
                "transport_assumption": "向未实现该方案的社区套用此效果需假设社区同质，"
                                        "属外推假设，不是试验内估计",
            }
        self._arm_evidence = evidence
        return evidence

    def slots_for(self, group, arm):
        fraction = self.model.fractions[group][arm]
        return round(len(self.model.members[group]) * fraction)

    # ---- 方案枚举 ----
    def enumerate_plans(self):
        groups = self.model.registered_groups
        choices = [sorted(self.model.arm_probs[g]) for g in groups]
        comparable, non_comparable = [], []
        over_budget_n = 0
        realized_support = self._realized_supported_cells()
        for arms in product(*choices):
            plan = dict(zip(groups, arms))
            slots = sum(self.slots_for(g, plan[g]) for g in groups)
            if slots > self.budget:
                over_budget_n += 1
                continue
            created_cells = self._cells_created_by_plan(plan)
            unsupported = sorted(c.name for c in created_cells if c not in realized_support)
            if unsupported:
                non_comparable.append(PlanResult(
                    plan=plan, slots=slots, comparable=False,
                    reasons=[f"方案会产生本轮没有任何实现对象的暴露格子: {unsupported}，"
                             "其收益无试验支持，保留缺口而不外推"],
                    unsupported_cells=unsupported))
            else:
                comparable.append(PlanResult(plan=plan, slots=slots))
        return comparable, non_comparable, over_budget_n

    def _realized_supported_cells(self):
        rows, _ = self.model.support_table()
        return {next(c for c in ALL_CELLS if c.name == r["cell"])
                for r in rows if r["realized_n"] > 0}

    def _cells_created_by_plan(self, plan):
        invited = self.model.invited_set(plan)
        cells = set()
        for pid, _ in self.trial.persons:
            cells.add(self.model.person_cell(pid, invited))
        return cells

    # ---- 可比方案的期望收益与不确定性 ----
    def score_comparable_plans(self):
        comparable, non_comparable, over_budget_n = self.enumerate_plans()
        rates = self.causal.itt_by_assignment()["community_rates"]
        by_arm = {}
        for row in rates:
            by_arm.setdefault(row["arm"], []).append(row)
        sizes = {g: len(self.model.members[g]) for g in self.model.registered_groups}
        base = self.base_arm
        control_rates = [r["rate"] for r in by_arm[base]]
        arm_rates = {arm: [r["rate"] for r in rows] for arm, rows in by_arm.items()}

        boot_totals = {i: [] for i in range(len(comparable))}
        for _ in range(self.bootstrap_reps):
            boot_control = self._mean(self.rng.choices(control_rates, k=len(control_rates)))
            boot_arm = {arm: self._mean(self.rng.choices(vals, k=len(vals)))
                        for arm, vals in arm_rates.items()}
            for i, result in enumerate(comparable):
                total = 0.0
                for g, arm in result.plan.items():
                    if arm == base:
                        continue
                    total += sizes[g] * (boot_arm[arm] - boot_control)
                boot_totals[i].append(total)

        evidence = self.arm_evidence()
        for i, result in enumerate(comparable):
            total = 0.0
            for g, arm in result.plan.items():
                if arm == base:
                    continue
                total += sizes[g] * evidence[arm]["effect_vs_control_rate"]
            vals = sorted(boot_totals[i])
            result.expected_gain = round(total, 3)
            result.ci95 = (round(vals[int(0.025 * self.bootstrap_reps)], 3),
                           round(vals[int(0.975 * self.bootstrap_reps)], 3))
        comparable.sort(key=lambda r: (r.expected_gain, -r.slots), reverse=True)
        return comparable, non_comparable, over_budget_n

    @staticmethod
    def _mean(values):
        return sum(values) / len(values)

    # ---- 登记设计之外的选项（只登记缺口，不给收益）----
    def unregistered_options(self):
        all_arms = sorted({row.arm for row in self.trial.design})
        options = []
        for group in self.model.registered_groups:
            registered = set(self.model.arm_probs[group])
            for arm in all_arms:
                if arm not in registered:
                    options.append({
                        "group": group, "arm": arm,
                        "status": "该社区登记概率为 0，抽签设计不支持辨认，保留缺口，不估计收益",
                    })
        return options

    def build_decision_pack(self):
        comparable, non_comparable, over_budget_n = self.score_comparable_plans()
        baseline = next((r for r in comparable
                         if all(a == self.base_arm for a in r.plan.values())), None)
        return {
            "budget_invitations": self.budget,
            "over_budget_plan_n": over_budget_n,
            "allocation_unit": "社区整体（社区 × 登记方案），无个人预测分数",
            "arm_evidence": self.arm_evidence(),
            "double_count_note": "方案效果取原分配 ITT，已包含受邀者之外的同伴带动收益；"
                                 "同伴暴露对比（E1-E0）仅作机制证据，不再叠加，"
                                 "故非受邀者收益不会被重复计算",
            "denominator_note": "期望增量到课 = 社区人数 × 社区等权方案率差；"
                                "推断单位为社区，不确定性由社区自助法给出",
            "baseline_plan": self._serialize(baseline) if baseline else None,
            "comparable_plans": [self._serialize(r) for r in comparable],
            "non_comparable_plans": [self._serialize(r) for r in non_comparable],
            "unregistered_options": self.unregistered_options(),
        }

    def query_plans(self, comparable_only=True):
        pack = self.build_decision_pack()
        if comparable_only:
            return pack["comparable_plans"]
        return pack["comparable_plans"], pack["non_comparable_plans"]

    @staticmethod
    def _serialize(result):
        return {
            "plan": dict(result.plan), "slots": result.slots,
            "expected_gain": result.expected_gain, "ci95": result.ci95,
            "comparable": result.comparable,
            "reasons": result.reasons or [],
            "unsupported_cells": result.unsupported_cells or [],
        }
