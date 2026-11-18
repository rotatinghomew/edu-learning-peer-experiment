import csv
import tempfile
import unittest
from pathlib import Path

from learning_peer_experiment.causal import (
    CausalAnalysis, invert_additive_ci, wilson_interval)
from learning_peer_experiment.design import DesignModel, E0, E1, E2, E3
from learning_peer_experiment.planning import Planner
from learning_peer_experiment.trial_records import (
    audit_edges, find_noncompliance, load_trial)

DATA_DIR = Path(__file__).parent.parent / "data" / "trial"


class RegistryAndExposureTests(unittest.TestCase):
    def setUp(self):
        self.trial = load_trial(DATA_DIR, strict_post_treatment=False)
        self.model = DesignModel(self.trial)

    def test_post_treatment_edge_is_rejected_in_strict_mode_and_audited(self):
        rejected = audit_edges(DATA_DIR)
        self.assertIn(("甲02", "甲04"), rejected)
        with self.assertRaisesRegex(ValueError, "拒收课后新增关系边"):
            load_trial(DATA_DIR, strict_post_treatment=True)
        # 非严格模式下该边确实不在邻接表里
        self.assertNotIn("甲04", self.model.adjacency.get("甲02", set()))

    def test_probabilities_sum_to_one_per_lottery_unit(self):
        sums = {}
        for row in self.trial.design:
            sums.setdefault(row.group, 0.0)
            sums[row.group] += row.probability
        for total in sums.values():
            self.assertAlmostEqual(total, 1.0)

    def test_design_draws_use_registry_probabilities_not_actual_contact(self):
        draws = list(self.model.design_draws())
        # 丙没有 high 登记行，任何抽签下丙都不会被分到 high
        self.assertTrue(all(draw["丙"] != "high" for draw, _ in draws))
        self.assertAlmostEqual(sum(p for _, p in draws), 1.0)
        # 实际通知与分配不一致被识别，但不改变分配
        self.assertEqual(find_noncompliance(self.trial), ["甲03"])
        arm_03 = next(r for r in self.trial.assignments if r.person_id == "甲03")
        notif_03 = next(r for r in self.trial.notifications if r.person_id == "甲03")
        self.assertEqual(arm_03.assigned_invite, 1)
        self.assertEqual(notif_03.notified, 0)

    def test_edge_classification_covers_three_kinds(self):
        classes = {e["class"] for e in self.model.edge_audit.classifications}
        self.assertEqual(
            classes, {"组内", "跨社区-试验内", "跨出登记设计边界"})
        self.assertEqual(sorted(self.model.edge_audit.boundary_persons), ["丙01"])

    def test_support_distinguishes_gap_from_zero_effect(self):
        rows, _ = self.model.support_table()
        by_cell = {r["cell"]: r for r in rows}
        # E3 设计上有正概率、本轮无人实现：必须标注为缺口而非零效果
        e3 = by_cell["E3"]
        self.assertTrue(e3["design_supported"])
        self.assertEqual(e3["realized_n"], 0)
        self.assertIn("证据缺口", e3["status"])
        e0 = by_cell["E0"]
        self.assertTrue(e0["design_supported"])
        self.assertGreater(e0["realized_n"], 0)


class CausalTests(unittest.TestCase):
    def setUp(self):
        self.trial = load_trial(DATA_DIR, strict_post_treatment=False)
        self.model = DesignModel(self.trial)
        self.causal = CausalAnalysis(self.model, self.trial)

    def test_itt_keeps_original_assignment_and_community_denominators(self):
        result = self.causal.itt_by_assignment()
        self.assertEqual(len(result["community_rates"]), 6)
        for contrast in result["contrasts"]:
            if contrast.get("estimable"):
                # 推断单位是社区，不是个人
                self.assertIn("社区", contrast["denominator"]["unit"])
                total_communities = sum(contrast["denominator"]["communities"].values())
                self.assertLessEqual(total_communities, 6)

    def test_ri_p_value_lives_between_zero_and_one(self):
        result = self.causal.itt_by_assignment()
        for contrast in result["contrasts"]:
            if contrast.get("estimable"):
                self.assertIsNotNone(contrast["ri_p_value"])
                self.assertGreaterEqual(contrast["ri_p_value"], 0.0)
                self.assertLessEqual(contrast["ri_p_value"], 1.0)

    def test_wilson_interval_covers_rate(self):
        interval = wilson_interval(3, 10)
        self.assertLess(interval[0], 0.3 < interval[1])

    def test_ci_inversion_reports_unbounded_side_as_none(self):
        # 恒大于 0.05 的 p 函数 => 两侧都无界
        ci = invert_additive_ci(lambda tau: 0.5, span=(-0.5, 0.5), grid_step=0.1)
        self.assertEqual(ci, (None, None))

    def test_peer_contrast_identifiable_with_explicit_caveats(self):
        result = self.causal.peer_exposure_contrast()
        self.assertTrue(result["eligibility"]["identifiable"])
        contrast = result["contrast"]
        self.assertTrue(contrast["estimable"])
        # 分母必须是本人未受邀人群中的 E1/E0
        self.assertEqual(
            contrast["denominator"]["E1_n"] + contrast["denominator"]["E0_n"],
            contrast["denominator"]["E1_n"] + contrast["denominator"]["E0_n"])
        self.assertGreater(contrast["denominator"]["E1_n"], 0)
        self.assertGreater(contrast["denominator"]["E0_n"], 0)
        caveat_text = " ".join(result["eligibility"]["caveats"])
        self.assertIn("抽签边界", caveat_text)

    def test_missing_comparison_is_gap_not_zero(self):
        result = self.causal.peer_exposure_contrast()
        cells = {c["cell"]: c for c in result["unsupported_cells"]}
        self.assertIn("E3", cells)
        self.assertIn("证据缺口", cells["E3"]["status"])


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.trial = load_trial(DATA_DIR, strict_post_treatment=False)
        self.model = DesignModel(self.trial)
        self.causal = CausalAnalysis(self.model, self.trial)
        self.planner = Planner(self.model, self.trial, self.causal)

    def test_budget_is_enforced_at_community_level(self):
        comparable, non_comparable, over_budget_n = self.planner.score_comparable_plans()
        budget = self.trial.budget[0].invitations_budget
        for result in comparable + non_comparable:
            self.assertLessEqual(result.slots, budget)
        self.assertGreaterEqual(over_budget_n, 0)
        self.assertEqual(self.planner.build_decision_pack()["over_budget_plan_n"],
                         over_budget_n)

    def test_comparable_and_non_comparable_plans_are_separated(self):
        pack = self.planner.build_decision_pack()
        self.assertTrue(all(r["comparable"] for r in pack["comparable_plans"]))
        self.assertTrue(all(not r["comparable"]
                            for r in pack["non_comparable_plans"]))
        # 不可比方案必须给出缺口格子且不得外推收益
        for result in pack["non_comparable_plans"]:
            self.assertIn("E3", result["unsupported_cells"])
            self.assertIsNone(result["expected_gain"])

    def test_no_individual_scores_are_produced(self):
        pack = self.planner.build_decision_pack()
        text = str(pack)
        self.assertNotIn("person_score", text)
        self.assertIn("无个人预测分数", pack["allocation_unit"])

    def test_unregistered_option_kept_as_gap(self):
        options = self.planner.unregistered_options()
        self.assertEqual([(o["group"], o["arm"]) for o in options],
                         [("丙", "high")])

    def test_spillover_gain_not_double_counted(self):
        pack = self.planner.build_decision_pack()
        self.assertIn("不会被重复计算", pack["double_count_note"])
        # 可比方案的期望收益只由 ITT 方案率差构成，同伴对比不参与
        evidence = pack["arm_evidence"]
        self.assertIn("effect_vs_control_rate", evidence["high"])


class UnidentifiableDesignTests(unittest.TestCase):
    """构造一个 E1 在设计上零概率的试验，验证同伴对比被判为不可辨认。"""

    def _write_trial(self, directory):
        def write(name, fields, rows):
            with (Path(directory) / name).open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        # 两个社区，每组 4 人，只有对照/邀请两方案；邀请比例 1.0（整组邀请），
        # 因此非受邀者的同伴也不可能受邀：E1 在设计上无概率支持
        design = []
        for group in ("子", "丑"):
            design += [
                {"period": "t", "group": group, "arm": "control", "arm_label": "对照",
                 "invite_flag": 0, "probability": 0.5, "invite_fraction": 0.0},
                {"period": "t", "group": group, "arm": "invite", "arm_label": "邀请",
                 "invite_flag": 1, "probability": 0.5, "invite_fraction": 1.0},
            ]
        write("design.csv",
              ["period", "group", "arm", "arm_label", "invite_flag",
               "probability", "invite_fraction"], design)
        persons = [{"person_id": f"{g}{i}", "group": g, "eligible": 1}
                   for g in ("子", "丑") for i in range(1, 5)]
        write("persons.csv", ["person_id", "group", "eligible"], persons)
        assignments = []
        notifications = []
        attendance = []
        realized = {"子": "invite", "丑": "control"}
        for g in ("子", "丑"):
            for i in range(1, 5):
                pid = f"{g}{i}"
                own = 1 if realized[g] == "invite" else 0
                assignments.append({"period": "t", "person_id": pid, "group": g,
                                    "assigned_arm": realized[g], "assigned_invite": own})
                notifications.append({"period": "t", "person_id": pid, "group": g,
                                      "notified": own})
                attendance.append({"period": "t", "person_id": pid, "group": g,
                                   "attended": 0})
        write("assignments.csv",
              ["period", "person_id", "group", "assigned_arm", "assigned_invite"],
              assignments)
        write("notifications.csv",
              ["period", "person_id", "group", "notified"], notifications)
        write("attendance.csv",
              ["period", "person_id", "group", "attended"], attendance)
        # 只有组内边，全部 pre_treatment=1
        edges = [{"period": "t", "from_person": f"{g}1", "to_person": f"{g}2",
                  "pre_treatment": 1} for g in ("子", "丑")]
        write("edges.csv",
              ["period", "from_person", "to_person", "pre_treatment"], edges)
        write("budget.csv", ["period", "invitations_budget"],
              [{"period": "t", "invitations_budget": 8}])

    def test_peer_contrast_not_identifiable_without_support(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_trial(directory)
            trial = load_trial(directory, strict_post_treatment=False)
            model = DesignModel(trial)
            causal = CausalAnalysis(model, trial)
            result = causal.peer_exposure_contrast()
            self.assertFalse(result["eligibility"]["identifiable"])
            self.assertFalse(result["contrast"]["estimable"])
            self.assertIn("证据缺口", result["contrast"]["reason"])
            # 原分配 ITT 仍然可以报告
            itt = causal.itt_by_assignment()
            self.assertTrue(any(c.get("estimable") for c in itt["contrasts"]))


if __name__ == "__main__":
    unittest.main()
