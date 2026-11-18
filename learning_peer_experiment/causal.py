"""因果效果：按原分配的 ITT 与设计可辨认时的同伴暴露对比。

两条证据线分开报告，口径互不混用：
1. 原分配 ITT：以社区（抽签单位）的登记方案为处理，社区共同受分配，
   推断使用登记设计下的随机化推断（RI）与加性效应反演置信区间，
   而不是把个人当成独立观测做普通 t 检验。
2. 同伴暴露对比：仅在设计对两个格子都给出正概率且本轮都有实现时可辨认；
   分母、支持范围与不可估计格子逐项写明。缺少可比对象一律记为证据缺口，
   不得解释为没有效果。
"""
import math
from dataclasses import dataclass

from .design import ALL_CELLS, E0, E1


def wilson_interval(successes, n, z=1.96):
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class Outcomes:
    y: dict
    group_of: dict


def build_outcomes(trial):
    y = {r.person_id: r.attended for r in trial.attendance}
    group_of = dict(trial.persons)
    return Outcomes(y, group_of)


def _draw_arm_stat(draw, outcomes, members, arm_a, arm_b):
    """社区等权的方案均值差（社区共同受分配，权重不按人数放大）。"""
    means_a, means_b = [], []
    for group, arm in draw.items():
        ids = members[group]
        mean = sum(outcomes.y[pid] for pid in ids) / len(ids)
        if arm == arm_a:
            means_a.append(mean)
        elif arm == arm_b:
            means_b.append(mean)
    if not means_a or not means_b:
        return None
    return sum(means_a) / len(means_a) - sum(means_b) / len(means_b)


def randomization_p_value(observed_stat, stat_fn, draws, shift_fn=None):
    """登记设计下的加权随机化 p 值；缺臂的抽签结果剔除后重新归一化权重。"""
    numerator, weight_total = 0.0, 0.0
    for draw, prob in draws:
        if shift_fn is not None:
            stat_fn_shift = shift_fn(draw)
        else:
            stat_fn_shift = stat_fn(draw)
        if stat_fn_shift is None:
            continue
        weight_total += prob
        if abs(stat_fn_shift) >= abs(observed_stat) - 1e-12:
            numerator += prob
    if weight_total == 0:
        return None
    return numerator / weight_total


def invert_additive_ci(p_value_at_shift, span=(-0.9, 0.9), grid_step=0.02,
                       tolerance=1e-4):
    """反演加性效应 τ 的双侧随机化 95% 置信区间。

    双侧 p(τ) 呈 U 形：τ 向两端远离 0 时统计量变极端、p 变小，因此存在两个
    p=0.05 的交点。先在网格上定位符号变化，再在括号内二分精修。
    若某一侧在搜索范围内 p 始终大于 0.05，则该侧无界，返回 None 并由调用方标注。
    """
    grid = []
    tau = span[0]
    while tau <= span[1] + 1e-12:
        grid.append(round(tau, 6))
        tau += grid_step

    pvals = []
    for tau in grid:
        p = p_value_at_shift(tau)
        if p is None:
            return None
        pvals.append(p)

    def bisect(lo, hi, rising):
        for _ in range(60):
            mid = (lo + hi) / 2
            p = p_value_at_shift(mid)
            if p is None:
                return None
            if rising:  # 该括号内 p 随 τ 上升
                if p > 0.05:
                    hi = mid
                else:
                    lo = mid
            else:       # 该括号内 p 随 τ 下降
                if p > 0.05:
                    lo = mid
                else:
                    hi = mid
            if hi - lo < tolerance:
                break
        return round((lo + hi) / 2, 4)

    low = high = None
    for i in range(len(grid) - 1):
        pl, pr = pvals[i], pvals[i + 1]
        if pl <= 0.05 < pr or (pl < 0.05 <= pr):
            low = bisect(grid[i], grid[i + 1], rising=True)
        if pl >= 0.05 > pr or (pl > 0.05 >= pr):
            high = bisect(grid[i], grid[i + 1], rising=False)
    return (low, high)


def _ci_boundary_note(ci):
    if ci is None:
        return "区间无法反演"
    low, high = ci
    if low is None and high is None:
        return ("双侧 95% 区间在效应搜索范围内无界：抽签单位过少使随机化分布的最小 "
                "p 值仍大于 0.05，应报告点估计与 p 值，不能声称存在或不存在效果")
    if low is None or high is None:
        return f"区间一侧无界（{ci}）：小样本离散随机化分布限制，缺口侧保留不确定性"
    return "双侧 95% 随机化置信区间（加性效应反演）"


class CausalAnalysis:
    def __init__(self, model, trial):
        self.model = model
        self.trial = trial
        self.outcomes = build_outcomes(trial)
        self.draws = list(model.design_draws())

    def _shift_p_value(self, tau, adjusted_persons, realized_draw, stat_fn):
        """加性尖锐零假设下的随机化 p 值。

        调整向量固定在本轮实际分配上（实际在 a 组/实际在 E1 的人员各减 τ），
        用调整后的结果在登记设计的全部抽签上重算统计量，参考值是调整结果在
        本轮实际抽签上的统计量。
        """
        saved = dict(self.outcomes.y)
        for pid in adjusted_persons:
            self.outcomes.y[pid] = saved[pid] - tau
        reference = stat_fn(realized_draw)
        value = randomization_p_value(reference, stat_fn, self.draws)
        self.outcomes.y = saved
        return value

    # ---------- 1. 按原分配的 ITT ----------
    def itt_by_assignment(self):
        members = self.model.members
        realized = self.model.realized_draw()
        community_rows = []
        for group in self.model.registered_groups:
            ids = members[group]
            n = len(ids)
            successes = sum(self.outcomes.y[pid] for pid in ids)
            wilson = wilson_interval(successes, n)
            community_rows.append({
                "group": group, "arm": realized[group], "n_persons": n,
                "attended": successes, "rate": successes / n,
                "wilson95": tuple(round(v, 4) for v in wilson) if wilson else None,
            })

        contrasts = []
        # 方案对从登记设计动态读取：以登记的对照方案为基准（invite_flag=0），
        # 其余邀请方案逐一对比，再补充邀请方案之间的对比
        arm_labels = {}
        for row in self.trial.design:
            arm_labels.setdefault(row.arm, row.arm_label)
        all_arms = sorted({row.arm for row in self.trial.design})
        control_arms = sorted({row.arm for row in self.trial.design
                               if row.invite_flag == 0})
        base_arm = control_arms[0] if control_arms else all_arms[0]
        invite_arms = [arm for arm in all_arms if arm != base_arm]
        pairs = [(arm, base_arm) for arm in invite_arms]
        pairs += [(invite_arms[i], invite_arms[j])
                  for i in range(len(invite_arms))
                  for j in range(i + 1, len(invite_arms))]
        for arm_a, arm_b in pairs:
            label = f"{arm_labels.get(arm_a, arm_a)} vs {arm_labels.get(arm_b, arm_b)}（原分配 ITT）"
            observed = _draw_arm_stat(realized, self.outcomes, members, arm_a, arm_b)
            if observed is None:
                contrasts.append({"label": label, "estimable": False,
                                  "reason": "本轮缺少对比所需的方案社区"})
                continue
            stat_fn = lambda draw, a=arm_a, b=arm_b: _draw_arm_stat(
                draw, self.outcomes, members, a, b)
            p = randomization_p_value(observed, stat_fn, self.draws)
            n_a_groups = [g for g in realized if realized[g] == arm_a]
            n_b_groups = [g for g in realized if realized[g] == arm_b]
            n_a = sum(len(members[g]) for g in n_a_groups)
            n_b = sum(len(members[g]) for g in n_b_groups)
            # 受限随机化推断的加性 CI：调整向量固定在本轮实际分配上，
            # 再在登记设计的所有抽签上检验尖锐零假设
            adjusted_persons = [pid for g in n_a_groups for pid in members[g]]

            def stat_on_draw(draw, a=arm_a, b=arm_b):
                return _draw_arm_stat(draw, self.outcomes, members, a, b)

            def shifted_p(tau):
                return self._shift_p_value(tau, adjusted_persons,
                                           realized, stat_on_draw)

            ci = invert_additive_ci(shifted_p)
            contrasts.append({
                "label": label,
                "estimable": True,
                "estimate": round(observed, 4),
                "ri_p_value": round(p, 4) if p is not None else None,
                "ri_ci95": ci,
                "ri_ci95_note": _ci_boundary_note(ci),
                "denominator": {
                    "communities": {arm_a: len(n_a_groups), arm_b: len(n_b_groups)},
                    "persons": {arm_a: n_a, arm_b: n_b},
                    "unit": "推断单位为社区（抽签单位），个人数仅作描述性分母",
                },
                "inference": "登记设计全枚举随机化推断；缺臂抽签结果按权重剔除并归一化",
            })
        return {"community_rates": community_rows, "contrasts": contrasts,
                "note": "ITT 是政策效果：包含组内同伴带动与跨边界影响，不把实际通知 D 当作分配"}

    # ---------- 2. 同伴暴露对比 ----------
    def peer_exposure_contrast(self):
        eligibility = self.model.peer_contrast_eligibility()
        support_rows, _ = self.model.support_table()
        realized_invited = self.model.invited_set(self.model.realized_draw())
        person_ids = [pid for pid, _ in self.trial.persons]
        realized_cells = {pid: self.model.person_cell(pid, realized_invited)
                          for pid in person_ids}
        internal_cells = {pid: self.model.person_cell_internal_only(pid, realized_invited)
                          for pid in person_ids}
        boundary_membership_changes = [
            pid for pid in person_ids if realized_cells[pid] != internal_cells[pid]
        ]

        cell_outcomes = {}
        for cell in ALL_CELLS:
            ids = [pid for pid in person_ids if realized_cells[pid] == cell]
            successes = sum(self.outcomes.y[pid] for pid in ids)
            cell_outcomes[cell.name] = {
                "n": len(ids), "attended": successes,
                "rate": successes / len(ids) if ids else None,
                "wilson95": (tuple(round(v, 4) for v in
                                   wilson_interval(successes, len(ids)))
                             if ids else None),
                "members": sorted(ids),
            }

        result = {
            "eligibility": eligibility,
            "cell_outcomes": cell_outcomes,
            "boundary_sensitivity": {
                "persons_whose_cell_changes_when_external_edges_dropped":
                    sorted(boundary_membership_changes),
                "interpretation": "为空表示跨边界边在本轮不改变任何暴露格子，"
                                  "但外部端点无抽签概率的限制仍然存在",
            },
        }

        if not eligibility["identifiable"]:
            result["contrast"] = {
                "estimable": False,
                "reason": "E0/E1 缺少设计概率支持或本轮没有可比对象；这是证据缺口而非零效果",
            }
            result["unsupported_cells"] = self._unsupported_cell_report(support_rows)
            return result

        e1_ids = cell_outcomes["E1"]["members"]
        e0_ids = cell_outcomes["E0"]["members"]
        observed = (cell_outcomes["E1"]["rate"] - cell_outcomes["E0"]["rate"])

        def exposure_stat(draw):
            invited = self.model.invited_set(draw)
            sums = {E1: [0, 0], E0: [0, 0]}
            for pid in person_ids:
                cell = self.model.person_cell(pid, invited)
                if cell in (E0, E1):
                    sums[cell][0] += self.outcomes.y[pid]
                    sums[cell][1] += 1
            if sums[E0][1] == 0 or sums[E1][1] == 0:
                return None
            return sums[E1][0] / sums[E1][1] - sums[E0][0] / sums[E0][1]

        p = randomization_p_value(observed, exposure_stat, self.draws)

        def shifted_p(tau):
            return self._shift_p_value(tau, e1_ids, self.model.realized_draw(),
                                       exposure_stat)

        ci = invert_additive_ci(shifted_p)
        result["contrast"] = {
            "estimable": True,
            "label": "同伴影响：非受邀者 E1 到课率 - E0 到课率",
            "estimate": round(observed, 4),
            "ri_p_value": round(p, 4) if p is not None else None,
            "ri_ci95": ci,
            "ri_ci95_note": _ci_boundary_note(ci),
            "denominator": {"E1_n": len(e1_ids), "E0_n": len(e0_ids),
                            "population": "本人未收到分配邀请的人员，暴露由入组前网络与抽签 Z 决定"},
            "inference": "登记设计全枚举随机化推断（Fisher 尖锐零假设 + 加性效应反演）",
        }
        result["unsupported_cells"] = self._unsupported_cell_report(support_rows)
        return result

    def _unsupported_cell_report(self, support_rows):
        report = []
        for row in support_rows:
            if row["status"].startswith("设计支持但本轮未实现") or \
               row["status"].startswith("设计无概率支持"):
                report.append({"cell": row["cell"], "status": row["status"],
                               "design_expected_n": row["design_expected_n"],
                               "realized_n": row["realized_n"]})
        return report
