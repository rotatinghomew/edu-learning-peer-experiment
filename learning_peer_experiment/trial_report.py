"""同伴带动试验的完整分析报告：登记校验、暴露支持、ITT、同伴对比、社区级决策。"""
import json
import tempfile
from pathlib import Path

from .causal import CausalAnalysis
from .design import DesignModel
from .planning import Planner
from .storage import replace_trial_records
from .trial_records import audit_edges, find_noncompliance, load_trial

TRIAL_DIR = Path(__file__).resolve().parent.parent / "data" / "trial"


def build_pack(trial_dir=TRIAL_DIR):
    rejected_edges = audit_edges(trial_dir)
    # 非严格加载：课后边已被剔除并在结果中留痕，分析只使用 pre_treatment=1 的边
    trial = load_trial(trial_dir, strict_post_treatment=False)
    model = DesignModel(trial)
    causal = CausalAnalysis(model, trial)
    planner = Planner(model, trial, causal)

    support_rows, person_support = model.support_table()
    return {
        "登记与口径校验": {
            "period": trial.period,
            "课后新增边_拒收": [{"from_person": a, "to_person": b} for a, b in rejected_edges],
            "不依从人员_D不等于Z": find_noncompliance(trial),
            "边分类": model.edge_audit.classifications,
            "跨出登记设计边界人员": sorted(model.edge_audit.boundary_persons),
            "抽签单位": model.registered_groups,
        },
        "暴露支持度": {
            "cells": [{k: v for k, v in row.items()} for row in support_rows],
            "说明": "design_expected_n 为登记设计下该格子的期望人数；"
                    "本轮未实现但设计为正的格子是证据缺口，不是效果为零",
        },
        "原分配ITT": causal.itt_by_assignment(),
        "同伴暴露对比": causal.peer_exposure_contrast(),
        "下一轮决策包": planner.build_decision_pack(),
    }


def main():
    with tempfile.TemporaryDirectory() as directory:
        trial = load_trial(TRIAL_DIR, strict_post_treatment=False)
        counts = replace_trial_records(Path(directory) / "trial.db", trial)
    pack = build_pack()
    print(json.dumps(pack, ensure_ascii=False, indent=2, default=str))
    print(f"\n[SQLite 落盘行数] {counts}")


if __name__ == "__main__":
    main()
