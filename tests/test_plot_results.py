"""The README's reason-code chart must aggregate the way RiskExplainer does."""

import pandas as pd

from scripts.plot_results import mean_abs_by_reason
from src.explainer import REASON_GROUPS


def test_dummies_of_one_block_offset_before_taking_abs():
    features = [name for names in REASON_GROUPS.values() for name in names]
    contribs = pd.DataFrame(0.0, index=[0], columns=features)
    contribs[["loan_grade_A", "loan_grade_B"]] = [0.5, -0.5]
    contribs["loan_int_rate"] = -0.3

    impact = mean_abs_by_reason(contribs)

    assert impact["loan_grade"] == 0.0
    assert impact["interest_rate"] == 0.3
