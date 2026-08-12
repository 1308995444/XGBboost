from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from catboost import CatBoostClassifier


ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "inference_manifest.json"
MODEL_PATH = ROOT / "final_catboost_inference.cbm"


@st.cache_resource
def load_assets():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    return manifest, model, explainer


MANIFEST, MODEL, EXPLAINER = load_assets()
FEATURES = MANIFEST["features"]
THRESHOLD = float(MANIFEST["frozen_threshold"])


LABELS = {
    "gender": "性别 / Gender",
    "gastric_disease": "胃病 / Gastric disease",
    "pain": "慢性疼痛 / Chronic pain",
    "sleep_duration_night": "夜间睡眠时间（小时） / Sleep duration at night (hours)",
    "retirement_status": "退休状态 / Retirement status",
    "self_rated_health": "自评健康 / Self-rated health",
    "adl_disability": "日常生活活动障碍 / ADL disability",
    "future_hope": "对未来的希望 / Future hope",
    "life_satisfaction": "生活满意度 / Life satisfaction",
    "education_level": "教育程度 / Education level",
    "chronic_disease_count": "慢性病数量 / Number of chronic diseases",
    "hearing_ability": "听力 / Hearing ability",
}

CATEGORY_LABELS = {
    "gender": {0: "0 - 女 / Female", 1: "1 - 男 / Male"},
    "gastric_disease": {0: "0 - 无 / No", 1: "1 - 有 / Yes"},
    "pain": {0: "0 - 无 / No", 1: "1 - 有 / Yes"},
    "retirement_status": {0: "0 - 未退休 / Not retired", 1: "1 - 已退休 / Retired"},
    "self_rated_health": {
        1: "1 - 很差 / Very poor",
        2: "2 - 较差 / Poor",
        3: "3 - 一般 / Fair",
        4: "4 - 较好 / Good",
        5: "5 - 很好 / Very good",
    },
    "adl_disability": {
        0: "0 - 无 / None",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6 - 完全依赖 / Complete dependence",
    },
    "future_hope": {
        1: "1 - 很低 / Very low",
        2: "2 - 较低 / Low",
        3: "3 - 较高 / High",
        4: "4 - 很高 / Very high",
    },
    "life_satisfaction": {
        1: "1 - 非常不满意 / Very dissatisfied",
        2: "2 - 不满意 / Dissatisfied",
        3: "3 - 一般 / Neutral",
        4: "4 - 满意 / Satisfied",
        5: "5 - 非常满意 / Very satisfied",
    },
    "education_level": {
        1: "1 - 小学以下 / Below primary",
        2: "2 - 小学 / Primary",
        3: "3 - 中学 / Secondary",
        4: "4 - 中学以上 / Above secondary",
    },
    "hearing_ability": {
        1: "1 - 很差 / Very poor",
        2: "2 - 较差 / Poor",
        3: "3 - 一般 / Fair",
        4: "4 - 较好 / Good",
        5: "5 - 很好 / Very good",
    },
}


def validate_record(raw_record: dict) -> tuple[dict | None, list[str]]:
    errors = []
    cleaned = {}
    schema = MANIFEST["input_schema"]

    missing = [name for name in FEATURES if raw_record.get(name) in (None, "")]
    if missing:
        errors.append("Missing required inputs / 缺少必填项: " + ", ".join(missing))

    for name in FEATURES:
        value = raw_record.get(name)
        if value in (None, ""):
            continue
        rule = schema[name]
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name}: a numeric value is required / 必须为数值")
            continue
        if not math.isfinite(numeric):
            errors.append(f"{name}: value must be finite / 必须为有限数值")
            continue

        if "allowed_values" in rule:
            if numeric != int(numeric) or int(numeric) not in rule["allowed_values"]:
                errors.append(f"{name}: invalid category / 分类值无效")
            else:
                cleaned[name] = int(numeric)
        elif rule.get("integer"):
            if numeric != int(numeric):
                errors.append(f"{name}: an integer is required / 必须为整数")
            elif not rule["minimum"] <= int(numeric) <= rule["maximum"]:
                errors.append(
                    f"{name}: must be from {rule['minimum']} to {rule['maximum']} / 超出允许范围"
                )
            else:
                cleaned[name] = int(numeric)
        else:
            if not rule["minimum_exclusive"] < numeric < rule["maximum_exclusive"]:
                errors.append(
                    f"{name}: must be >{rule['minimum_exclusive']} and "
                    f"<{rule['maximum_exclusive']} / 超出允许范围"
                )
            else:
                cleaned[name] = numeric

    return (cleaned if not errors else None), errors


def transform_for_inference(record: dict) -> np.ndarray:
    preprocessing = MANIFEST["preprocessing"]
    transformed = []

    for name in preprocessing["categorical_order"]:
        categories = preprocessing["categorical_categories"][name]
        ordinal_value = categories.index(record[name])
        # The final training pipeline used one-hot encoding with drop='first'.
        transformed.extend(float(ordinal_value == index) for index in range(1, len(categories)))

    for name in preprocessing["numerical_order"]:
        mean = float(preprocessing["numerical_mean"][name])
        scale = float(preprocessing["numerical_scale"][name])
        transformed.append((float(record[name]) - mean) / scale)

    array = np.asarray([transformed], dtype=float)
    if array.shape[1] != int(MANIFEST["transformed_feature_count"]):
        raise RuntimeError("The inference feature contract does not match the frozen model.")
    return array


def aggregate_shap_values(shap_values: np.ndarray) -> pd.DataFrame:
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, 1]
    if values.ndim == 2:
        values = values[0]
    if values.ndim != 1:
        raise RuntimeError(f"Unexpected SHAP value shape: {values.shape}")

    rows = []
    for name in FEATURES:
        indices = MANIFEST["shap_feature_groups"][name]
        contribution = float(values[indices].sum())
        rows.append(
            {
                "Feature": LABELS[name],
                "SHAP contribution": contribution,
                "Absolute contribution": abs(contribution),
            }
        )
    return pd.DataFrame(rows).sort_values("Absolute contribution", ascending=True)


st.set_page_config(page_title="Current depressive-symptom screening", layout="centered")
st.title("Current Depressive-Symptom Screening Model")
st.subheader("当前抑郁症状筛查模型（仅供研究使用）")
st.caption(
    f"Model release {MANIFEST['release_version']} | Frozen OOF Youden threshold: {THRESHOLD:.6f}"
)

st.info(
    """
**隐私与数据处理 / Privacy and data handling**

本应用代码不会将输入写入文件或数据库，也不会主动保留输入。输入仅用于当前会话计算。
请勿输入姓名、身份证号、联系方式或其他可直接识别个人身份的信息。托管平台的数据处理仍受其平台政策约束。

The application code does not write inputs to files or databases and does not intentionally
retain them beyond the active session. Do not enter names, identification numbers, contact
details, or other directly identifiable information. Hosting-platform processing remains
subject to the platform's policies.
"""
)

st.warning(
    """
**免责声明 / Disclaimer**

本工具仅估计**当前抑郁症状**的筛查概率，不是抑郁障碍诊断，也不预测未来发病。
本模型尚未在独立机构或国家进行外部验证，也尚未专门验证农村人群。结果不能替代 CESD-10
确认性评估、专业临床判断或治疗决策。

This tool estimates a screening probability for **current depressive symptoms**. It is not a
diagnosis of depressive disorder and does not predict future onset. The model has not undergone
independent external validation in another institution or country and has not been specifically
validated for rural populations. Results do not replace confirmatory CESD-10 assessment,
professional clinical judgement, or treatment decisions.
"""
)

st.header("Enter all 12 required values / 请填写全部12项必填信息")

raw_record = {}
with st.form("prediction_form", clear_on_submit=False):
    for name in FEATURES:
        rule = MANIFEST["input_schema"][name]
        if "allowed_values" in rule:
            raw_record[name] = st.selectbox(
                LABELS[name],
                options=rule["allowed_values"],
                index=None,
                placeholder="Select a value / 请选择",
                format_func=lambda value, feature=name: CATEGORY_LABELS[feature][value],
                key=f"input_{name}",
            )
        elif rule.get("integer"):
            raw_record[name] = st.text_input(
                LABELS[name],
                value="",
                placeholder=f"Integer {rule['minimum']}-{rule['maximum']} / 请输入整数",
                key=f"input_{name}",
            )
        else:
            raw_record[name] = st.text_input(
                LABELS[name],
                value="",
                placeholder=(
                    f">{rule['minimum_exclusive']} and <{rule['maximum_exclusive']} "
                    "/ 请输入范围内数值"
                ),
                key=f"input_{name}",
            )

    submitted = st.form_submit_button("Calculate screening probability / 计算筛查概率")

if submitted:
    record, validation_errors = validate_record(raw_record)
    if validation_errors:
        st.error("Please correct the following inputs / 请更正以下输入：")
        for error in validation_errors:
            st.write(f"- {error}")
    else:
        transformed = transform_for_inference(record)
        positive_probability = float(MODEL.predict_proba(transformed)[0, 1])
        above_threshold = positive_probability >= THRESHOLD

        st.metric(
            "Probability of current depressive symptoms / 当前抑郁症状概率",
            f"{positive_probability * 100:.2f}%",
        )
        if above_threshold:
            st.warning(
                "Above the frozen research threshold; confirmatory CESD-10 assessment is required. "
                "/ 高于冻结研究阈值，需进一步完成 CESD-10 确认性评估。"
            )
        else:
            st.success(
                "Below the frozen research threshold. This does not exclude depressive symptoms. "
                "/ 低于冻结研究阈值，但不能据此排除抑郁症状。"
            )

        st.subheader("Model-attribution summary / 模型归因摘要")
        shap_values = EXPLAINER.shap_values(transformed)
        shap_table = aggregate_shap_values(shap_values)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        colors = ["#3B528B" if value < 0 else "#5DC863" for value in shap_table["SHAP contribution"]]
        ax.barh(shap_table["Feature"], shap_table["SHAP contribution"], color=colors)
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_xlabel("SHAP contribution to model output")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "SHAP values describe attribution within the fitted model. They do not establish "
            "causal effects or identify treatment targets. / SHAP值仅解释模型归因，不代表因果效应。"
        )

