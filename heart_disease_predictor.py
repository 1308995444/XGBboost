import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import catboost
from matplotlib import font_manager

# 模型加载
model = joblib.load('cat.pkl')

# 特征定义
feature_ranges = {
    "gender": {
        "type": "categorical",
        "options": [0, 1],
        "desc": "性别 / Gender (0: 女 / Female, 1: 男 / Male)"
    },
    "gastric_disease": {
        "type": "categorical",
        "options": [0, 1],
        "desc": "胃病 / Gastric disease (0: 无 / No, 1: 有 / Yes)"
    },
    "pain": {
        "type": "categorical",
        "options": [0, 1],
        "desc": "慢性疼痛 / Chronic pain (0: 无 / No, 1: 有 / Yes)"
    },
    "sleep_duration_night": {
        "type": "numerical",
        "min": 0.0,
        "max": 24.0,
        "default": 8.0,
        "desc": "睡眠时间 / Sleep duration at night (hours)",
        "step": 0.1,
        "format": "%.1f"
    },
    "retirement_status": {
        "type": "categorical",
        "options": [0, 1],
        "desc": "退休状态 / Retirement status (0: 未退休 / Not retired, 1: 已退休 / Retired)"
    },
    "self_rated_health": {
        "type": "categorical",
        "options": [1, 2, 3, 4, 5],
        "desc": "自评健康 / Self-rated health (1-5: 很差 / Very poor 到 很好 / Very good)"
    },
    "adl_disability": {
        "type": "categorical",
        "options": [0, 1, 2, 3, 4, 5, 6],
        "desc": "日常生活活动障碍 / ADL disability (0-6: 无 / None 到 完全依赖 / Complete dependence)"
    },
    "future_hope": {
        "type": "categorical",
        "options": [1, 2, 3, 4],
        "desc": "对未来的希望 / Future hope (1-4: 很低 / Very low 到 很高 / Very high)"
    },
    "life_satisfaction": {
        "type": "categorical",
        "options": [1, 2, 3, 4, 5],
        "desc": "生活满意度 / Life satisfaction (1-5: 非常不满意 / Very dissatisfied 到 非常满意 / Very satisfied)"
    },
    "education_level": {
        "type": "categorical",
        "options": [1, 2, 3, 4],
        "desc": "教育程度 / Education level (1: 小学以下 / Below Primary, 2: 小学 / Primary, 3: 中学 / Secondary, 4: 中学以上 / Above Secondary)"
    },
    "chronic_disease_count": {
        "type": "numerical",
        "min": 0.0,
        "max": 50.0,
        "default": 0.0,
        "desc": "慢性病数量 / Number of chronic diseases (integer)",
        "step": 1.0,
        "format": "%.0f"
    },
    "hearing_ability": {
        "type": "categorical",
        "options": [1, 2, 3, 4, 5],
        "desc": "听力 / Hearing ability (1-5: 很差 / Very poor 到 很好 / Very good)"
    }
}

# 界面布局
st.title("Depression Risk-Prediction Model with SHAP Visualization")
st.subheader("抑郁风险预测模型与 SHAP 可视化")

st.info(
    """
**隐私与数据保存政策 / Privacy & Data Retention Policy**

我们澄清：本工具运行过程中**不保存用户输入数据**。  
预测所需数据仅用于**当次会话计算**，不会进行长期存储、数据库写入、日志留存或后续复用。  
请勿输入可直接识别个人身份的敏感信息。

We clarify that this tool **does not store user input data** during operation.  
The data required for prediction is used **only for the current session computation** and is not subject to long-term storage, database writing, log retention, or subsequent reuse.  
Please do not enter any personally identifiable sensitive information.
    """
)

st.warning(
    """
**免责声明 / Disclaimer**

本工具仅用于**辅助风险提示与初筛参考**，不能替代临床诊断或医生判断。  
最终诊断与治疗决策应由**有资质的临床人员**结合病史、体征、实验室检查及其他必要检查结果综合作出。

This tool is intended **only for auxiliary risk prompting and preliminary screening reference**.  
It cannot replace clinical diagnosis or a physician’s judgment.  
Final diagnosis and treatment decisions should be made by **qualified clinical professionals** based on medical history, physical examination, laboratory tests, and other necessary findings.
    """
)

st.header("Enter the following feature values / 请输入以下特征值：")

# 输入表单
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=properties["desc"],
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            step=float(properties.get("step", 1.0)),
            format=properties.get("format", "%f"),
            key=f"num_{feature}"
        )
    else:
        value = st.selectbox(
            label=properties["desc"],
            options=properties["options"],
            key=f"cat_{feature}"
        )
    feature_values.append(value)

features = np.array([feature_values])

# 预测与解释
if st.button("Predict / 预测"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 结果显示
    risk_label_en = "High risk" if predicted_class == 1 else "Low risk"
    risk_label_cn = "高风险" if predicted_class == 1 else "低风险"

    text_en = f"Predicted probability: {probability:.2f}% ({risk_label_en})"
    text_cn = f"预测概率：{probability:.2f}%（{risk_label_cn}）"

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.text(
        0.5, 0.7,
        text_en + "\n" + text_cn,
        fontsize=14,
        ha='center',
        va='center',
        fontname='Arial'
    )
    ax.axis('off')
    st.pyplot(fig)

    # SHAP解释
    explainer = shap.TreeExplainer(model)
    feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())
    shap_values = explainer.shap_values(feature_df)

    if isinstance(shap_values, list):
        shap_values_class = shap_values[predicted_class][0]
        expected_value = explainer.expected_value[predicted_class]
    else:
        shap_values_class = shap_values[0]
        expected_value = explainer.expected_value

    st.subheader("SHAP Explanation / SHAP 解释")

    plt.figure()
    shap.force_plot(
        expected_value,
        shap_values_class,
        feature_df,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())
