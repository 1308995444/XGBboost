import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 模型加载 Model Loading
# XGBoost模型路径 Path to XGBoost model
model = joblib.load('XGB.pkl')  

# 特征定义 Feature Definition
# 包含所有特征及其取值范围/选项的字典 
# Dictionary containing all features and their value ranges/options
feature_ranges = {
    # 性别 (1:男/Male, 2:女/Female)
    'gender': {"type": "categorical", "options": [1, 2], "desc": "性别/Gender<br>(1:男/Male, 2:女/Female)"},
    
    # 自评健康 (1-5: 很差/Very poor 到 很好/Very good)
    'srh': {"type": "categorical", "options": [1,2,3,4,5], "desc": "自评健康/Self-rated health"},
    
    # 日常活动能力 (0-6: 无/None 到 完全依赖/Complete dependence)
    'adlab_c': {"type": "categorical", "options": [0,1,2,3,4,5,6], "desc": "日常活动能力/Activities of daily living"},
    
    # 关节炎 (0:无/No, 1:有/Yes)
    'arthre': {"type": "categorical", "options": [0, 1], "desc": "关节炎/Arthritis<br>(0-无/No, 1-有/Yes)"},
    
    # 消化系统问题 (0:无/No, 1:有/Yes) 
    'digeste': {"type": "categorical", "options": [0, 1], "desc": "消化系统问题/Digestive issues<br>(0-无/No, 1-有/Yes)"},
    
    # 退休状态 (0:未退休/Not retired, 1:已退休/Retired)
    'retire': {"type": "categorical", "options": [0, 1], "desc": "退休状态/Retirement status<br>(0-未退休/Not retired, 1-已退休/Retired)"},
    
    # 生活满意度 (1-5: 非常不满意/Very dissatisfied 到 非常满意/Very satisfied)
    'satlife': {"type": "categorical", "options": [1,2,3,4,5], "desc": "生活满意度/Life satisfaction"},
    
    # 睡眠时长 (小时/hours)
    'sleep': {"type": "numerical", "min": 0.000, "max": 24.000, "default": 8.000, "desc": "睡眠时长/Sleep duration"},
    
    # 残疾状况 (0:无/No, 1:有/Yes)
    'disability': {"type": "categorical", "options": [0, 1], "desc": "残疾/Disability<br>(0-无/No, 1-有/Yes)"},
    
    # 互联网使用 (0:不使用/No, 1:使用/Yes)
    'internet': {"type": "categorical", "options": [0, 1], "desc": "互联网使用/Internet use<br>(0-无/No, 1-有/Yes)"},
    
    # 希望程度 (1-4: 很低/Very low 到 很高/Very high)
    'hope': {"type": "categorical", "options": [1,2,3,4], "desc": "希望程度/Hope level"},
    
    # 跌倒史 (0:无/No, 1:有/Yes)
    'fall_down': {"type": "categorical", "options": [0, 1], "desc": "跌倒史/Fall history<br>(0-无/No, 1-有/Yes)"},
    
    # 近距离视力 (1-5: 很差/Very poor 到 很好/Very good)
    'eyesight_close': {"type": "categorical", "options": [1,2,3,4,5], "desc": "视力/Near vision"},
    
    # 听力 (1-5: 很差/Very poor 到 很好/Very good)
    'hear': {"type": "categorical", "options": [1,2,3,4,5], "desc": "听力/Hearing"},
    
    # 教育程度 (1-4: 小学/Primary 到 大学/University)
    'edu': {"type": "categorical", "options": [1,2,3,4],"desc": "教育程度/Education<br>level(1-小学以下/Below Primary, 2-小学/Primary, 3-中学/Secondary, 4-中学以上/Above Secondary)"},
    
    # 养老金 (0:无/No, 1:有/Yes)
    'pension': {"type": "categorical", "options": [0, 1], "desc": "养老保险/Pension<br>(0-无/No, 1-有/Yes)"},
    
    # 慢性疼痛 (0:无/No, 1:有/Yes)
    'pain': {"type": "categorical", "options": [0, 1], "desc": "慢性疼痛/Chronic pain<br>(0-无/No, 1-有/Yes)"}
}

# 界面布局 UI Layout
st.title("Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

# 输入表单 Input Form
feature_values = []
for feature, properties in feature_ranges.items():
    label = f"{properties['desc']} ({feature})"
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{label} ({properties['min']}-{properties['max']})", 
            min_value=float(properties["min"]), 
            max_value=float(properties["max"]), 
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=label,
            options=properties["options"],
        )
    feature_values.append(value)

features = np.array([feature_values])

# 预测与解释 Prediction & Explanation
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 结果显示 Result Display
    text_en = f"Predicted probability: {probability:.2f}% ({'High risk' if predicted_class == 1 else 'Low risk'})"
    text_cn = f"预测概率: {probability:.2f}% ({'高风险' if predicted_class == 1 else '低风险'})"
    
    fig, ax = plt.subplots(figsize=(10,2))
    ax.text(0.5, 0.7, text_en, 
            fontsize=14, ha='center', va='center', fontname='Arial')
    ax.text(0.5, 0.3, text_cn,
            fontsize=14, ha='center', va='center', fontname='SimHei')
    ax.axis('off')
    st.pyplot(fig)

    # SHAP解释 SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))
    
    if isinstance(shap_values, list):
        shap_values_class = shap_values[predicted_class][0]
        expected_value = explainer.expected_value[predicted_class]
    else:
        shap_values_class = shap_values[0]
        expected_value = explainer.expected_value

    feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())

    st.subheader("特征影响分析/Feature Impact Analysis")
    plt.figure()
    shap_plot = shap.force_plot(
        expected_value,
        shap_values_class,
        feature_df,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())