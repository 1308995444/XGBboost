import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 模型加载
model = joblib.load('XGB.pkl')

# 特征定义
feature_ranges = {
    'gender': {"type": "categorical", "options": [1, 2],},
    'srh': {"type": "categorical", "options": [1,2,3,4,5]},
    'adlab_c': {"type": "categorical", "options": [0,1,2,3,4,5,6]},
    'arthre': {"type": "categorical", "options": [0, 1]},
    'digeste': {"type": "categorical", "options": [0, 1]},
    'retire': {"type": "categorical", "options": [0, 1]},
    'satlife': {"type": "categorical", "options": [1,2,3,4,5]},
    'sleep': {"type": "numerical", "min": 0.000, "max": 24.000, "default": 8.000},
    'disability': {"type": "categorical", "options": [0, 1]},
    'internet': {"type": "categorical", "options": [0, 1]},
    'hope': {"type": "categorical", "options": [1,2,3,4]},
    'fall_down': {"type": "categorical", "options": [0, 1]},
    'eyesight_close': {"type": "categorical", "options": [1,2,3,4,5]},
    'hear': {"type": "categorical", "options": [1,2,3,4,5]},
    'edu': {"type": "categorical", "options": [1,2,3,4]},
    'pension': {"type": "categorical", "options": [0, 1]},
    'pain': {"type": "categorical", "options": [0, 1]},
}

# 界面布局
st.title("Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

# 输入表单
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature}({properties['min']} - {properties['max']})", 
            min_value=float(properties["min"]), 
            max_value=float(properties["max"]), 
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)", 
            options=properties["options"],
        )
    feature_values.append(value)

features = np.array([feature_values])

# 预测与解释
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 结果显示
    text = f"Based on feature values, predicted possibility of XGB is {probability:.2f}%" 
    fig, ax = plt.subplots(figsize=(8,1))
    ax.text(0.5, 0.5, text, 
            fontsize=16,
            ha='center', va='center',
            fontname='Times New Roman',
            transform=ax.transAxes)
    ax.axis('off')
    st.pyplot(fig)

    # SHAP解释
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))
    
    if isinstance(shap_values, list):
        shap_values_class = shap_values[predicted_class][0]
        expected_value = explainer.expected_value[predicted_class]
    else:
        shap_values_class = shap_values[0]
        expected_value = explainer.expected_value

    feature_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())

    plt.figure()
    shap_plot = shap.force_plot(
        expected_value,
        shap_values_class,
        feature_df,
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())