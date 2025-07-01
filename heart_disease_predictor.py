import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 1. 环境预配置
st.set_page_config(layout="wide")
plt.style.use('seaborn-v0_8')
try:
    font_manager.fontManager.addfont('SimHei.ttf')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    pass

# 2. 模型加载优化（带缓存）
@st.cache_resource
def load_model():
    try:
        return joblib.load('XGB.pkl')
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None

model = load_model()
if model is None:
    st.stop()

# 3. 特征定义优化（添加中文标签和单位说明）
feature_config = {
    'gender': {"type": "categorical", "options": [1, 2], "label": "性别", "options_label": ["男", "女"]},
    'srh': {"type": "categorical", "options": [1,2,3,4,5], "label": "自评健康", "options_label": ["很差","差","一般","好","很好"]},
    'adlab_c': {"type": "categorical", "options": [0,1,2,3,4,5,6], "label": "日常活动能力"},
    'sleep': {"type": "numerical", "min": 0, "max": 24, "default": 8, "label": "睡眠时长(小时)", "step": 0.5},
    # 其他特征类似添加...
}

# 4. 界面布局优化
st.title("🧠 心脏病风险预测模型")
st.markdown("""<style>div[data-testid="stNumberInput"] > label {font-weight:bold}</style>""", unsafe_allow_html=True)

# 5. 动态生成输入表单（两列布局）
col1, col2 = st.columns(2)
feature_values = {}

with col1:
    for i, (feature, config) in enumerate(feature_config.items()):
        if i % 2 == 0:
            if config["type"] == "numerical":
                feature_values[feature] = st.number_input(
                    label=config.get("label", feature),
                    min_value=float(config["min"]),
                    max_value=float(config["max"]),
                    value=float(config["default"]),
                    step=float(config.get("step", 1.0))
                )
            else:
                options = config["options"]
                labels = config.get("options_label", [str(x) for x in options])
                selected = st.selectbox(
                    label=config.get("label", feature),
                    options=options,
                    format_func=lambda x: dict(zip(options, labels)).get(x, x)
                )
                feature_values[feature] = selected

with col2:
    for i, (feature, config) in enumerate(feature_config.items()):
        if i % 2 == 1:
            if config["type"] == "numerical":
                feature_values[feature] = st.number_input(
                    label=config.get("label", feature),
                    min_value=float(config["min"]),
                    max_value=float(config["max"]),
                    value=float(config["default"]),
                    step=float(config.get("step", 1.0))
                )
            else:
                options = config["options"]
                labels = config.get("options_label", [str(x) for x in options])
                selected = st.selectbox(
                    label=config.get("label", feature),
                    options=options,
                    format_func=lambda x: dict(zip(options, labels)).get(x, x)
                )
                feature_values[feature] = selected

# 6. 预测与解释优化
if st.button("🚀 开始预测", use_container_width=True):
    with st.spinner("正在计算..."):
        try:
            # 6.1 预测结果
            features = pd.DataFrame([feature_values])
            proba = model.predict_proba(features)[0]
            pred_class = model.predict(features)[0]
            
            # 6.2 结果展示
            st.success(f"预测结果: {'高风险' if pred_class == 1 else '低风险'} (置信度: {max(proba)*100:.1f}%)")
            
            # 6.3 SHAP解释
            st.subheader("特征影响分析")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(features)
            
            # 创建两个独立的图表
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### 各特征贡献度(瀑布图)")
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap_values[0], max_display=15, show=False)
                st.pyplot(fig1)
                plt.close(fig1)
            
            with col2:
                st.write("#### 特征重要性排序")
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                shap.plots.bar(shap_values, max_display=15, show=False)
                st.pyplot(fig2)
                plt.close(fig2)
            
            # 6.4 原始数据展示
            with st.expander("📊 查看原始数据"):
                st.dataframe(features.style.highlight_max(axis=0))
                
        except Exception as e:
            st.error(f"预测过程中发生错误: {str(e)}")