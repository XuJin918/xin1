# 导入核心库
import streamlit as st  
import joblib  
import numpy as np  
import pandas as pd  
import shap  
import matplotlib.pyplot as plt  
from lime.lime_tabular import LimeTabularExplainer  
import warnings
warnings.filterwarnings('ignore')

# ===================== 1. 基础配置 =====================
# 加载训练好的随机森林模型（确保RF.pkl与脚本同目录）
model = joblib.load('RF.pkl')  

# 加载测试数据（用于LIME解释器，确保X_test.csv与脚本同目录）
X_test = pd.read_csv('X_test.csv')  

# 定义特征名称（替换为业务相关列名，与编码规则对应）
feature_names = [
    "硬的食物", "睡眠时长", "心理咨询", "洗手扶手", "多药",
    "安全警示", "是否住院", "经济", "PHQ", "锻炼次数",
    "ACEzong", "教育程度", "健身区", "童年健康", "童年经济"
]  

# ===================== 2. Streamlit页面配置 =====================
st.set_page_config(page_title="衰弱风险预测器", layout="wide")
st.title("衰弱风险预测器")  
st.markdown("### 请填写以下信息，点击预测获取衰弱风险评估结果")

# ===================== 3. 特征输入组件（按编码规则设计） =====================
# 1. 硬的食物（0：完全没问题，1：有问题）
hard_food = st.selectbox(
    "硬的食物食用情况：",
    options=[0, 1],
    format_func=lambda x: "完全没问题" if x == 0 else "有问题"
)

# 2. 睡眠时长（0：正常，1：异常）
sleep_hours = st.selectbox(
    "睡眠时长：",
    options=[0, 1],
    format_func=lambda x: "正常" if x == 0 else "异常"
)

# 3. 心理咨询（0：否，1：是）
psychological_counseling = st.selectbox(
    "是否接受心理咨询：",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)

# 4. 洗手扶手（0：无，1：有）
handrail = st.selectbox(
    "是否有洗手扶手：",
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

# 5. 多药（0：否，1：是）
multiple_drugs = st.selectbox(
    "是否服用多种药物：",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)

# 6. 安全警示（0：否，1：是）
safety_warning = st.selectbox(
    "是否有安全警示风险：",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)

# 7. 是否住院（0：否，1：是）
hospitalization = st.selectbox(
    "是否有住院史：",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)

# 8. 经济状况（0：贫困，1：非贫困）
economy = st.selectbox(
    "经济状况：",
    options=[0, 1],
    format_func=lambda x: "贫困" if x == 0 else "非贫困"
)

# 9. PHQ（0：否，1：是）
phq = st.selectbox(
    "PHQ量表评估结果：",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)

# 10. 锻炼次数（0：无体育锻炼，1：有体育锻炼）
exercise_times = st.selectbox(
    "是否有体育锻炼：",
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

# 11. ACEzong（0：否，1：是）
acezong = st.selectbox(
    "ACEzong评估结果：",
    options=[0, 1],
    format_func=lambda x: "否" if x == 0 else "是"
)

# 12. 教育程度（0：小学及以下，1：初中及以上）
education = st.selectbox(
    "教育程度：",
    options=[0, 1],
    format_func=lambda x: "小学及以下" if x == 0 else "初中及以上"
)

# 13. 健身区（0：无，1：有）
fitness_area = st.selectbox(
    "是否有健身区：",
    options=[0, 1],
    format_func=lambda x: "无" if x == 0 else "有"
)

# 14. 童年健康（0：不差，1：差）
childhood_health = st.selectbox(
    "童年健康状况：",
    options=[0, 1],
    format_func=lambda x: "不差" if x == 0 else "差"
)

# 15. 童年经济（0：非贫困，1：贫困）
childhood_economy = st.selectbox(
    "童年经济状况：",
    options=[0, 1],
    format_func=lambda x: "非贫困" if x == 0 else "贫困"
)

# ===================== 4. 数据处理与预测 =====================
# 整合用户输入特征
feature_values = [
    hard_food, sleep_hours, psychological_counseling, handrail, multiple_drugs,
    safety_warning, hospitalization, economy, phq, exercise_times,
    acezong, education, fitness_area, childhood_health, childhood_economy
]
# 转换为模型输入格式
features = np.array([feature_values])  

# 预测按钮逻辑
if st.button("预测"):
    # 模型预测
    predicted_class = model.predict(features)[0]  # 0：低风险，1：高风险
    predicted_proba = model.predict_proba(features)[0]  # 概率值

    # 显示预测结果（中文适配）
    st.subheader("📊 预测结果")
    risk_label = "高风险" if predicted_class == 1 else "低风险"
    st.write(f"**衰弱风险等级：{predicted_class}（{risk_label}）**")
    st.write(f"**风险概率：** 低风险概率 {predicted_proba[0]:.2%} | 高风险概率 {predicted_proba[1]:.2%}")

    # 生成个性化建议（中文）
    st.subheader("💡 健康建议")
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"模型预测您的衰弱风险为高风险（概率{probability:.1f}%）。"
            "建议尽快前往医疗机构进行全面的衰弱评估，重点关注营养摄入（如硬食食用困难）、睡眠质量、心理健康（PHQ评估）等方面，"
            "同时可根据自身情况增加适宜的体育锻炼，改善生活环境（如加装洗手扶手）。"
        )
    else:
        advice = (
            f"模型预测您的衰弱风险为低风险（概率{probability:.1f}%）。"
            "建议保持现有健康生活方式，定期进行健康体检，关注童年健康/经济等潜在影响因素，"
            "持续维持规律锻炼和良好的经济、睡眠状况。"
        )
    st.write(advice)

    # ===================== 5. SHAP解释（修复matplotlib报错，适配Streamlit无IPython环境） =====================
    st.subheader("🔍 SHAP特征贡献解释")
    explainer_shap = shap.TreeExplainer(model)
    # 计算SHAP值（适配分类模型）
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 移除shap.initjs()，改用新版SHAP API生成force plot HTML
    base_value = explainer_shap.expected_value[predicted_class]
    shap_val = shap_values[predicted_class]

    # 直接生成force plot的HTML（无需initjs）
    force_plot = shap.plots.force(
        base_value=base_value,
        shap_values=shap_val,
        features=pd.DataFrame([feature_values], columns=feature_names),
        feature_names=feature_names,
        out_names="衰弱风险" if predicted_class == 1 else "无衰弱风险",
        show=False  # 不立即显示，生成HTML对象
    )

    # 转换为HTML字符串并在Streamlit中显示
    shap_html = force_plot.html()
    st.components.v1.html(f"<div>{shap_html}</div>", height=300, scrolling=True)
    
    # ===================== 6. LIME解释（适配业务特征） =====================
    st.subheader("🔍 LIME特征贡献解释")
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=feature_names,
        class_names=['低衰弱风险', '高衰弱风险'],  # 适配业务类别
        mode='classification'
    )
    # 生成LIME解释
    lime_exp = lime_explainer.explain_instance(
        data_row=features.flatten(),
        predict_fn=model.predict_proba,
        num_features=10  # 显示前10个重要特征
    )
    # 显示LIME解释（HTML格式）
    lime_html = lime_exp.as_html(show_table=True)
    st.components.v1.html(lime_html, height=600, scrolling=True)