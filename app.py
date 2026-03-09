import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# 设置页面配置
st.set_page_config(
    page_title="Neonatal Health Prediction System",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover { background-color: #45a049; }
    .prediction-result {
        background-color: #e8f5e8;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 4px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 1. 加载模型
#    PyCaret 的 save_model() 本质是 joblib.dump()，
#    保存文件名为 my_best_pipeline0106.pkl
#    直接用 joblib.load() 即可，无需安装 pycaret
# ============================================================
model = None
model_path = 'my_best_pipeline0106.pkl'

if not os.path.exists(model_path):
    st.error(f"❌ 模型文件未找到：{model_path}")
    st.info("请确保 my_best_pipeline0106.pkl 已上传到 GitHub 仓库根目录。")
else:
    try:
        model = joblib.load(model_path)
        st.success("✅ 模型加载成功")
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")

# 2. 页面标题
st.title('👶 Neonatal Health Prediction System')
st.markdown('### Machine Learning-based Neonatal Health Risk Prediction Tool')
st.write('---')

# 3. 输入部分
with st.container():
    st.subheader('📋 Patient Information Input')
    with st.expander('Input Parameter Instructions', expanded=False):
        st.write('Please enter the relevant information about the newborn.')

    col1, col2 = st.columns(2)

    with col1:
        # 特征1: 胎龄
        gestational_age = st.number_input(
            'Gestational Age (weeks) / 胎龄（周）',
            min_value=22.0, max_value=42.0,
            value=36.0, step=0.1,
            help="Newborn's gestational age"
        )

        # 特征2: C反应蛋白水平
        crp = st.number_input(
            'C-Reactive Protein (mg/L) / C反应蛋白水平',
            min_value=0.0, max_value=500.0,
            value=5.0, step=0.1,
            help="Newborn's C-Reactive Protein level"
        )

        # 特征3: 接受气管插管
        endotracheal = st.selectbox(
            'Receipt of Endotracheal Intubation / 接受气管插管',
            [0, 1],
            format_func=lambda x: 'Yes / 是' if x == 1 else 'No / 否',
            help='Whether the newborn received endotracheal intubation'
        )

        # 特征4: 使用儿茶酚胺类药物
        catecholamines = st.selectbox(
            'Use of Catecholamines / 使用儿茶酚胺类药物',
            [0, 1],
            format_func=lambda x: 'Yes / 是' if x == 1 else 'No / 否',
            help='Whether the newborn used catecholamines'
        )

    with col2:
        # 特征5: 5分钟Apgar评分
        apgar_5min = st.number_input(
            'Apgar Score (5 min) / 5分钟Apgar评分',
            min_value=0, max_value=10,
            value=8, step=1,
            help="Newborn's Apgar score at 5 minutes"
        )

# 4. 预测逻辑
st.write('---')
if st.button('🔍 Start Prediction', key='predict_btn'):
    if model is None:
        st.error("模型未加载，无法预测。请检查模型文件是否存在。")
    else:
        try:
            # ✅ 列名与训练数据完全一致（中文）
            input_data = {
                '胎龄':           [gestational_age],
                'C反应蛋白水平':   [crp],
                '接受气管插管':    [endotracheal],
                '使用儿茶酚胺类药物': [catecholamines],
                '5分钟Apgar评分': [apgar_5min],
            }
            input_df = pd.DataFrame(input_data)

            # 预测
            prediction = model.predict(input_df)

            # 获取概率
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(input_df)[0][1] * 100
            else:
                probability = None

            # 显示结果
            st.subheader('📊 Prediction Result')
            if probability is not None:
                risk_level = "⚠️ High Risk / 高风险" if probability >= 50 else "✅ Low Risk / 低风险"
                st.markdown(
                    f"""
                    <div class="prediction-result">
                        <h4>Probability of sIVH: <strong>{probability:.2f}%</strong></h4>
                        <p>Risk Level / 风险等级: <strong>{risk_level}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.progress(int(probability))
            else:
                result_text = 'High Risk / 高风险' if prediction[0] == 1 else 'Low Risk / 低风险'
                st.markdown(
                    f"""
                    <div class="prediction-result">
                        <h4>Prediction Result: <strong>{result_text}</strong></h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"预测出错: {str(e)}")
            st.write(f"Error details: {type(e).__name__}: {str(e)}")

# 5. 说明
st.write('---')
st.subheader('ℹ️ About the System')
st.write('This system is based on machine learning algorithms for neonatal health risk prediction.')
st.write('⚠️ Note: This is an auxiliary tool only and cannot replace professional medical diagnosis.')
