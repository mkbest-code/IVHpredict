import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 设置页面配置
st.set_page_config(
    page_title="Neonatal Health Prediction System",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加自定义CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .prediction-result {
        background-color: #e8f5e8;
        border-left: 4px solid #4CAF50;
        padding: 15px;
        border-radius: 4px;
        margin-top: 20px;
    }
    .input-container {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 1. 加载模型
model = None
try:
    from pycaret.classification import load_model
    model = load_model('my_best_pipeline0106')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.write("Please make sure PyCaret is installed and the model file exists.")

# 2. 页面标题
st.title('👶 Neonatal Health Prediction System')
st.markdown('### Machine Learning-based Neonatal Health Risk Prediction Tool')
st.write('---')

# 3. 输入部分
with st.container():
    st.subheader('📋 Patient Information Input')
    with st.expander('Input Parameter Instructions', expanded=False):
        st.write('Please enter the relevant information about the newborn, and the system will predict health risks based on these parameters.')

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            # 特征1: 胎龄
            gestational_age = st.number_input(
                'Gestational Age (weeks) / 胎龄（周）',
                value=36.0,
                step=0.1,
                help="Newborn's gestational age"
            )

            # 特征2: C反应蛋白水平
            crp = st.number_input(
                'C-Reactive Protein (mg/L) / C反应蛋白水平',
                value=5.0,
                step=0.1,
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
                value=8,
                step=1,
                help="Newborn's Apgar score at 5 minutes"
            )

# 4. 预测逻辑
st.write('---')
if st.button('🔍 Start Prediction', key='predict_btn'):
    if model is None:
        st.error("Model not loaded. Please check that the model file exists.")
    else:
        try:
            # ✅ 关键：列名必须与训练数据完全一致（中文）
            input_data = {
                '胎龄': [gestational_age],
                'C反应蛋白水平': [crp],
                '接受气管插管': [endotracheal],
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
            with st.container():
                st.subheader('📊 Prediction Result')
                if probability is not None:
                    st.markdown(
                        f"""
                        <div class="prediction-result">
                            <h4>Probability of sIVH: <strong>{probability:.2f}%</strong></h4>
                            <p>Based on the input newborn information, the system calculates the probability of sIVH as:
                            <strong>{probability:.2f}%</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="prediction-result">
                            <h4>Prediction Result</h4>
                            <p>Based on the input newborn information, the system predicts:
                            <strong>{'High Risk' if prediction[0] == 1 else 'Low Risk'}</strong></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.write("Please check the input values and try again.")
            st.write(f"Error details: {type(e).__name__}: {str(e)}")

# 5. 说明部分
st.write('---')
with st.container():
    st.subheader('ℹ️ About the System')
    st.write('This system is based on machine learning algorithms, predicting potential health risks by analyzing neonatal clinical data.')
    st.write('Note: This system is only an auxiliary tool and cannot replace professional medical diagnosis.')
