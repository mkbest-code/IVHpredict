import streamlit as st
import pickle
import pandas as pd

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
import numpy as np

# 尝试加载模型
model = None
try:
    # 使用PyCaret的load_model函数加载模型
    from pycaret.classification import load_model
    model = load_model('my_best_pipeline0106')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.write("Please make sure PyCaret is installed and the model file exists.")

# 2. 设置网页标题和副标题
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
            # 特征1: Gestational age (胎龄)
            gestational_age = st.number_input(
                'Gestational Age (weeks)', 
                value=38.0, 
                step=0.1,
                help='Newborn\'s gestational age'
            )
            
       # 特征2: Apgar 5 min (Apgar评分5分钟)
            apgar_1min = st.number_input(
                'Apgar Score (5 min)', 
                value=8, 
                step=1,
                help='Newborn\'s Apgar score at 5 minute'
            )

            # 特征3:  C-Reactive Protein (C反应蛋白水平)
             C-Reactive Protein = st.number_input(
                ' C-Reactive Protein (mg/L)', 
                value=5.0, 
                step=0.1,
                help='Newborn\'s  C-Reactive Protein'
            )
            
            # 特征4: Receipt of Endotracheal Intubation (接受气管插管)
            rds = st.selectbox(
                'Receipt of Endotracheal Intubation', 
                [0, 1], 
                format_func=lambda x: 'Yes' if x == 1 else 'No',
                help='Whether the newborn received endotracheal intubation'
            )
            
            # 特征5: Use of Catecholamines (使用儿茶酚胺类药物)
            rds = st.selectbox(
                'Use of Catecholamines ', 
                [0, 1], 
                format_func=lambda x: 'Yes' if x == 1 else 'No',
                help='Whether the newborn used  catecholamines '
            )
        

     
            
            
            
# 4. 预测逻辑
st.write('---')
if st.button('🔍 Start Prediction', key='predict_btn'):
    try:
        # 按照模型训练时的特征顺序组织输入数据
        input_data = {
            'Gestational age': [胎龄], 
            'C-Reactive Protein': [C反应蛋白水平], 
            'Receipt of Endotracheal Intubation': [接受气管插管], 
            'Use of Catecholamines': [使用儿茶酚胺类药物], 
            'Apgar 5 min': [5分钟Apgar评分], 
        }
        
        # 转换为DataFrame格式
        input_df = pd.DataFrame(input_data)
        
        # 直接使用模型进行预测
        prediction = model.predict(input_df)
        
        # 获取患病概率
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_df)[0][1] * 100  # 获取第二类（患病）的概率
        else:
            probability = None
        
        # 显示预测结果
        with st.container():
            st.subheader('📊 Prediction Result')
            with st.container():
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

# 5. 信息部分
st.write('---')
with st.container():
    st.subheader('ℹ️ About the System')
    st.write('This system is based on machine learning algorithms, predicting potential health risks by analyzing neonatal clinical data.')
    st.write('Note: This system is only an auxiliary tool and cannot replace professional medical diagnosis.')