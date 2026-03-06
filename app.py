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
    with open('my_best_pipeline666.pkl', 'rb') as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")
    st.write(f"Model type: {type(model)}")
    st.write(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
    
    # 检查是否有predict方法
    if hasattr(model, 'predict'):
        st.write("Model has predict method")
    else:
        st.write("Model does not have predict method")
        
    # 检查是否是numpy数组
    if isinstance(model, np.ndarray):
        st.write(f"Model is a numpy array with shape: {model.shape}")
        st.write("Warning: Model is a numpy array, not a trained model object")
        
except Exception as e:
    st.error(f"Error loading model: {str(e)}")

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
            
            # 特征2: Birthweight-kg (出生体重)
            birth_weight = st.number_input(
                'Birth Weight (kg)', 
                value=3.0, 
                step=0.1,
                help='Newborn\'s birth weight'
            )
            
            # 特征3: Head circumference-cm (头围)
            head_circumference = st.number_input(
                'Head Circumference (cm)', 
                value=35.0, 
                step=0.1,
                help='Newborn\'s head circumference'
            )
            
            # 特征4: Chest circumference-cm (胸围)
            chest_circumference = st.number_input(
                'Chest Circumference (cm)', 
                value=33.0, 
                step=0.1,
                help='Newborn\'s chest circumference'
            )
        
        with col2:
            # 特征5: Apgar 1 min (Apgar评分1分钟)
            apgar_1min = st.number_input(
                'Apgar Score (1 min)', 
                value=8, 
                step=1,
                help='Newborn\'s Apgar score at 1 minute'
            )
            
            # 特征6: RDS (新生儿呼吸窘迫综合征)
            rds = st.selectbox(
                'Respiratory Distress Syndrome (RDS)', 
                [0, 1], 
                format_func=lambda x: 'Yes' if x == 1 else 'No',
                help='Whether the newborn has respiratory distress syndrome'
            )
            
            # 特征7: Invasive mechanical ventilation (有创呼吸机通气时间)
            invasive_ventilation = st.number_input(
                'Invasive Mechanical Ventilation (days)', 
                value=0.0, 
                step=0.1,
                help='Days of invasive mechanical ventilation'
            )
            
            # 特征8: Non-invasive mechanical ventilation (无创呼吸机通气时间)
            non_invasive_ventilation = st.number_input(
                'Non-invasive Mechanical Ventilation (days)', 
                value=0.0, 
                step=0.1,
                help='Days of non-invasive mechanical ventilation'
            )

# 4. 预测逻辑
st.write('---')
if st.button('🔍 Start Prediction', key='predict_btn'):
    try:
        # 按照模型训练时的特征顺序组织输入数据
        input_data = {
            'Gestational age': [gestational_age], 
            'Birthweight-kg': [birth_weight], 
            'Head circumference-cm': [head_circumference], 
            'Chest circumference-cm': [chest_circumference], 
            'Apgar 1 min': [apgar_1min], 
            'RDS': [rds], 
            'Invasive mechanical ventilation': [invasive_ventilation], 
            'Non-invasive mechanical ventilation': [non_invasive_ventilation]
        }
        
        # 转换为DataFrame格式
        input_df = pd.DataFrame(input_data)
        
        # 预测
        prediction = model.predict(input_df)
        
        # 显示预测结果
        with st.container():
            st.subheader('📊 Prediction Result')
            with st.container():
                st.markdown(
                    f"""
                    <div class="prediction-result">
                        <h4>Prediction Result: <strong>{'High Risk' if prediction[0] == 1 else 'Low Risk'}</strong></h4>
                        <p>Based on the input newborn information, the system predicts the newborn\'s health risk level as:
                        <strong>{'High Risk' if prediction[0] == 1 else 'Low Risk'}</strong></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.write("Please check the input values and try again.")

# 5. 信息部分
st.write('---')
with st.container():
    st.subheader('ℹ️ About the System')
    st.write('This system is based on machine learning algorithms, predicting potential health risks by analyzing neonatal clinical data.')
    st.write('Note: This system is only an auxiliary tool and cannot replace professional medical diagnosis.')