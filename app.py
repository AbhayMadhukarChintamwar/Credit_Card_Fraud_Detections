import streamlit as st
import pickle
import numpy as np

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ----------------------------
# Custom CSS
# ----------------------------

st.markdown("""
<style>

.main{
     background: linear-gradient(135deg,#ffffff,#0f172a,#1e293b);
}

h1{
    color:white;
    text-align:center;
    font-size:45px;
}

h3{
    color:white;
}

div[data-testid="stSidebar"]{
    background:#111827;
}

.stButton>button{

    width:100%;
    height:55px;

    font-size:22px;
    font-weight:bold;

    border-radius:15px;

    color:white;

    background:linear-gradient(90deg,#06b6d4,#3b82f6);

    border:none;

}

.stButton>button:hover{

    background:linear-gradient(90deg,#2563eb,#06b6d4);
}

div[data-testid="metric-container"]{

    background:white;

    border-radius:15px;

    padding:15px;

    box-shadow:0px 8px 20px rgba(0,0,0,.2);
}

input{

    border-radius:10px;

}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Model
# ----------------------------

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# ----------------------------
# Sidebar
# ----------------------------


st.sidebar.title("Project Details")

st.sidebar.info("""
### Credit Card Fraud Detection

Machine Learning Model

Algorithm :
- Logistic Regression

Features :
- Time
- V1 - V28
- Amount

Developed using Streamlit
""")

# ----------------------------
# Header
# ----------------------------

st.markdown("<h1 style='color:#2E8B57'>💳 Credit Card Fraud Detection System</h1>",unsafe_allow_html=True)

st.markdown(
"<center><h4 style='color:white'>Predict whether a transaction is Fraud or Genuine.</h4></center>",
unsafe_allow_html=True)

st.write("")

# ----------------------------
# Metrics
# ----------------------------

c1,c2,c3=st.columns(3)

with c1:
    st.metric("Features","30")

with c2:
    st.metric("Algorithms","Logistic Regression")

with c3:
    st.metric("Prediction","Real Time")

st.divider()

# ----------------------------
# Input Form
# ----------------------------

col1,col2=st.columns(2)

inputs=[]

with col1:

    st.subheader("Transaction Details")

    inputs.append(st.number_input("Time"))

    for i in range(1,15):
        inputs.append(
            st.number_input(f"V{i}",format="%.6f")
        )

with col2:

    st.subheader("Transaction Features")

    for i in range(15,29):
        inputs.append(
            st.number_input(f"V{i}",format="%.6f")
        )

    inputs.append(st.number_input("Amount"))

st.write("")
st.write("")

# ----------------------------
# Prediction
# ----------------------------

if st.button("🔍 Predict Transaction"):

    data=np.array(inputs).reshape(1,-1)

    data=scaler.transform(data)

    prediction=model.predict(data)

    st.divider()

    if prediction[0]==1:

        st.error("🚨 Fraudulent Transaction Detected")

        st.balloons()

    else:

        st.success("✅ Genuine Transaction")

        st.snow()

st.divider()

st.caption("© 2026 Credit Card Fraud Detection | Machine Learning Project")
st.caption("Made ❤️ with Abhay")
