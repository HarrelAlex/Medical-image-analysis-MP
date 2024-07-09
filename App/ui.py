import streamlit as st
from PIL import Image
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models
import main
from main import calculate_risk
import time 
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


tumormodel = models.resnet50(weights='IMAGENET1K_V1')
num_ftrs = tumormodel.fc.in_features

tumormodel.fc = nn.Linear(num_ftrs, 4)

tumormodel = tumormodel.to(device)
tumormodel.load_state_dict(torch.load('tumormod.pth'))



pmodel = tf.keras.models.load_model('pmod.keras')


if "age" not in st.session_state:
    st.session_state.age = 25
if "gender" not in st.session_state:
    st.session_state.gender = 'Male'
if "medical_history" not in st.session_state:
    st.session_state.medical_history = []
if "lifestyle_factors" not in st.session_state:
    st.session_state.lifestyle_factors = []
if "blood_pressure" not in st.session_state:
    st.session_state.blood_pressure = 120
if "cholesterol" not in st.session_state:
    st.session_state.cholesterol = 180
if "symptoms" not in st.session_state:
    st.session_state.symptoms = []
if "risk_score" not in st.session_state:
    st.session_state.risk_score = 0
if "rf" not in st.session_state:
    st.session_state.rf = 'NA'
if 'details' not in st.session_state:
    st.session_state.details = []
if "med_report" not in st.session_state:
    st.session_state.med_report = []


def answer_gen():
        for word in st.session_state.answer.split():
            yield word + " "
            time.sleep(0.05)      


def cb():
    if query := st.chat_input("Ask a question"):
                st.write(st.session_state.med_report)
                with st.chat_message("human"):
                    st.markdown(query) 
                st.session_state.messages.append({"role": "human", "content": query})
                if "answer" not in st.session_state:
                    st.session_state.answer = []
                st.session_state.answer = main.generate_med_report_qna(st.session_state.med_report, query)
                with st.chat_message("assistant"):
                    st.session_state.answer = st.write_stream(answer_gen())
                st.session_state.messages.append({"role": "assistant", "content": st.session_state.answer})     


st.title('Medical Image Analysis')
st.markdown("\n1) Upload the image \n2) Enter the details of the patient \n3) Check for the disease")
with st.expander('About'):
    st.write('This app analyzes medical images to assist in diagnosing conditions. Users can upload medical scans, and the system could provide insights or connect them with relevant medical resources.')
    st.write('This Web app can take the CT scan of brain, X-ray scan of lungs as input')    


def upload_img():            
    submitted= False
    ans=' '           
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            image.save("input/upload/img.jpeg")
            submitted = True
            
        if submitted:
            st.session_state.age = st.number_input('Enter patient age:', min_value=0, max_value=120, value=st.session_state.age)
            st.session_state.gender = st.selectbox('Select patient gender:', ['Male', 'Female', 'Other'], index=['Male', 'Female', 'Other'].index(st.session_state.gender))
            st.session_state.medical_history = st.multiselect('Select medical history factors:', ['Family History', 'Previous Diagnosis'], default=st.session_state.medical_history)
            st.session_state.lifestyle_factors = st.multiselect('Select lifestyle factors:', ['Smoking', 'Alcohol'], default=st.session_state.lifestyle_factors)
            st.session_state.blood_pressure = st.number_input('Enter blood pressure:', min_value=0, max_value=300, value=st.session_state.blood_pressure)
            st.session_state.cholesterol = st.number_input('Enter cholesterol level:', min_value=0, max_value=500, value=st.session_state.cholesterol)
            st.session_state.symptoms = st.multiselect('Select symptoms:', ['Severe Headache', 'Vision Problems'], default=st.session_state.symptoms) 
            col1, col2 = st.columns(2)
            try:
                with col1:
                    if st.button('Check for Pneumonia'):
                        ans=main.test(pmodel)                
                with col2:
                    if st.button('Check for Brain Tumor'):
                        ans=main.visualize_model_predictions(tumormodel,img_path='Input/upload/img.jpeg')    
            except Exception as e:
                print(e)
                ans="Wrong Image for analysis"
    if ans != ' ':
        st.session_state.details = {
            'Disease': [ans],
        'Patient Age': [st.session_state.age],
        'Patient Gender': [st.session_state.gender],
        'Medical History': [st.session_state.medical_history],
        'Lifestyle Factors': [st.session_state.lifestyle_factors],
        'Blood Pressure': [st.session_state.blood_pressure], 
        'Cholesterol': [st.session_state.cholesterol],
        'Symptoms': [st.session_state.symptoms]}
        
        details_df = pd.DataFrame(st.session_state.details)
        st.markdown(f"***:red[{"Disease Found- "+ans}]***")   
        st.session_state.risk_score = calculate_risk(ans, st.session_state.age, st.session_state.gender, st.session_state.medical_history, st.session_state.lifestyle_factors, st.session_state.blood_pressure, st.session_state.cholesterol, st.session_state.symptoms)
        if st.session_state.risk_score >= 90:
            st.session_state.rf = 'High Risk'
        elif st.session_state.risk_score <= 30:
            st.session_state.rf = 'Minimal Risk'
        else:
            st.session_state.rf = 'Moderate Risk'
            
        st.title('Patient Details')
        st.table(details_df)
        st.markdown(f'***Risk Score: {st.session_state.risk_score}***')
        st.session_state.med_report=main.generate_med_report(ans,st.session_state.rf) 
        st.write(st.session_state.med_report)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        else:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
    



upload_img()
cb()

