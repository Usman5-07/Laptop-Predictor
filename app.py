import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('bestPipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())


type = st.selectbox('Type', df['TypeName'].unique())

ram = st.selectbox('RAM (in GBs)', [2,4,6,8,12,16,32,64])

weight = st.number_input('Weight of laptop')

touch = st.selectbox('TouchScreen', ['Yes', 'No'])

ips = st.selectbox('IPS', ['Yes', 'No'])

screenResolution = st.selectbox('Screen Resolution', ['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

screenSize = st.number_input("Input Screen Size")

cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GBs)', [0,128,256,513,1024,2048])
ssd = st.selectbox('SSD (in GBs)', [0,128,256,513,1024])
gpu = st.selectbox('GPU', df['Gpu Brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

if st.button('Predict Price'):
    xRes = int(screenResolution.split('x')[0])
    yRes = int(screenResolution.split('x')[1])
    ppi = ((xRes**2) + (yRes**2))**0.5 / screenSize

    if touch == 'Yes':
        touch = 1
    else:
        touch = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    query = np.array([company,type,ram,weight,touch,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = query.reshape(1,12)
    price = np.exp(pipe.predict(query))
    st.title(price)

