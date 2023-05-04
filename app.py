import xgboost as xgb
from pathlib import Path
import datetime
import streamlit as st
import pandas as pd
import math

##page layout
st.set_page_config(layout="centered", page_icon="🏦", page_title=" Customer Segmentation In Bank Marketing")
st.title("🏦 Customer Segmentation In Bank Marketing")
st.image("https://echovme.in/wp-content/uploads/2022/07/social-media-digital-marketing-strategies-for-banks.jpg", caption=None)
st.caption("Time deposits are the main source of income for a bank. Time deposits are cash investments held at financial institutions. Your money is invested for an agreed rate of interest over a certain period or period. Telephone marketing campaigns are still one of the most effective ways to reach people. However, they require a large investment of hiring a call center to actually run this campaign. That's why it's so important to identify customers who are most likely to convert first so they can be targeted specifically through calls. The data relates to a direct marketing campaign (phone calls) from a Portuguese banking institution. The purpose of classification is to predict whether a client will subscribe to a time deposit.")
st.subheader(
    ":blue[Will client will subscribe to term deposit?]"
)
st.info("Whilst every effort has been taken during the development of these tools for them to be as accurate and reliable as possible it is important that the user understands they are still a prediction and not an absolute. Any decisions taken whist using these tools are the responsibility of the user and no liability whatsoever will be taken by the developers/authors of the tools or the website owner.",icon="ℹ️")
st.write("##")
st.write("##")

# declaring variables
dict_job = {'admin':0, 'blue-collar':1, 'entrepreneur':2, 'housemaid':3, 'management':4, 'retired':5, 'self-employed':6, 'services':7, 'student':8, 'technician':9, 'unemployed':10, 'unknown':11}
dict_marital = {'divorced':0, 'married':1, 'single':2}
dict_contact = {'cellular':0, 'telephone':1, 'unknown':2}
dict_education = {'primary':0,'secondary':1,'tertiary':2,'unknown':3
                  }
dict_month = {'April':0, 'August':1, 'December':2, 'February':3, 'January':4, 'July':5, 'June':6, 'March':7, 'May':8, 'November':9, 'October':10, 'September':11}
dict_binary = {'no':0, 'yes':1}

# user input

st.subheader("Customer Infomation")

# selecting job
job_option = st.selectbox(
        "Select Client's Occupation",
        (dict_job.keys()),
)

# selecting marital status
marital_option = st.selectbox(
        "Select Client's Marital Status",
        (dict_marital.keys()),
)

education_option = st.selectbox(
        "Select Client's Highest Education Level",
        (dict_education.keys()),
)

default_radio = st.radio(
    "has credit in default?",
    dict_binary.keys()
)

balance = st.number_input(
    "Average yearly balance in euros(K)",
    min_value=0
)

# housing loan
housing_loan = st.radio(
    "has housing loan?",
    dict_binary.keys()
)

# personal loan
personal_loan = st.radio(
    "has personal loan?",
    dict_binary.keys()
)

st.subheader("Current Campaign Info")

# contact communication type
contact = st.radio(
    "contact communication type",
    dict_contact.keys()
)

# last contact day of the month
last_contact_month = st.selectbox('Select last contact month', dict_month.keys())
# last contact month of year
last_contact_day = st.selectbox('Select last contact day of the month', list(range(1,32)))
# last contact duration, in seconds 
last_contact_duration = st.time_input('Last contact duration(in minutes)', datetime.time(3, 30))
# campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
count_contact = st.number_input("Number of contacts performed: ",min_value=1)
# pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
day_passed = st.number_input("Number of days that passed by after the client was last contacted from a previous campaign(-1 if the client is not previously contacted)",min_value=-1)
# previous: number of contacts performed before this campaign and for this client (numeric)
count_prev_contact = st.number_input("Number of contacts performed: ",min_value=0)

# Model
model = xgb.XGBClassifier()
model.load_model('./model.json')

def predict():
    x = [dict_job[job_option],dict_marital[marital_option],dict_education[education_option],dict_binary[default_radio],balance,dict_binary[housing_loan],dict_binary[personal_loan],dict_contact[contact],last_contact_day,dict_month[last_contact_month],last_contact_duration.hour*60+last_contact_duration.minute ,count_contact,day_passed,count_prev_contact]
    df = pd.DataFrame([x] , columns = model.feature_names_in_)
    prediction = model.predict(df)
    prediction_prob = model.predict_proba(df)
    if(prediction[0] == 1):
        st.write('This customer segment :green[will deposit] with probability of ', "%.2f" % (prediction_prob[0][1]*100), '%')
    else:
        st.write('This customer segment :red[will not deposit] with probability of ', "%.2f" % (prediction_prob[0][0]*100), '%')    
        
if st.button('Classify'): 
    predict()
