import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

@st.cache
def load_model():
	return joblib.load('lgbm_imp.pkl')

@st.cache
def load_scaler():
	return joblib.load('scaler_imp.pkl')

# @st.cache
# def load_features():
# 	return np.load('feat_imp_cols.npy', allow_pickle=True)

# @st.cache
# def load_ohe_cols():
# 	return np.load('ohe_cols.npy', allow_pickle=True)

def input_features():
	country = st.sidebar.selectbox('Country', ('PRT', 'Other'))
	agent = st.sidebar.selectbox('Agent', ('9', '240', '14', 'Other'))
	assigned_room_type = st.sidebar.selectbox('Assigned Room Type', ('I', 'Other'))
	arrival_date_month = st.sidebar.selectbox('Arrival Date Month', ('January', 
		'February', 'March', 'April', 'May', 'June', 'July', 
		'August', 'September', 'October', 'November', 'December'))
	meal = st.sidebar.selectbox('Meal', ('BB', 'SC', 'Other'))
	customer_type = st.sidebar.selectbox('Customer Type', ('Contract', 'Other'))

	return pd.DataFrame({
		'adr': [st.sidebar.number_input('adr')],
		'lead_time': [st.sidebar.number_input('lead_time')],
		'stays_in_week_nights': [st.sidebar.number_input('stays_in_week_nights')],
		'total_of_special_requests': [st.sidebar.number_input('total_of_special_requests')],
		'stays_in_weekend_nights': [st.sidebar.number_input('stays_in_weekend_nights')],
		'booking_changes': [st.sidebar.number_input('booking_changes')],
		'adults': [st.sidebar.number_input('adults')],
		'previous_bookings_not_canceled': [st.sidebar.number_input('previous_bookings_not_canceled')],
		'previous_cancellations': [st.sidebar.number_input('previous_cancellations')],
		'required_car_parking_spaces': [st.sidebar.number_input('required_car_parking_spaces')],
		'children': [st.sidebar.number_input('children')],
		'country_PRT': [1] if country == 'PRT' else [0],
		'agent_9': [1] if agent == '9' else [0],
		'agent_240': [1] if agent == '14' else [0],
		'assigned_room_type_I': [1] if assigned_room_type == 'I' else [0],
		'days_in_waiting_list': [st.sidebar.number_input('days_in_waiting_list')],
		'agent_14': [1] if agent == '14' else [0],
		'arrival_date_month_July': [1] if arrival_date_month == 'July' else [0],
		'customer_type_Contract': [1] if customer_type == 'Contract' else [0],
		'arrival_date_month_April': [1] if arrival_date_month == 'April' else [0],
		'arrival_date_month_August': [1] if arrival_date_month == 'August' else [0],
		'meal_BB': [1] if meal == 'BB' else [0],
		'meal_SC': [1] if meal == 'SC' else [0]
		})


model = load_model()
scl = load_scaler()
# features = load_features()
# ohe_cols = load_ohe_cols()

# main page
st.title("Hotel Demand Booking")

# input
input_df = input_features()

# prediction
st.subheader("Booking Cancellation Prediction")
scaled_input = scl.transform(input_df)
st.write("Cancel" if model.predict(scaled_input) == 1 else "Not Cancel")
