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
		'adr': [st.sidebar.number_input('adr', min_value=0.00, max_value=5500.00, step=0.01)],
		'lead_time': [st.sidebar.number_input('lead_time', min_value=0, max_value=750, step=1)],
		'stays_in_week_nights': [st.sidebar.number_input('stays_in_week_nights', min_value=0, max_value=50, step=1)],
		'total_of_special_requests': [st.sidebar.number_input('total_of_special_requests', min_value=0, max_value=5, step=1)],
		'stays_in_weekend_nights': [st.sidebar.number_input('stays_in_weekend_nights', min_value=0, max_value=20, step=1)],
		'booking_changes': [st.sidebar.number_input('booking_changes', min_value=0, max_value=25, step=1)],
		'adults': [st.sidebar.number_input('adults', min_value=0, max_value=60, step=1)],
		'previous_bookings_not_canceled': [st.sidebar.number_input('previous_bookings_not_canceled', min_value=0, max_value=75, step=1)],
		'previous_cancellations': [st.sidebar.number_input('previous_cancellations', min_value=0, max_value=30, step=1)],
		'required_car_parking_spaces': [st.sidebar.number_input('required_car_parking_spaces', min_value=0, max_value=10, step=1)],
		'children': [st.sidebar.number_input('children', min_value=0, max_value=15, step=1)],
		'country_PRT': [1] if country == 'PRT' else [0],
		'agent_9': [1] if agent == '9' else [0],
		'agent_240': [1] if agent == '14' else [0],
		'assigned_room_type_I': [1] if assigned_room_type == 'I' else [0],
		'days_in_waiting_list': [st.sidebar.number_input('days_in_waiting_list', min_value=0, max_value=400, step=1)],
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
