import streamlit as st
import joblib

model = joblib.load('diabetes_model.joblib')
scaler = joblib.load('scaler.joblib')
