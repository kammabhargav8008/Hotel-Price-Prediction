import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# -----------------------------
# Generate synthetic hotel data
# -----------------------------
def generate_data(n=1000):
    cities = ["Hyderabad", "Bangalore", "Chennai", "Mumbai"]
    room_types = ["Standard", "Deluxe", "Suite"]

    data = []
    for _ in range(n):
        city = np.random.choice(cities)
        room = np.random.choice(room_types)
        guests = np.random.randint(1, 5)
        stay = np.random.randint(1, 7)
        is_weekend = np.random.choice([0, 1])

        base_price = {
            "Hyderabad": 2000,
            "Bangalore": 2500,
            "Chennai": 2300,
            "Mumbai": 3000
        }[city]

        room_multiplier = {
            "Standard": 1.0,
            "Deluxe": 1.4,
            "Suite": 2.0
        }[room]

        price = base_price * room_multiplier
        price += guests * 300
        price += stay * 100
        price += is_weekend * 800
        price += np.random.randint(-300, 300)

        data.append([city, room, guests, stay, is_weekend, int(price)])

    return pd.DataFrame(
        data,
        columns=["city", "room_type", "guests", "stay_duration", "is_weekend", "price"]
    )
@st.cache_resource
def train_model():
    df = generate_data()

    label_encoders = {}
    for col in ["city", "room_type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model, label_encoders

model, label_encoders = train_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Hotel Price Predictor")
st.title("🏨 Hotel Price Predictor")
st.write("Predict hotel room prices using Machine Learning")

city = st.selectbox("City", ["Hyderabad", "Bangalore", "Chennai", "Mumbai"])
room_type = st.selectbox("Room Type", ["Standard", "Deluxe", "Suite"])
guests = st.slider("Number of Guests", 1, 6, 2)

checkin = st.date_input("Check-in Date", date.today())
checkout = st.date_input("Check-out Date", date.today())

if st.button("Predict Price"):
    stay_duration = (checkout - checkin).days

    if stay_duration <= 0:
        st.error("Check-out date must be after check-in date")
    else:
        is_weekend = 1 if checkin.weekday() >= 5 else 0

        input_df = pd.DataFrame({
            "city": [city],
            "room_type": [room_type],
            "guests": [guests],
            "stay_duration": [stay_duration],
            "is_weekend": [is_weekend]
        })

        for col in ["city", "room_type"]:
            input_df[col] = label_encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]

        st.success(f"💰 Estimated price per night: ₹{int(prediction)}")
