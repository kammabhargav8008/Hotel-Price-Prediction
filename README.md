🏨 Hotel Price Predictor

Hotel Price Predictor is a machine learning–based web application that estimates hotel room prices based on city, room type, number of guests, stay duration, and weekend effects. The project demonstrates an end-to-end ML workflow including data generation, model training, and real-time prediction using a user-friendly web interface.

🚀 Features
- Predicts hotel room prices per night
- Considers location, room type, guests, and stay duration
- Detects weekend pricing impact
- Real-time predictions using a trained ML model
- Simple and interactive web UI

🛠️ Tech Stack
- **Programming Language:** Python  
- **Machine Learning:** XGBoost, Scikit-learn  
- **Web Framework:** Streamlit  
- **Data Handling:** Pandas, NumPy  

📊 Machine Learning Approach
- Synthetic hotel pricing data generation
- Feature engineering (stay duration, weekend flag)
- Label encoding for categorical features
- Regression modeling using XGBoost
- Model caching for faster predictions

📁 Project Structure
hotel-price-predictor/

├── app.py # Complete ML + Streamlit app
├── requirements.txt # Project dependencies
└── README.md # Project documentation


▶️ How to Run Locally

1. Clone the repository
bash
git clone https://github.com/kammabhargav8008/hotel-price-prediction.git
cd hotel-price-predictor

pip install -r requirements.txt

streamlit run app.py

