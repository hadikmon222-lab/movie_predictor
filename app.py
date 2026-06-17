import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load your best trained model
@st.cache_resource
def load_model():
    # Make sure 'best_model.pkl' is saved from your notebook using:
    # joblib.dump(your_best_model, 'best_model.pkl')
    try:
        return joblib.load('best_model.pkl')
    except FileNotFoundError:
        st.error("Error: 'best_model.pkl' not found. Please dump your model from Colab and place it in this directory.")
        return None

model = load_model()

# 2. Configure Web Interface Layout
st.set_page_config(page_title="Movie Rating Predictor", page_icon="🎬", layout="centered")

st.title("🎬 Top-Rated Movies Rating Predictor")
st.write("This app predicts the **Vote Average** of a movie based on its metadata and popularity trends.")
st.divider()

# 3. Input Interface Sections
st.subheader("📊 Enter Movie Features")

col1, col2 = st.columns(2)

with col1:
    # Popularity metric (Observed max in dataset was ~567.76)
    popularity = st.number_input(
        "Movie Popularity Score", 
        min_value=0.0000, 
        max_value=600.0000, 
        value=4.2168, 
        step=0.1,
        help="The TMDB popularity metrics score."
    )
    
    # Vote count metric (Observed max was ~38665)
    vote_count = st.number_input(
        "Total Vote Count", 
        min_value=0, 
        max_value=50000, 
        value=2148, 
        step=10,
        help="Total number of user reviews submitted for this movie."
    )

with col2:
    # Release date broken down exactly like your dataframe splits
    release_year = st.slider("Release Year", min_value=1900, max_value=2026, value=2015)
    release_month = st.slider("Release Month", min_value=1, max_value=12, value=6)
    release_day = st.slider("Release Day", min_value=1, max_value=31, value=15)

st.divider()

# Optional fields for user interaction (matching text features found in dataset)
st.subheader("📝 Movie Details (Optional)")
movie_title = st.text_input("Movie Title", value="My Awesome Sci-Fi Movie")
movie_overview = st.text_area("Overview / Plot Summary", value="An exciting story about data scientists predicting the future...")

# 4. Handle Prediction Actions
if st.button("🔮 Predict Movie Rating", type="primary"):
    if model is not None:
        # Create a DataFrame matching the precise feature set expected by your model
        input_data = pd.DataFrame([{
            'popularity': popularity,
            'vote_count': vote_count,
            'release_year': release_year,
            'release_month': release_month,
            'release_day': release_day
        }])
        
        # NOTE: If you applied a RobustScaler, MinMaxScaler, or log transformations 
        # to features like popularity or vote_count in your notebook before training,
        # you MUST load that scaler object here and use: scaler.transform(input_data)
        
        try:
            # Generate the prediction
            prediction = model.predict(popularity_class)
            
            # Constrain predictions to standard rating boundaries if outliers occur
            predicted_rating = max(0.0, min(10.0, prediction[0]))
            
            # 5. Display the Output Response
            st.success(f"### 🎉 Predicted Vote Average: **{predicted_rating:.2f} / 10**")
            
            # Quick interactive feedback component
            if predicted_rating >= 7.5:
                st.balloons()
                st.write("🌟 **High Rating Potential!** This looks like a critically acclaimed blockbuster recipe.")
            elif predicted_rating >= 6.0:
                st.write("👍 **Decent Profile.** Expected to perform average among viewers.")
            else:
                st.write("📉 **Lower Audience Agreement.** The current parameters yield a lower rating trend.")
                
        except Exception as e:
            st.error(f"Prediction failed. Ensure the input data features exactly match your trained model expectations. Error detail: {e}")
