import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model = load_model("lstm_model.h5")

# Load dataset
df = pd.read_csv("data/cleaned_ILINet.csv")

# Ensure 'year' is a datetime type
if "YEAR" in df.columns:
    df.rename(columns={"YEAR": "year"}, inplace=True)
    df["year"] = pd.to_datetime(df["year"], errors='coerce')

# Preprocessing
scaler = MinMaxScaler()
df["% WEIGHTED ILI"] = scaler.fit_transform(df[["% WEIGHTED ILI"]])

# Streamlit UI
st.set_page_config(page_title="Disease Outbreak Prediction", layout="wide")
st.title("📊 Disease Outbreak Prediction System")
st.markdown("This tool predicts the spread of diseases based on past data trends.")

# Region Selection
st.subheader("🌍 Select a Region")
regions = ["India", "United States", "United Kingdom", "Australia", "Canada"]
selected_region = st.selectbox("Choose a region:", regions)

# Filter data based on selected region
region_data = df[df["Region"] == selected_region] if "Region" in df.columns else df

# Layout for better UI
col1, col2 = st.columns([2, 1])

# Plot historical data
with col1:
    st.subheader(f"📈 Historical Trends - {selected_region}")
    fig = px.line(region_data, x="year", y="% WEIGHTED ILI", title=f"Influenza-like Illness Trends in {selected_region}", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# Prediction Section
with col2:
    st.subheader("🔮 Predict Future Outbreaks")
    days = st.slider("Select number of future days to predict:", 1, 30, 7)

    # Get last sequence of data for prediction
    if len(region_data) >= 10:
        input_data = region_data["% WEIGHTED ILI"].values[-10:].reshape(1, 10, 1)
        predictions = []
        
        with st.spinner("Predicting... Please wait."):
            for _ in range(days):
                pred = model.predict(input_data)
                predictions.append(pred[0][0])  
                input_data = np.roll(input_data, -1)
                input_data[0, -1, 0] = pred[0][0]  

        # Transform predictions back to original scale
        predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        # Display results
        pred_df = pd.DataFrame({"Day": np.arange(1, days+1), "Predicted ILI (%)": predicted_values.flatten()})
        st.dataframe(pred_df, height=200)

        # Plot Predictions
        st.subheader("📊 Future Predictions Trend")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pred_df["Day"], y=pred_df["Predicted ILI (%)"], mode='lines+markers', line=dict(color='red')))
        fig2.update_layout(title=f"Predicted Future ILI Trends in {selected_region}", xaxis_title="Days Ahead", yaxis_title="Predicted ILI (%)")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.success("✅ Prediction Complete!")
        st.balloons()
    else:
        st.error(f"Not enough data for prediction in {selected_region}. Need at least 10 historical data points.")

# Disease News Section
st.subheader("📰 Latest Disease Outbreak News")
news_api_key = "0056df10504d493188bae5b4bb973ab5"  # Replace with your API key
news_url = f"https://newsapi.org/v2/everything?q={selected_region} disease outbreak OR virus OR epidemic&language=en&sortBy=publishedAt&apiKey={news_api_key}"

try:
    response = requests.get(news_url)
    news_data = response.json()

    if news_data["status"] == "ok":
        articles = news_data["articles"][:5]  # Show latest 5 articles

        for article in articles:
            st.markdown(f"### [{article['title']}]({article['url']})")
            st.write(f"🗞 **Source:** {article['source']['name']}")
            st.write(f"📅 **Published:** {article['publishedAt'][:10]}")
            st.write(f"📝 {article['description']}")
            st.image(article["urlToImage"], width=500)
            st.markdown("---")
    else:
        st.error("⚠️ Could not fetch news. Please check your API key or try again later.")

except Exception as e:
    st.error(f"⚠️ Error fetching news: {e}")
