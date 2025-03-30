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

# Streamlit UI
st.set_page_config(page_title="Disease Outbreak Prediction", layout="wide")
st.title("📊 Disease Outbreak Prediction System")
st.markdown("This tool predicts the spread of diseases based on past data trends.")

# 🌍 Enhanced Region Selection UI
st.sidebar.header("🌍 Select a Region")
regions = ["India", "USA", "UK", "Australia", "Canada"]
selected_region = st.sidebar.radio("Choose a region:", regions)

# Filter dataset for selected region
if "REGION" in df.columns:
    df = df[df["REGION"] == selected_region]

# Preprocessing
scaler = MinMaxScaler()
# Convert column to numeric, forcing errors='coerce' to convert invalid values to NaN
df["% WEIGHTED ILI"] = pd.to_numeric(df["% WEIGHTED ILI"], errors="coerce")

# Handle NaN values (Fill with median to retain data distribution)
df["% WEIGHTED ILI"].fillna(df["% WEIGHTED ILI"].median(), inplace=True)

# Apply scaling
df["% WEIGHTED ILI"] = scaler.fit_transform(df[["% WEIGHTED ILI"]])


# Layout for better UI
col1, col2 = st.columns([2, 1])

# 📈 Historical Data Visualization
with col1:
    st.subheader(f"📈 Historical Trends in {selected_region}")
    fig = px.line(df, x="year", y="% WEIGHTED ILI", title=f"Influenza-like Illness Trends in {selected_region}", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# 🔮 Prediction Section
with col2:
    st.subheader(f"🔮 Predict Future Outbreaks in {selected_region}")
    days = st.slider("Select number of future days to predict:", 1, 30, 7)

    if len(df) >= 10:
        input_data = df["% WEIGHTED ILI"].values[-10:].reshape(1, 10, 1)
        predictions = []
        
        with st.spinner("Predicting... Please wait."):
            for _ in range(days):
                pred = model.predict(input_data)
                predictions.append(pred[0][0])  
                input_data = np.roll(input_data, -1)
                input_data[0, -1, 0] = pred[0][0]  

        predicted_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        pred_df = pd.DataFrame({"Day": np.arange(1, days+1), "Predicted ILI (%)": predicted_values.flatten()})
        st.dataframe(pred_df, height=200)

        st.subheader(f"📊 Future Predictions Trend for {selected_region}")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pred_df["Day"], y=pred_df["Predicted ILI (%)"], mode='lines+markers', line=dict(color='red')))
        fig2.update_layout(title=f"Predicted Future ILI Trends in {selected_region}", xaxis_title="Days Ahead", yaxis_title="Predicted ILI (%)")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.success("✅ Prediction Complete!")
        st.balloons()
    else:
        st.error(f"Not enough data for prediction in {selected_region}. Need at least 10 historical data points.")

# 📰 Disease News Section
st.subheader(f"📰 Latest Disease Outbreak News in {selected_region}")
news_api_key = "0056df10504d493188bae5b4bb973ab5"
news_url = f"https://newsapi.org/v2/everything?q={selected_region} disease outbreak OR virus OR epidemic&language=en&sortBy=publishedAt&apiKey={news_api_key}"

try:
    response = requests.get(news_url)
    news_data = response.json()

    if news_data["status"] == "ok":
        articles = news_data["articles"][:5]

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