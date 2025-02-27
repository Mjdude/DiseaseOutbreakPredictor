import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# Layout for better UI
col1, col2 = st.columns([2, 1])

# Plot historical data
with col1:
    st.subheader("📈 Historical Trends")
    fig = px.line(df, x="year", y="% WEIGHTED ILI", title="Influenza-like Illness Trends", markers=True)
    st.plotly_chart(fig, use_container_width=True)

# Prediction Section
with col2:
    st.subheader("🔮 Predict Future Outbreaks")
    days = st.slider("Select number of future days to predict:", 1, 30, 7)

    # Get last sequence of data for prediction
    if len(df) >= 10:
        input_data = df["% WEIGHTED ILI"].values[-10:].reshape(1, 10, 1)
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
        fig2.update_layout(title="Predicted Future ILI Trends", xaxis_title="Days Ahead", yaxis_title="Predicted ILI (%)")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.success("✅ Prediction Complete!")
        st.balloons()
    else:
        st.error("Not enough data for prediction. Need at least 10 historical data points.")
