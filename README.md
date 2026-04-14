# Menstrual Cycle Arrival Prediction using LSTM

This repository contains a deep learning project that predicts the number of days remaining until the start of the next menstrual cycle. By leveraging time-series data including hormonal fluctuations and daily symptoms, the model provides a personalized prediction based on biological markers.

## 🚀 Project Overview
The goal of this project is to create a data-driven approach to cycle tracking. By using an **LSTM (Long Short-Term Memory)** network, the model learns the specific hormonal and physical cues that signal the onset of the menstrual phase.

## 📊 Dataset Description
The model is trained on `processed_period_data.csv`, which includes 5,730 daily records across multiple users.
- **Hormone Markers:** LH (Luteinizing Hormone), Estrogen, and PdG (Progesterone).
- **Physical Symptoms:** Cramps, fatigue, sore breasts, bloating, and headaches.
- **Psychological Markers:** Stress levels and mood swings.
- **Lifestyle Factors:** Activity levels (sedentary/light) and sleep quality.

## 🧠 Model Architecture
LSTMs are highly effective for this task as they can retain information about previous days in the cycle to predict future events.
- **Input:** 10-day sliding window of physiological data.
- **Network:** - LSTM Layer (64 units)
  - Dropout Layer (0.2)
  - LSTM Layer (32 units)
  - Dense Regression Layer (Output: Days to Period)
- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)

## 📈 Key Visualizations
Included in the analysis are:
1. **Hormone Trends:** Mapping average LH, Estrogen, and PdG levels across cycle phases.
2. **Symptom Heatmaps:** Correlation of symptoms like cramps and bloating with specific phases.
3. **Loss Curves:** Documentation of model convergence during training.
4. **Actual vs. Predicted:** Comparison of the model's countdown versus the actual cycle start.

## 🛠️ Requirements
- Python 3.8+
- TensorFlow / Keras
- Pandas & NumPy
- Scikit-learn
- Matplotlib & Seaborn

## 📝 Disclaimer
This tool is for educational and research purposes only and should not be used as a substitute for professional medical advice or as a primary method of birth control.
