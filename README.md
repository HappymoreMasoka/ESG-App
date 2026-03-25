🌍 ESG Risk Intelligence Dashboard (v1)

An end-to-end Machine Learning application for predicting Environmental, Social, and Governance (ESG) scores, with interactive analytics and NLP insights.

🚀 Overview

This project is a production-style ML system that predicts ESG scores for companies using structured data and provides insights through an interactive dashboard.

It moves beyond notebooks into a modular, deployable ML architecture, integrating:

✅ 3 Machine Learning models (E, S, G)
✅ Feature engineering pipeline
✅ Real-time prediction dashboard (Streamlit)
✅ NLP-based ESG insights
✅ Deployment-ready setup (Docker)
🧠 What Problem This Solves

Traditional ESG ratings are:

opaque
inconsistent
not easily actionable

This platform provides:

transparent ML-based ESG scoring
real-time predictions
explainable insights for decision-making
🏗️ Architecture
User Input
   ↓
Feature Engineering
   ↓
3 ML Models (E, S, G)
   ↓
ESG Aggregation
   ↓
Dashboard + NLP Insights
⚙️ Features
📊 ESG Prediction Engine
Environmental Score Model
Social Score Model
Governance Score Model
Combined ESG Score
🧮 Feature Engineering
Emissions intensity
Energy & water efficiency
Financial ratios
Temporal features (date-based)
🧠 NLP Insights
Extract ESG-related keywords from text
Basic sentiment analysis
Risk alerts detection
📈 Dashboard
Real-time predictions
Risk categorization (Low / Medium / High)
Input-driven analysis
Company-level ESG profile

📁 Project Structure
esg-risk-intelligence-dashboard/
├── app/
│   ├── main.py
│   ├── predict.py
│   ├── nlp.py
│   ├── utils.py
│   └── assets/
├── models/
│   ├── env_model.joblib
│   ├── social_model.joblib
│   └── gov_model.joblib
├── data/
│   └── sample_input.csv
├── Dockerfile
├── requirements.txt
├── README.md
🤖 Models

Three supervised learning models were trained:

Model	Target	Description
Environmental	Env_score	Emissions, energy, water usage
Social	Social_score	Workforce, safety, diversity
Governance	Gov_score	Board structure, ethics, compliance

Each model uses:

Feature engineering
Preprocessing pipelines
XGBoost regression
🧪 How to Run Locally
1. Clone repo
git clone https://github.com/HappymoreMasoka/ESG-App.git
cd ESG-App
2. Install dependencies
pip install -r requirements.txt
3. Run app
python -m streamlit run app/main.py
4. Open in browser
http://localhost:8501
🐳 Run with Docker
Build image
docker build -t esg-dashboard .
Run container
docker run -p 7860:7860 esg-dashboard

Then open:

http://localhost:7860
☁️ Deployment (Free Options)
Streamlit Community Cloud (fastest)
Hugging Face Spaces (Docker supported)
Render (free tier)
🔍 Example Use Case

A user inputs:

emissions data
workforce metrics
governance indicators

The system outputs:

ESG score
E, S, G breakdown
risk classification
NLP insights from reports
📌 Version 1 Scope

This version focuses on:

core ML prediction pipeline
working dashboard
deployable system
🔮 Future Improvements
SHAP explainability
Industry benchmarking
Time-series ESG trends
Advanced NLP (transformers)
API (FastAPI)
OpenTelemetry observability
👤 Author

Happymore Masoka
Machine Learning Engineer
