from pathlib import Path
import pandas as pd
import streamlit as st

from predict import ESGPredictor
from nlp import analyze_esg_text
from utils import risk_label
from telemetry import setup_telemetry

setup_telemetry()

st.set_page_config(
    page_title="ESG Risk Intelligence Dashboard v2",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


@st.cache_resource
def load_predictor() -> ESGPredictor:
    return ESGPredictor()


def build_input_dataframe(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([inputs])

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f4f7f6;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #062f2a 0%, #0a4d43 100%);
        }

        section[data-testid="stSidebar"] * {
            color: white !important;
        }

        .main-header {
            background: linear-gradient(90deg, #062f2a 0%, #0e6251 60%, #0a8f6a 100%);
            padding: 1.4rem 1.8rem;
            border-radius: 18px;
            margin-bottom: 1.2rem;
            box-shadow: 0 6px 20px rgba(0,0,0,0.10);
        }

        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
            font-weight: 800;
        }

        .main-header p {
            color: #d7f3ec;
            margin-top: 0.4rem;
            margin-bottom: 0;
            font-size: 1rem;
        }

        .section-card {
            background: white;
            border-radius: 18px;
            padding: 1.2rem 1.2rem;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            margin-bottom: 1rem;
            border: 1px solid #e7eceb;
        }

        .metric-card {
            border-radius: 18px;
            padding: 1rem 1.1rem;
            color: white;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            min-height: 110px;
        }

        .metric-card h4 {
            margin: 0 0 0.35rem 0;
            font-size: 1rem;
            font-weight: 700;
        }

        .metric-card .value {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
        }

        .metric-card .sub {
            margin-top: 0.25rem;
            font-size: 0.9rem;
            opacity: 0.95;
        }

        .overall-card {
            background: linear-gradient(135deg, #0b6b57 0%, #16a085 100%);
        }

        .env-card {
            background: linear-gradient(135deg, #1b8f4b 0%, #58c472 100%);
        }

        .social-card {
            background: linear-gradient(135deg, #1f6feb 0%, #5b9cff 100%);
        }

        .gov-card {
            background: linear-gradient(135deg, #7b3fe4 0%, #a77bff 100%);
        }

        .latency-banner {
            background: #e8f1ff;
            color: #1359b3;
            padding: 0.9rem 1rem;
            border-radius: 14px;
            font-weight: 600;
            border-left: 5px solid #1f6feb;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }

        .risk-pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }

        .pill-low {
            background: #dff6e9;
            color: #157347;
        }

        .pill-medium {
            background: #fff3cd;
            color: #946200;
        }

        .pill-high {
            background: #f8d7da;
            color: #b02a37;
        }

        .subheading {
            font-size: 1.5rem;
            font-weight: 800;
            color: #1f2937;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid #e8ecef;
            padding: 0.8rem 1rem;
            border-radius: 14px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.04);
        }

        /* ============================= */
        /* 🔥 INPUT FIELD IMPROVEMENTS 🔥 */
        /* ============================= */

        /* Text inside inputs */
        input, textarea {
            color: #d32f2f !important;  /* RED TEXT */
            font-weight: 600;
        }

        /* Number inputs */
        div[data-baseweb="input"] input {
            color: #d32f2f !important;
        }

        /* Select dropdown */
        div[data-baseweb="select"] span {
            color: #d32f2f !important;
        }

        /* Input box styling */
        input {
            background-color: #ffffff !important;
            border-radius: 10px !important;
            border: 2px solid #d32f2f !important;
        }

        /* Focus effect */
        input:focus, textarea:focus {
            border: 2px solid #16a085 !important;
            box-shadow: 0 0 6px rgba(22,160,133,0.5);
        }

        /* Placeholder */
        input::placeholder, textarea::placeholder {
            color: #999 !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

def render_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>ESG Risk Intelligence Dashboard v2</h1>
            <p>ML-powered ESG scoring, analytics, NLP insights, and observability</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_card(title: str, value: float, risk: str, css_class: str):
    pill_class = "pill-medium"
    if risk == "Low Risk":
        pill_class = "pill-low"
    elif risk == "High Risk":
        pill_class = "pill-high"

    st.markdown(
        f"""
        <div class="metric-card {css_class}">
            <h4>{title}</h4>
            <div class="value">{value}</div>
            <div class="sub">/100</div>
            <div class="risk-pill {pill_class}">{risk}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_custom_css()
    render_header()

    predictor = load_predictor()

    with st.sidebar:
        st.header("Prediction Input")
        company_name = st.text_input("Company Name", "ABC Manufacturing Ltd.")
        industry = st.selectbox(
            "Industry",
            ["Manufacturing", "Energy", "Technology", "Finance", "Healthcare", "Industrial"]
        )
        date_value = st.date_input("Analysis Date")

        st.markdown("### Environmental")
        scope_1 = st.number_input("Scope 1", min_value=0.0, value=3000.0)
        scope_2 = st.number_input("Scope 2", min_value=0.0, value=1200.0)
        co2_emissions = st.number_input("CO2 Emissions", min_value=0.0, value=4500.0)
        energy_use = st.number_input("Energy Use", min_value=0.0, value=20000.0)
        water_use = st.number_input("Water Use", min_value=0.0, value=15000.0)
        water_recycle = st.number_input("Water Recycle", min_value=0.0, value=4000.0)
        toxic_chem_red = st.number_input("Toxic Chemical Reduction", min_value=0.0, value=1.0)
        recycling_initiatives = st.number_input("Recycling Initiatives", min_value=0.0, value=1.0)

        st.markdown("### Social")
        injury_rate = st.number_input("Injury Rate", min_value=0.0, value=1.2)
        women_employees = st.number_input("Women Employees %", min_value=0.0, value=35.0)
        human_rights = st.number_input("Human Rights", min_value=0.0, value=1.0)
        strikes = st.number_input("Strikes", min_value=0.0, value=0.0)
        turnover_empl = st.number_input("Turnover Employees", min_value=0.0, value=12.0)

        st.markdown("### Governance")
        board_size = st.number_input("Board Size", min_value=0.0, value=10.0)
        shareholder_rights = st.number_input("Shareholder Rights", min_value=0.0, value=1.0)
        board_gen_div = st.number_input("Board Gender Diversity %", min_value=0.0, value=40.0)
        bribery = st.number_input("Bribery Incidents", min_value=0.0, value=0.0)

        st.markdown("### Financial")
        market_cap = st.number_input("Market Cap", min_value=0.0, value=500000000.0)
        total_assets = st.number_input("Total Assets", min_value=0.0, value=1000000000.0)
        net_income = st.number_input("Net Income", value=80000000.0)
        return_on_asset = st.number_input("RETURN_ON_ASSET", value=0.08)
        quick_ratio = st.number_input("QUICK_RATIO", value=1.5)
        asset_growth = st.number_input("ASSET_GROWTH", value=0.10)
        fncl_lvrg = st.number_input("FNCL_LVRG", value=0.45)
        pe_ratio = st.number_input("PE_RATIO", value=18.0)
        bvps = st.number_input("BVPS", value=12.4)
        shares = st.number_input("Shares", min_value=0.0, value=12000000.0)

        run_prediction = st.button("Run ESG Prediction", use_container_width=True)

    report_text = st.text_area(
        "NLP Insights Input",
        height=180,
        placeholder="Paste annual report or sustainability report text here..."
    )

    if run_prediction:
        input_payload = {
            "Identifier (RIC)": "N/A",
            "Company Name": company_name,
            "Date": str(date_value),
            "BVPS": bvps,
            "Market_cap": market_cap,
            "Shares": shares,
            "Industry": industry,
            "Net_income": net_income,
            "RETURN_ON_ASSET": return_on_asset,
            "QUICK_RATIO": quick_ratio,
            "ASSET_GROWTH": asset_growth,
            "FNCL_LVRG": fncl_lvrg,
            "PE_RATIO": pe_ratio,
            "Scope_1": scope_1,
            "Scope_2": scope_2,
            "CO2_emissions": co2_emissions,
            "Energy_use": energy_use,
            "Water_use": water_use,
            "Water_recycle": water_recycle,
            "Toxic_chem_red": toxic_chem_red,
            "Injury_rate": injury_rate,
            "Women_Employees": women_employees,
            "Human_Rights": human_rights,
            "Strikes": strikes,
            "Turnover_empl": turnover_empl,
            "Board_Size": board_size,
            "Shareholder_Rights": shareholder_rights,
            "Board_gen_div": board_gen_div,
            "Bribery": bribery,
            "Recycling_Initiatives": recycling_initiatives,
            "Total_assets": total_assets,
        }

        input_df = build_input_dataframe(input_payload)
        predictions = predictor.predict(input_df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_score_card(
                "Environmental (E)",
                predictions["environmental_score"],
                risk_label(predictions["environmental_score"]),
                "env-card"
            )
        with col2:
            render_score_card(
                "Social (S)",
                predictions["social_score"],
                risk_label(predictions["social_score"]),
                "social-card"
            )
        with col3:
            render_score_card(
                "Governance (G)",
                predictions["governance_score"],
                risk_label(predictions["governance_score"]),
                "gov-card"
            )
        with col4:
            render_score_card(
                "Overall ESG",
                predictions["esg_score"],
                risk_label(predictions["esg_score"]),
                "overall-card"
            )

        st.markdown(
            f"""
            <div class="latency-banner">
                Prediction latency: {predictions['latency_ms']} ms
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([1.4, 1])

        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="subheading">Prediction Summary</div>', unsafe_allow_html=True)
            st.json({
                "Company": company_name,
                "Industry": industry,
                "Overall Risk": risk_label(predictions["esg_score"]),
                "Environmental Risk": risk_label(predictions["environmental_score"]),
                "Social Risk": risk_label(predictions["social_score"]),
                "Governance Risk": risk_label(predictions["governance_score"]),
            })
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="subheading">Input Snapshot</div>', unsafe_allow_html=True)
            st.dataframe(input_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="subheading">NLP Analysis</div>', unsafe_allow_html=True)

            if report_text.strip():
                nlp_result = analyze_esg_text(report_text)
                st.write("**Sentiment:**", nlp_result["sentiment"])
                st.write("**Environmental keywords:**", nlp_result["environmental_keywords"])
                st.write("**Social keywords:**", nlp_result["social_keywords"])
                st.write("**Governance keywords:**", nlp_result["governance_keywords"])
                st.write("**Alerts:**", nlp_result["alerts"] if nlp_result["alerts"] else "None")
            else:
                st.info("Paste report text to generate NLP insights.")

            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.info("Enter your ESG inputs in the sidebar, then click Run ESG Prediction.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()