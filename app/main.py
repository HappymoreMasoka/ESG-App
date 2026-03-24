from pathlib import Path
import pandas as pd
import streamlit as st

from predict import ESGPredictor
from nlp import analyze_esg_text
from utils import risk_label

st.set_page_config(page_title="ESG Risk Intelligence Dashboard", layout="wide")

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@st.cache_resource
def load_predictor() -> ESGPredictor:
    return ESGPredictor()


def build_input_dataframe(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame([inputs])


def main() -> None:
    st.title("ESG Risk Intelligence Dashboard")
    st.caption("ML-powered ESG scoring and analytics platform")

    predictor = load_predictor()

    with st.sidebar:
        st.header("Company Input")

        company_name = st.text_input("Company Name", "ABC Manufacturing Ltd.")
        industry = st.selectbox(
            "Industry",
            ["Manufacturing", "Energy", "Technology", "Finance", "Healthcare", "Industrial"]
        )
        date_value = st.date_input("Analysis Date")

        st.subheader("Environmental Inputs")
        scope_1 = st.number_input("Scope 1", min_value=0.0, value=3000.0)
        scope_2 = st.number_input("Scope 2", min_value=0.0, value=1200.0)
        co2_emissions = st.number_input("CO2 Emissions", min_value=0.0, value=4500.0)
        energy_use = st.number_input("Energy Use", min_value=0.0, value=20000.0)
        water_use = st.number_input("Water Use", min_value=0.0, value=15000.0)
        water_recycle = st.number_input("Water Recycle", min_value=0.0, value=4000.0)
        toxic_chem_red = st.number_input("Toxic Chemical Reduction", min_value=0.0, value=1.0)
        recycling_initiatives = st.number_input("Recycling Initiatives", min_value=0.0, value=1.0)

        st.subheader("Social Inputs")
        injury_rate = st.number_input("Injury Rate", min_value=0.0, value=1.2)
        women_employees = st.number_input("Women Employees %", min_value=0.0, value=35.0)
        human_rights = st.number_input("Human Rights", min_value=0.0, value=1.0)
        strikes = st.number_input("Strikes", min_value=0.0, value=0.0)
        turnover_empl = st.number_input("Turnover Employees", min_value=0.0, value=12.0)

        st.subheader("Governance Inputs")
        board_size = st.number_input("Board Size", min_value=0.0, value=10.0)
        shareholder_rights = st.number_input("Shareholder Rights", min_value=0.0, value=1.0)
        board_gen_div = st.number_input("Board Gender Diversity %", min_value=0.0, value=40.0)
        bribery = st.number_input("Bribery Incidents", min_value=0.0, value=0.0)

        st.subheader("Financial Inputs")
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

    col1, col2 = st.columns([2, 1])

    with col2:
        report_text = st.text_area(
            "NLP Insights Input",
            height=220,
            placeholder="Paste annual report, sustainability report, or ESG-related text here..."
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

        with col1:
            a, b, c, d = st.columns(4)
            a.metric("Environmental (E)", predictions["environmental_score"])
            b.metric("Social (S)", predictions["social_score"])
            c.metric("Governance (G)", predictions["governance_score"])
            d.metric("Overall ESG", predictions["esg_score"])

            st.subheader("Prediction Summary")
            st.write(
                {
                    "Company": company_name,
                    "Industry": industry,
                    "Overall Risk": risk_label(predictions["esg_score"]),
                    "Environmental Risk": risk_label(predictions["environmental_score"]),
                    "Social Risk": risk_label(predictions["social_score"]),
                    "Governance Risk": risk_label(predictions["governance_score"]),
                }
            )

            st.subheader("Input Snapshot")
            st.dataframe(input_df, use_container_width=True)

        with col2:
            if report_text.strip():
                nlp_result = analyze_esg_text(report_text)
                st.subheader("NLP Analysis")
                st.write("**Sentiment:**", nlp_result["sentiment"])
                st.write("**Environmental keywords:**", nlp_result["environmental_keywords"])
                st.write("**Social keywords:**", nlp_result["social_keywords"])
                st.write("**Governance keywords:**", nlp_result["governance_keywords"])
                st.write("**Alerts:**", nlp_result["alerts"] if nlp_result["alerts"] else "None")
            else:
                st.info("Paste report text to generate NLP insights.")

    else:
        st.info("Enter company data in the sidebar and click 'Run ESG Prediction'.")


if __name__ == "__main__":
    main()