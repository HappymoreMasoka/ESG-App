from pathlib import Path
import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["date_year"] = df["Date"].dt.year
        df["date_month"] = df["Date"].dt.month
        df["date_quarter"] = df["Date"].dt.quarter
        df["date_is_year_end"] = df["Date"].dt.is_year_end.astype(float)
        df = df.drop(columns=["Date"])
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    if {"CO2_emissions", "Total_assets"}.issubset(df.columns):
        df["emissions_intensity"] = df["CO2_emissions"] / (df["Total_assets"] + eps)

    if {"Energy_use", "Total_assets"}.issubset(df.columns):
        df["energy_intensity"] = df["Energy_use"] / (df["Total_assets"] + eps)

    if {"Water_use", "Total_assets"}.issubset(df.columns):
        df["water_intensity"] = df["Water_use"] / (df["Total_assets"] + eps)

    if {"Water_recycle", "Water_use"}.issubset(df.columns):
        df["water_recycle_ratio"] = df["Water_recycle"] / (df["Water_use"] + eps)

    if {"Scope_1", "Scope_2"}.issubset(df.columns):
        df["total_scope_emissions"] = df["Scope_1"] + df["Scope_2"]

    if {"Net_income", "Total_assets"}.issubset(df.columns):
        df["profitability_ratio"] = df["Net_income"] / (df["Total_assets"] + eps)

    if "Total_assets" in df.columns:
        df["log_total_assets"] = pd.Series(df["Total_assets"]).clip(lower=0).apply(
            lambda x: __import__("math").log1p(x)
        )

    if "Market_cap" in df.columns:
        df["log_market_cap"] = pd.Series(df["Market_cap"]).clip(lower=0).apply(
            lambda x: __import__("math").log1p(x)
        )

    return df


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    df = add_date_features(df)
    df = add_engineered_features(df)
    return df


class ESGPredictor:
    def __init__(self) -> None:
        self.env_model = joblib.load(MODEL_DIR / "env_model.joblib")
        self.social_model = joblib.load(MODEL_DIR / "social_model.joblib")
        self.gov_model = joblib.load(MODEL_DIR / "gov_model.joblib")

    def predict(self, input_df: pd.DataFrame) -> dict:
        prepared = prepare_input(input_df)

        env_score = float(self.env_model.predict(prepared)[0])
        social_score = float(self.social_model.predict(prepared)[0])
        gov_score = float(self.gov_model.predict(prepared)[0])

        esg_score = 0.33 * env_score + 0.33 * social_score + 0.34 * gov_score

        return {
            "environmental_score": round(env_score, 2),
            "social_score": round(social_score, 2),
            "governance_score": round(gov_score, 2),
            "esg_score": round(esg_score, 2),
        }