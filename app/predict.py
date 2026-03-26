from pathlib import Path
import math
import time
import joblib
import pandas as pd

from telemetry import tracer, prediction_counter, prediction_latency

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
        df["log_total_assets"] = df["Total_assets"].clip(lower=0).apply(math.log1p)

    if "Market_cap" in df.columns:
        df["log_market_cap"] = df["Market_cap"].clip(lower=0).apply(math.log1p)

    return df


def prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    with tracer.start_as_current_span("prepare_input_features"):
        df = add_date_features(df)
        df = add_engineered_features(df)
        return df


class ESGPredictor:
    def __init__(self) -> None:
        with tracer.start_as_current_span("load_models"):
            self.env_model = joblib.load(MODEL_DIR / "env_model.joblib")
            self.social_model = joblib.load(MODEL_DIR / "social_model.joblib")
            self.gov_model = joblib.load(MODEL_DIR / "gov_model.joblib")

    def predict(self, input_df: pd.DataFrame) -> dict:
        start = time.perf_counter()

        with tracer.start_as_current_span("predict_esg_scores") as span:
            prepared = prepare_input(input_df)

            env_score = float(self.env_model.predict(prepared)[0])
            social_score = float(self.social_model.predict(prepared)[0])
            gov_score = float(self.gov_model.predict(prepared)[0])

            esg_score = 0.33 * env_score + 0.33 * social_score + 0.34 * gov_score

            duration_ms = (time.perf_counter() - start) * 1000

            prediction_counter.add(1)
            prediction_latency.record(duration_ms)

            span.set_attribute("prediction.environmental_score", env_score)
            span.set_attribute("prediction.social_score", social_score)
            span.set_attribute("prediction.governance_score", gov_score)
            span.set_attribute("prediction.esg_score", esg_score)
            span.set_attribute("prediction.latency_ms", duration_ms)

            return {
                "environmental_score": round(env_score, 2),
                "social_score": round(social_score, 2),
                "governance_score": round(gov_score, 2),
                "esg_score": round(esg_score, 2),
                "latency_ms": round(duration_ms, 2),
            }