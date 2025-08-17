# DataForSEO Labs — Keyword Ideas + Intent Planner (Live, trial-safe)
# - Seed keyword -> up to 20 suggestions with volume/CPC/competition
# - Search Intent (language-only, no location needed)
# - Group by intent with CTR/CVR assumptions + blended totals
# - Password gate (session-state) + caching + raw-debug

import os
import math
import streamlit as st
import pandas as pd
import numpy as np
import requests # Keep requests for the fallback client, just in case.

# Import the official RestClient
from Client.client import RestClient

st.set_page_config(page_title="Labs Keyword Ideas + Intent (Live)", layout="wide")
st.title("DataForSEO Labs — Keyword Ideas → Intent Planner")

# --- Password Gate ---
# Ensures the application is not publicly accessible without a password.
if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    password_input = st.text_input("Password", type="password")
    if st.button("Enter"):
        # Compare the input with the secret stored in Streamlit Cloud.
        if password_input == st.secrets.get("APP_PASSWORD"):
            st.session_state.authed = True
            st.rerun() # Rerun the script to show the main app
        else:
            st.error("The password you entered is incorrect.")
    st.stop() # Stop execution until authenticated

# --- API Credentials ---
# Fetches DataForSEO credentials securely from Streamlit secrets.
DATAFORSEO_LOGIN = st.secrets.get("DATAFORSEO_LOGIN")
DATAFORSEO_PASSWORD = st.secrets.get("DATAFORSEO_PASSWORD")

if not (DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD):
    st.error("DataForSEO API credentials are not found. Please set DATAFORSEO_LOGIN and DATAFORSEO_PASSWORD in your Streamlit secrets.")
    st.stop()

# --- API Client Initialisation ---
# Use the provided RestClient to interact with the DataForSEO API.
try:
    client = RestClient(DATAFORSEO_LOGIN, DATAFORSEO_PASSWORD)
    LABS_BASE_URL = "/v3/dataforseo_labs"
except Exception as e:
    st.error(f"Failed to initialise the API client: {e}")
    st.stop()

# --- Helper Functions ---

def make_api_post_request(endpoint: str, payload: list[dict]) -> dict:
    """
    Makes a POST request to a DataForSEO Labs endpoint and handles errors.
    """
    try:
        response = client.post(f"{LABS_BASE_URL}{endpoint}", payload)
        if response and response.get("status_code") == 20000:
            return response
        else:
            st.error(f"API Error on {endpoint}: {response.get('status_message', 'Unknown error')}")
            return {}
    except Exception as e:
        st.error(f"An exception occurred while calling the API endpoint {endpoint}: {e}")
        return {}

def extract_items_from_response(response: dict) -> list[dict]:
    """
    Safely extracts the 'items' list from a standard DataForSEO API response.
    """
    try:
        return response["tasks"][0]["result"][0]["items"]
    except (KeyError, IndexError, TypeError):
        return []

def calculate_monthly_average_searches(monthly_searches: list[dict]) -> float:
    """
    Calculates the average search volume from a list of monthly search data.
    """
    if not monthly_searches:
        return np.nan
    volumes = [
        m.get("search_volume") for m in monthly_searches if isinstance(m, dict) and "search_volume" in m
    ]
    return np.mean(volumes) if volumes else np.nan

def safe_average(series: pd.Series) -> float:
    """
    Calculates a safe average of a pandas Series, ignoring non-numeric values.
    """
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    return numeric_series.mean() if not numeric_series.empty else np.nan

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Inputs")
    seed_keyword = st.text_input("Seed Keyword", value="remortgage")
    language_code = st.text_input("Language Code (e.g., en, fr, de)", value="en")

    use_location_for_suggestions = st.toggle(
        "Use Location for Suggestions",
        value=True,
        help="Keyword Suggestions can optionally be filtered by location. This does not apply to Search Intent."
    )
    location_name = "United Kingdom"
    if use_location_for_suggestions:
        location_name = st.text_input("Location Name", value="United Kingdom")

    limit = st.slider("Max Keyword Ideas", 5, 20, 20, step=5)
    usd_to_gbp_rate = st.number_input("USD to GBP Exchange Rate", 0.1, 2.0, 0.79, 0.01)

    st.divider()
    st.caption("CTR/CVR Assumptions by Intent")
    intents = ["informational", "navigational", "commercial", "transactional"]
    ctr_defaults = {"informational": 0.02, "navigational": 0.04, "commercial": 0.05, "transactional": 0.06}
    cvr_defaults = {"informational": 0.01, "navigational": 0.02, "commercial": 0.03, "transactional": 0.08}
    ctrs, cvrs = {}, {}
    for intent in intents:
        col1, col2 = st.columns(2)
        with col1:
            ctrs[intent] = st.number_input(f"{intent.title()} CTR", 0.0, 1.0, ctr_defaults[intent], 0.005, format="%.3f", key=f"ctr_{intent}")
        with col2:
            cvrs[intent] = st.number_input(f"{intent.title()} CVR", 0.0, 1.0, cvr_defaults[intent], 0.005, format="%.3f", key=f"cvr_{intent}")

    st.divider()
    show_raw_data = st.toggle("Show Raw API Data (for debugging)", value=False)


# --- REFACTORED API FUNCTION ---

@st.cache_data(ttl=3600, show_spinner="Fetching keywords and analysing intent...")
def get_keywords_and_intent(seed: str, lang_code: str, loc_name: str | None, limit: int) -> tuple[pd.DataFrame, dict, dict]:
    """
    A single function to fetch keyword suggestions and then immediately fetch their search intent.
    This avoids caching conflicts between two separate functions.
    """
    # --- Part 1: Get Keyword Suggestions ---
    sug_payload_item = {
        "keyword": seed.strip(),
        "language_code": lang_code.strip(),
        "limit": limit,
        "include_seed_keyword": False,
    }
    if loc_name:
        sug_payload_item["location_name"] = loc_name.strip()

    sug_payload = [sug_payload_item]
    sug_response = make_api_post_request("/google/keyword_suggestions/live", sug_payload)
    sug_items = extract_items_from_response(sug_response)

    if not sug_items:
        return pd.DataFrame(), sug_response, {}

    rows = []
    for item in sug_items:
        info = item.get("keyword_info", {})
        cpc = info.get("cpc")
        rows.append({
            "keyword": item.get("keyword"),
            "search_volume": info.get("search_volume"),
            "cpc_usd": cpc.get("cpc") if isinstance(cpc, dict) else cpc,
            "competition": info.get("competition"),
            "avg_monthly_searches": calculate_monthly_average_searches(info.get("monthly_searches"))
        })

    df_kw = pd.DataFrame(rows).dropna(subset=["keyword"]).drop_duplicates(subset=["keyword"]).reset_index(drop=True)
    df_kw['search_volume'] = df_kw['search_volume'].fillna(df_kw['avg_monthly_searches'])
    
    # Standardise the keyword column immediately
    df_kw['keyword'] = df_kw['keyword'].str.lower().str.strip()
    
    # --- Part 2: Get Search Intent ---
    keyword_list = df_kw["keyword"].tolist()
    if not keyword_list:
        return df_kw, sug_response, {}

    intent_payload_item = {
        "keywords": keyword_list,
        "language_code": lang_code.strip()
    }
    intent_payload = [intent_payload_item]
    intent_response = make_api_post_request("/google/search_intent/live", intent_payload)
    intent_items = extract_items_from_response(intent_response)

    intent_rows = []
    for item in intent_items:
        intent_info = item.get("search_intent_info", {})
        intent_rows.append({
            "keyword": item.get("keyword"),
            "intent": intent_info.get("main_intent"),
            "intent_probability": intent_info.get("probability")
        })
    df_intent = pd.DataFrame(intent_rows)

    if not df_intent.empty:
        df_intent['keyword'] = df_intent['keyword'].str.lower().str.strip()
        # --- Part 3: Merge the data ---
        df_merged = pd.merge(df_kw, df_intent, on="keyword", how="left")
        return df_merged, sug_response, intent_response
    else:
        # If intent call fails, return suggestions df with empty intent columns
        df_kw['intent'] = None
        df_kw['intent_probability'] = None
        return df_kw, sug_response, intent_response


# --- Main Application Logic ---

if st.button("Fetch Ideas & Analyse Intent", type="primary"):
    loc_for_sug = location_name if use_location_for_suggestions else None
    
    # Call the single, combined function
    df_merged, raw_sug, raw_int = get_keywords_and_intent(seed_keyword, language_code, loc_for_sug, limit)

    if df_merged.empty:
        st.warning("No keyword suggestions were returned. Please try a different seed keyword or adjust the settings.")
        st.stop()

    df_merged["cpc_gbp"] = (pd.to_numeric(df_merged["cpc_usd"], errors="coerce") * usd_to_gbp_rate).round(2)
    st.subheader("Keyword Ideas & Metrics")
    st.dataframe(df_merged[['keyword', 'search_volume', 'cpc_gbp', 'competition', 'intent']], use_container_width=True)


    if show_raw_data:
        with st.expander("Debug: Raw API Data"):
            st.write("Keyword Suggestions Response:")
            st.json(raw_sug)
            st.write("Search Intent Response:")
            st.json(raw_int)

    # Check for any keywords that failed to match
    unmatched_keywords = df_merged[df_merged['intent'].isna()]['keyword'].tolist()
    if unmatched_keywords:
        with st.expander(f"Debug: {len(unmatched_keywords)} keywords could not be assigned an intent"):
            st.write(unmatched_keywords)

    # Filter out rows where intent is missing for accurate summary
    summary_df = df_merged.dropna(subset=['intent'])

    # Summarise by Intent
    if not summary_df.empty:
        summary = summary_df.groupby("intent").agg(
            keywords=("keyword", "count"),
            total_volume=("search_volume", "sum"),
            avg_cpc_gbp=("cpc_gbp", safe_average),
            avg_intent_prob=("intent_probability", safe_average)
        ).reset_index().rename(columns={"intent": "Intent"})

        summary["CTR"] = summary["Intent"].map(ctrs)
        summary["CVR"] = summary["Intent"].map(cvrs)
        summary["Clicks"] = (summary["total_volume"] * summary["CTR"]).round(0)
        summary["Avg CPC £"] = summary["avg_cpc_gbp"].round(2)
        summary["Spend £"] = (summary["Clicks"] * summary["Avg CPC £"]).round(2)
        summary["Conversions"] = (summary["Clicks"] * summary["CVR"]).round(0)

        st.subheader("Grouped by Search Intent")
        st.dataframe(summary.fillna("—"), use_container_width=True)

        # Blended Overview
        blended_overview = pd.DataFrame({
            "Total Keywords": [int(summary["keywords"].sum())],
            "Total Volume": [int(summary["total_volume"].sum())],
            "Blended Avg CPC £": [round(summary["avg_cpc_gbp"].mean(), 2) if not summary["avg_cpc_gbp"].isnull().all() else 0],
            "Blended CTR": [round(summary["CTR"].mean(), 3)],
            "Total Clicks": [int(summary["Clicks"].sum())],
            "Blended CVR": [round(summary["CVR"].mean(), 3)],
            "Total Conversions": [int(summary["Conversions"].sum())],
            "Total Spend £": [round(summary["Spend £"].sum(), 2)]
        })
        st.subheader("Blended Overview (All Keywords)")
        st.dataframe(blended_overview, use_container_width=True)
    else:
        st.warning("Could not generate intent summary as no intent data was returned.")


    # Download Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download Detailed Data (CSV)",
            df_merged.to_csv(index=False).encode("utf-8"),
            "keyword_intent_details.csv",
            "text/csv"
        )
    if not summary_df.empty:
        with col2:
            st.download_button(
                "Download Intent Summary (CSV)",
                summary.to_csv(index=False).encode("utf-8"),
                "intent_summary.csv",
                "text/csv"
            )
        with col3:
            st.download_button(
                "Download Blended Overview (CSV)",
                blended_overview.to_csv(index=False).encode("utf-8"),
                "blended_overview.csv",
                "text/csv"
            )

    # Cost Estimation
    num_keywords = len(df_merged)
    cost_sug = 0.01 + num_keywords * 0.0001
    cost_int = 0.001 + num_keywords * 0.0001
    approx_cost = cost_sug + cost_int
    st.caption(f"Approximate API cost for this run: ${approx_cost:.4f} for {num_keywords} keywords (estimate only).")

