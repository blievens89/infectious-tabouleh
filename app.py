# DataForSEO Labs — Keyword Ideas + Intent Planner (Live, trial-safe)
# - Seed keyword -> up to 20 suggestions with volume/CPC/competition
# - Search Intent (language-only, no location needed)
# - Group by intent with CTR/CVR assumptions + blended totals
# - Password gate (session-state) + caching + raw-debug

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Import the official RestClient
from Client.client import RestClient

st.set_page_config(page_title="Labs Keyword Ideas + Intent (Live)", layout="wide")
st.title("DataForSEO Labs — Keyword & Intent Planner")

# --- Initialise Session State ---
# This ensures the variables exist on the first run
if 'results' not in st.session_state:
    st.session_state.results = None

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
    BASE_URL = "/v3"
except Exception as e:
    st.error(f"Failed to initialise the API client: {e}")
    st.stop()

# --- Helper Functions ---

def make_api_post_request(endpoint: str, payload: dict) -> dict:
    """
    Makes a POST request to a DataForSEO endpoint and handles errors.
    Note: The payload must be a dictionary as expected by the official RestClient.
    """
    try:
        response = client.post(f"{BASE_URL}{endpoint}", payload)
        # A successful response should have a 20000 status_code.
        if response and response.get("status_code") == 20000:
            return response
        else:
            st.error(f"API Error on {endpoint}: {response.get('status_message', 'Unknown error')}")
            st.json(response) # Show the full error response for debugging
            return {}
    except Exception as e:
        st.error(f"An exception occurred while calling the API endpoint {endpoint}: {e}")
        return {}

def extract_items_from_response(response: dict) -> list[dict]:
    """
    Safely extracts the 'items' list from a standard DataForSEO API response.
    """
    try:
        # Check if the task was successful before trying to access results
        if response.get("tasks_error", 1) > 0:
            st.warning("The API task returned an error. See raw response for details.")
            return []
        return response["tasks"][0]["result"]
    except (KeyError, IndexError, TypeError):
        return []

def safe_average(series: pd.Series) -> float:
    """
    Calculates a safe average of a pandas Series, ignoring non-numeric values.
    """
    numeric_series = pd.to_numeric(series, errors='coerce').dropna()
    return numeric_series.mean() if not numeric_series.empty else np.nan

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Inputs")
    
    # NEW: Analysis mode selector
    analysis_mode = st.radio(
        "Analysis Mode",
        ("Generate from Seed Keyword", "Analyse My Keyword List"),
        key="analysis_mode"
    )

    language_code = st.text_input("Language Code (e.g., en, fr, de)", value="en")
    location_name = st.text_input("Location Name", value="United Kingdom")
    
    # Conditional UI based on analysis mode
    if analysis_mode == "Generate from Seed Keyword":
        seed_keyword = st.text_input("Seed Keyword", value="remortgage")
        limit = st.slider("Max Keyword Ideas", 10, 300, 50, step=10)
        uploaded_keywords = None
    else:
        st.subheader("Your Keywords")
        pasted_keywords = st.text_area("Paste keywords here (one per line)")
        uploaded_file = st.file_uploader("Or upload a TXT/CSV file", type=['txt', 'csv'])
        
        uploaded_keywords = []
        if pasted_keywords:
            uploaded_keywords.extend(pasted_keywords.strip().split('\n'))
        if uploaded_file:
            stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
            # Read all lines and strip them
            lines = [line.strip() for line in stringio.readlines()]
            uploaded_keywords.extend(lines)
        
        # Deduplicate
        uploaded_keywords = list(dict.fromkeys(filter(None, uploaded_keywords)))


    usd_to_gbp_rate = st.number_input("USD to GBP Exchange Rate", 0.1, 2.0, 0.79, 0.01)

    st.divider()
    st.caption("CTR/CVR Assumptions by Intent")
    intents = ["informational", "navigational", "commercial", "transactional"]
    ctr_defaults = {"informational": 0.03, "navigational": 0.03, "commercial": 0.04, "transactional": 0.04}
    cvr_defaults = {"informational": 0.015, "navigational": 0.015, "commercial": 0.03, "transactional": 0.03}
    ctrs, cvrs = {}, {}
    for intent in intents:
        col1, col2 = st.columns(2)
        with col1:
            ctrs[intent] = st.number_input(f"{intent.title()} CTR", 0.0, 1.0, ctr_defaults[intent], 0.005, format="%.3f", key=f"ctr_{intent}")
        with col2:
            cvrs[intent] = st.number_input(f"{intent.title()} CVR", 0.0, 1.0, cvr_defaults[intent], 0.005, format="%.3f", key=f"cvr_{intent}")

    st.divider()
    show_raw_data = st.toggle("Show Raw API Data (for debugging)", value=False)


# --- API FUNCTIONS ---

@st.cache_data(ttl=3600, show_spinner="Fetching keyword suggestions...")
def get_keyword_suggestions(seed: str, lang_code: str, loc_name: str, limit: int) -> pd.DataFrame:
    payload_item = {
        "keyword": seed.strip(),
        "language_code": lang_code.strip(),
        "location_name": loc_name.strip(),
        "limit": limit,
    }
    post_data = {0: payload_item}
    response = make_api_post_request("/dataforseo_labs/google/keyword_suggestions/live", post_data)
    items = extract_items_from_response(response)
    
    if not items or 'items' not in items[0]:
        return pd.DataFrame()

    rows = []
    for item in items[0]['items']:
        info = item.get("keyword_info", {})
        cpc = info.get("cpc")
        rows.append({
            "keyword": item.get("keyword"),
            "search_volume": info.get("search_volume"),
            "cpc_usd": cpc.get("cpc") if isinstance(cpc, dict) else cpc,
            "competition": info.get("competition"),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=3600, show_spinner="Fetching metrics for your keywords...")
def get_keyword_metrics(keywords: list, lang_code: str, loc_name: str) -> pd.DataFrame:
    payload_item = {
        "keywords": keywords,
        "language_code": lang_code.strip(),
        "location_name": loc_name.strip(),
    }
    post_data = {0: payload_item}
    response = make_api_post_request("/keywords_data/google_ads/search_volume/live", post_data)
    items = extract_items_from_response(response)

    rows = []
    for item in items:
        rows.append({
            "keyword": item.get("keyword"),
            "search_volume": item.get("search_volume"),
            "cpc_usd": item.get("cpc"),
            "competition": item.get("competition"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner="Analysing search intent...")
def get_search_intent(keywords: list) -> pd.DataFrame:
    payload_item = {"keywords": keywords}
    post_data = {0: payload_item}
    response = make_api_post_request("/dataforseo_labs/google/search_intent/live", post_data)
    items = extract_items_from_response(response)

    if not items or 'items' not in items[0]:
        return pd.DataFrame()

    rows = []
    for item in items[0]['items']:
        intent_info = item.get("keyword_intent", {})
        rows.append({
            "keyword_clean": (item.get("keyword") or "").lower().strip(),
            "intent": intent_info.get("label"),
            "intent_probability": intent_info.get("probability")
        })
    return pd.DataFrame(rows)


# --- Main Application Logic ---

if st.button("Analyse Keywords", type="primary"):
    
    df_metrics = pd.DataFrame()

    if analysis_mode == "Generate from Seed Keyword":
        df_metrics = get_keyword_suggestions(seed_keyword, language_code, location_name, limit)
    elif uploaded_keywords:
        # Chunk keywords into batches of 1000 for the API
        keyword_chunks = [uploaded_keywords[i:i + 1000] for i in range(0, len(uploaded_keywords), 1000)]
        results_list = []
        for chunk in keyword_chunks:
            results_list.append(get_keyword_metrics(chunk, language_code, location_name))
        df_metrics = pd.concat(results_list, ignore_index=True)

    if df_metrics.empty:
        st.warning("Could not retrieve keyword metrics. Please check your inputs or try different keywords.")
        st.session_state.results = None
    else:
        df_metrics['keyword_clean'] = df_metrics['keyword'].str.lower().str.strip()
        
        # Get intent for the retrieved keywords
        intent_keywords = df_metrics['keyword_clean'].tolist()
        df_intent = get_search_intent(intent_keywords)

        if not df_intent.empty:
            df_merged = pd.merge(df_metrics, df_intent, on="keyword_clean", how="left")
            df_merged = df_merged.drop(columns=['keyword_clean'])
        else:
            df_merged = df_metrics.drop(columns=['keyword_clean'])
            df_merged['intent'] = 'unknown'
            df_merged['intent_probability'] = None

        st.session_state.results = {"df_merged": df_merged.dropna(subset=['keyword'])}


# --- Display Results (if they exist in session state) ---
if st.session_state.results:
    df_merged = st.session_state.results["df_merged"]

    df_merged["cpc_gbp"] = (pd.to_numeric(df_merged["cpc_usd"], errors="coerce") * usd_to_gbp_rate).round(2)
    st.subheader("Keyword Analysis Results")
    st.dataframe(df_merged[['keyword', 'search_volume', 'cpc_gbp', 'competition', 'intent']], use_container_width=True)

    unmatched_keywords = df_merged[df_merged['intent'].isna()]['keyword'].tolist()
    if unmatched_keywords:
        with st.expander(f"Debug: {len(unmatched_keywords)} keywords could not be assigned an intent"):
            st.write(unmatched_keywords)

    summary_df = df_merged.dropna(subset=['intent'])

    if not summary_df.empty and 'unknown' not in summary_df['intent'].unique():
        summary = summary_df.groupby("intent").agg(
            keywords=("keyword", "count"),
            total_volume=("search_volume", "sum"),
            avg_cpc_gbp=("cpc_gbp", safe_average),
        ).reset_index().rename(columns={"intent": "Intent"})

        summary["CTR"] = summary["Intent"].map(ctrs)
        summary["CVR"] = summary["Intent"].map(cvrs)
        summary["Clicks"] = (summary["total_volume"] * summary["CTR"]).round(0)
        summary["Avg CPC £"] = summary["avg_cpc_gbp"].round(2)
        summary["Spend £"] = (summary["Clicks"] * summary["Avg CPC £"]).round(2)
        summary["Conversions"] = (summary["Clicks"] * summary["CVR"]).round(0)
        summary["CPA £"] = (summary["Spend £"] / summary["Conversions"]).replace([np.inf, -np.inf], 0).round(2)

        st.subheader("Grouped by Search Intent")
        st.dataframe(summary.fillna("—"), use_container_width=True)

        # --- NEW: Visualisation Section ---
        st.subheader("Performance by Intent")
        
        chart_metric = st.selectbox(
            "Choose a metric to visualise",
            ("Total Volume", "Clicks", "Spend £", "Conversions", "CPA £")
        )

        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('Intent:N', sort='-y'),
            y=alt.Y(f'{chart_metric}:Q', title=chart_metric),
            tooltip=['Intent', chart_metric]
        ).properties(
            title=f'{chart_metric} by Search Intent'
        )
        st.altair_chart(chart, use_container_width=True)


        # --- Blended Overview ---
        total_keywords = summary["keywords"].sum()
        total_volume = summary["total_volume"].sum()
        total_clicks = summary["Clicks"].sum()
        total_spend = summary["Spend £"].sum()
        total_conversions = summary["Conversions"].sum()

        blended_cpc = total_spend / total_clicks if total_clicks > 0 else 0
        blended_ctr = total_clicks / total_volume if total_volume > 0 else 0
        blended_cvr = total_conversions / total_clicks if total_clicks > 0 else 0
        blended_cpa = total_spend / total_conversions if total_conversions > 0 else 0
        
        blended_overview = pd.DataFrame({
            "Total Keywords": [int(total_keywords)],
            "Total Volume": [int(total_volume)],
            "Weighted Avg CPC £": [round(blended_cpc, 2)],
            "Weighted CTR": [round(blended_ctr, 3)],
            "Total Clicks": [int(total_clicks)],
            "Weighted CVR": [round(blended_cvr, 3)],
            "Total Conversions": [int(total_conversions)],
            "Total Spend £": [round(total_spend, 2)],
            "Blended CPA £": [round(blended_cpa, 2)]
        })
        st.subheader("Blended Overview (Weighted)")
        st.dataframe(blended_overview, use_container_width=True)
    else:
        st.warning("Could not generate intent summary as no intent data was returned.")

    # --- Download Buttons ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("Download Detailed Data (CSV)", df_merged.to_csv(index=False).encode("utf-8"), "keyword_intent_details.csv", "text/csv", key="d1")
    if not summary_df.empty and 'unknown' not in summary_df['intent'].unique():
        with col2:
            st.download_button("Download Intent Summary (CSV)", summary.to_csv(index=False).encode("utf-8"), "intent_summary.csv", "text/csv", key="d2")
        with col3:
            st.download_button("Download Blended Overview (CSV)", blended_overview.to_csv(index=False).encode("utf-8"), "blended_overview.csv", "text/csv", key="d3")

    num_keywords = len(df_merged)
    cost_sug = 0.01 + num_keywords * 0.0001
    cost_int = 0.001 + num_keywords * 0.0001
    approx_cost = cost_sug + cost_int
    st.caption(f"Approximate API cost for this run: ${approx_cost:.4f} for {num_keywords} keywords (estimate only).")

