# Streamlit • DataForSEO Labs • Intent Planner (Live, credit-friendly)
# - Seed keyword -> up to N suggestions (default 20)
# - Metrics: search volume, CPC (USD->GBP), competition
# - Search Intent (Live) per keyword
# - Group by intent with CTR/CVR assumptions and blended overview
# - Uses DataForSEO RestClient (preferred); falls back to minimal client if missing
# - Password gate + caching to protect secrets and credits

import os, math
import requests
import pandas as pd
import numpy as np
import streamlit as st

# ----------------------------
# Password gate (set APP_PASSWORD in secrets)
# ----------------------------
def gate():
    pw = st.text_input("Password", type="password")
    if pw != st.secrets.get("APP_PASSWORD"):
        st.info("Enter password to continue.")
        st.stop()

st.set_page_config(page_title="Labs Intent Planner (Live)", layout="wide")
st.title("DataForSEO Labs — Intent Planner (Live)")

# Uncomment to enable the gate (recommended for public Streamlit URLs)
gate()

# ----------------------------
# Secrets / credentials
# ----------------------------
LOGIN = st.secrets.get("DATAFORSEO_LOGIN", os.getenv("DATAFORSEO_LOGIN", ""))
PASSWORD = st.secrets.get("DATAFORSEO_PASSWORD", os.getenv("DATAFORSEO_PASSWORD", ""))
if not (LOGIN and PASSWORD):
    st.error("Missing DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD in Streamlit Secrets (or env).")
    st.stop()

# ----------------------------
# RestClient (as per DataForSEO docs)
# ----------------------------
try:
    # Preferred: official client from https://cdn.dataforseo.com/v3/examples/python/python_Client.zip
    from client import RestClient  # noqa: E402
except Exception:
    # Fallback minimal client so you can run without downloading their zip immediately
    class RestClient:  # type: ignore
        def __init__(self, login, password, base_url="https://api.dataforseo.com"):
            self.login = login
            self.password = password
            self.base_url = base_url.rstrip("/")
        def post(self, path, payload):
            url = f"{self.base_url}{path}"
            r = requests.post(url, auth=(self.login, self.password), json=payload, timeout=60)
            try:
                r.raise_for_status()
            except requests.HTTPError as e:
                try:
                    msg = r.json()
                except Exception:
                    msg = r.text[:500]
                raise RuntimeError(f"POST {path} -> HTTP {r.status_code}: {msg}") from e
            return r.json()

client = RestClient(LOGIN, PASSWORD)

LABS_BASE = "/v3/dataforseo_labs"

# ----------------------------
# Helpers
# ----------------------------
def labs_post(endpoint_path: str, payload: list[dict]) -> dict:
    """Call a Labs Live endpoint with RestClient; returns JSON."""
    return client.post(f"{LABS_BASE}{endpoint_path}", payload)

def extract_items(resp: dict) -> list[dict]:
    """Flatten Labs response to items list."""
    try:
        tasks = resp.get("tasks", [])
        if not tasks:
            return []
        result = tasks[0].get("result", [])
        if not result:
            return []
        items = result[0].get("items", [])
        return items or []
    except Exception:
        return []

def monthly_avg(monthly_searches: list[dict]) -> float:
    if not monthly_searches:
        return float("nan")
    vals = [m.get("search_volume") for m in monthly_searches if isinstance(m, dict)]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return float(np.mean(vals)) if vals else float("nan")

def safe_avg(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.mean() if not s.empty else np.nan

# ----------------------------
# Sidebar settings
# ----------------------------
with st.sidebar:
    st.header("Settings")

    seed_keyword = st.text_input("Seed keyword", value="sell property fast")
    # UK default (you can change): 2826 per DataForSEO location codes
    location_code = st.number_input("location_code (UK=2826)", min_value=1, value=2826, step=1)
    language_name = st.text_input("language_name", value="English")

    # Hard cap to protect trial credits
    limit = st.slider("Max keyword ideas (protect trial credit)", 5, 20, 20, step=5,
                      help="Labs returns metrics with suggestions; 20 ideas keeps costs tiny.")

    # Currency conversion
    usd_to_gbp = st.number_input("USD→GBP rate", min_value=0.1, max_value=2.0, value=0.78, step=0.01)

    st.divider()
    st.caption("CTR/CVR assumptions by intent")
    intents = ["informational", "navigational", "commercial", "transactional"]
    ctr_defaults = {"informational": 0.02, "navigational": 0.04, "commercial": 0.05, "transactional": 0.06}
    cvr_defaults = {"informational": 0.01, "navigational": 0.02, "commercial": 0.03, "transactional": 0.08}
    ctrs, cvrs = {}, {}
    for i in intents:
        c1, c2 = st.columns(2)
        with c1:
            ctrs[i] = st.number_input(f"{i.title()} CTR", min_value=0.0, max_value=1.0,
                                      value=ctr_defaults[i], step=0.005, format="%.3f")
        with c2:
            cvrs[i] = st.number_input(f"{i.title()} CVR", min_value=0.0, max_value=1.0,
                                      value=cvr_defaults[i], step=0.005, format="%.3f")

    st.divider()
    budget = st.number_input("Optional monthly budget (£)", min_value=0.0, value=0.0, step=100.0,
                             help="If >0, we’ll show a simple budget allocation check.")

# ----------------------------
# Cached calls (avoid double-billing on reruns)
# ----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_keyword_suggestions(seed: str, loc: int, lang: str, limit: int) -> pd.DataFrame:
    payload = [{
        "keyword": seed,
        "location_code": int(loc),
        "language_name": lang,
        "limit": int(limit),
        "include_seed_keyword": False,
        "include_serp_info": False,
        "ignore_synonyms": False,
        "include_clickstream_data": False,
        "exact_match": False
    }]
    resp = labs_post("/google/keyword_suggestions/live", payload)
    items = extract_items(resp)
    rows = []
    for it in items:
        kw = it.get("keyword")
        info = it.get("keyword_info") or {}
        vol = info.get("search_volume")
        # CPC may be nested dict or number depending on endpoint
        cpc_val = info.get("cpc")
        if isinstance(cpc_val, dict):
            cpc_usd = cpc_val.get("usd")
        else:
            cpc_usd = cpc_val
        comp = info.get("competition")
        hist = info.get("monthly_searches") or it.get("monthly_searches")
        avg_month = monthly_avg(hist)
        rows.append({
            "keyword": kw,
            "search_volume": vol if vol is not None else (avg_month if not math.isnan(avg_month) else None),
            "cpc_usd": cpc_usd,
            "competition": comp
        })
    df = pd.DataFrame(rows).dropna(subset=["keyword"]).reset_index(drop=True)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def get_search_intent(keywords: list[str], loc: int | None, lang_name: str | None, lang_code: str | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """
    DataForSEO Labs · Search Intent (Live)
    - Only requires keywords + (language_name or language_code)
    - location_code is optional; omit when None
    Returns: (df, raw_response, payload_used)
    """
    # Clean keywords
    kws = sorted(list({(k or "").strip() for k in keywords if isinstance(k, str)}))
    if not kws:
        return pd.DataFrame(columns=["keyword","intent","intent_probability"]), {"note": "no keywords"}, {}

    # Build payload with only required/valid fields
    payload_item = {
        "keywords": kws[:1000]
    }
    if lang_code:         # preferred if you have it (e.g., "en")
        payload_item["language_code"] = lang_code
    elif lang_name:       # fallback (e.g., "English")
        payload_item["language_name"] = lang_name.strip()

    if loc:               # include ONLY if provided
        payload_item["location_code"] = int(loc)

    payload = [payload_item]

    resp = labs_post("/google/search_intent/live", payload)
    items = extract_items(resp)

    rows = []
    for it in items:
        kw = it.get("keyword")
        si = it.get("search_intent_info") or {}
        rows.append({
            "keyword": kw,
            "intent": si.get("main_intent"),
            "intent_probability": si.get("probability")
        })
    df = pd.DataFrame(rows)
    return df, resp, payload_item

# ----------------------------
# Run
# ----------------------------
run = st.button("Fetch ideas & analyse intent")

if run:
    with st.spinner("Fetching keyword ideas (Live)…"):
        df_kw = get_keyword_suggestions(seed_keyword.strip(), location_code, language_name, limit)

    if df_kw.empty:
        st.warning("No keyword ideas returned. Try another seed or adjust locale.")
        st.stop()

    # Currency conversion
    if "cpc_usd" in df_kw.columns:
        df_kw["cpc_gbp"] = (pd.to_numeric(df_kw["cpc_usd"], errors="coerce") * usd_to_gbp).round(2)

    st.subheader("Keyword ideas with metrics")
    st.dataframe(df_kw, use_container_width=True)

    # Intent classification
kw_list = df_kw["keyword"].dropna().astype(str).tolist()

# Pass None for loc if you want to omit it
use_loc_for_intent = False  # or drive this from a sidebar toggle
loc_for_intent = location_code if use_loc_for_intent else None

# You can pass language_name OR language_code ("en")
language_code = "en"        # keep it simple; set from a dropdown if you prefer
df_intent, raw_int, sent_payload = get_search_intent(
    kw_list,
    loc_for_intent,
    lang_name=language_name,
    lang_code=language_code
)
    # Merge
    df = df_kw.merge(df_intent, on="keyword", how="left")

    # Per-intent summary
    by_intent = (df
                 .groupby("intent", dropna=False)
                 .agg(
                     keywords=("keyword", "count"),
                     total_volume=("search_volume", "sum"),
                     avg_cpc_gbp=("cpc_gbp", safe_avg),
                     avg_intent_prob=("intent_probability", safe_avg)
                 )
                 .reset_index()
                 .rename(columns={"intent": "Intent"}))

    # Apply CTR/CVR & compute spend, clicks, conversions
    by_intent["CTR"] = by_intent["Intent"].map(ctrs).fillna(0.02)
    by_intent["CVR"] = by_intent["Intent"].map(cvrs).fillna(0.02)
    by_intent["Clicks"] = (by_intent["total_volume"] * by_intent["CTR"]).round(0)
    by_intent["Avg CPC £"] = by_intent["avg_cpc_gbp"].round(2)
    by_intent["Spend £"] = (by_intent["Clicks"] * by_intent["Avg CPC £"]).round(2)
    by_intent["Conversions"] = (by_intent["Clicks"] * by_intent["CVR"]).round(0)

    st.subheader("Grouped by intent")
    st.dataframe(by_intent.fillna("—"), use_container_width=True)

    # Blended overview (all terms combined)
    st.subheader("Blended overview (all terms)")
    blended = pd.DataFrame({
        "keywords": [int(by_intent["keywords"].sum())],
        "total_volume": [int(by_intent["total_volume"].sum() if by_intent["total_volume"].notna().any() else 0)],
        "avg_cpc_gbp": [round(by_intent["avg_cpc_gbp"].mean(skipna=True), 2)],
        "CTR": [round(by_intent["CTR"].mean(skipna=True), 3)],
        "Clicks": [int(by_intent["Clicks"].sum())],
        "CVR": [round(by_intent["CVR"].mean(skipna=True), 3)],
        "Conversions": [int(by_intent["Conversions"].sum())],
        "Spend £": [round(by_intent["Spend £"].sum(), 2)]
    })
    st.dataframe(blended, use_container_width=True)

    # Optional budget fit
    if budget and budget > 0:
        st.subheader("Budget fit (optional)")
        needs = by_intent["Spend £"].replace(0, np.nan)
        if needs.notna().any():
            weights = needs / needs.sum()
        else:
            weights = pd.Series([1/len(by_intent)]*len(by_intent), index=by_intent.index)
        alloc = (weights * budget).fillna(0)
        feasible_clicks = np.where(by_intent["Avg CPC £"] > 0, alloc / by_intent["Avg CPC £"], 0)
        feasible_convs = feasible_clicks * by_intent["CVR"]
        budget_df = pd.DataFrame({
            "Intent": by_intent["Intent"],
            "Budget £": alloc.round(2),
            "Avg CPC £": by_intent["Avg CPC £"].round(2),
            "CTR": by_intent["CTR"],
            "CVR": by_intent["CVR"],
            "Clicks (budget)": np.round(feasible_clicks, 0).astype(int),
            "Conversions (budget)": np.round(feasible_convs, 0).astype(int),
        })
        st.dataframe(budget_df, use_container_width=True)

    # Downloads
    st.download_button("Download detailed rows (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="labs_keywords_with_intent.csv", mime="text/csv")
    st.download_button("Download intent summary (CSV)", by_intent.to_csv(index=False).encode("utf-8"),
                       file_name="intent_summary.csv", mime="text/csv")
    st.download_button("Download blended overview (CSV)", blended.to_csv(index=False).encode("utf-8"),
                       file_name="blended_overview.csv", mime="text/csv")

    # Rough cost estimate (trial safety)
    n_items = len(df_kw)
    # Typical published micro-prices (subject to change) — suggestions + search_intent
    EST = {
        "suggest_task": 0.01,     # per request
        "suggest_item": 0.0001,   # per keyword returned
        "intent_task": 0.001,     # per request (cheaper than others)
        "intent_item": 0.0001     # per keyword
    }
    approx_cost = EST["suggest_task"] + n_items * EST["suggest_item"] + EST["intent_task"] + n_items * EST["intent_item"]
    st.caption(f"Approx. cost this run: ~${approx_cost:.3f} for {n_items} keywords (estimate only).")
