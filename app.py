# DataForSEO Labs — Keyword Ideas + Intent Planner (Live, trial-safe)
# - Seed keyword -> up to 20 suggestions with volume/CPC/competition
# - Search Intent (language-only, no location needed)
# - Group by intent with CTR/CVR assumptions + blended totals
# - Password gate (session-state) + caching + raw-debug

import os, math
import requests
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Labs Keyword Ideas + Intent (Live)", layout="wide")
st.title("DataForSEO Labs — Keyword Ideas → Intent Planner")

# ----------------------------
# One-time password gate (APP_PASSWORD is YOUR chosen gate code, not the API password)
# ----------------------------
if "authed" not in st.session_state:
    st.session_state.authed = False

if not st.session_state.authed:
    pw = st.text_input("Password", type="password")
    if st.button("Enter"):
        if pw == st.secrets.get("APP_PASSWORD"):
            st.session_state.authed = True
            st.experimental_rerun()
        else:
            st.error("Wrong password")
    st.stop()

# ----------------------------
# Credentials (Streamlit Secrets)
# ----------------------------
LOGIN = st.secrets.get("DATAFORSEO_LOGIN", os.getenv("DATAFORSEO_LOGIN", ""))
PASSWORD = st.secrets.get("DATAFORSEO_PASSWORD", os.getenv("DATAFORSEO_PASSWORD", ""))
if not (LOGIN and PASSWORD):
    st.error("Missing DATAFORSEO_LOGIN / DATAFORSEO_PASSWORD in secrets (or env).")
    st.stop()

# ----------------------------
# RestClient (official) or fallback
# ----------------------------
try:
    from Client.client import RestClient  # official client you uploaded
except Exception:
    class RestClient:
        def __init__(self, login, password, base_url="https://api.dataforseo.com"):
            self.login, self.password, self.base_url = login, password, base_url.rstrip("/")
        def post(self, path, payload):
            url = f"{self.base_url}{path}"
            r = requests.post(url, auth=(self.login, self.password), json=payload, timeout=60)
            try: r.raise_for_status()
            except requests.HTTPError as e:
                try: msg = r.json()
                except Exception: msg = r.text[:800]
                raise RuntimeError(f"POST {path} -> HTTP {r.status_code}: {msg}") from e
            return r.json()

client = RestClient(LOGIN, PASSWORD)
LABS_BASE = "/v3/dataforseo_labs"

# ----------------------------
# Helpers
# ----------------------------
def labs_post(endpoint: str, payload: list[dict]) -> dict:
    return client.post(f"{LABS_BASE}{endpoint}", payload)

def extract_items(resp: dict) -> list[dict]:
    try:
        tasks = resp.get("tasks") or []
        result = tasks[0].get("result") or []
        return result[0].get("items") or []
    except Exception:
        return []

def monthly_avg(monthly_searches: list[dict]) -> float:
    if not monthly_searches: return float("nan")
    vals = [m.get("search_volume") for m in monthly_searches if isinstance(m, dict)]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return float(np.mean(vals)) if vals else float("nan")

def safe_avg(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.mean() if not s.empty else np.nan

# ----------------------------
# Sidebar (trial-safe defaults)
# ----------------------------
with st.sidebar:
    st.header("Inputs")

    seed_keyword = st.text_input("Seed keyword", value="remortgage")
    # For suggestions we’ll use language_code 'en' (safe default).
    language_code = st.text_input("language_code (e.g., en, fr, de)", value="en")
    # Location for suggestions is OPTIONAL; many users like to scope to UK by name.
    use_location_for_suggestions = st.toggle("Use a location for suggestions", value=True,
        help="Keyword Suggestions accepts location_name or location_code. If off, Labs uses defaults / all locations.")
    location_name = "United Kingdom"
    if use_location_for_suggestions:
        location_name = st.text_input("location_name (optional for Suggestions)", value="United Kingdom")

    limit = st.slider("Max ideas (protect trial balance)", 5, 20, 20, step=5)

    usd_to_gbp = st.number_input("USD→GBP rate", min_value=0.1, max_value=2.0, value=0.78, step=0.01)

    st.divider()
    st.caption("CTR/CVR assumptions by intent")
    intents = ["informational", "navigational", "commercial", "transactional"]
    ctr_defaults = {"informational":0.02, "navigational":0.04, "commercial":0.05, "transactional":0.06}
    cvr_defaults = {"informational":0.01, "navigational":0.02, "commercial":0.03, "transactional":0.08}
    ctrs, cvrs = {}, {}
    for i in intents:
        c1, c2 = st.columns(2)
        with c1:
            ctrs[i] = st.number_input(f"{i.title()} CTR", 0.0, 1.0, ctr_defaults[i], 0.005, format="%.3f", key=f"ctr_{i}")
        with c2:
            cvrs[i] = st.number_input(f"{i.title()} CVR", 0.0, 1.0, cvr_defaults[i], 0.005, format="%.3f", key=f"cvr_{i}")

    st.divider()
    budget = st.number_input("Optional monthly budget (£)", 0.0, step=100.0, value=0.0)
    show_raw = st.toggle("Show raw API payloads/responses (debug)", value=False)

# ----------------------------
# Cached calls (protect credits)
# ----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_keyword_suggestions(seed: str, lang_code: str, loc_name: str | None, limit: int) -> tuple[pd.DataFrame, dict, dict]:
    """Labs Keyword Suggestions (Live) — returns df + raw resp + payload."""
    payload_item = {
        "keyword": (seed or "").strip(),
        "limit": int(limit),
        # Prefer language_code; language_name is also OK but not needed if code is set.
        "language_code": (lang_code or "en").strip(),
        "include_seed_keyword": False,
        "include_serp_info": False,
        "ignore_synonyms": False,
        "include_clickstream_data": False,
        "exact_match": False
    }
    if loc_name:  # Optional — many like 'United Kingdom'
        payload_item["location_name"] = loc_name.strip()

    payload = [payload_item]
    resp = labs_post("/google/keyword_suggestions/live", payload)
    items = extract_items(resp)

    rows = []
    for it in items:
        kw = (it.get("keyword") or "").strip()
        info = it.get("keyword_info") or {}
        vol = info.get("search_volume")
        # CPC may be number or object; treat consistently
        cpc_val = info.get("cpc")
        cpc_usd = cpc_val.get("usd") if isinstance(cpc_val, dict) else cpc_val
        comp = info.get("competition")
        hist = info.get("monthly_searches") or it.get("monthly_searches")
        avg_m = monthly_avg(hist)
        rows.append({
            "keyword": kw,
            "search_volume": vol if vol is not None else (avg_m if not math.isnan(avg_m) else None),
            "cpc_usd": cpc_usd,
            "competition": comp
        })

    df = (pd.DataFrame(rows)
            .dropna(subset=["keyword"])
            .assign(keyword=lambda d: d["keyword"].str.strip())
            .drop_duplicates(subset=["keyword"])
            .reset_index(drop=True))

    return df, resp, payload_item

@st.cache_data(ttl=3600, show_spinner=False)
def get_search_intent(keywords: list[str], lang_code: str, lang_name: str | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """Labs Search Intent (Live) — only needs keywords + language_code/name."""
    kws = sorted(list({(k or "").strip() for k in keywords if isinstance(k, str)}))
    if not kws:
        return pd.DataFrame(columns=["keyword","intent","intent_probability"]), {"note":"no keywords"}, {}

    payload_item = {"keywords": kws[:1000]}
    if lang_code:
        payload_item["language_code"] = lang_code.strip()
    elif lang_name:
        payload_item["language_name"] = (lang_name or "").strip()
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
if st.button("Fetch ideas & analyse intent"):
    # 1) Suggestions (limit 20)
    with st.spinner("Getting keyword suggestions…"):
        loc_name_for_sug = location_name if use_location_for_suggestions else None
        df_kw, raw_sug, payload_sug = get_keyword_suggestions(seed_keyword, language_code, loc_name_for_sug, limit)

    if show_raw:
        with st.expander("Payload → keyword_suggestions"):
            st.json(payload_sug)
        with st.expander("RAW response ← keyword_suggestions"):
            st.json(raw_sug)

    if df_kw.empty:
        st.warning("No suggestions returned. Try a simpler seed, increase limit, or remove location.")
        st.stop()

    df_kw["cpc_gbp"] = (pd.to_numeric(df_kw["cpc_usd"], errors="coerce") * usd_to_gbp).round(2)
    st.subheader("Keyword ideas with metrics")
    st.dataframe(df_kw, use_container_width=True)

    # 2) Intent (omit location; language-only as required by docs)
    with st.spinner("Classifying search intent…"):
        kw_list = df_kw["keyword"].dropna().astype(str).tolist()
        df_intent, raw_int, payload_int = get_search_intent(kw_list, language_code, lang_name="English")

    if show_raw:
        with st.expander("Payload → search_intent"):
            st.json(payload_int)
        with st.expander("RAW response ← search_intent"):
            st.json(raw_int)

    # Debug: show any keywords lacking an intent row
    got = set(df_intent["keyword"].dropna().astype(str))
    sent = set(kw_list)
    missing = sorted(list(sent - got))
    if missing:
        with st.expander(f"Debug: {len(missing)} keywords missing intent (click to view)"):
            st.write(missing)

    # Merge + group
    df = df_kw.merge(df_intent, on="keyword", how="left")

    def summarise(df_):
        return (df_
                .groupby("intent", dropna=False)
                .agg(
                    keywords=("keyword", "count"),
                    total_volume=("search_volume", "sum"),
                    avg_cpc_gbp=("cpc_gbp", safe_avg),
                    avg_intent_prob=("intent_probability", safe_avg)
                )
                .reset_index()
                .rename(columns={"intent":"Intent"}))

    by_intent = summarise(df)
    by_intent["CTR"] = by_intent["Intent"].map(ctrs).fillna(0.02)
    by_intent["CVR"] = by_intent["Intent"].map(cvrs).fillna(0.02)
    by_intent["Clicks"] = (by_intent["total_volume"] * by_intent["CTR"]).round(0)
    by_intent["Avg CPC £"] = by_intent["avg_cpc_gbp"].round(2)
    by_intent["Spend £"] = (by_intent["Clicks"] * by_intent["Avg CPC £"]).round(2)
    by_intent["Conversions"] = (by_intent["Clicks"] * by_intent["CVR"]).round(0)

    st.subheader("Grouped by intent")
    st.dataframe(by_intent.fillna("—"), use_container_width=True)

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

    # Downloads
    st.download_button("Download detailed rows (CSV)",
        df.to_csv(index=False).encode("utf-8"), "labs_keywords_with_intent.csv", "text/csv")
    st.download_button("Download intent summary (CSV)",
        by_intent.to_csv(index=False).encode("utf-8"), "intent_summary.csv", "text/csv")
    st.download_button("Download blended overview (CSV)",
        blended.to_csv(index=False).encode("utf-8"), "blended_overview.csv", "text/csv")

    # Cost estimate (very rough; just to reassure on trial usage)
    n = len(df_kw)
    EST = {"sug_task":0.01, "sug_item":0.0001, "intent_task":0.001, "intent_item":0.0001}
    approx_cost = EST["sug_task"] + n*EST["sug_item"] + EST["intent_task"] + n*EST["intent_item"]
    st.caption(f"Approximate API cost this run: ~${approx_cost:.3f} for {n} keywords (estimate only).")
