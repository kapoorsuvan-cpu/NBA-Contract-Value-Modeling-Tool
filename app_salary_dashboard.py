"""
Streamlit dashboard for NBA Salary Predictor
- Top players table now includes raw numeric Predicted_SALARY (dollars)
- Removed the 'Derived metrics (sample)' section
- Keeps salary parsing, model training, calibration, and plotting
- FIX: Predicted_SALARY now computed via bulk prediction on full df_perf
  (row-by-row prediction caused all values to collapse near the mean ~$14M)
"""

from pathlib import Path
import io
from typing import Optional, Dict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# Try import user module
try:
    from nba_salary_predictor import NBADataCollector, SalaryPredictionModel
except Exception as e:
    st.error(
        "Could not import nba_salary_predictor.py from the repo. "
        "Ensure that file exists in the same directory and contains "
        "NBADataCollector and SalaryPredictionModel classes."
    )
    st.stop()

st.set_page_config(page_title="NBA Salary Predictor", layout="wide")
st.title("NBA Salary Predictor — Dashboard")

# -------------------------
# Sidebar & inputs
# -------------------------
st.sidebar.header("Data input")

use_merged_upload = st.sidebar.checkbox("Upload merged stats+salary CSV (recommended)", value=False)

# Always show the fetch-from-API option
fetch_stats = st.sidebar.checkbox("Fetch current season stats from NBA API (may rate-limit)", value=False)

if use_merged_upload:
    merged_file = st.sidebar.file_uploader("Merged CSV (must include player stats + SALARY column)", type=["csv"])
    salary_file = None
    stats_file = None
else:
    merged_file = None
    salary_file = st.sidebar.file_uploader("Salary CSV (e.g., nba_salaries_2023_24.csv)", type=["csv"])
    stats_file = st.sidebar.file_uploader("Optional: Player stats CSV (if not using NBA API)", type=["csv"])

train_button = st.sidebar.button("Train / Retrain model")
st.sidebar.markdown("---")
st.sidebar.write("Model numeric features expected: PTS, REB, AST, BLK, STL, TS_PCT, WIN_SHARES, EPA")
st.sidebar.write("If you want the app to fetch live stats, add `nba_api` to requirements.txt before deploying.")

# -------------------------
# Utilities
# -------------------------
@st.cache_data(show_spinner=False)
def read_csv_buffer(buf):
    return pd.read_csv(buf, index_col=False, encoding="utf-8-sig")

def require_columns(df, cols):
    return [c for c in cols if c not in df.columns]

def _ensure_salary_columns(salary_df: pd.DataFrame) -> pd.DataFrame:
    """Try to create LAST_NAME, TEAM and numeric Salary columns from common variants."""
    df = salary_df.copy()
    lower_map = {c.lower(): c for c in df.columns}

    # Find player col
    player_col = None
    for cand in ("player", "player_name", "name", "playername"):
        if cand in lower_map:
            player_col = lower_map[cand]
            break

    # Find team col
    team_col = None
    for cand in ("team", "team_abbreviation", "tm"):
        if cand in lower_map:
            team_col = lower_map[cand]
            break

    if "LAST_NAME" not in df.columns:
        if player_col:
            df["LAST_NAME"] = (
                df[player_col].astype(str)
                .str.replace(r'\s+(Jr\.|Sr\.|II|III|IV|V|der)$', '', regex=True)
                .str.strip()
                .str.split().str[-1]
            )
        else:
            df["LAST_NAME"] = ""

    if "TEAM" not in df.columns:
        if team_col:
            df["TEAM"] = df[team_col].astype(str).str.strip()
        else:
            df["TEAM"] = ""

    # find a salary-like column
    if "Salary" not in df.columns and "SALARY" not in df.columns:
        found = None
        for cand in ("salary", "sal", "amount", "contract_amount"):
            if cand in lower_map:
                found = lower_map[cand]
                break
        if found:
            df["Salary"] = df[found]
        else:
            df["Salary"] = pd.NA

    df["LAST_NAME"] = df["LAST_NAME"].astype(str).str.strip()
    df["TEAM"] = df["TEAM"].astype(str).str.strip()
    return df

def parse_salary_to_numeric(s):
    """Parse messy salary strings like '$53.32M', '53,320,000', '53320000' into float dollars."""
    if pd.isna(s):
        return np.nan
    # if already numeric
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)
    s = str(s).strip()
    if s == "":
        return np.nan
    # remove currency symbols and spaces/commas
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    # handle trailing M or K
    try:
        if s.endswith(("M","m")):
            num = float(s[:-1])
            return num * 1_000_000.0
        if s.endswith(("K","k")):
            num = float(s[:-1])
            return num * 1_000.0
        # try plain float
        return float(s)
    except Exception:
        # try to remove non-digit chars then float
        import re
        digits = re.sub(r'[^\d.]', '', s)
        if digits == "":
            return np.nan
        try:
            return float(digits)
        except Exception:
            return np.nan

def normalize_stats_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map common variants of NBA stat column names to expected canonical names.
    Returns a copy with renamed columns.
    """
    df = df.copy()
    lower = {c.lower(): c for c in df.columns}
    colmap = {}

    mapping = {
        "PTS": ["pts", "points", "ppg", "pts_per_game", "pts_pg"],
        "REB": ["reb", "rebounds", "rpg", "rebounds_per_game", "reb_pg"],
        "AST": ["ast", "assists", "apg", "ast_pg"],
        "BLK": ["blk", "blocks", "bpg"],
        "STL": ["stl", "steals", "spg"],
        "FGA": ["fga", "fga_per_game"],
        "FGM": ["fgm", "fgm_per_game"],
        "FTA": ["fta", "fta_per_game"],
        "TS_PCT": ["ts_pct", "ts%", "true_shooting_pct", "true_shooting_percentage"],
        "GP": ["gp", "games", "games_played"],
        "FG3_PCT": ["fg3_pct", "fg3%", "three_pt_pct"],
    }

    for canonical, variants in mapping.items():
        for v in variants:
            if v.lower() in lower:
                colmap[lower[v.lower()]] = canonical
                break

    if colmap:
        df = df.rename(columns=colmap)
    return df

# -------------------------
# Load / prepare data
# -------------------------
st.header("Data loading")
collector = NBADataCollector(season="2023-24", salary_file="nba_salaries_2023_24.csv")
merged_df = None

# 1) merged upload path (user supplied file with both stats & salary)
if use_merged_upload and merged_file is not None:
    try:
        merged_df = read_csv_buffer(merged_file)
        st.success(f"Loaded merged file with {len(merged_df)} rows.")
    except Exception as e:
        st.error(f"Failed to read merged CSV: {e}")
        st.stop()

    # normalize columns
    merged_df = normalize_stats_columns(merged_df)

    # check required stat cols
    expected_raw = ["PTS", "REB", "AST", "BLK", "STL", "FGA", "FGM", "FTA", "GP"]
    missing_raw = [c for c in expected_raw if c not in merged_df.columns]

    # if merged file is actually salary-only, allow recovery if fetch_stats is enabled
    if missing_raw:
        st.warning("Uploaded merged CSV is missing stat columns. Attempting recovery...")
        if fetch_stats:
            st.info("Fetching stats from NBA API and merging with uploaded salary file.")
            try:
                salary_df = _ensure_salary_columns(merged_df)
            except Exception as e:
                st.error(f"Failed to normalize uploaded salary CSV: {e}")
                st.stop()
            with st.spinner("Fetching NBA stats..."):
                stats_df = collector.get_player_stats()
            if stats_df is None:
                st.error("Failed to fetch stats from NBA API. Please upload proper merged CSV or a separate stats CSV.")
                st.stop()
            collector.player_stats = stats_df
            collector.salary_data = salary_df
            with st.spinner("Merging fetched stats with uploaded salary CSV..."):
                merged_df = collector.merge_stats_with_salaries()
            if merged_df is None or len(merged_df) == 0:
                st.error("Merging failed. Inspect uploaded salary CSV names/team abbreviations.")
                st.stop()
            else:
                st.success(f"Merged dataset contains {len(merged_df)} players (after API fetch).")
        else:
            st.error(
                "Uploaded merged CSV does not contain the required stat columns. "
                "Either upload a merged CSV with stats, or enable 'Fetch current season stats from NBA API' "
                "so the app can fetch stats and merge with your uploaded salary file."
            )
            st.stop()

# 2) salary + (stats upload or API) path
else:
    if salary_file is None:
        st.warning("Upload a salary CSV (or use merged upload) in the sidebar to proceed.")
        st.stop()

    try:
        salary_df = read_csv_buffer(salary_file)
        st.success(f"Loaded salary file with {len(salary_df)} rows.")
    except Exception as e:
        st.error(f"Failed to read salary CSV: {e}")
        st.stop()

    # normalize salary CSV
    try:
        salary_df = _ensure_salary_columns(salary_df)
    except Exception as e:
        st.error(f"Failed to normalize salary CSV: {e}")
        st.stop()

    # load stats either via upload or via API (if requested)
    stats_df = None
    if stats_file is not None:
        try:
            stats_df = read_csv_buffer(stats_file)
            st.success(f"Loaded stats CSV with {len(stats_df)} rows.")
        except Exception as e:
            st.error(f"Failed to read stats CSV: {e}")
            st.stop()
    else:
        stats_df = None

    if stats_df is None and fetch_stats:
        st.info("Fetching stats from NBA API. This may take a minute and can hit rate limits.")
        with st.spinner("Fetching NBA stats from nba_api..."):
            stats_df = collector.get_player_stats()
        if stats_df is None:
            st.error("Failed to fetch stats from NBA API. Try uploading a stats CSV instead.")
            st.stop()
        else:
            st.success(f"Fetched {len(stats_df)} player stats from NBA API.")

    if stats_df is None:
        st.error("No player stats available. Either upload a stats CSV or enable NBA API fetch.")
        st.stop()

    # merge using collector logic
    collector.player_stats = stats_df
    collector.salary_data = salary_df

    with st.spinner("Merging stats and salaries..."):
        merged_df = collector.merge_stats_with_salaries()

    if merged_df is None or len(merged_df) == 0:
        st.error("Merging failed. Inspect names/team abbreviations in your files.")
        st.stop()
    else:
        st.success(f"Merged dataset contains {len(merged_df)} players.")

# ---------- normalize stats columns BEFORE advanced metrics ----------
merged_df = normalize_stats_columns(merged_df)

# Ensure Salary numeric column exists and convert it
if 'SALARY' not in merged_df.columns and 'Salary' in merged_df.columns:
    merged_df['SALARY'] = merged_df['Salary']

if 'SALARY' not in merged_df.columns:
    # last resort: try to find some salary-like column
    lower_map = {c.lower(): c for c in merged_df.columns}
    for cand in ("salary","sal","amount","contract_amount"):
        if cand in lower_map:
            merged_df['SALARY'] = merged_df[lower_map[cand]]
            break

if 'SALARY' not in merged_df.columns:
    st.error("Merged dataset does not include a Salary column (expected 'SALARY' or 'Salary').")
    st.stop()

# parse salary strings into numeric dollars
merged_df['SALARY'] = merged_df['SALARY'].apply(parse_salary_to_numeric)

# check for NaN after parse
if merged_df['SALARY'].isnull().all():
    st.error("After parsing, Salary column has no numeric values. Make sure salary values are present and formatted like '$53.32M' or '53320000'.")
    st.stop()

# Check required raw columns for derived metrics
expected_raw = ["PTS", "REB", "AST", "BLK", "STL", "FGA", "FGM", "FTA", "GP"]
missing_raw = [c for c in expected_raw if c not in merged_df.columns]

if missing_raw:
    st.error(
        "Your merged stats are missing required columns needed to compute derived metrics:\n\n"
        f"{missing_raw}\n\n"
        "Available columns in your merged dataset:\n\n"
        f"{list(merged_df.columns)}\n\n"
        "Please upload a stats CSV with standard column names or enable NBA API fetch."
    )
    st.stop()

# Now compute derived metrics (safe)
try:
    merged_df = collector.add_advanced_metrics(merged_df)
except KeyError as e:
    st.error(f"add_advanced_metrics failed - missing column: {e}. Available columns: {list(merged_df.columns)}")
    st.stop()
except Exception as e:
    st.error(f"add_advanced_metrics failed: {e}")
    st.stop()

st.subheader("Preview (first 6 rows)")
st.dataframe(merged_df.head(6))

# -------------------------
# Train model & calibrate
# -------------------------
st.header("Model training & diagnostics")

@st.cache_resource
def _train_and_calibrate(df: pd.DataFrame, optimize=True):
    model_obj = SalaryPredictionModel()
    X, y = model_obj.prepare_features(df)
    X_test, y_test, y_pred_test = model_obj.train(X, y, optimize=optimize)

    # calibrate predictions -> actual (simple linear calibrator)
    try:
        preds_full = model_obj.model.predict(model_obj.scaler.transform(X))
        calibrator = LinearRegression()
        calibrator.fit(preds_full.reshape(-1,1), y)
        model_obj._calibrator = calibrator
    except Exception:
        model_obj._calibrator = None

    df_perf = model_obj.create_performance_metric(df.copy())
    return model_obj, X_test, y_test, y_pred_test, df_perf

do_train = train_button or not st.session_state.get("trained_once", False)
if do_train:
    with st.spinner("Training (this can take 20-60s)..."):
        try:
            model_obj, X_test, y_test, y_pred, df_perf = _train_and_calibrate(merged_df, optimize=True)
            st.session_state["trained_once"] = True
            st.success("Model trained and calibrated.")
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()
else:
    st.info("Click 'Train / Retrain model' in the sidebar to (re)train.")
    if not st.session_state.get("trained_once", False):
        st.stop()

if model_obj is None:
    st.error("Model object not available after training.")
    st.stop()

def predict_salary_calibrated(model_obj, stats: Dict[str,float]):
    X = np.array([[stats[f] for f in model_obj.feature_names]])
    Xs = model_obj.scaler.transform(X)
    raw = model_obj.model.predict(Xs)[0]
    if getattr(model_obj, "_calibrator", None) is not None:
        try:
            cal = float(model_obj._calibrator.predict(np.array([[raw]]))[0])
        except Exception:
            cal = raw
    else:
        cal = raw
    return float(cal), float(cal*0.92), float(cal*1.08)

# -------------------------
# Feature importance (percentage)
# -------------------------
st.subheader("Feature importance (percentage of total)")
fi_df = pd.DataFrame(model_obj.feature_importance).copy() if getattr(model_obj, "feature_importance", None) is not None else pd.DataFrame({
    "Feature": model_obj.feature_names,
    "importance": [1.0/len(model_obj.feature_names)]*len(model_obj.feature_names)
})
if 'Importance' in fi_df.columns and 'importance' not in fi_df.columns:
    fi_df = fi_df.rename(columns={'Importance':'importance'})

fi_df['abs_importance'] = fi_df['importance'].abs()
total = fi_df['abs_importance'].sum()
if total <= 0 or np.isnan(total):
    fi_df['importance_pct'] = 100.0 / len(fi_df)
else:
    fi_df['importance_pct'] = fi_df['abs_importance'] / total * 100.0

table_df = fi_df[['Feature','importance_pct']].sort_values('importance_pct', ascending=False).reset_index(drop=True)
plot_df = table_df.sort_values('importance_pct', ascending=True)

fig_fi = px.bar(
    plot_df,
    x='importance_pct',
    y='Feature',
    orientation='h',
    text=plot_df['importance_pct'].round(1).astype(str) + '%',
    labels={'importance_pct': 'Importance (%)'},
    title='Feature importance (percentage of total)'
)
fig_fi.update_traces(marker_color='lightskyblue', marker_line_color='black', marker_line_width=1)
fig_fi.update_layout(height=380, margin=dict(l=160, r=20, t=50, b=40))
st.plotly_chart(fig_fi, use_container_width=True)
st.table(table_df.round(2))

# -------------------------
# Predicted vs Actual (no statsmodels required)
# -------------------------
st.subheader("Predicted vs Actual Salaries (test set)")
try:
    test_df = pd.DataFrame({"actual": y_test, "predicted": y_pred})
    if getattr(model_obj, "_calibrator", None) is not None:
        test_df['predicted_calibrated'] = model_obj._calibrator.predict(test_df['predicted'].values.reshape(-1,1))
    else:
        test_df['predicted_calibrated'] = test_df['predicted']

    test_df['actual_m'] = test_df['actual']/1e6
    test_df['predicted_m'] = test_df['predicted_calibrated']/1e6

    fig_pa = px.scatter(
        test_df,
        x='actual_m',
        y='predicted_m',
        labels={'actual_m':'Actual Salary (M$)','predicted_m':'Predicted Salary (M$)'},
        title='Predicted vs Actual Salaries (test set)'
    )
    minv = min(test_df['actual_m'].min(), test_df['predicted_m'].min())
    maxv = max(test_df['actual_m'].max(), test_df['predicted_m'].max())
    fig_pa.add_shape(type="line", x0=minv, y0=minv, x1=maxv, y1=maxv, line=dict(color="red", dash="dash"))
    st.plotly_chart(fig_pa, use_container_width=True)

    r2_val = r2_score(test_df['actual'], test_df['predicted_calibrated']) if len(test_df)>0 else np.nan
    mae_val = mean_absolute_error(test_df['actual'], test_df['predicted_calibrated']) if len(test_df)>0 else np.nan
    st.metric("Test MAE", f"${mae_val:,.0f}")
    st.metric("Test R²", f"{r2_val:.3f}")
except Exception as e:
    st.error(f"Unable to create Pred vs Actual plot: {e}")

# -------------------------
# Salary vs Performance
# -------------------------
st.subheader("Salary vs Performance Metric")
try:
    df_perf['salary_m'] = df_perf['SALARY']/1e6
    fig_sp = px.scatter(
        df_perf,
        x='salary_m',
        y='PERFORMANCE_METRIC',
        size='PERFORMANCE_METRIC',
        color='PERFORMANCE_METRIC',
        hover_data=['PLAYER_NAME','TEAM_ABBREVIATION','SALARY'],
        labels={'salary_m':'Salary (M$)','PERFORMANCE_METRIC':'Performance Metric'},
        title='Salary vs Performance Metric'
    )
    st.plotly_chart(fig_sp, use_container_width=True)
except Exception as e:
    st.error(f"Unable to create Salary vs Performance plot: {e}")

# -------------------------
# Top players table with predicted salary
# FIX: Use bulk prediction on the full df_perf dataset instead of row-by-row.
# Row-by-row prediction caused the scaler and calibrator to receive single-point
# inputs, collapsing all predictions near the mean (~$14M). Bulk prediction
# mirrors how the model was trained and produces accurate per-player estimates.
# -------------------------
st.subheader("Top players by performance metric (top 10)")
try:
    top_n = 10
    display_cols = ['PLAYER_NAME','TEAM_ABBREVIATION','PTS','REB','AST','PERFORMANCE_METRIC','SALARY']
    present_cols = [c for c in display_cols if c in df_perf.columns]

    # Bulk-predict over the full df_perf so the scaler sees the full distribution
    try:
        X_full, _ = model_obj.prepare_features(df_perf)
        X_full_scaled = model_obj.scaler.transform(X_full)
        raw_preds_full = model_obj.model.predict(X_full_scaled)

        if getattr(model_obj, "_calibrator", None) is not None:
            calibrated_preds = model_obj._calibrator.predict(
                raw_preds_full.reshape(-1, 1)
            ).flatten()
        else:
            calibrated_preds = raw_preds_full

        df_perf = df_perf.copy()
        df_perf['Predicted_SALARY'] = calibrated_preds.astype(float)

    except Exception as pred_err:
        st.warning(f"Bulk prediction failed: {pred_err}")
        df_perf['Predicted_SALARY'] = np.nan

    top_players = df_perf.nlargest(top_n, 'PERFORMANCE_METRIC')[present_cols + ['Predicted_SALARY']].copy()
    top_players['PERFORMANCE_METRIC'] = top_players['PERFORMANCE_METRIC'].round(2)

    st.dataframe(top_players.reset_index(drop=True))

except Exception as e:
    st.error(f"Error building top players table: {e}")

# -------------------------
# Interactive predictor
# -------------------------
st.header("Interactive predictor — input stats & get predicted salary")
with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        pts = st.number_input("PTS", min_value=0.0, max_value=50.0, value=15.0, step=0.5)
        reb = st.number_input("REB", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
        ast = st.number_input("AST", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
    with c2:
        blk = st.number_input("BLK", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        stl = st.number_input("STL", min_value=0.0, max_value=5.0, value=0.7, step=0.1)
        ts_pct = st.number_input("TS_PCT (0-1)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    with c3:
        win_shares = st.number_input("WIN_SHARES", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
        epa = st.number_input("EPA", min_value=-20.0, max_value=50.0, value=1.0, step=0.1)
    submit = st.form_submit_button("Predict salary")

if submit:
    stats = {'PTS': pts, 'REB': reb, 'AST': ast, 'BLK': blk, 'STL': stl, 'TS_PCT': ts_pct, 'WIN_SHARES': win_shares, 'EPA': epa}
    try:
        pred, low, high = predict_salary_calibrated(model_obj, stats)
        st.success(f"Predicted salary (raw dollars): {pred:.2f}  (Range: {low:.2f} — {high:.2f})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------
# Downloads & footer
# -------------------------
st.header("Export & downloads")
st.download_button(label="Download feature importance CSV", data=table_df.to_csv(index=False).encode('utf-8'), file_name="feature_importance.csv")
st.download_button(label="Download merged dataset", data=merged_df.to_csv(index=False).encode('utf-8'), file_name="merged_dataset.csv")

st.markdown("---")
st.caption("Notes: Predicted_SALARY column is raw numeric dollars (no string formatting). Predictions are calibrated by a simple linear fit on model predictions -> actuals to reduce systematic bias.")
