# app_salary_dashboard.py
"""
Streamlit dashboard for NBA Salary Predictor
Uses classes and functions from nba_salary_predictor.py in the same repo.

Features:
- Load/upload salary CSV and optionally stats CSV
- Option to fetch current season stats via nba_api (may hit rate limits)
- Train salary model (RandomForest / GradientBoosting selection)
- Show feature importance (percentage), predicted vs actual, salary vs performance
- Top players table
- Interactive prediction form (user inputs stats -> predicted salary + range)
"""

from pathlib import Path
import io
import time
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Try to import your existing code (nba_salary_predictor.py must be in the repo)
try:
    from nba_salary_predictor import (
        NBADataCollector,
        SalaryPredictionModel,
    )
except Exception as e:
    st.error(
        "Could not import nba_salary_predictor.py from the repo. Make sure the file exists "
        "in the same directory as this dashboard and has the classes NBADataCollector and SalaryPredictionModel."
    )
    st.stop()

st.set_page_config(page_title="NBA Salary Predictor", layout="wide", initial_sidebar_state="expanded")
st.title("NBA Salary Predictor — Dashboard")

# -------------------------
# Sidebar: data control
# -------------------------
st.sidebar.header("Data input")

use_merged_upload = st.sidebar.checkbox("Upload merged stats+salary CSV (recommended)", value=False)
merged_file = None
salary_file = None
stats_file = None

if use_merged_upload:
    merged_file = st.sidebar.file_uploader("Merged CSV (must include player stats + SALARY column)", type=["csv"])
else:
    salary_file = st.sidebar.file_uploader("Salary CSV (e.g., nba_salaries_2023_24.csv)", type=["csv"])
    stats_file = st.sidebar.file_uploader("Optional: Player stats CSV (if not using NBA API)", type=["csv"])
    fetch_stats = st.sidebar.checkbox("Fetch current season stats from NBA API (may rate-limit)", value=False)

train_button = st.sidebar.button("Train / Retrain model")
st.sidebar.markdown("---")
st.sidebar.write("Instructions:")
st.sidebar.write("- For reliable results upload a merged CSV with salary and stats columns.")
st.sidebar.write("- Alternatively upload salary CSV and allow the app to fetch stats from the NBA API.")
st.sidebar.write("- The model expects these numeric features: PTS, REB, AST, BLK, STL, TS_PCT, WIN_SHARES, EPA")

# -------------------------
# Utility functions
# -------------------------
@st.cache_data(show_spinner=False)
def read_csv_buffer(buf) -> pd.DataFrame:
    return pd.read_csv(buf, index_col=False, encoding="utf-8-sig")

def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    return missing

# -------------------------
# Load or fetch data
# -------------------------
st.header("Data loading")

merged_df = None
collector = NBADataCollector(season='2023-24', salary_file='nba_salaries_2023_24.csv')

if use_merged_upload and merged_file:
    try:
        merged_df = read_csv_buffer(merged_file)
        st.success(f"Loaded merged file with {len(merged_df)} rows.")
    except Exception as e:
        st.error(f"Failed to read merged CSV: {e}")
        st.stop()
else:
    # load salary CSV (required)
    if salary_file is None:
        st.warning("Upload a salary CSV in the sidebar to proceed (or use merged upload).")
        st.stop()
    try:
        salary_df = read_csv_buffer(salary_file)
        st.success(f"Loaded salary file with {len(salary_df)} rows.")
    except Exception as e:
        st.error(f"Failed to read salary CSV: {e}")
        st.stop()

    # If stats file uploaded, read it
    if stats_file is not None:
        try:
            stats_df = read_csv_buffer(stats_file)
            st.success(f"Loaded stats CSV with {len(stats_df)} rows.")
        except Exception as e:
            st.error(f"Failed to read stats CSV: {e}")
            st.stop()
    else:
        stats_df = None

    # If user asked to fetch stats from NBA API
    if stats_df is None and fetch_stats:
        st.info("Fetching stats from NBA API. This may take a minute and can hit rate limits.")
        with st.spinner("Fetching NBA stats..."):
            stats_df = collector.get_player_stats()
        if stats_df is None:
            st.error("Failed to fetch stats from NBA API. Try uploading a stats CSV instead.")
            st.stop()
        else:
            st.success(f"Fetched {len(stats_df)} player stats from NBA API.")

    # If we now have stats_df, attempt to merge using collector logic
    if stats_df is None:
        st.error("No player stats available. Either upload a stats CSV or enable NBA API fetch.")
        st.stop()

    # We have salary_df and stats_df — prepare and merge using collector functions
    collector.player_stats = stats_df
    collector.salary_data = salary_df

    with st.spinner("Merging stats and salaries..."):
        merged_df = collector.merge_stats_with_salaries()

    if merged_df is None or len(merged_df) == 0:
        st.error("Merging failed. Inspect console logs / check last names and team abbreviations for matching.")
        st.stop()
    else:
        st.success(f"Merged dataset contains {len(merged_df)} players.")

# -------------------------
# Add advanced metrics (WIN_SHARES, EPA)
# -------------------------
st.subheader("Derived metrics")
with st.spinner("Computing advanced metrics..."):
    merged_df = collector.add_advanced_metrics(merged_df)

# Quick preview and checks
st.write("Preview of merged dataset (first 8 rows):")
st.dataframe(merged_df.head(8))

required_features = ['PTS', 'REB', 'AST', 'BLK', 'STL', 'TS_PCT', 'WIN_SHARES', 'EPA', 'SALARY']
missing = require_columns(merged_df, required_features)
if missing:
    st.error(f"Missing required columns in merged dataset: {missing}. The dashboard needs these to train the model.")
    st.stop()

# -------------------------
# Train model (cached)
# -------------------------
st.header("Model training & diagnostics")

@st.cache_resource
def train_salary_model(df: pd.DataFrame, optimize=True):
    """
    Trains the SalaryPredictionModel from the merged dataframe.
    Returns: (model_object, X_test, y_test, y_pred_test, trained_df_with_perf)
    """
    model_obj = SalaryPredictionModel()
    X, y = model_obj.prepare_features(df)
    X_test, y_test, y_pred = model_obj.train(X, y, optimize=optimize)

    # create performance metric and attach to df copy
    df_perf = model_obj.create_performance_metric(df.copy())

    return model_obj, X_test, y_test, y_pred, df_perf

# Trigger training: either on button or cached previous run
do_train = train_button
if not st.session_state.get("trained_once", False):
    # we also train automatically first time (unless user wants manual only)
    do_train = True

if do_train:
    with st.spinner("Training salary model (this can take 20-60s depending on dataset and model)..."):
        try:
            model_obj, X_test, y_test, y_pred, df_perf = train_salary_model(merged_df, optimize=True)
            st.session_state["trained_once"] = True
            st.success("Model trained successfully.")
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()
else:
    st.info("Click 'Train / Retrain model' in the sidebar to (re)train.")

# -------------------------
# Diagnostics: feature importance (percentage)
# -------------------------
st.subheader("Feature importance (percentage of total)")
fi_df = model_obj.feature_importance.copy()
if 'Importance' in fi_df.columns:
    fi_df = fi_df.rename(columns={'Importance':'importance'})
fi_df['abs_importance'] = fi_df['importance'].abs()
total = fi_df['abs_importance'].sum()
if total == 0:
    fi_df['importance_pct'] = 0.0
else:
    fi_df['importance_pct'] = fi_df['abs_importance'] / total * 100.0

# Plot with Plotly
fig_fi = px.bar(
    fi_df.sort_values('importance_pct'),
    x='importance_pct',
    y='Feature',
    orientation='h',
    text=fi_df['importance_pct'].round(2),
    labels={'importance_pct': 'Importance (%)'},
    title='Feature importance (percentage of total)'
)
fig_fi.update_layout(height=340, margin=dict(l=60, r=20, t=40, b=20))
st.plotly_chart(fig_fi, use_container_width=True)

# show table
fi_table = fi_df[['Feature', 'importance_pct']].sort_values('importance_pct', ascending=False).reset_index(drop=True)
fi_table['importance_pct'] = fi_table['importance_pct'].round(2)
st.table(fi_table)

# -------------------------
# Predictions vs Actual
# -------------------------
st.subheader("Predicted vs Actual Salaries (test set)")
try:
    test_df = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred
    })
    test_df['actual_m'] = test_df['actual'] / 1e6
    test_df['predicted_m'] = test_df['predicted'] / 1e6

    fig_pa = px.scatter(
        test_df,
        x='actual_m',
        y='predicted_m',
        labels={'actual_m': 'Actual Salary (M$)', 'predicted_m': 'Predicted Salary (M$)'},
        title='Predicted vs Actual Salaries (test set)',
        trendline='ols',
        hover_data=[]
    )
    # add perfect prediction line
    minv = min(test_df['actual_m'].min(), test_df['predicted_m'].min())
    maxv = max(test_df['actual_m'].max(), test_df['predicted_m'].max())
    fig_pa.add_shape(type="line", x0=minv, y0=minv, x1=maxv, y1=maxv, line=dict(color="red", dash="dash"))
    st.plotly_chart(fig_pa, use_container_width=True)

    # show metrics
    r2 = float(np.nan if len(y_test)==0 else np.nan_to_num(np.corrcoef(y_test, y_pred)[0,1])**2)
    mae = float(np.mean(np.abs(y_test - y_pred)))
    st.metric("Test MAE", f"${mae:,.0f}")
    st.metric("Test R² (approx)", f"{r2:.3f}")
except Exception as e:
    st.error(f"Unable to create Pred vs Actual plot: {e}")

# -------------------------
# Salary vs Performance
# -------------------------
st.subheader("Salary vs Performance Metric")
try:
    df_perf['salary_m'] = df_perf['SALARY'] / 1e6
    fig_sp = px.scatter(
        df_perf,
        x='salary_m',
        y='PERFORMANCE_METRIC',
        size='PERFORMANCE_METRIC',
        color='PERFORMANCE_METRIC',
        hover_data=['PLAYER_NAME', 'TEAM_ABBREVIATION', 'SALARY'],
        labels={'salary_m': 'Salary (M$)', 'PERFORMANCE_METRIC': 'Performance Metric'},
        title='Salary vs Performance Metric'
    )
    st.plotly_chart(fig_sp, use_container_width=True)
except Exception as e:
    st.error(f"Unable to create Salary vs Performance plot: {e}")

# -------------------------
# Top players by performance table
# -------------------------
st.subheader("Top players by performance metric")
top_n = st.slider("Number of top players to show", min_value=5, max_value=30, value=10)
top_players = df_perf.nlargest(top_n, 'PERFORMANCE_METRIC')[[
    'PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS', 'REB', 'AST', 'PERFORMANCE_METRIC', 'SALARY'
]].copy()
top_players['SALARY'] = top_players['SALARY'].apply(lambda x: f"${x/1e6:.2f}M")
top_players['PERFORMANCE_METRIC'] = top_players['PERFORMANCE_METRIC'].round(2)
st.dataframe(top_players.reset_index(drop=True))

# -------------------------
# Interactive salary predictor
# -------------------------
st.header("Interactive predictor — input stats & get predicted salary")

with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        pts = st.number_input("Points per game (PTS)", min_value=0.0, max_value=50.0, value=15.0, step=0.5)
        reb = st.number_input("Rebounds per game (REB)", min_value=0.0, max_value=20.0, value=5.0, step=0.5)
        ast = st.number_input("Assists per game (AST)", min_value=0.0, max_value=20.0, value=3.0, step=0.5)
    with c2:
        blk = st.number_input("Blocks per game (BLK)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        stl = st.number_input("Steals per game (STL)", min_value=0.0, max_value=5.0, value=0.7, step=0.1)
        ts_pct = st.number_input("True Shooting % (TS_PCT, 0-1)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    with c3:
        win_shares = st.number_input("Win Shares", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
        epa = st.number_input("EPA", min_value=-20.0, max_value=50.0, value=1.0, step=0.1)
        player_name = st.text_input("Player name (optional, for labeling)")

    submit = st.form_submit_button("Predict salary")

if submit:
    # prepare stats dict
    stats = {
        'PTS': float(pts),
        'REB': float(reb),
        'AST': float(ast),
        'BLK': float(blk),
        'STL': float(stl),
        'TS_PCT': float(ts_pct),
        'WIN_SHARES': float(win_shares),
        'EPA': float(epa)
    }
    try:
        prediction, lower, upper = model_obj.predict_salary(stats)
        st.success(f"Predicted salary: ${prediction:,.0f}   (Range: ${lower:,.0f} — ${upper:,.0f})")

        # Show contributions (approx): feature importance * scaled value
        # We'll compute simple contribution proxy = weight_pct * (feature_normalized * coefproxy)
        fi = model_obj.feature_importance.copy()
        fi = fi.set_index('Feature')
        # normalize features using training df ranges (df_perf)
        contribs = []
        for feat in model_obj.feature_names:
            val = stats[feat]
            # normalize to 0-1 using min/max from training df
            fmin = df_perf[feat].min()
            fmax = df_perf[feat].max()
            norm = 0.0 if fmax <= fmin else (val - fmin) / (fmax - fmin)
            weight = fi.loc[feat]['Importance'] if feat in fi.index else 0.0
            contribs.append((feat, weight * norm))
        contrib_df = pd.DataFrame(contribs, columns=['feature', 'contribution_proxy'])
        if contrib_df['contribution_proxy'].abs().sum() > 0:
            contrib_df['pct'] = contrib_df['contribution_proxy'].abs() / contrib_df['contribution_proxy'].abs().sum() * 100.0
        else:
            contrib_df['pct'] = 0.0
        contrib_df = contrib_df.sort_values('pct', ascending=False).reset_index(drop=True)
        st.subheader("Rough contribution proxy (how input features drive prediction)")
        st.table(contrib_df[['feature', 'pct']].rename(columns={'pct': 'contribution (%)'}).round(2))

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------
# Download artifacts
# -------------------------
st.header("Export & downloads")
if st.button("Download trained model feature importance (CSV)"):
    st.download_button(
        label="Download feature importance CSV",
        data=fi_df[['Feature', 'importance_pct']].to_csv(index=False).encode('utf-8'),
        file_name='salary_feature_importance.csv',
        mime='text/csv'
    )

if st.button("Download merged dataset (CSV)"):
    st.download_button(
        label="Download merged dataset",
        data=merged_df.to_csv(index=False).encode('utf-8'),
        file_name='merged_salary_stats.csv',
        mime='text/csv'
    )

st.markdown("---")
st.caption("Notes: Predictions assume the same preprocessing used during training. The contribution proxy is a heuristic to help interpret which features moved the prediction; for rigorous explanations use permutation importance or SHAP.")
