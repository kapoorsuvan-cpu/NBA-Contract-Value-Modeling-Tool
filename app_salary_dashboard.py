# app_salary_dashboard.py
"""
Streamlit dashboard for NBA Salary Predictor
Uses classes and functions from nba_salary_predictor.py in the same repo.

This updated version:
- fixes feature importance percentage/labeling
- removes Plotly trendline to avoid statsmodels dependency
- moves the Derived metrics UI to the bottom (computation still runs early)
- fixes Top players slider bug and adds Predicted Salary column
- defensive checks / clearer error messages
"""

from pathlib import Path
import io
import time
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import r2_score, mean_absolute_error

# Import your existing module
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

def _ensure_salary_columns(salary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize salary_df columns so downstream merge code expects:
      - LAST_NAME (extracted from a Player-like column)
      - TEAM (team abbreviation)
      - Salary (numeric)
    This returns a cleaned DataFrame.
    """
    df = salary_df.copy()
    cols_lower = {c.lower(): c for c in df.columns}

    # find player-like column (Player, PLAYER, name, player_name, etc.)
    player_col = None
    for candidate in ("player", "player_name", "name", "playername", "PLAYER", "Player"):
        if candidate.lower() in cols_lower:
            player_col = cols_lower[candidate.lower()]
            break

    # find team-like column
    team_col = None
    for candidate in ("team", "team_abbreviation", "team_abbrev", "tm", "team code", "team_name", "Team"):
        if candidate.lower() in cols_lower:
            team_col = cols_lower[candidate.lower()]
            break

    # Create LAST_NAME if missing
    if "LAST_NAME" not in df.columns:
        if player_col is not None:
            df['LAST_NAME'] = (
                df[player_col]
                .astype(str)
                .str.replace(r'\s+(Jr\.|Sr\.|II|III|IV|V|der|IV\.)$', '', regex=True)
                .str.strip()
                .str.split()
                .str[-1]
            )
            st.sidebar.info(f"Created LAST_NAME from `{player_col}`")
        else:
            st.sidebar.error("Salary CSV missing a player name column (e.g. 'Player'). Cannot create LAST_NAME.")
            df['LAST_NAME'] = ""

    # Create TEAM if missing
    if "TEAM" not in df.columns:
        if team_col is not None:
            df['TEAM'] = df[team_col].astype(str).str.strip()
            st.sidebar.info(f"Created TEAM from `{team_col}`")
        else:
            df['TEAM'] = ""
            st.sidebar.warning("Salary CSV missing a team column; TEAM created empty. Matching may be weaker.")

    # Ensure Salary numeric column exists
    if 'Salary' not in df.columns:
        found_salary = None
        for candidate in ("salary", "sal", "contract_amount", "amount", "Salary", "PAY"):
            if candidate.lower() in cols_lower:
                found_salary = cols_lower[candidate.lower()]
                break
        if found_salary:
            df['Salary'] = pd.to_numeric(df[found_salary].astype(str).str.replace(r'[^\d.-]', '' , regex=True), errors='coerce')
            st.sidebar.info(f"Using `{found_salary}` as Salary column")
        else:
            st.sidebar.error("Salary CSV missing a Salary column (e.g. 'Salary'). The merge requires numeric Salary values.")
            df['Salary'] = pd.NA

    # Trim/strip
    df['LAST_NAME'] = df['LAST_NAME'].astype(str).str.strip()
    df['TEAM'] = df['TEAM'].astype(str).str.strip()

    return df

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

    # Normalize salary_df
    try:
        salary_df = _ensure_salary_columns(salary_df)
    except Exception as e:
        st.error(f"Failed while normalizing salary CSV columns: {e}")
        st.stop()

    # Debug: show salary columns and sample
    st.sidebar.write("Salary CSV columns:", salary_df.columns.tolist())
    st.sidebar.write("Salary CSV sample (first 5 rows):")
    st.sidebar.dataframe(salary_df.head(5))

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

    # Ensure stats available
    if stats_df is None:
        st.error("No player stats available. Either upload a stats CSV or enable NBA API fetch.")
        st.stop()

    # Assign into collector and merge
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
# Compute advanced metrics (needed for training/visuals)
# -------------------------
# (We compute early but will display 'Derived metrics' at the bottom)
merged_df = collector.add_advanced_metrics(merged_df)

# Quick preview (small) — will also show full derived metrics later at bottom
st.write("Preview of merged dataset (first 6 rows):")
st.dataframe(merged_df.head(6))

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

# Trigger training
do_train = train_button
if not st.session_state.get("trained_once", False):
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

# Safety: ensure model_obj defined
if 'model_obj' not in locals():
    st.error("Model object not available. Training must complete successfully to use the dashboard.")
    st.stop()

# -------------------------
# Diagnostics: feature importance (percentage)
# -------------------------
st.subheader("Feature importance (percentage of total)")
fi_df = model_obj.feature_importance.copy()

# Be tolerant of column names
if 'Importance' in fi_df.columns and 'importance' not in fi_df.columns:
    fi_df = fi_df.rename(columns={'Importance': 'importance'})

# if model returned zero importances for some reason, avoid divide-by-zero
fi_df['abs_importance'] = fi_df['importance'].abs()
total = fi_df['abs_importance'].sum()
if total <= 0 or np.isnan(total):
    # fallback: evenly distribute
    n = len(fi_df)
    fi_df['importance_pct'] = 100.0 / n
else:
    fi_df['importance_pct'] = fi_df['abs_importance'] / total * 100.0

# Now sort properly for plotting (largest at top)
plot_df = fi_df.sort_values('importance_pct', ascending=True).copy()

# Plot with Plotly (text mapping uses the same sorted df)
fig_fi = px.bar(
    plot_df,
    x='importance_pct',
    y='Feature',
    orientation='h',
    text=plot_df['importance_pct'].round(1).astype(str) + '%',
    labels={'importance_pct': 'Importance (%)'},
    title='Feature importance (percentage of total)'
)
fig_fi.update_traces(marker_color='lightskyblue', marker_line_color='black', marker_line_width=1.0)
fig_fi.update_layout(height=380, margin=dict(l=140, r=20, t=50, b=40), xaxis_tickformat=',.0f')
st.plotly_chart(fig_fi, use_container_width=True)

# show table (sorted descending)
fi_table = fi_df[['Feature', 'importance_pct']].sort_values('importance_pct', ascending=False).reset_index(drop=True)
fi_table['importance_pct'] = fi_table['importance_pct'].round(2)
st.table(fi_table)

# -------------------------
# Predictions vs Actual (no statsmodels)
# -------------------------
st.subheader("Predicted vs Actual Salaries (test set)")
try:
    test_df = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred
    })
    test_df['actual_m'] = test_df['actual'] / 1e6
    test_df['predicted_m'] = test_df['predicted'] / 1e6

    # Use Plotly scatter without built-in trendline
    fig_pa = px.scatter(
        test_df,
        x='actual_m',
        y='predicted_m',
        labels={'actual_m': 'Actual Salary (M$)', 'predicted_m': 'Predicted Salary (M$)'},
        title='Predicted vs Actual Salaries (test set)',
        hover_data=[]
    )
    # add perfect prediction line
    minv = min(test_df['actual_m'].min(), test_df['predicted_m'].min())
    maxv = max(test_df['actual_m'].max(), test_df['predicted_m'].max())
    fig_pa.add_shape(type="line", x0=minv, y0=minv, x1=maxv, y1=maxv, line=dict(color="red", dash="dash"))
    st.plotly_chart(fig_pa, use_container_width=True)

    # compute metrics properly
    r2_val = r2_score(y_test, y_pred) if len(y_test) > 0 else np.nan
    mae_val = mean_absolute_error(y_test, y_pred) if len(y_test) > 0 else np.nan
    st.metric("Test MAE", f"${mae_val:,.0f}")
    st.metric("Test R²", f"{r2_val:.3f}")
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
# Top players by performance table (with predicted salary)
# -------------------------
st.subheader("Top players by performance metric")
# Guard: ensure df_perf and model_obj exist
if 'df_perf' not in locals() or df_perf is None:
    st.error("Performance dataframe not available.")
else:
    # slider
    top_n = st.slider("Number of top players to show", min_value=5, max_value=30, value=10, key="top_players_slider")

    try:
        top_players = df_perf.nlargest(top_n, 'PERFORMANCE_METRIC')[[
            'PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS', 'REB', 'AST', 'PERFORMANCE_METRIC', 'SALARY'
        ]].copy()

        # Add predicted salary column using model_obj
        def _predict_row_salary(r):
            stats = {
                'PTS': float(r['PTS']),
                'REB': float(r['REB']),
                'AST': float(r['AST']),
                'BLK': float(r.get('BLK', 0.0)),
                'STL': float(r.get('STL', 0.0)),
                'TS_PCT': float(r.get('TS_PCT', 0.0)),
                'WIN_SHARES': float(r.get('WIN_SHARES', 0.0)),
                'EPA': float(r.get('EPA', 0.0)),
            }
            try:
                pred, low, high = model_obj.predict_salary(stats)
                return pred
            except Exception:
                return np.nan

        top_players['Predicted_SALARY'] = top_players.apply(_predict_row_salary, axis=1)
        top_players['SALARY'] = top_players['SALARY'].apply(lambda x: f"${(x/1e6):.2f}M" if pd.notna(x) else "N/A")
        top_players['Predicted_SALARY'] = top_players['Predicted_SALARY'].apply(lambda x: f"${(x/1e6):.2f}M" if pd.notna(x) else "N/A")
        top_players['PERFORMANCE_METRIC'] = top_players['PERFORMANCE_METRIC'].round(2)

        st.dataframe(top_players.reset_index(drop=True))
    except Exception as e:
        st.error(f"Unable to build top players table: {e}")

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

        # Rough contribution proxy (same approach used earlier)
        fi_local = model_obj.feature_importance.copy()
        if 'Importance' in fi_local.columns and 'importance' not in fi_local.columns:
            fi_local = fi_local.rename(columns={'Importance': 'importance'})
        fi_local = fi_local.set_index('Feature')

        contribs = []
        for feat in model_obj.feature_names:
            val = stats[feat]
            fmin = df_perf[feat].min() if feat in df_perf.columns else 0.0
            fmax = df_perf[feat].max() if feat in df_perf.columns else 1.0
            norm = 0.0 if fmax <= fmin else (val - fmin) / (fmax - fmin)
            weight = fi_local.loc[feat]['importance'] if (feat in fi_local.index) else 0.0
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
# Downloads
# -------------------------
st.header("Export & downloads")
st.download_button(
    label="Download feature importance CSV",
    data=fi_df[['Feature', 'importance_pct']].sort_values('importance_pct', ascending=False).to_csv(index=False).encode('utf-8'),
    file_name='salary_feature_importance.csv',
    mime='text/csv'
)
st.download_button(
    label="Download merged dataset (CSV)",
    data=merged_df.to_csv(index=False).encode('utf-8'),
    file_name='merged_salary_stats.csv',
    mime='text/csv'
)

st.markdown("---")

# -------------------------
# Derived metrics display (moved to bottom by request)
# -------------------------
st.subheader("Derived metrics (WIN_SHARES, EPA) — sample")
try:
    sample_cols = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS', 'REB', 'AST', 'WIN_SHARES', 'EPA']
    present = [c for c in sample_cols if c in merged_df.columns]
    st.dataframe(merged_df[present].head(15))
except Exception:
    st.write("Derived metrics not available to preview.")

st.caption("Notes: WIN_SHARES and EPA are simplified / demo calculations. For formal modeling, consider canonical formulas or external advanced metrics sources.")

st.caption("Notes: Predictions assume the same preprocessing used during training. The contribution proxy is a heuristic to help interpret which features moved the prediction; for rigorous explanations use permutation importance or SHAP.")
