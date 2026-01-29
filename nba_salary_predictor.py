"""
NBA Salary Prediction Model - Fixed Name Matching
Predicts player salaries based on performance metrics including:
- Basic stats: Points, Rebounds, Assists, Blocks, Steals, TS%
- Advanced metrics: Win Shares, EPA
"""

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import players
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class NBADataCollector:
    """Handles fetching and processing NBA player statistics and salary data"""
    
    def __init__(self, season='2023-24', salary_file='nba_salaries_2023_24.csv'):
        self.season = season
        self.salary_file = salary_file
        self.player_stats = None
        self.salary_data = None
        
    def get_player_stats(self) -> pd.DataFrame:
        """Fetch current season player statistics"""
        print("Fetching player statistics from NBA API...")
        
        try:
            # Get player stats for the season
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=self.season,
                per_mode_detailed='PerGame',
                season_type_all_star='Regular Season'
            )
            
            df = stats.get_data_frames()[0]
            
            # Keep relevant columns
            columns_to_keep = [
                'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'GP', 'MIN',
                'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 
                'FT_PCT', 'FGA', 'FG3A', 'FTA', 'FGM'
            ]
            
            df = df[columns_to_keep]
            
            # Calculate True Shooting Percentage (TS%)
            # TS% = PTS / (2 * (FGA + 0.44 * FTA))
            df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
            df['TS_PCT'] = df['TS_PCT'].fillna(0)
            df['TS_PCT'] = df['TS_PCT'].clip(0, 1)  # Cap at 100%
            
            # Filter for players with at least 20 games played
            df = df[df['GP'] >= 20]
            
            # Extract last name for matching
            df['LAST_NAME'] = df['PLAYER_NAME'].str.split().str[-1]
            
            print(f"Retrieved stats for {len(df)} players")
            self.player_stats = df
            return df
            
        except Exception as e:
            print(f"Error fetching player stats: {e}")
            print("You may have hit the API rate limit. Wait a few minutes and try again.")
            return None
    
    def load_salary_data(self) -> pd.DataFrame:
        """Load salary data from CSV file"""
        print(f"Loading salary data from {self.salary_file}...")
        
        try:
            # Load the salary CSV
            salary_df = pd.read_csv(self.salary_file)
            
            print(f"Loaded salary data for {len(salary_df)} players")
            
            # The Player column contains last names (and some Jr., Sr., etc.)
            # Clean it up
            salary_df['LAST_NAME'] = salary_df['Player'].str.replace(
                r'\s+(Jr\.|Sr\.|II|III|IV|V|der)$', '', regex=True
            ).str.strip()
            
            # Also keep team for better matching when there are duplicate last names
            salary_df['TEAM'] = salary_df['Team']
            
            self.salary_data = salary_df
            return salary_df
            
        except Exception as e:
            print(f"Error loading salary data: {e}")
            return None
    
    def merge_stats_with_salaries(self) -> pd.DataFrame:
        """Merge player statistics with salary data using smart matching"""
        print("\nMerging statistics with salary data...")
        
        if self.player_stats is None or self.salary_data is None:
            print("Error: Missing stats or salary data")
            return None
        
        stats_df = self.player_stats.copy()
        salary_df = self.salary_data.copy()
        
        # Create a mapping dictionary for better matching
        merged_list = []
        matched_salary_indices = set()
        
        for idx, stats_row in stats_df.iterrows():
            stats_last = stats_row['LAST_NAME']
            stats_team = stats_row['TEAM_ABBREVIATION']
            
            # Try to find matching player in salary data
            # First try: exact last name + team match
            matches = salary_df[
                (salary_df['LAST_NAME'] == stats_last) & 
                (salary_df['TEAM'] == stats_team)
            ]
            
            # Second try: just last name (for players who changed teams)
            if len(matches) == 0:
                matches = salary_df[salary_df['LAST_NAME'] == stats_last]
            
            # If we found exactly one match, use it
            if len(matches) == 1:
                match_idx = matches.index[0]
                if match_idx not in matched_salary_indices:
                    matched_salary_indices.add(match_idx)
                    row_data = stats_row.to_dict()
                    row_data['SALARY'] = matches.iloc[0]['Salary']
                    row_data['SALARY_TEAM'] = matches.iloc[0]['TEAM']
                    merged_list.append(row_data)
            
            # If multiple matches, try to be smarter
            elif len(matches) > 1:
                # Prefer the one with matching team
                team_match = matches[matches['TEAM'] == stats_team]
                if len(team_match) == 1:
                    match_idx = team_match.index[0]
                    if match_idx not in matched_salary_indices:
                        matched_salary_indices.add(match_idx)
                        row_data = stats_row.to_dict()
                        row_data['SALARY'] = team_match.iloc[0]['Salary']
                        row_data['SALARY_TEAM'] = team_match.iloc[0]['TEAM']
                        merged_list.append(row_data)
        
        if len(merged_list) == 0:
            print("Error: No players matched!")
            print("\nDebug info:")
            print("Sample stats last names:", stats_df['LAST_NAME'].head(10).tolist())
            print("Sample salary last names:", salary_df['LAST_NAME'].head(10).tolist())
            return None
        
        merged = pd.DataFrame(merged_list)
        
        print(f"Successfully matched {len(merged)} players")
        
        # Show unmatched players from stats
        unmatched_stats = stats_df[~stats_df['LAST_NAME'].isin([m['LAST_NAME'] for m in merged_list])]
        if len(unmatched_stats) > 0:
            print(f"\nNote: {len(unmatched_stats)} players from stats not found in salary data")
            if len(unmatched_stats) <= 10:
                print("Some unmatched players:")
                for name in unmatched_stats['PLAYER_NAME'].head(10):
                    print(f"  - {name}")
        
        return merged
    
    def add_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Win Shares and EPA (estimated plus/minus)
        Note: These are simplified calculations for demonstration
        """
        print("Calculating advanced metrics...")
        
        # Simplified Win Shares calculation
        offensive_contribution = (
            df['PTS'] * 0.032 +
            df['AST'] * 0.034 +
            df['FGM'] * 0.017 +
            df['REB'] * 0.007
        )
        
        defensive_contribution = (
            df['STL'] * 0.052 +
            df['BLK'] * 0.051 +
            df['REB'] * 0.003
        )
        
        # Penalize inefficiency
        missed_shots = df['FGA'] - df['FGM']
        efficiency_penalty = missed_shots * 0.015
        
        df['WIN_SHARES'] = (
            (offensive_contribution + defensive_contribution - efficiency_penalty) * 
            (df['GP'] / 82)  # Adjust for games played
        ).clip(lower=0)
        
        # Simplified EPA (Estimated Plus/Minus Achievement)
        df['EPA'] = (
            df['PTS'] * 0.45 +
            df['AST'] * 0.60 +
            df['REB'] * 0.25 +
            df['STL'] * 0.55 +
            df['BLK'] * 0.55 +
            df['TS_PCT'] * 15 -
            missed_shots * 0.25
        ) / 10
        
        # Normalize EPA to a reasonable range
        df['EPA'] = df['EPA'].clip(-10, 15)
        
        print("Advanced metrics calculated")
        return df


class SalaryPredictionModel:
    """Machine learning model to predict NBA player salaries"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'PTS', 'REB', 'AST', 'BLK', 'STL', 'TS_PCT', 'WIN_SHARES', 'EPA'
        ]
        self.feature_importance = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and scale features for the model"""
        X = df[self.feature_names].values
        y = df['SALARY'].values
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, optimize=True):
        """
        Train the model with optional hyperparameter optimization
        """
        print("\n" + "="*60)
        print("TRAINING SALARY PREDICTION MODEL")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set: {len(X_train)} players")
        print(f"Test set: {len(X_test)} players")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if optimize:
            # Try multiple models and select the best
            models = {
                'Random Forest': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            }
            
            best_score = -np.inf
            best_model = None
            best_name = None
            
            print("\nEvaluating models...")
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                print(f"  {name}: R² = {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name
            
            self.model = best_model
            print(f"\n✓ Best model: {best_name} (R² = {best_score:.4f})")
        else:
            # Use Gradient Boosting as default
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        print("\n" + "-"*60)
        print("MODEL PERFORMANCE")
        print("-"*60)
        print(f"Training Set:")
        print(f"  R² Score: {r2_score(y_train, y_pred_train):.4f}")
        print(f"  MAE: ${mean_absolute_error(y_train, y_pred_train):,.0f}")
        
        print(f"\nTest Set:")
        print(f"  R² Score: {r2_score(y_test, y_pred_test):.4f}")
        print(f"  MAE: ${mean_absolute_error(y_test, y_pred_test):,.0f}")
        print(f"  RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_test)):,.0f}")
        print("-"*60)
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance:")
        for idx, row in self.feature_importance.iterrows():
            print(f"  {row['Feature']:15s}: {row['Importance']:.4f}")
        
        return X_test_scaled, y_test, y_pred_test
    
    def predict_salary(self, stats: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Predict salary for given stats
        Returns: (prediction, lower_bound, upper_bound)
        """
        # Create feature array
        features = np.array([[
            stats['PTS'], stats['REB'], stats['AST'], stats['BLK'],
            stats['STL'], stats['TS_PCT'], stats['WIN_SHARES'], stats['EPA']
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Calculate confidence interval (using ±8% as a tight range)
        lower_bound = prediction * 0.92
        upper_bound = prediction * 1.08
        
        return prediction, lower_bound, upper_bound
    
    def create_performance_metric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a composite performance metric based on weighted features
        """
        # Use feature importance as weights
        weights = dict(zip(
            self.feature_importance['Feature'],
            self.feature_importance['Importance']
        ))
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate composite metric
        df['PERFORMANCE_METRIC'] = 0
        for feature, weight in weights.items():
            # Normalize each feature to 0-100 scale
            feature_min = df[feature].min()
            feature_max = df[feature].max()
            if feature_max > feature_min:
                normalized = (df[feature] - feature_min) / (feature_max - feature_min) * 100
                df['PERFORMANCE_METRIC'] += normalized * weight
        
        return df


class SalaryVisualizer:
    """Create visualizations for salary predictions"""
    
    @staticmethod
    def plot_salary_vs_performance(df: pd.DataFrame, save_path: str = None):
        """Plot salary vs performance metric"""
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Create scatter plot
        scatter = ax.scatter(
            df['SALARY']/1e6, 
            df['PERFORMANCE_METRIC'], 
            alpha=0.6, 
            s=120, 
            c=df['PERFORMANCE_METRIC'], 
            cmap='viridis', 
            edgecolors='black', 
            linewidth=0.7
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Performance Metric', fontsize=13, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(df['SALARY']/1e6, df['PERFORMANCE_METRIC'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['SALARY'].min()/1e6, df['SALARY'].max()/1e6, 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2.5, label='Trend Line')
        
        ax.set_xlabel('Salary (Millions $)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Performance Metric', fontsize=14, fontweight='bold')
        ax.set_title('NBA Player Salary vs Performance Metric (2023-24)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        
        # Annotate top performers
        top_performers = df.nlargest(5, 'PERFORMANCE_METRIC')
        for _, row in top_performers.iterrows():
            ax.annotate(
                row['PLAYER_NAME'], 
                xy=(row['SALARY']/1e6, row['PERFORMANCE_METRIC']),
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=9, 
                alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.4),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', alpha=0.6)
            )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_feature_importance(feature_importance: pd.DataFrame, save_path: str = None):
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(11, 7))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
        
        bars = ax.barh(
            feature_importance['Feature'], 
            feature_importance['Importance'],
            color=colors,
            edgecolor='black', 
            linewidth=1.5
        )
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width, 
                bar.get_y() + bar.get_height()/2, 
                f'  {width:.3f}',
                ha='left', 
                va='center', 
                fontsize=11, 
                fontweight='bold'
            )
        
        ax.set_xlabel('Importance', fontsize=13, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=13, fontweight='bold')
        ax.set_title('Feature Importance for NBA Salary Prediction', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Feature importance plot saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_predictions_vs_actual(y_test, y_pred, save_path: str = None):
        """Plot predicted vs actual salaries"""
        fig, ax = plt.subplots(figsize=(11, 9))
        
        ax.scatter(
            y_test/1e6, 
            y_pred/1e6, 
            alpha=0.6, 
            s=100, 
            edgecolors='black', 
            linewidth=0.7,
            c='steelblue'
        )
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min()) / 1e6
        max_val = max(y_test.max(), y_pred.max()) / 1e6
        ax.plot(
            [min_val, max_val], 
            [min_val, max_val], 
            'r--', 
            linewidth=2.5, 
            label='Perfect Prediction',
            alpha=0.8
        )
        
        # Calculate and display metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        textstr = f'R² = {r2:.3f}\nMAE = ${mae/1e6:.2f}M'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(
            0.05, 0.95, 
            textstr, 
            transform=ax.transAxes, 
            fontsize=12,
            verticalalignment='top', 
            bbox=props
        )
        
        ax.set_xlabel('Actual Salary (Millions $)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted Salary (Millions $)', fontsize=14, fontweight='bold')
        ax.set_title('Predicted vs Actual NBA Salaries (2023-24)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Predictions plot saved to {save_path}")
        
        plt.show()


def interactive_prediction(model: SalaryPredictionModel):
    """Interactive function to get salary predictions for user input"""
    print("\n" + "="*70)
    print(" "*15 + "NBA SALARY PREDICTOR - INTERACTIVE MODE")
    print("="*70)
    
    while True:
        print("\nEnter player statistics (or type 'quit' to exit):")
        print("-"*70)
        
        try:
            pts = input("Points per game (PTS): ")
            if pts.lower() == 'quit':
                break
            pts = float(pts)
            
            reb = float(input("Rebounds per game (REB): "))
            ast = float(input("Assists per game (AST): "))
            blk = float(input("Blocks per game (BLK): "))
            stl = float(input("Steals per game (STL): "))
            ts_pct = float(input("True Shooting % (TS_PCT, e.g., 0.58): "))
            win_shares = float(input("Win Shares (WIN_SHARES): "))
            epa = float(input("EPA: "))
            
            stats = {
                'PTS': pts,
                'REB': reb,
                'AST': ast,
                'BLK': blk,
                'STL': stl,
                'TS_PCT': ts_pct,
                'WIN_SHARES': win_shares,
                'EPA': epa
            }
            
            prediction, lower, upper = model.predict_salary(stats)
            
            print("\n" + "="*70)
            print(" "*20 + "SALARY PREDICTION RESULTS")
            print("="*70)
            print(f"  Predicted Salary:     ${prediction:,.0f}")
            print(f"  Salary Range (±8%):   ${lower:,.0f} - ${upper:,.0f}")
            print("="*70)
            
        except ValueError as e:
            print(f"\n⚠ Invalid input. Please enter numerical values.")
        except Exception as e:
            print(f"\n⚠ An error occurred: {e}")
    
    print("\n" + "="*70)
    print(" "*18 + "Thank you for using NBA Salary Predictor!")
    print("="*70)


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print(" "*15 + "NBA SALARY PREDICTION MODEL")
    print(" "*18 + "2023-24 Season Analysis")
    print("="*70)
    
    # Step 1: Collect Data
    collector = NBADataCollector(
        season='2023-24',
        salary_file='nba_salaries_2023_24.csv'
    )
    
    # Load salary data
    salary_data = collector.load_salary_data()
    if salary_data is None:
        print("Error: Could not load salary data. Make sure 'nba_salaries_2023_24.csv' is in the same directory.")
        return
    
    # Get player stats
    player_stats = collector.get_player_stats()
    if player_stats is None:
        print("Error: Could not fetch player stats from NBA API.")
        return
    
    # Merge stats with salaries
    merged_data = collector.merge_stats_with_salaries()
    if merged_data is None or len(merged_data) == 0:
        print("Error: Could not merge stats with salary data.")
        return
    
    # Add advanced metrics
    merged_data = collector.add_advanced_metrics(merged_data)
    
    print(f"\n✓ Final dataset: {len(merged_data)} players with complete data")
    
    # Step 2: Train Model
    model = SalaryPredictionModel()
    X, y = model.prepare_features(merged_data)
    X_test, y_test, y_pred = model.train(X, y, optimize=True)
    
    # Step 3: Create performance metric
    merged_data = model.create_performance_metric(merged_data)
    
    # Step 4: Visualizations
    visualizer = SalaryVisualizer()
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    visualizer.plot_feature_importance(
        model.feature_importance,
        save_path='feature_importance.png'
    )
    
    visualizer.plot_predictions_vs_actual(
        y_test, y_pred,
        save_path='predictions_vs_actual.png'
    )
    
    visualizer.plot_salary_vs_performance(
        merged_data,
        save_path='salary_vs_performance.png'
    )
    
    # Show some example players
    print("\n" + "="*70)
    print("TOP 10 PLAYERS BY PERFORMANCE METRIC")
    print("="*70)
    top_players = merged_data.nlargest(10, 'PERFORMANCE_METRIC')[
        ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'PTS', 'REB', 'AST', 
         'PERFORMANCE_METRIC', 'SALARY']
    ].copy()
    top_players['SALARY'] = top_players['SALARY'].apply(lambda x: f"${x/1e6:.2f}M")
    top_players['PERFORMANCE_METRIC'] = top_players['PERFORMANCE_METRIC'].round(2)
    print(top_players.to_string(index=False))
    
    # Step 5: Interactive Predictions
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE!")
    print("="*70)
    
    choice = input("\nWould you like to make interactive salary predictions? (yes/no): ")
    if choice.lower() in ['yes', 'y']:
        interactive_prediction(model)
    
    print("\n✓ Program complete. Check the generated PNG files for visualizations.")
    print("  - feature_importance.png")
    print("  - predictions_vs_actual.png")
    print("  - salary_vs_performance.png")


if __name__ == "__main__":
    main()
