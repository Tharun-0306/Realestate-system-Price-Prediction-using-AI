"""
Real Estate Price Prediction - Model Training Pipeline
Uses the King County House Sales dataset (Kaggle)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

# ─── Try to import XGBoost / LightGBM ────────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed. Run: pip install lightgbm")

# ─── 1. Load Data ─────────────────────────────────────────────────────────────

def load_data(filepath: str = "kc_house_data.csv") -> pd.DataFrame:
   
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print("⚠️  Dataset not found. Generating synthetic data for demo...")
        return generate_synthetic_data()

def generate_synthetic_data(n=5000) -> pd.DataFrame:
    """Generate realistic synthetic housing data for demonstration."""
    np.random.seed(42)
    df = pd.DataFrame({
        'bedrooms':      np.random.choice([2,3,4,5,6], n, p=[0.1,0.4,0.35,0.12,0.03]),
        'bathrooms':     np.random.choice([1,1.5,2,2.5,3,3.5,4], n),
        'sqft_living':   np.random.randint(800, 6000, n),
        'sqft_lot':      np.random.randint(2000, 50000, n),
        'floors':        np.random.choice([1,1.5,2,2.5,3], n, p=[0.3,0.1,0.45,0.05,0.1]),
        'waterfront':    np.random.choice([0,1], n, p=[0.93,0.07]),
        'view':          np.random.choice([0,1,2,3,4], n, p=[0.7,0.1,0.1,0.06,0.04]),
        'condition':     np.random.choice([1,2,3,4,5], n, p=[0.01,0.1,0.65,0.2,0.04]),
        'grade':         np.random.choice(range(4,13), n),
        'sqft_above':    np.random.randint(600, 4000, n),
        'sqft_basement': np.random.randint(0, 2000, n),
        'yr_built':      np.random.randint(1900, 2015, n),
        'yr_renovated':  np.random.choice([0]*8 + list(range(1970, 2020)), n),
        'zipcode':       np.random.randint(98001, 98200, n),
        'lat':           np.random.uniform(47.1, 47.8, n),
        'long':          np.random.uniform(-122.5, -121.3, n),
        'sqft_living15': np.random.randint(800, 4000, n),
        'sqft_lot15':    np.random.randint(2000, 20000, n),
    })

    # Generate price based on features (realistic formula)
    df['price'] = (
        150000
        + df['sqft_living'] * 120
        + df['bedrooms'] * 12000
        + df['bathrooms'] * 10000
        + df['grade'] * 18000
        + df['condition'] * 8000
        + df['waterfront'] * 180000
        + df['view'] * 22000
        + df['floors'] * 7000
        - (2024 - df['yr_built']) * 700
        + df['sqft_basement'] * 45
        + np.where(df['yr_renovated'] > 2000, 45000, 0)
        + (df['lat'] - 47.0) * 45000
        + np.random.normal(0, 30000, n)  # noise
    ).clip(lower=80000)

    print(f"✅ Generated synthetic dataset: {n:,} rows")
    return df

# ─── 2. Clean Data ────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle NaN values, outliers, and bad data before feature engineering.
    This is the most important step — NaN in any column will crash sklearn models.
    """
    df = df.copy()

    initial_rows = len(df)

    # ── Drop rows where the TARGET is missing (can't train without it) ─────────
    if 'price' in df.columns:
        df = df.dropna(subset=['price'])
        df = df[df['price'] > 0]

    # ── Known numeric columns and their sensible fill values ───────────────────
    numeric_fills = {
        'bedrooms':      df['bedrooms'].median()   if 'bedrooms'      in df else 3,
        'bathrooms':     df['bathrooms'].median()  if 'bathrooms'     in df else 2,
        'sqft_living':   df['sqft_living'].median()if 'sqft_living'   in df else 1800,
        'sqft_lot':      df['sqft_lot'].median()   if 'sqft_lot'      in df else 6000,
        'floors':        1.0,
        'waterfront':    0,       # assume not waterfront if unknown
        'view':          0,       # assume no view if unknown
        'condition':     3,       # average condition
        'grade':         7,       # average grade
        'sqft_above':    df['sqft_living'].median()if 'sqft_living'   in df else 1800,
        'sqft_basement': 0,       # assume no basement if unknown
        'yr_built':      1980,
        'yr_renovated':  0,       # 0 means never renovated
        'zipcode':       df['zipcode'].mode()[0]   if 'zipcode'       in df and len(df) else 98001,
        'lat':           df['lat'].median()        if 'lat'           in df else 47.5,
        'long':          df['long'].median()       if 'long'          in df else -122.0,
        'sqft_living15': df['sqft_living15'].median() if 'sqft_living15' in df else 1800,
        'sqft_lot15':    df['sqft_lot15'].median()    if 'sqft_lot15'    in df else 5000,
    }

    for col, fill_val in numeric_fills.items():
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"  ⚠️  Filling {nan_count:,} NaN in '{col}' with {fill_val}")
            df[col] = df[col].fillna(fill_val)

    # ── Fill any remaining numeric NaNs with column median ─────────────────────
    remaining_numeric = df.select_dtypes(include=[np.number]).columns
    still_nan = df[remaining_numeric].isna().sum()
    still_nan = still_nan[still_nan > 0]
    if len(still_nan) > 0:
        print(f"  ⚠️  Filling remaining NaN columns with median: {list(still_nan.index)}")
        df[remaining_numeric] = df[remaining_numeric].fillna(df[remaining_numeric].median())

    # ── Remove extreme outlier rows (bedroom=33, price=$0 etc.) ───────────────
    if 'bedrooms' in df.columns:
        df = df[df['bedrooms'] <= 20]
    if 'price' in df.columns:
        # Remove bottom 0.5% and top 1% price outliers
        low  = df['price'].quantile(0.005)
        high = df['price'].quantile(0.99)
        df = df[(df['price'] >= low) & (df['price'] <= high)]

    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"  🗑️  Dropped {dropped:,} bad rows ({initial_rows:,} → {len(df):,})")

    # ── Final sanity check ─────────────────────────────────────────────────────
    nan_remaining = df.isna().sum().sum()
    if nan_remaining > 0:
        print(f"  ❌ WARNING: {nan_remaining} NaN values still remain after cleaning!")
    else:
        print(f"  ✅ Data clean — 0 NaN values in {len(df):,} rows")

    return df

# ─── 3. Feature Engineering ───────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Age features
    df['house_age'] = 2024 - df['yr_built']
    df['years_since_renovation'] = np.where(
        df['yr_renovated'] > 0,
        2024 - df['yr_renovated'],
        df['house_age']
    )
    df['was_renovated'] = (df['yr_renovated'] > 0).astype(int)

    # Area ratios
    df['basement_ratio'] = df['sqft_basement'] / df['sqft_living'].clip(lower=1)
    df['lot_ratio'] = df['sqft_living'] / df['sqft_lot'].clip(lower=1)
    df['neighbor_ratio'] = df['sqft_living'] / df['sqft_living15'].clip(lower=1)

    # Interaction features
    df['bath_per_bed'] = df['bathrooms'] / df['bedrooms'].clip(lower=1)
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['luxury_score'] = df['grade'] * df['condition']
    df['premium_features'] = df['waterfront'] * 3 + df['view']

    # Drop original date columns
    df.drop(columns=['yr_built', 'yr_renovated'], inplace=True, errors='ignore')

    return df

# ─── 3. Preprocessing ─────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    # ── Step 1: Drop ID/date columns ──────────────────────────────────────────
    drop_cols = ['id', 'date']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Step 2: Clean NaN / outliers BEFORE any math ─────────────────────────
    print("\n🧹 Cleaning data...")
    df = clean_data(df)

    # ── Step 3: Feature engineering ──────────────────────────────────────────
    df = engineer_features(df)

    # ── Step 4: Split X / y ───────────────────────────────────────────────────
    X = df.drop(columns=['price'])
    y = df['price']

    # ── Step 5: Safety-net imputer (catches any NaN created during engineering)
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # Confirm zero NaN before splitting
    remaining = X_imputed.isna().sum().sum()
    if remaining > 0:
        raise ValueError(f"❌ {remaining} NaN values survived imputation — check your data!")
    print(f"  ✅ All features clean after imputation")

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    # ── Step 6: Scale for Linear Regression ──────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_imputed.columns.tolist()

# ─── 4. Model Training ────────────────────────────────────────────────────────

def train_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    models = {}
    results = {}

    print("\n📊 Training Models...\n" + "="*50)

    # Linear Regression (baseline)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models['Linear Regression'] = (lr, True)  # True = needs scaled input

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = (rf, False)

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)
    models['Gradient Boosting'] = (gb, False)

    if HAS_XGB:
        xgb_model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        models['XGBoost'] = (xgb_model, False)

    if HAS_LGB:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=50, random_state=42, n_jobs=-1
        )
        lgb_model.fit(X_train, y_train)
        models['LightGBM'] = (lgb_model, False)

    # ─── Evaluate ─────────────────────────────────────────────────────────────
    best_model_name = None
    best_r2 = -np.inf

    for name, (model, scaled) in models.items():
        X_t = X_test_scaled if scaled else X_test
        y_pred = model.predict(X_t)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

        print(f"🔹 {name:<25} MAE: ${mae:>10,.0f}  RMSE: ${rmse:>10,.0f}  R²: {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    print(f"\n🏆 Best model: {best_model_name} (R² = {best_r2:.4f})")
    return models, results, best_model_name

# ─── 5. Feature Importance ────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names: list, model_name: str):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        fi_df = fi_df.sort_values('importance', ascending=False).head(15)
        print(f"\n📊 Top 15 Feature Importances ({model_name}):")
        for _, row in fi_df.iterrows():
            bar = "█" * int(row['importance'] * 500)
            print(f"  {row['feature']:<25} {bar} {row['importance']:.4f}")
    return fi_df if hasattr(model, 'feature_importances_') else None

# ─── 6. Save Model ────────────────────────────────────────────────────────────

def save_model(model, scaler, feature_names: list, path: str = "models/"):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(f"{path}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"{path}/features.pkl", 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"\n💾 Model saved to {path}/")

# ─── Main ─────────────────────────────────────────────────────────────────────

import os

if __name__ == "__main__":
    print("🏠 Real Estate Price Prediction - Training Pipeline")
    print("="*55)

    df = load_data()

    print(f"\n📈 Price Statistics:")
    print(f"  Min:    ${df['price'].min():>12,.0f}")
    print(f"  Max:    ${df['price'].max():>12,.0f}")
    print(f"  Mean:   ${df['price'].mean():>12,.0f}")
    print(f"  Median: ${df['price'].median():>12,.0f}")

    X_train, X_test, X_train_s, X_test_s, y_train, y_test, scaler, features = preprocess(df)
    print(f"\n✅ Features engineered: {len(features)}")
    print(f"✅ Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    models, results, best = train_models(X_train, X_test, X_train_s, X_test_s, y_train, y_test)

    best_model, needs_scale = models[best]
    plot_feature_importance(best_model, features, best)

    save_model(best_model, scaler, features)

    print("\n✅ Training complete! Run the API with:")
    print("   uvicorn backend.main:app --reload")

