"""
Exploratory Data Analysis (EDA) Notebook
Real Estate Price Prediction Project
Run as a script or paste sections into Jupyter notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.style.use('dark_background')
ACCENT = '#4F7FFF'
sns.set_palette([ACCENT, '#7B5EFF', '#22C55E', '#F59E0B', '#F87171'])

# ─── Load Data ────────────────────────────────────────────────────────────────
def load_or_generate(path="data/kc_house_data.csv"):
    try:
        df = pd.read_csv(path)
        print(f"✅ Loaded: {df.shape}")
        return df
    except:
        print("⚠️  Generating synthetic dataset...")
        np.random.seed(42)
        n = 3000
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
        df['price'] = (
            150000 + df['sqft_living'] * 120 + df['bedrooms'] * 12000 +
            df['bathrooms'] * 10000 + df['grade'] * 18000 + df['condition'] * 8000 +
            df['waterfront'] * 180000 + df['view'] * 22000 + df['floors'] * 7000 -
            (2024 - df['yr_built']) * 700 + df['sqft_basement'] * 45 +
            np.where(df['yr_renovated'] > 2000, 45000, 0) +
            (df['lat'] - 47.0) * 45000 + np.random.normal(0, 30000, n)
        ).clip(lower=80000)
        return df

df = load_or_generate()

# ─── 1. Basic Info ────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("📊 DATASET OVERVIEW")
print("="*55)
print(df.describe().to_string())
print(f"\n🔍 Missing values:\n{df.isnull().sum()[df.isnull().sum()>0].to_string() or 'None!'}")

# ─── 2. Price Distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Price Distribution Analysis", fontsize=16, fontweight='bold', color='white')

axes[0].hist(df['price']/1000, bins=50, color=ACCENT, alpha=0.8, edgecolor='none')
axes[0].set_title("Raw Price Distribution", color='white')
axes[0].set_xlabel("Price ($K)", color='gray')
axes[0].set_ylabel("Count", color='gray')
axes[0].axvline(df['price'].median()/1000, color='#22C55E', linestyle='--', label=f"Median ${df['price'].median()/1000:.0f}K")
axes[0].legend()

axes[1].hist(np.log(df['price']), bins=50, color='#7B5EFF', alpha=0.8, edgecolor='none')
axes[1].set_title("Log-Transformed Price (more normal)", color='white')
axes[1].set_xlabel("log(Price)", color='gray')
axes[1].set_ylabel("Count", color='gray')

plt.tight_layout()
plt.savefig("notebooks/price_distribution.png", dpi=150, bbox_inches='tight', facecolor='#080E1A')
print("\n💾 Saved: notebooks/price_distribution.png")

# ─── 3. Correlation Heatmap ───────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
price_corr = corr['price'].drop('price').sort_values(ascending=False)

print("\n📈 TOP CORRELATIONS WITH PRICE:")
for feat, val in price_corr.head(10).items():
    bar = "█" * int(abs(val) * 30)
    print(f"  {feat:<25} {bar} {val:+.3f}")

plt.figure(figsize=(12, 9))
top_feats = list(price_corr.head(12).index) + ['price']
mask = np.triu(np.ones_like(corr[top_feats].loc[top_feats], dtype=bool))
sns.heatmap(corr[top_feats].loc[top_feats], mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Matrix", fontsize=14, fontweight='bold', color='white', pad=20)
plt.tight_layout()
plt.savefig("notebooks/correlation_heatmap.png", dpi=150, bbox_inches='tight', facecolor='#080E1A')
print("💾 Saved: notebooks/correlation_heatmap.png")

# ─── 4. Price by Key Features ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Price by Key Features", fontsize=16, fontweight='bold', color='white')

# Bedrooms
bed_median = df.groupby('bedrooms')['price'].median() / 1000
axes[0,0].bar(bed_median.index, bed_median.values, color=ACCENT, alpha=0.85, edgecolor='none')
axes[0,0].set_title("Median Price by Bedrooms", color='white')
axes[0,0].set_xlabel("Bedrooms", color='gray')
axes[0,0].set_ylabel("Price ($K)", color='gray')

# Grade
grade_median = df.groupby('grade')['price'].median() / 1000
axes[0,1].bar(grade_median.index, grade_median.values, color='#7B5EFF', alpha=0.85, edgecolor='none')
axes[0,1].set_title("Median Price by Grade", color='white')
axes[0,1].set_xlabel("Building Grade", color='gray')
axes[0,1].set_ylabel("Price ($K)", color='gray')

# sqft_living vs price (scatter)
sample = df.sample(min(1000, len(df)), random_state=42)
axes[1,0].scatter(sample['sqft_living'], sample['price']/1000, alpha=0.3, s=15, color='#22C55E')
axes[1,0].set_title("Living Area vs Price", color='white')
axes[1,0].set_xlabel("Living Area (sqft)", color='gray')
axes[1,0].set_ylabel("Price ($K)", color='gray')

# Waterfront vs non-waterfront
wf = df.groupby('waterfront')['price'].median() / 1000
axes[1,1].bar(['No Waterfront', 'Waterfront'], wf.values, color=[ACCENT, '#F59E0B'], alpha=0.85, edgecolor='none')
axes[1,1].set_title("Waterfront Price Premium", color='white')
axes[1,1].set_ylabel("Median Price ($K)", color='gray')
for bar, val in zip(axes[1,1].patches, wf.values):
    axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f"${val:.0f}K", ha='center', color='white', fontsize=11, fontweight='bold')

for ax in axes.flat:
    ax.tick_params(colors='gray')
    ax.spines['bottom'].set_color((1, 1, 1, 0.1))
    ax.spines['left'].set_color((1, 1, 1, 0.1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("notebooks/price_by_features.png", dpi=150, bbox_inches='tight', facecolor='#080E1A')
print("💾 Saved: notebooks/price_by_features.png")

print("\n✅ EDA complete! Charts saved to notebooks/")
print("👉 Next step: Run models/train.py to train and evaluate the ML models")
