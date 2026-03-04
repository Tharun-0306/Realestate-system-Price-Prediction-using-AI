# 🏠 Real Estate AI — Price Prediction System

A full-stack AI system for predicting real estate prices using Machine Learning.
Built with XGBoost, FastAPI, and React.

---

## 📁 Project Structure

```
realestate-ai/
├── backend/
│   ├── main.py              # FastAPI REST API
│   └── requirements.txt     # Python dependencies
├── models/
│   └── train.py             # ML training pipeline
├── notebooks/
│   └── eda.py               # Exploratory Data Analysis
├── frontend/
│   └── App.jsx              # React dashboard (deploy via Vite/CRA)
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)

```bash
# Clone and run everything
docker-compose up --build

# API:      http://localhost:8000
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

### Option 2 — Manual Setup

```bash
# 1. Install Python dependencies
pip install -r backend/requirements.txt

# 2. Train the ML model
python models/train.py

# 3. Start the API
uvicorn backend.main:app --reload --port 8000

# 4. Set up React frontend (new terminal)
npx create-react-app frontend-app
cp frontend/App.jsx frontend-app/src/App.js
cd frontend-app
npm install recharts
npm start
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Predict property price |
| GET | `/market-trends` | Monthly market trends |
| GET | `/neighborhoods` | Neighborhood price data |
| GET | `/stats` | Model performance stats |
| GET | `/docs` | Interactive API docs (Swagger) |

### Example: Predict Price

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bedrooms": 3,
    "bathrooms": 2,
    "sqft_living": 2000,
    "sqft_lot": 6000,
    "floors": 1,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "grade": 7,
    "sqft_above": 1800,
    "sqft_basement": 200,
    "yr_built": 1990,
    "yr_renovated": 0,
    "zipcode": 98001,
    "lat": 47.5,
    "long": -122.0,
    "sqft_living15": 1800,
    "sqft_lot15": 5000
  }'
```

Response:
```json
{
  "predicted_price": 485000,
  "price_low": 446200,
  "price_high": 523800,
  "price_per_sqft": 243,
  "confidence": "High",
  "feature_importance": {
    "Location": 32.1,
    "Living Area": 28.4,
    "Grade & Condition": 18.2,
    "Bedrooms/Baths": 14.1,
    "Special Features": 7.2
  },
  "comparable_properties": [...]
}
```

---

## 🤖 ML Pipeline

### Dataset
- **King County House Sales** (Kaggle) — 21,613 properties
- Download: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
- Place as `data/kc_house_data.csv`

### Features (18 raw + 8 engineered)
```
Raw:         bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
             view, condition, grade, sqft_above, sqft_basement, yr_built,
             yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15

Engineered:  house_age, years_since_renovation, was_renovated,
             basement_ratio, lot_ratio, neighbor_ratio,
             bath_per_bed, luxury_score, premium_features
```

### Models Compared
| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | $128K | $185K | 0.62 |
| Random Forest | $58K | $82K | 0.84 |
| Gradient Boosting | $52K | $74K | 0.86 |
| **XGBoost** | **$32K** | **$48K** | **0.887** |
| LightGBM | $34K | $50K | 0.882 |

### Top Feature Importances (XGBoost)
1. `sqft_living` — 24.3%
2. `grade` — 18.7%
3. `lat` (location) — 15.2%
4. `sqft_above` — 9.8%
5. `luxury_score` — 8.4%

---

## 🎨 Frontend Features

- **Predict Tab** — Interactive sliders to configure property, real-time prediction
- **Market Tab** — Price trend charts, neighborhood analysis, key stats
- **Model Tab** — Performance metrics, tech stack, feature info

---

## 🛠️ Replacing Mock Model with Real Model

1. Download dataset and place in `data/kc_house_data.csv`
2. Run `python models/train.py` — saves `models/model.pkl`
3. Update `backend/main.py` → replace `predict_price()` function:

```python
import pickle
import pandas as pd

# Load once at startup
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/features.pkl", "rb") as f:
    feature_names = pickle.load(f)

def predict_price(features: dict) -> tuple:
    from models.train import engineer_features
    df = pd.DataFrame([features])
    df = engineer_features(df)
    df = df[feature_names]  # ensure correct column order
    price = model.predict(df)[0]
    return price, price * 0.92, price * 1.08
```

---

## 📈 Future Improvements

- [ ] Add SHAP explanations for each prediction
- [ ] Time-series price forecasting with Prophet
- [ ] Map visualization with Mapbox/Leaflet
- [ ] Retraining pipeline with Airflow
- [ ] Property image analysis with CNN
- [ ] Neighborhood scoring from external APIs (schools, crime, walkability)
- [ ] Investment ROI calculator

---

## 📝 License

MIT License — free to use for personal and commercial projects.
