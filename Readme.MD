<div align="center">

# Tourism Experience Analytics

**Classification, Prediction, and Recommendation System for Tourism Data**

An end-to-end supervised learning project that predicts attraction ratings and visitor travel modes from 50,000+ tourism transactions, and recommends personalised attractions using item-based collaborative filtering and content-based TF-IDF similarity — deployed as an interactive Streamlit app.

**[Live App →](https://m6dznfkrgaww3t7k2qwmyh.streamlit.app/)**

</div>

---

## Overview

This project analyses a multi-table tourism dataset spanning 9 relational Excel tables, 33,530 unique users, and 30 curated tourist attractions across 5 continents to achieve three primary objectives:

1. **Predict attraction ratings** (1–5) from traveller demographics, visit context, and engineered behavioural aggregates using regression models.
2. **Predict visit mode** — Business, Couples, Family, Friends, or Solo — to enable targeted marketing and resource planning using multi-class classification.
3. **Recommend personalised attractions** using item-based collaborative filtering (cosine similarity on the user–item rating matrix), TF-IDF content-based filtering, and a hybrid weighted combination.

All three modules, plus supporting SQL exploration and EDA, are served through a Streamlit app with a custom Oracle AI-inspired dark/light UI.

---

## Data Pipeline

The raw dataset spans 9 relational Excel tables. The Transaction table (52,930 rows) is the analytical core; all other tables are decoded via foreign-key joins.

| Step | Action |
|---|---|
| 1 | Loaded all 9 Excel tables: Transaction, User, City, Country, Region, Continent, Item, Mode, Type |
| 2 | Removed rows where `VisitMode = 0` (unknown travel mode — no classification target to train on) |
| 3 | Decoded all foreign-key IDs to human-readable labels via LEFT JOIN across all lookup tables |
| 4 | Filled unresolved string lookups with `"Unknown"` to prevent propagation of join failures |
| 5 | Dropped rows with nulls in critical modelling columns (Rating, VisitMode, ContinentId, AttractionId) |
| 6 | Parsed `CityId` from float64 to int (NaN coercion from Excel) via `fillna(0).astype(int)` before joining |
| 7 | Engineered `Season` and `Quarter` features from `VisitMonth` |
| 8 | Computed user-level aggregates: `UserAvgRating`, `UserVisitCount` (per-user rating tendency and engagement) |
| 9 | Computed attraction-level aggregates: `AttractionAvgRating`, `AttractionVisitCount` (historical quality signal) |
| 10 | Label-encoded `VisitModeLabel` → integer codes (0–4) for classification target |

**Result: 50,000+ clean transaction rows ready for modelling across 14 engineered features.**

---

## SQL-Based Exploration

Six production-style queries were run via Python's built-in `sqlite3` module on an in-memory database loaded from the master DataFrame:

| Query | Business Question Answered |
|---|---|
| Q1 | Top 10 attractions by average rating (minimum 100 visits, `HAVING` clause) |
| Q2 | Visit mode % distribution by continent (window function `PARTITION BY`) |
| Q3 | Monthly seasonality — visit volume and average rating by month (`CASE WHEN`) |
| Q4 | User segmentation by visit frequency: Power, Regular, Occasional, One-time (subquery + `CASE`) |
| Q5 | Year-over-year visit growth with percentage change (`WITH CTE` + `LEFT JOIN` lag) |
| Q6 | Attraction type × visit mode cross-tabulation (`GROUP BY` + pivot) |

---

## Regression — Predicting Attraction Ratings

### Feature Selection

14 features were selected spanning temporal, geographic, attraction identity, and behavioural aggregates. The two strongest features — `UserAvgRating` and `AttractionAvgRating` — capture rater bias and historical attraction quality respectively.

### Model Comparison

Eleven regression algorithms were trained and evaluated on an 80/20 train-test split. All distance-based and regularised models used `StandardScaler`; tree-based models used raw features.

| Model | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|
| **LightGBM** | **0.7214** | **0.5531** | **0.3389** |
| XGBoost | 0.7261 | 0.5572 | 0.3301 |
| Random Forest | 0.7289 | 0.5598 | 0.3248 |
| Extra Trees | 0.7318 | 0.5621 | 0.3195 |
| Gradient Boosting | 0.7412 | 0.5703 | 0.3012 |
| Decision Tree | 0.7589 | 0.5864 | 0.2696 |
| Ridge | 0.8024 | 0.6287 | 0.1802 |
| Linear Regression | 0.8031 | 0.6294 | 0.1786 |
| ElasticNet | 0.8099 | 0.6341 | 0.1646 |
| Lasso | 0.8112 | 0.6355 | 0.1619 |
| KNN | 0.8244 | 0.6489 | 0.1347 |

**LightGBM was selected as the final regression model** — it achieved the lowest RMSE and highest R² among all candidates. Learning curves confirmed no significant overfitting. Residuals are approximately normally distributed around zero with no heteroskedasticity, validated by Q-Q plot inspection.

### Per-Class RMSE

| True Rating | RMSE |
|---|---|
| 1 | 1.14 |
| 2 | 1.08 |
| 3 | 0.89 |
| 4 | 0.72 |
| 5 | 0.61 |

Low ratings (1–2) are harder to predict precisely due to their rarity in the dataset (< 5% of all transactions), making them the most actionable targets for service improvement despite the lower model accuracy on that segment.

---

## Classification — Predicting Visit Mode

### Model Comparison

Eleven classification algorithms were trained on the same 14-feature set (with `Rating` added as an input feature) with stratified 80/20 split and `class_weight="balanced"` for all tree-based models.

| Model | Accuracy ↑ | Precision ↑ | Recall ↑ | F1 (Weighted) ↑ |
|---|---|---|---|---|
| **Extra Trees** | **0.5124** | **0.5138** | **0.5124** | **0.5129** |
| Random Forest | 0.5091 | 0.5103 | 0.5091 | 0.5096 |
| LightGBM | 0.5067 | 0.5079 | 0.5067 | 0.5072 |
| XGBoost | 0.5048 | 0.5061 | 0.5048 | 0.5053 |
| Gradient Boosting | 0.4996 | 0.5008 | 0.4996 | 0.5001 |
| Decision Tree | 0.4712 | 0.4726 | 0.4712 | 0.4718 |
| Logistic Regression | 0.3841 | 0.3858 | 0.3841 | 0.3848 |
| KNN | 0.3674 | 0.3689 | 0.3674 | 0.3681 |
| Naive Bayes | 0.3241 | 0.3258 | 0.3241 | 0.3247 |

**Extra Trees was selected as the final classification model** — it achieved the highest weighted F1 score across all five visit mode classes. The classification task is inherently harder than regression: the same traveller demographic profile can correspond to any visit mode depending on trip purpose and context outside the dataset, capping achievable accuracy at roughly 51%. Multi-class ROC curves (One-vs-Rest) and per-class precision-recall curves confirmed that Family and Friends modes have the strongest signal while Business mode — the smallest class — is the hardest to recover.

### 5-Fold Cross-Validation

| Fold | F1 (Weighted) |
|---|---|
| 1 | 0.5113 |
| 2 | 0.5131 |
| 3 | 0.5119 |
| 4 | 0.5108 |
| 5 | 0.5147 |
| **Mean** | **0.5124** |
| **Std** | **0.0014** |

Low variance across folds confirms stable generalisation with no significant data ordering effects.

---

## Recommendation System

Built as **item-based collaborative filtering**: a user–attraction rating matrix (33,530 users × 30 attractions) is transposed so each attraction is represented by its rating pattern across users, and cosine similarity is computed between attractions. Given a user ID, the top-N unvisited attractions most similar to their visited history are returned. A secondary content-based recommender (TF-IDF over attraction name, type, and city) is also built for cold-start users, and a hybrid system (α × collaborative + (1-α) × content) combines both signals.

### Offline Evaluation (Leave-One-Out at K=5)

One visited attraction per eligible user was held out and the recommender was asked to recover it in its Top-5 recommendations, evaluated against a "most popular attractions" baseline:

| Approach | Precision@5 | Recall@5 |
|---|---|---|
| **Item-Based Collaborative Filtering** | **0.0148** | **0.0740** |
| Popularity Baseline (Top-5 global) | 0.0031 | 0.0155 |

- Evaluated on 1,000 randomly sampled eligible users (2+ visit history).
- **Coverage: 100%** — all 30 attractions in the catalogue appeared in at least one Top-5 list, a natural consequence of the small, curated 30-item catalogue.
- The collaborative filter outperforms the popularity baseline by roughly **4.8×** on both metrics, confirming it captures genuine co-visit signal. The hybrid system further improves Recall@5 for cold-start users by falling back to content-based recommendations when visit history is sparse (1–2 attractions).

---

## Repository Structure

```
Tourism-Experience-Analytics/
├── models/
│   ├── best_regression_model.pkl
│   ├── best_classification_model.pkl
│   ├── regression_scaler.pkl
│   ├── classification_scaler.pkl
│   ├── label_encoder_mode.pkl
│   ├── collab_similarity.pkl
│   ├── content_similarity.pkl
│   ├── user_item_matrix.pkl
│   ├── feature_meta.pkl
│   ├── master_df.csv
│   └── df_item.csv
├── .gitignore
├── README.md
├── app.py
├── requirements.txt
└── tourism_experience_analytics.ipynb
```

---

## Running Locally

```bash
git clone https://github.com/Mohit-1307/Tourism-Experience-Analytics.git
cd Tourism-Experience-Analytics

pip install -r requirements.txt
```

Extract the Tourism Dataset zip into a `Tourism Dataset/` folder at the project root, then run the notebook end-to-end to generate all model artifacts:

```bash
jupyter notebook tourism_experience_analytics.ipynb
```

Once the `models/` folder is populated, launch the app:

```bash
streamlit run app.py
```

The app expects all trained artifacts inside `models/`. These are produced by running `tourism_experience_analytics.ipynb` end-to-end.

---

## Tech Stack

- **Data / ML:** pandas, numpy, scikit-learn (Extra Trees, LightGBM, XGBoost, Random Forest, and 7 others), scipy
- **SQL Exploration:** sqlite3 (in-memory database, 6 business queries with CTEs and window functions)
- **Recommendation:** cosine similarity (collaborative filtering), TF-IDF (content-based), hybrid weighted combination
- **Explainability:** SHAP (beeswarm, bar, waterfall plots), permutation importance fallback
- **Visualization:** matplotlib, seaborn, plotly
- **App:** Streamlit (Oracle AI-inspired dark/light theme)
- **Model persistence:** joblib

---

## Author

**Sita Bharatula**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sita280603)
[![GitHub](https://img.shields.io/badge/GitHub-121011?style=flat-square&logo=github&logoColor=white)](https://github.com/maha2806)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:bhsita18@gmail.com)

---

<div align="center">

*If this project was useful, a ⭐ on the repository is appreciated.*

</div>
