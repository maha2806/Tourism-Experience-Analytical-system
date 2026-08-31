# Tourism Experience Analytics

Classification, Prediction, and Recommendation System built on the Bali Tourism Dataset
(52,930 transactions, 33,530 users, 30 attractions, 2013–2022).

## Contents

| File / Folder | Description |
|---|---|
| `Tourism_Experience_Analytics.ipynb` | Full EDA + modeling notebook (executed, 20+ charts, regression, classification, recommendation) |
| `streamlit_app.py` | Interactive app: predict visit mode, predict rating, get recommendations |
| `data_prep.py` | Standalone script that cleans & merges the 9 raw tables into `data/cleaned_dataset.csv` |
| `train_models.py` | Standalone script that trains and saves all models to `models/` |
| `data/raw/` | Original Excel source tables |
| `data/cleaned_dataset.csv` | Cleaned, merged, feature-engineered dataset (analysis-ready) |
| `models/` | Saved model artifacts (regression, classification, similarity matrices, encoders) |
| `requirements.txt` | Python dependencies |

## How to run

```bash
pip install -r requirements.txt

# 1. Rebuild the cleaned dataset (already included, but reproducible)
python data_prep.py

# 2. Retrain all models (already included, but reproducible)
python train_models.py

# 3. Launch the app
streamlit run streamlit_app.py
```

Open the notebook in Jupyter / VS Code / Colab to see the full EDA narrative, all 22 charts
with "why / insight / business impact" write-ups, and the model comparison tables.

## Model summary

- **Regression (Rating prediction):** Gradient Boosting Regressor, R² ≈ 0.75, RMSE ≈ 0.49
- **Classification (VisitMode prediction):** Random Forest Classifier, weighted F1 ≈ 0.47
  (visit mode is only weakly determined by the available demographic/attraction features —
  reported honestly rather than overstated)
- **Recommendation:** Item-based collaborative filtering (cosine similarity on the user-item
  rating matrix) with a content-based (attraction type + city) fallback for cold-start users

## Notes on the data

- Only 30 attractions are covered, all located in Bali, Indonesia — despite the raw `Item`
  table's geography columns suggesting a global dataset, the actual `Transaction` table only
  references these 30. An enrichment file (`Updated_Item.xlsx`, 1,698 rows) exists but is not
  referenced by any transaction, so it was left out of the modeling pipeline.
- ~7,600 (user, attraction) pairs repeat across different years — these are legitimate
  return visits, not duplicate rows, and were kept.
