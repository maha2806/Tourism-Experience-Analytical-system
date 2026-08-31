"""
Tourism Experience Analytics - Streamlit App
Predicts visit mode, predicts attraction rating, and recommends attractions.

Run with:  streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Tourism Experience Analytics", page_icon="🏝️", layout="wide")

# ---------------- Load artifacts ----------------
@st.cache_resource
def load_artifacts():
    reg_model = joblib.load("models/regression_model.pkl")
    clf_model = joblib.load("models/classification_model.pkl")
    le_target = joblib.load("models/visitmode_label_encoder.pkl")
    encoders = joblib.load("models/feature_encoders.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    item_sim = pd.read_csv("models/item_similarity_matrix.csv", index_col=0)
    content_sim = pd.read_csv("models/content_similarity_matrix.csv", index_col=0)
    attraction_meta = pd.read_csv("models/attraction_meta.csv")
    df = pd.read_csv("data/cleaned_dataset.csv")
    return reg_model, clf_model, le_target, encoders, feature_cols, item_sim, content_sim, attraction_meta, df

reg_model, clf_model, le_target, encoders, feature_cols, item_sim, content_sim, attraction_meta, df = load_artifacts()

attraction_avg = df.groupby("Attraction")["Rating"].mean()
user_item = df.pivot_table(index="UserId", columns="Attraction", values="Rating", aggfunc="mean").fillna(0)

# ---------------- Sidebar: user profile input ----------------
st.sidebar.title("🧭 Your Travel Profile")

continent_opts = sorted(df["UserContinent"].unique())
continent = st.sidebar.selectbox("Continent", continent_opts)

region_opts = sorted(df[df["UserContinent"] == continent]["UserRegion"].unique())
region = st.sidebar.selectbox("Region", region_opts)

country_opts = sorted(df[df["UserRegion"] == region]["UserCountry"].unique())
country = st.sidebar.selectbox("Country", country_opts)

visit_year = st.sidebar.slider("Visit Year", 2013, 2026, 2024)
visit_month = st.sidebar.selectbox("Visit Month", list(range(1, 13)), index=6)
season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
season = season_map[visit_month]

attraction_opts = sorted(df["Attraction"].unique())
attraction = st.sidebar.selectbox("Attraction you're considering", attraction_opts)

known_user_id = st.sidebar.number_input(
    "Returning user? Enter your User ID (optional, leave 0 for a new visitor)",
    min_value=0, value=0, step=1
)

st.sidebar.markdown("---")
st.sidebar.caption("Built on 52,930 real Bali tourism transactions across 30 attractions.")

# ---------------- Header ----------------
st.title("🏝️ Tourism Experience Analytics")
st.markdown("Get a predicted **visit mode**, a predicted **satisfaction rating**, and **personalized attraction recommendations** based on your profile.")

# ---------------- Build feature row ----------------
attraction_row = df[df["Attraction"] == attraction].iloc[0]

def safe_encode(encoder, value):
    """Encode a category, falling back to the most common training class if unseen."""
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return encoder.transform([encoder.classes_[0]])[0]

row = {
    "UserContinent_enc": safe_encode(encoders["UserContinent"], continent),
    "UserRegion_enc": safe_encode(encoders["UserRegion"], region),
    "UserCountry_enc": safe_encode(encoders["UserCountry"], country),
    "AttractionType_enc": safe_encode(encoders["AttractionType"], attraction_row["AttractionType"]),
    "Season_enc": safe_encode(encoders["Season"], season),
    "VisitYear": visit_year,
    "VisitMonth": visit_month,
    "AttractionAvgRating": attraction_row["AttractionAvgRating"],
    "AttractionNumVisits": attraction_row["AttractionNumVisits"],
}

if known_user_id and known_user_id in df["UserId"].values:
    user_row = df[df["UserId"] == known_user_id].iloc[0]
    row["UserAvgRating"] = user_row["UserAvgRating"]
    row["UserNumVisits"] = user_row["UserNumVisits"]
else:
    row["UserAvgRating"] = df["Rating"].mean()
    row["UserNumVisits"] = 1

X_input = pd.DataFrame([row])[feature_cols]

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["🎯 Predict Visit Mode", "⭐ Predict Rating", "✨ Recommendations"])

with tab1:
    st.subheader("Predicted Visit Mode")
    pred_mode_enc = clf_model.predict(X_input)[0]
    pred_mode = le_target.inverse_transform([pred_mode_enc])[0]
    proba = clf_model.predict_proba(X_input)[0]
    proba_df = pd.DataFrame({"Visit Mode": le_target.classes_, "Probability": proba}).sort_values("Probability", ascending=False)

    st.success(f"Most likely visit mode: **{pred_mode}**")
    st.bar_chart(proba_df.set_index("Visit Mode"))
    st.caption("Note: this classifier's overall F1-score is ~0.47 on held-out data — treat this as a directional signal, not a certainty, and use it to guide (not replace) marketing decisions.")

with tab2:
    st.subheader(f"Predicted Rating for '{attraction}'")
    pred_rating = float(np.clip(reg_model.predict(X_input)[0], 1, 5))
    st.metric("Predicted Rating", f"{pred_rating:.2f} / 5")
    st.progress(pred_rating / 5)
    st.caption(f"This attraction's historical average rating is {attraction_avg.get(attraction, np.nan):.2f} / 5.")

with tab3:
    st.subheader("Recommended Attractions For You")

    def recommend_for_user(user_id, top_n=5):
        if user_id not in user_item.index or user_item.loc[user_id].sum() == 0:
            return attraction_meta.merge(attraction_avg.rename("AvgRating"), on="Attraction") \
                                   .sort_values("AvgRating", ascending=False)["Attraction"].head(top_n).tolist()
        user_ratings = user_item.loc[user_id]
        rated = user_ratings[user_ratings > 0]
        scores = pd.Series(0.0, index=user_item.columns)
        for att, rating in rated.items():
            scores += item_sim[att] * rating
        scores = scores.drop(rated.index, errors="ignore")
        return scores.sort_values(ascending=False).head(top_n).index.tolist()

    uid = known_user_id if known_user_id else -1
    recs = recommend_for_user(uid, top_n=6)

    cols = st.columns(3)
    for i, rec in enumerate(recs):
        meta = attraction_meta[attraction_meta["Attraction"] == rec].iloc[0]
        with cols[i % 3]:
            st.markdown(f"**{rec}**")
            st.caption(f"{meta['AttractionType']} · {meta['AttractionCity']}")
            st.caption(f"⭐ {attraction_avg.get(rec, np.nan):.2f} avg rating")

    if uid in user_item.index and user_item.loc[uid].sum() > 0:
        st.info("Recommendations personalized from your visit history (collaborative filtering).")
    else:
        st.info("Showing top-rated attractions (no visit history found) — recommendations personalize once you have rated visits.")

    st.markdown("---")
    st.subheader(f"Similar to '{attraction}'")
    if attraction in content_sim.index:
        similar = content_sim[attraction].drop(attraction).sort_values(ascending=False).head(5)
        st.table(similar.rename("Similarity Score"))
