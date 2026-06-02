import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data
from recommender import Recommender
from ml_model import MLModel 

st.set_page_config(
    page_title="AI Movie Recommender",
    layout="wide"
)


st.markdown("""
<style>
body {
    background-color: #0e0e0e;
    color: white;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #e50914;
    margin-bottom: 20px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #bbb;
    margin-bottom: 40px;
}

.card {
    background-color: #1c1c1c;
    padding: 15px;
    border-radius: 12px;
    margin: 10px;
    transition: 0.3s;
}

.card:hover {
    transform: scale(1.05);
    background-color: #2a2a2a;
}

.score {
    color: #e50914;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)


movies = load_data()


st.markdown("<div class='title'> AI Movie Recommender</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Find movies using Hybrid AI (CSP + A* + ML)</div>", unsafe_allow_html=True)


st.markdown("###  Choose Your Preferences")

col1, col2, col3 = st.columns(3)

with col1:
    genre = st.text_input("Genre", "Action")

with col2:
    year = st.slider("Minimum Year", 1980, 2023, 2010)

with col3:
    duration = st.slider("Max Duration", 60, 240, 150)

recommend_btn = st.button(" Get Recommendations")

st.markdown("---")


if recommend_btn:

    rec = Recommender(movies)
    results = rec.recommend(genre, year, duration)

    if not results:
        st.warning("No movies found. Try different filters.")
    else:
        st.markdown("## Recommended For You")

        cols = st.columns(5)

        for i, (score, m, exp) in enumerate(results):

            with cols[i % 5]:

                st.markdown(f"""
                <div class="card">
                    <h4>{m['title']}</h4>
                    <p class="score"> {round(score,2)}</p>
                    <p> {m['genres'][:30]}</p>
                    <p> {m['year']}</p>
                    <p>⏱ {int(m['runtime'])} min</p>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("Why recommended?"):
                    st.write(exp)


st.markdown("---")
st.subheader("AI Insights")

# Training ML model
ml = MLModel(movies[:1000])  # limit for speed
ml.train()

# Preparing data
X, y = ml.prepare()
clusters = ml.kmeans.predict(X)

df_vis = pd.DataFrame(X, columns=["Year", "Runtime", "Popularity"])
df_vis["Cluster"] = clusters
df_vis["Rating"] = y


st.markdown("### Movie Clusters (K-means)")

fig1, ax1 = plt.subplots()
sns.scatterplot(
    data=df_vis,
    x="Year",
    y="Rating",
    hue="Cluster",
    palette="viridis",
    ax=ax1
)

st.pyplot(fig1)


st.markdown("### Actual vs Predicted Ratings")

predictions = []

for i in range(min(100, len(X))):  
    pred = ml.ann.predict([X[i]])[0]
    predictions.append(pred)

df_vis_small = df_vis.iloc[:len(predictions)].copy()
df_vis_small["Predicted"] = predictions

fig2, ax2 = plt.subplots()
ax2.scatter(df_vis_small["Rating"], df_vis_small["Predicted"])
ax2.set_xlabel("Actual Rating")
ax2.set_ylabel("Predicted Rating")
ax2.set_title("Prediction Accuracy")

st.pyplot(fig2)

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888;'>Built using Hybrid AI | CSP + Search + Machine Learning</p>",
    unsafe_allow_html=True
)