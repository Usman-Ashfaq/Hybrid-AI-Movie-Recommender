# 🎬 Hybrid AI Movie Recommendation & Decision System

A hybrid intelligent system that integrates classical AI search algorithms, constraint satisfaction, and machine learning to provide explainable and personalized movie recommendations.

## 🚀 Features
* **Constraint-Based Filtering:** Uses CSP to filter movies based on strict user criteria.
* **Intelligent Search:** Implements BFS, DFS, and A* algorithms to explore the movie space optimally.
* **ML Integration:** Utilizes K-Means clustering for user/movie grouping and ANN for rating predictions.
* **Explainable AI:** Provides clear reasoning for why a specific movie was recommended.
* **Multi-User Support:** Includes an optional Minimax implementation for resolving group decision conflicts.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** NumPy, Pandas
* **ML/AI:** Scikit-learn, TensorFlow/PyTorch
* **UI:** Streamlit
* **Dataset:** IMDb / TMDB / Kaggle CSV

## ⚙️ How It Works
1. **Input:** User provides preferences.
2. **Constraint Satisfaction:** The CSP module filters the dataset.
3. **Search & Heuristics:** A* search finds the best candidates based on heuristic evaluation.
4. **Predictive Analytics:** K-Means clusters preferences while an ANN predicts the rating.
5. **Output:** A ranked list of movies with "Why this recommendation?" explanations.

## 📦 Getting Started
1. Clone the repo: `git clone [your-repo-link]`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`
