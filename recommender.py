from csp import CSPFilter
from search import AStarSearch
from ml_model import MLModel
from search import bfs, dfs

class Recommender:
    def __init__(self, movies):
        self.movies = movies

    def recommend(self, genre, min_year, max_duration):

        # CSP
        csp = CSPFilter(self.movies)
        filtered = csp.apply_constraints(genre, min_year, max_duration)

        if not filtered:
            return []

        # BFS / DFS
        bfs_list = bfs(filtered)
        dfs_list = dfs(filtered)

        # A*
        search = AStarSearch(filtered)
        searched = search.search(min_year, max_duration)

        # ML
        ml = MLModel(filtered)
        ml.train()

        results = []

        for m in searched[:100]:

            predicted = ml.predict(m)
            cluster = ml.get_cluster(m)

            # Decision function
            score = (
                0.4 * float(m['vote_average']) +
                0.4 * predicted +
                0.2 * (cluster + 1)
            )

            explanation = f"""
✔ Matches genre: {genre}
✔ Year close to preference
✔ Duration within limit
✔ Popular among similar users (Cluster {cluster})
✔ Actual rating: {m['vote_average']}
✔ Predicted rating: {round(predicted,2)}
"""

            results.append((score, m, explanation))

        results.sort(reverse=True, key=lambda x: x[0])

        return results[:5]