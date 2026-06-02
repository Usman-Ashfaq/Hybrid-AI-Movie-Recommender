import heapq

def bfs(movies):
    queue = []
    result = []

    for m in movies:
        queue.append(m)

    while queue:
        node = queue.pop(0)
        result.append(node)

    return result


def dfs(movies):
    stack = []
    result = []

    for m in movies:
        stack.append(m)

    while stack:
        node = stack.pop()
        result.append(node)

    return result



class AStarSearch:
    def __init__(self, movies):
        self.movies = movies

    def heuristic(self, m, user_year, max_duration):
        rating_score = 10 - float(m['vote_average'])
        year_penalty = abs(int(m['year']) - user_year) / 10
        duration_penalty = max(0, float(m['runtime']) - max_duration) / 50
        popularity_bonus = float(m['popularity']) / 100

        return rating_score + year_penalty + duration_penalty - popularity_bonus

    def search(self, user_year, max_duration):
        heap = []

        for m in self.movies:
            g = 0
            h = self.heuristic(m, user_year, max_duration)
            f = g + h

            heapq.heappush(heap, (f, m))

        result = []
        while heap:
            result.append(heapq.heappop(heap)[1])

        return result