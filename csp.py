class CSPFilter:
    def __init__(self, movies):
        self.movies = movies

    def apply_constraints(self, genre, min_year, max_duration):
        result = []

        for m in self.movies:
            if genre.lower() not in m['genres'].lower():
                continue

            if int(m['year']) < min_year:
                continue

            if float(m['runtime']) > max_duration:
                continue

            result.append(m)

        return result