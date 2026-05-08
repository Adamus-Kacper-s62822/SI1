import argparse
import math
import random
from typing import List, Tuple


def read_distance_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    cities = []
    matrix = []

    for line in lines:
        parts = line.split()
        city = parts[0]
        distances = list(map(int, parts[1:]))
        cities.append(city)
        matrix.append(distances)

    n = len(cities)

    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError(
                f"Błąd danych w wierszu {i + 1} ({cities[i]}): "
                f"oczekiwano {n} odległości, otrzymano {len(row)}"
            )

    return cities, matrix


def tour_length(tour: List[int], matrix: List[List[int]]) -> int:
    total = 0
    n = len(tour)
    for i in range(n):
        total += matrix[tour[i]][tour[(i + 1) % n]]
    return total


def nearest_neighbour(matrix: List[List[int]], start: int = 0) -> List[int]:
    n = len(matrix)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start

    while unvisited:
        nxt = min(unvisited, key=lambda x: matrix[current][x])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    return tour


def random_tour(n: int) -> List[int]:
    tour = list(range(n))
    random.shuffle(tour)
    return tour


def ordered_fill(child: List[int], source: List[int], start_pos: int) -> List[int]:
    n = len(source)
    pos = start_pos % n
    for city in source:
        if city not in child:
            while child[pos] != -1:
                pos = (pos + 1) % n
            child[pos] = city
    return child


def aex_crossover(parent1: List[int], parent2: List[int], matrix: List[List[int]]) -> List[int]:
    n = len(parent1)
    child = []
    used = set()

    current = parent1[0]
    child.append(current)
    used.add(current)

    while len(child) < n:
        next1 = None
        next2 = None

        i1 = parent1.index(current)
        cand1 = parent1[(i1 + 1) % n]
        if cand1 not in used:
            next1 = cand1

        i2 = parent2.index(current)
        cand2 = parent2[(i2 + 1) % n]
        if cand2 not in used:
            next2 = cand2

        candidates = [c for c in [next1, next2] if c is not None]

        if len(candidates) == 2:
            current = min(candidates, key=lambda c: matrix[child[-1]][c])
        elif len(candidates) == 1:
            current = candidates[0]
        else:
            remaining = [c for c in parent1 if c not in used]
            current = min(remaining, key=lambda c: matrix[child[-1]][c])

        child.append(current)
        used.add(current)

    return child


def hgrex_crossover(parent1: List[int], parent2: List[int], matrix: List[List[int]]) -> List[int]:
    n = len(parent1)
    child = []
    used = set()

    current = random.choice(parent1)
    child.append(current)
    used.add(current)

    while len(child) < n:
        neighbors = set()

        i1 = parent1.index(current)
        for cand in [parent1[(i1 - 1) % n], parent1[(i1 + 1) % n]]:
            if cand not in used:
                neighbors.add(cand)

        i2 = parent2.index(current)
        for cand in [parent2[(i2 - 1) % n], parent2[(i2 + 1) % n]]:
            if cand not in used:
                neighbors.add(cand)

        if neighbors:
            current = min(neighbors, key=lambda c: matrix[child[-1]][c])
        else:
            remaining = [c for c in range(n) if c not in used]
            current = min(remaining, key=lambda c: matrix[child[-1]][c])

        child.append(current)
        used.add(current)

    return child


def swap_mutation(tour: List[int], mutation_rate: float = 0.1) -> List[int]:
    mutated = tour[:]
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


def tournament_selection(population: List[List[int]], matrix: List[List[int]], k: int = 3) -> List[int]:
    selected = random.sample(population, k)
    return min(selected, key=lambda t: tour_length(t, matrix))


def genetic_algorithm(
    matrix: List[List[int]],
    crossover_type: str,
    population_size: int = 100,
    generations: int = 300,
    mutation_rate: float = 0.1,
) -> List[int]:
    n = len(matrix)
    population = [random_tour(n) for _ in range(population_size)]
    best = min(population, key=lambda t: tour_length(t, matrix))

    for _ in range(generations):
        new_population = []

        elite = min(population, key=lambda t: tour_length(t, matrix))
        new_population.append(elite)

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, matrix)
            parent2 = tournament_selection(population, matrix)

            if crossover_type == "aex":
                child = aex_crossover(parent1, parent2, matrix)
            elif crossover_type == "hgrex":
                child = hgrex_crossover(parent1, parent2, matrix)
            else:
                raise ValueError(f"Nieznany typ crossover: {crossover_type}")

            child = swap_mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population
        generation_best = min(population, key=lambda t: tour_length(t, matrix))
        if tour_length(generation_best, matrix) < tour_length(best, matrix):
            best = generation_best

    return best


def format_tour(tour: List[int], cities: List[str]) -> str:
    names = [cities[i] for i in tour]
    names.append(cities[tour[0]])
    return " -> ".join(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Ścieżka do pliku z macierzą odległości")
    parser.add_argument("--seed", type=int, default=42, help="Seed losowania")
    parser.add_argument("--population", type=int, default=100, help="Wielkość populacji")
    parser.add_argument("--generations", type=int, default=300, help="Liczba pokoleń")
    args = parser.parse_args()

    random.seed(args.seed)

    cities, matrix = read_distance_file(args.file)

    nn_tour = nearest_neighbour(matrix, start=0)
    aex_tour = genetic_algorithm(
        matrix,
        crossover_type="aex",
        population_size=args.population,
        generations=args.generations,
    )
    hgrex_tour = genetic_algorithm(
        matrix,
        crossover_type="hgrex",
        population_size=args.population,
        generations=args.generations,
    )

    results = [
        ("nearestNeighbour", nn_tour),
        ("aex", aex_tour),
        ("hgrex", hgrex_tour),
    ]

    for name, tour in results:
        print(f"\nAlgorytm: {name}")
        print(f"Długość trasy: {tour_length(tour, matrix)}")
        print(f"Trasa: {format_tour(tour, cities)}")


if __name__ == "__main__":
    main()