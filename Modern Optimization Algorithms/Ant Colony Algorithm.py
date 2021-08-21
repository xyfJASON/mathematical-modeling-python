import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm


class AntColonyAlgorithm_TSP:
    def __init__(self,
                 n_cities: int,
                 n_ants: int,
                 T: int,
                 dis: np.ndarray,
                 eta: np.ndarray,
                 alpha: float = 1,
                 beta: float = 5,
                 rho: float = 0.1,
                 Q: float = 1) -> None:
        """
        :param n_cities: number of cities
        :param n_ants: number of ants
        :param T: maximum iterations
        :param dis: distance matrix
        :param eta: heuristic factor
        :param alpha: factor
        :param beta: factor
        :param rho: factor
        :param Q: factor
        """
        assert eta.shape == dis.shape == (n_cities, n_cities)
        self.n_cities = n_cities
        self.n_ants = n_ants
        self.T = T
        self.dis = dis
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.tau = np.ones((n_cities, n_cities))

    def single_ant_tour(self) -> Tuple[float, np.ndarray]:
        distance = 0.0
        init = np.random.randint(0, self.n_cities)
        now = init
        route = [now]
        while len(route) < self.n_cities:
            prob = (self.tau[now] ** self.alpha) * (self.eta[now] ** self.beta)
            prob[np.array(route)] = 0
            prob = prob / prob.sum()
            nxt = np.random.choice(self.n_cities, replace=False, p=prob)
            route.append(nxt)
            distance += self.dis[now, nxt]
            now = nxt
        distance += self.dis[now, init]
        return distance, np.array(route)

    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        dists = np.zeros(self.n_ants)
        routes = np.zeros((self.n_ants, self.n_cities), dtype=int)
        delta_tau = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_ants):
            dists[i], routes[i] = self.single_ant_tour()
            for u in range(len(routes[i])):
                v = (u + 1) % len(routes[i])
                delta_tau[routes[i, u], routes[i, v]] += self.Q / dists[i]
        self.tau = (1 - self.rho) * self.tau + delta_tau
        return np.min(dists), routes[np.argmin(dists)]

    def run(self) -> Tuple[float, np.ndarray]:
        best_dist = np.inf
        best_route = np.zeros(self.n_cities, dtype=int)
        with tqdm(range(self.T)) as pbar:
            for _ in pbar:
                dist, route = self.step()
                if best_dist > dist:
                    best_dist, best_route = dist, route
                pbar.set_postfix({'best dist': best_dist})
        return best_dist, best_route


def TSP():
    """
    For this problem, the best result is 564
    refer to https://www.math.uwaterloo.ca/tsp/vlsi/xqf131.tour.html
    """
    with open('tsp_data.txt') as f:
        data = [list(map(int, line.strip().split())) for line in f.readlines()]
        data = np.array(data)
    n = 131

    def distance(_i, _j):
        """ distance between the ith row and the jth row """
        return np.sqrt((data[_i, 1] - data[_j, 1]) ** 2 + (data[_i, 2] - data[_j, 2]) ** 2)

    dis_mat = np.zeros((n, n))
    for _i in range(n):
        for _j in range(n):
            dis_mat[_i, _j] = distance(_i, _j)
    eta_mat = 1.0 / (dis_mat + np.diag([1e10] * n))

    def evaluate(route: np.ndarray) -> float:
        dist = 0.0
        for i in range(route.shape[0]):
            j = (i + 1) % route.shape[0]
            dist += dis_mat[route[i], route[j]]
        return dist

    solver = AntColonyAlgorithm_TSP(n_cities=n,
                                    n_ants=50,
                                    T=250,
                                    dis=dis_mat,
                                    eta=eta_mat,
                                    alpha=1,
                                    beta=5,
                                    rho=0.1,
                                    Q=1)
    best_dist, best_route = solver.run()
    evaluate_dist = evaluate(best_route)
    assert(best_dist == evaluate_dist), '%f != %f' % (best_dist, evaluate_dist)
    print('best dist:', best_dist)
    print('best route:')
    print(best_route)
    print('Pheromone along the route:')
    phe = []
    for _i in range(len(best_route)):
        _j = (_i + 1) % len(best_route)
        phe.append(solver.tau[best_route[_i], best_route[_j]])
    print(phe)
    print('Maximum pheromone:', np.max(solver.tau))
    print('Minimum pheromone:', np.min(solver.tau))

    def plot_route(route) -> None:
        fig, ax = plt.subplots(1, 1)
        ax.scatter(data[:, 1], data[:, 2])
        ax.plot(np.hstack((data[route, 1], data[route, 1])),
                np.hstack((data[route, 2], data[route, 2])))

    plot_route(best_route)
    plt.show()


TSP()
