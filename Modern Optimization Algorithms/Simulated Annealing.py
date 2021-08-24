import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


class SimulatedAnnealing:
    """
    To run simulated annealing, inherit this class, then
    override 'gen_init_state_energy` and `next_state_energy`.
    """
    def __init__(self,
                 init_T: float,
                 end_T: float,
                 cool_factor: float,
                 steps_per_T: int = 1) -> None:
        self.init_T = init_T
        self.end_T = end_T
        self.cool_factor = cool_factor
        self.steps_per_T = steps_per_T
        self.init_state, self.init_energy = self.gen_init_state_energy()
        self.record_energy = ([], [])  # 0: history; 1: best

    def reset(self) -> None:
        self.init_state, self.init_energy = self.gen_init_state_energy()
        self.record_energy = ([], [])

    def gen_init_state_energy(self) -> Tuple[object, float]:
        raise NotImplementedError

    def next_state_energy(self, cur_state: object, cur_energy: float) -> Tuple[object, float]:
        raise NotImplementedError

    def step(self, cur_T: float, cur_state: object, cur_energy: float) -> Tuple[object, float]:
        next_state, next_energy = self.next_state_energy(cur_state, cur_energy)
        delta_energy = max(0.0, next_energy - cur_energy)
        prob = np.exp(-delta_energy / cur_T)
        if np.random.rand() <= prob:
            return next_state, next_energy
        else:
            return cur_state, cur_energy

    def run(self) -> Tuple[object, float]:
        cur_T = self.init_T
        cur_state = self.init_state
        cur_energy = self.init_energy
        best_state = self.init_state
        best_energy = self.init_energy
        while cur_T > self.end_T:
            for _ in range(self.steps_per_T):
                cur_state, cur_energy = self.step(cur_T, cur_state, cur_energy)
                if best_energy > cur_energy:
                    best_state, best_energy = cur_state, cur_energy
                self.record_energy[0].append(cur_energy)
                self.record_energy[1].append(best_energy)
            cur_T *= self.cool_factor
        return best_state, best_energy

    def plot(self, history: bool = True, best: bool = True) -> None:
        assert history or best
        fig, ax = plt.subplots(1, 1)
        length = len(self.record_energy[0])
        if history:
            ax.plot(range(length), self.record_energy[0], label='history', c='dodgerblue')
        if best:
            ax.plot(range(length), self.record_energy[1], label='best', c='darkorange')
        ax.set_title('Energy curve')
        ax.set_xlabel('Iters')
        ax.set_ylabel('Energy')
        plt.legend()
        plt.show()


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

    class Solver(SimulatedAnnealing):
        def gen_init_state_energy(self) -> Tuple[np.ndarray, float]:
            best_init_state, best_init_energy = np.arange(n), np.inf
            for _ in range(10):
                init_state = np.random.permutation(n)
                init_energy = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    init_energy += distance(init_state[i], init_state[j])
                if init_energy < best_init_energy:
                    best_init_state, best_init_energy = init_state, init_energy
            return best_init_state, best_init_energy

        def next_state_energy(self, cur_state: np.ndarray, cur_energy: float) -> Tuple[np.ndarray, float]:
            u, v = np.random.choice(n+1, 2, replace=False)
            u, v = (v, u) if u > v else (u, v)
            next_state = cur_state.copy()
            next_state[u:v] = next_state[u:v][::-1]  # [u, v)
            next_energy = cur_energy
            if u > 0:
                next_energy -= distance(cur_state[u], cur_state[u-1])
                next_energy += distance(cur_state[v-1], cur_state[u-1])
            if v < n:
                next_energy -= distance(cur_state[v], cur_state[v-1])
                next_energy += distance(cur_state[v], cur_state[u])
            return next_state, next_energy

    solver = Solver(init_T=100,
                    end_T=1e-30,
                    cool_factor=0.999,
                    steps_per_T=1)
    res = solver.run()
    print(res)
    solver.plot()

    def plot_route(res_state):
        fig, ax = plt.subplots(1, 1)
        ax.scatter(data[:, 1], data[:, 2])
        ax.plot(np.hstack((data[res_state, 1], data[res_state[0:1], 1])),
                np.hstack((data[res_state, 2], data[res_state[0:1], 2])))
        plt.show()

    plot_route(res[0])


def Ex():
    def F(x, y):
        return 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2)-10*(x/5-x**3-y**5)*np.exp(-x**2-y**2)-1/3**np.exp(-(x+1)**2-y**2)

    class Solver(SimulatedAnnealing):
        def gen_init_state_energy(self) -> Tuple[np.ndarray, float]:
            return np.array([0, 0]), -F(0, 0)

        def next_state_energy(self, cur_state: np.ndarray, cur_energy: float) -> Tuple[np.ndarray, float]:
            next_state = cur_state + np.random.randn(2) / 10.
            next_state = np.clip(next_state, [-3, -3], [3, 3])
            return next_state, -F(next_state[0], next_state[1])

    solver = Solver(init_T=100,
                    end_T=1e-30,
                    cool_factor=0.996,
                    steps_per_T=1)
    res = solver.run()
    print(res[0], -res[1])
    solver.plot()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    # ax.scatter3D(res[2][:, 0], res[2][:, 1], res[2][:, 2], s=5, c='black')
    # ax.plot_surface(X, Y, F(X, Y), cmap='RdBu_r', alpha=0.8)
    # plt.show()


TSP()
# Ex()
