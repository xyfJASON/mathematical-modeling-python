import numpy as np
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
        :param alpha: coefficient
        :param beta: coefficient
        :param rho: coefficient
        :param Q: coefficient
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

    def single_ant_tour(self) -> tuple[float, np.ndarray]:
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

    def step(self) -> tuple[np.ndarray, np.ndarray]:
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

    def run(self) -> tuple[float, np.ndarray]:
        best_dist = np.inf
        best_route = np.zeros(self.n_cities, dtype=int)
        with tqdm(range(self.T)) as pbar:
            for _ in pbar:
                dist, route = self.step()
                if best_dist > dist:
                    best_dist, best_route = dist, route
                pbar.set_postfix({'best dist': best_dist})
        return best_dist, best_route


class GeneticAlgorithm:
    """
    To run genetic algorithms, inherit this class, then
    override `gen_init_population_code`, `calc_fitness` and `mutate`.
    Override `crossover` if needed.
    """
    def __init__(self,
                 sz_population: int,
                 n_generations: int,
                 rate_cross: float,
                 rate_mutate: float) -> None:
        """
        :param sz_population: size of population (number of individuals)
        :param n_generations: number of generations to evolve
        :param rate_cross: crossing over rate
        :param rate_mutate: mutation rate
        """
        self.sz_population = sz_population
        self.n_generations = n_generations
        self.rate_cross = rate_cross
        self.rate_mutate = rate_mutate
        self.init_population_code = self.gen_init_population_code()

    def gen_init_population_code(self):
        raise NotImplementedError

    def calc_fitness(self, population_code) -> np.ndarray:
        """ Calculate the fitness of a population.
        Better individuals have higher fitness scores.
        """
        raise NotImplementedError

    def mutate(self, code: np.ndarray) -> np.ndarray:
        """ Define the mutation process of an individual.
        Given an coded individual, return the mutated version of it.
        """
        raise NotImplementedError

    def crossover(self, code1: np.ndarray, code2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Define the crossing over process between two individuals.
        Given two coded individuals, return the crossed version of them.
        Can be overrided if needed.
        """
        assert len(code1.shape) == len(code2.shape) == 1
        assert code1.shape[0] == code2.shape[0]
        pos = np.random.randint(0, code1.shape[0])
        result1 = np.concatenate((code1[:pos], code2[pos:]))
        result2 = np.concatenate((code2[:pos], code1[pos:]))
        return result1, result2

    def crossover_and_mutate(self, population_code: np.ndarray) -> np.ndarray:
        assert len(population_code.shape) == 2
        next_population_code = []
        for code in population_code:
            new_code = code.copy()
            if np.random.rand() <= self.rate_cross:
                code2 = population_code[np.random.randint(0, self.sz_population)]
                new_code, _ = self.crossover(code, code2)
            if np.random.rand() <= self.rate_mutate:
                new_code = self.mutate(new_code)
            next_population_code.append(new_code)
        return np.array(next_population_code)

    def select(self, population_code: np.ndarray, fitness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert len(population_code.shape) == 2 and len(fitness.shape) == 1
        assert population_code.shape[0] == fitness.shape[0]
        # idx = np.argsort(-fitness)[:self.sz_population]
        idx = np.random.choice(population_code.shape[0],
                               size=self.sz_population,
                               replace=True,
                               p=(fitness / fitness.sum()))
        return population_code[idx], fitness[idx]

    def run(self) -> tuple[np.ndarray, np.ndarray]:
        """
        :return: final population and fitness
        """
        cur_population_code = self.init_population_code
        for _ in tqdm(range(self.n_generations)):
            next_population_code = self.crossover_and_mutate(cur_population_code)
            cur_population_code = np.concatenate((cur_population_code, next_population_code))
            fitness = self.calc_fitness(cur_population_code)
            cur_population_code, cur_fitness = self.select(cur_population_code, fitness)
        return cur_population_code, self.calc_fitness(cur_population_code)


class ParticleSwarmOptimization:
    """
    To run PSO, inherit this class, then
    override 'gen_init_particles` and `evaluate_particles`.
    """
    def __init__(self,
                 n_particle: int,
                 T: int,
                 c1: float = 2,
                 c2: float = 2,
                 omega_init: float = 0.9,
                 omega_end: float = 0.4,
                 lb: np.ndarray = None,
                 ub: np.ndarray = None) -> None:
        """
        :param n_particle: number of particles
        :param T: maximum iterations
        :param c1: factor
        :param c2: factor
        :param omega_init: factor
        :param omega_end: factor
        :param lb: lower bound of particles
        :param ub: upper bound of particles
        """
        self.n_particle = n_particle
        self.T = T
        self.c1 = c1
        self.c2 = c2
        self.omega_init = omega_init
        self.omega_end = omega_end
        self.lb, self.ub = lb, ub
        self.v, self.p = self.gen_init_particles()
        assert self.v.shape == self.p.shape and len(self.v.shape) == 2
        assert self.v.shape[0] == n_particle
        self.n_dim = self.v.shape[1]
        assert self.lb.shape == self.ub.shape == self.v.shape
        self.record_gbest = []
        self.record_scores = []

    def reset(self) -> None:
        self.v, self.p = self.gen_init_particles()
        self.n_dim = self.v.shape[1]
        self.record_gbest = []
        self.record_scores = []

    def gen_init_particles(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return two ndarray of the same size [n_particle, K]
        The first ndarray is the velocity of particles
        The second ndarray is the position of particles
        """
        raise NotImplementedError

    def evaluate_particles(self, p: np.ndarray) -> np.ndarray:
        """
        :param p: [N, k], position of particles
        :return: [N], scores of each particle, the higher the better
        """
        raise NotImplementedError

    def run(self) -> np.ndarray:
        pbest = self.p.copy()  # [N, K]
        pbest_scores = self.evaluate_particles(pbest)  # [N]
        gbest = pbest[np.argmax(pbest_scores)]  # [K]
        gbest_scores = np.max(pbest_scores)  # [1]
        for t in tqdm(range(self.T)):
            omega = (self.omega_init - self.omega_end) * (self.T - t) / self.T + self.omega_end
            self.v = (
                    omega * self.v +
                    self.c1 * np.random.rand(self.n_particle, 1) * (pbest - self.p) +
                    self.c2 * np.random.rand(self.n_particle, 1) * (gbest - self.p)
            )
            self.p += self.v
            self.p = np.clip(self.p, self.lb, self.ub)
            scores = self.evaluate_particles(self.p)  # [N]
            update_mask = (pbest_scores < scores)
            pbest[update_mask] = self.p[update_mask]
            pbest_scores[update_mask] = scores[update_mask]
            if gbest_scores < np.max(pbest_scores):
                gbest_scores = np.max(pbest_scores)
                gbest = pbest[np.argmax(pbest_scores)]
            self.record_gbest.append(gbest_scores)
            self.record_scores.append(scores)
        return gbest

    def plot(self, history: bool = True, best: bool = True) -> None:
        fig, ax = plt.subplots(1, 1)
        assert history or best
        ax.set_title('Score curve')
        ax.set_xlabel('Iters')
        ax.set_ylabel('Score')
        if history:
            record = np.vstack(self.record_scores)
            ax.errorbar(x=range(1, self.T+1),
                        y=record.mean(axis=1),
                        yerr=np.max(record, axis=1)-record.mean(axis=1),
                        fmt='.',
                        c='dodgerblue', label='history')
        if best:
            ax.plot(range(1, self.T+1), self.record_gbest, label='best', c='darkorange')
        plt.legend()
        plt.show()


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

    def gen_init_state_energy(self) -> tuple[object, float]:
        raise NotImplementedError

    def next_state_energy(self, cur_state: object, cur_energy: float) -> tuple[object, float]:
        raise NotImplementedError

    def step(self, cur_T: float, cur_state: object, cur_energy: float) -> tuple[object, float]:
        next_state, next_energy = self.next_state_energy(cur_state, cur_energy)
        delta_energy = max(0.0, next_energy - cur_energy)
        prob = np.exp(-delta_energy / cur_T)
        if np.random.rand() <= prob:
            return next_state, next_energy
        else:
            return cur_state, cur_energy

    def run(self) -> tuple[object, float]:
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
