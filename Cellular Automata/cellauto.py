import numpy as np
import matplotlib.pyplot as plt


class CellularAutomata2D:
    """
    To use cellular automata in 2D space, inherit this class,
    then override one of `next_state_global` and `next_state_single_cell`.
    Override `gen_init_state` if random binary initialization is not desired.
    """
    def __init__(self, sz_grid: tuple[int, int]) -> None:
        assert len(sz_grid) == 2
        self.sz_grid = sz_grid

    def gen_init_state(self) -> np.ndarray:
        """ Generate initial state
        Generate binary code randomly by default, override if needed.
        """
        return np.random.randint(0, 2, size=self.sz_grid)

    def next_state_global(self, cur_state: np.ndarray) -> np.ndarray:
        """ Return next state of all cells """
        raise NotImplementedError

    def next_state_single_cell(self, cur_state: np.ndarray, pos: tuple[int, int] or int) -> int:
        """ Return next state of the single cell at `pos` """
        raise NotImplementedError

    def plot(self, state: np.ndarray, it: int, pause_time: float = 0.1) -> None:
        ploty, plotx = np.nonzero(state)
        ploty = self.sz_grid[0] - 1 - ploty
        plt.plot(plotx, ploty, 'sk')
        plt.xticks([]); plt.yticks([])
        plt.xlim(-0.5, self.sz_grid[1]-0.5)
        plt.ylim(-0.5, self.sz_grid[0]-0.5)
        plt.xlabel('time step: %d' % it)
        plt.pause(pause_time)
        plt.cla()

    def run(self,  max_iters: int,
            show: bool = False, pause_time: float = 0.1) -> list[np.ndarray]:
        state = self.gen_init_state()
        assert state.shape == self.sz_grid
        states = [state]
        if show:
            self.plot(state, 0, pause_time)
        for it in range(max_iters):
            try:
                state = self.next_state_global(state)
            except NotImplementedError:
                next_state = state.copy()
                for i in range(self.sz_grid[0]):
                    for j in range(self.sz_grid[1]):
                        next_state[i, j] = self.next_state_single_cell(state, (i, j))
                state = next_state
            states.append(state)
            if show:
                self.plot(state, it+1, pause_time)
        return states


def main():
    class LifeGame(CellularAutomata2D):
        def gen_init_state(self) -> np.ndarray:
            init = np.zeros(self.sz_grid)
            init[[1, 1, 2, 2, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
                  7, 7, 7, 7, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
                  11, 11, 11, 11, 11, 13, 13, 13, 13, 14, 14, 15, 15],
                 [5, 11, 5, 11, 5, 6, 10, 11, 1, 2, 3, 6, 7, 9, 10, 13, 14, 15, 3, 5, 7, 9, 11, 13, 5, 6, 10, 11,
                  5, 6, 10, 11, 3, 5, 7, 9, 11, 13, 1, 2, 3, 6, 7, 9, 10, 13, 14, 15, 5, 6, 10, 11, 5, 11, 5, 11]] = 1
            return init

        def next_state_single_cell(self, cur_state: np.ndarray, pos: tuple[int, int]) -> int:
            cnt_alive = 0
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == 0 and j == 0:
                        continue
                    cnt_alive += cur_state[(pos[0] + i) % self.sz_grid[0],
                                           (pos[1] + j) % self.sz_grid[1]]
            return 1 if cnt_alive == 3 else (cur_state[pos[0], pos[1]] if cnt_alive == 2 else 0)

    game = LifeGame((20, 20))
    game.run(max_iters=20, show=True, pause_time=0.1)


if __name__ == '__main__':
    main()
