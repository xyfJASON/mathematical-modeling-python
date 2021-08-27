import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class SI:
    """
    Only consider susceptibles and infectives.
    Infectives can't recover.
    """
    def __init__(self,
                 N: int,
                 r: int,
                 beta: float,
                 I0: int) -> None:
        """
        :param N: total population, fixed
        :param r: number of contacts per person per time
        :param beta: probability of disease transmission in a contact
        :param I0: initial infectives population
        """
        self.N = N
        self.r = r
        self.beta = beta
        self.I0 = I0

    def predict(self, t: np.ndarray) -> np.ndarray:
        return self.N * self.I0 / (self.I0 + (self.N - self.I0) * np.exp(-self.r*self.beta*t))

    def show(self, t_begin: float, t_end: float) -> None:
        fig, ax = plt.subplots(1, 1)
        ax.set_title('SI Model\n' +
                     r'$r=%d,\,\beta=%.6f$' % (self.r, self.beta))
        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction')
        ax.set_xlim(t_begin, t_end)
        ax.set_ylim(0, 1)

        plot_x = np.linspace(t_begin, t_end, 100)
        plot_I = self.predict(plot_x)
        plot_S = self.N - plot_I
        ax.plot(plot_x, plot_I / self.N, label='Infectives')
        ax.plot(plot_x, plot_S / self.N, label='Susceptibles')
        plt.legend()
        plt.show()


class SIS:
    """
    Only consider susceptibles and infectives.
    Infectives can recover and may be infected again.
    """
    def __init__(self,
                 N: int,
                 r: int,
                 beta: float,
                 gamma: float,
                 I0: int) -> None:
        """
        :param N: total population, fixed
        :param r: number of contacts per person per time
        :param beta: probability of disease transmission in a contact
        :param gamma: probability of recovery
        :param I0: initial infectives population
        """
        self.N = N
        self.r = r
        self.beta = beta
        self.gamma = gamma
        self.I0 = I0

    def predict(self, t: np.ndarray) -> np.ndarray:
        rbg = self.r * self.beta - self.gamma
        Nrbg = self.N * rbg / (self.r * self.beta)
        return Nrbg / (1 + (Nrbg / self.I0 - 1) * np.exp(-rbg * t))

    def show(self, t_begin: float, t_end: float) -> None:
        fig, ax = plt.subplots(1, 1)
        ax.set_title('SIS Model\n' +
                     r'$r=%d,\,\beta=%.6f,\,\gamma=%.6f$' % (self.r, self.beta, self.gamma))
        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction')
        ax.set_xlim(t_begin, t_end)
        ax.set_ylim(0, 1)

        plot_x = np.linspace(t_begin, t_end, 100)
        plot_I = self.predict(plot_x)
        plot_S = self.N - plot_I
        ax.plot(plot_x, plot_I / self.N, label='Infectives')
        ax.plot(plot_x, plot_S / self.N, label='Susceptibles')
        plt.legend()
        plt.show()


class SIR:
    """
    Consider susceptibles, infectives and removed.
    Infectives can recover and won't be infected again.
    """
    def __init__(self,
                 N: int,
                 r: int,
                 beta: float,
                 gamma: float,
                 I0: int,
                 R0: int) -> None:
        """
        :param N: total population, fixed
        :param r: number of contacts per person per time
        :param beta: probability of disease transmission in a contact
        :param gamma: probability of recovery
        :param I0: initial infectives population
        :param R0: initial removed population
        """
        self.N = N
        self.r = r
        self.beta = beta
        self.gamma = gamma
        self.I0 = I0
        self.R0 = R0

    def predict(self, t: np.ndarray) -> np.ndarray:
        def fun(_, y):
            """ y = [S, I, R] """
            return np.array([-self.r * self.beta * y[1] * y[0] / self.N,
                             self.r * self.beta * y[1] * y[0] / self.N - self.gamma * y[1],
                             self.gamma * y[1]])

        res = solve_ivp(fun=fun,
                        t_span=(0, np.max(t)),
                        y0=np.array([self.N - self.I0 - self.R0, self.I0, self.R0]),
                        method='RK45',
                        t_eval=t)
        return res.y

    def show(self, t_begin: float, t_end: float) -> None:
        fig, ax = plt.subplots(1, 1)
        ax.set_title('SIR Model\n' +
                     r'$r=%d,\,\beta=%.6f,\,\gamma=%.6f$' % (self.r, self.beta, self.gamma))
        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction')
        ax.set_xlim(t_begin, t_end)
        ax.set_ylim(0, 1)

        plot_x = np.linspace(t_begin, t_end, 100)
        plot_S = self.predict(plot_x)
        plot_S, plot_I, plot_R = plot_S[0], plot_S[1], plot_S[2]
        ax.plot(plot_x, plot_I / self.N, label='Infectives')
        ax.plot(plot_x, plot_S / self.N, label='Susceptibles')
        ax.plot(plot_x, plot_R / self.N, label='Removed')
        plt.legend()
        plt.show()


class SEIR:
    """
    Consider susceptibles, exposed, infectives and removed.
    Infectives can recover and won't be infected again.
    Exposed cannot infect others.
    """
    def __init__(self,
                 N: int,
                 r: int,
                 beta: float,
                 sigma: float,
                 gamma: float,
                 E0: int,
                 I0: int,
                 R0: int) -> None:
        """
        :param N: total population, fixed
        :param r: number of contacts per person per time
        :param beta: probability of disease transmission in a contact
        :param sigma: probability of exposed -> infectives
        :param gamma: probability of recovery
        :param E0: initial exposed population
        :param I0: initial infectives population
        :param R0: initial removed population
        """
        self.N = N
        self.r = r
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.E0 = E0
        self.I0 = I0
        self.R0 = R0
        self.S0 = self.N - E0 - I0 - R0

    def predict(self, t: np.ndarray) -> np.ndarray:
        def fun(_, y):
            """ y = [S, E, I, R] """
            return np.array([-self.r * self.beta * y[2] * y[0] / self.N,
                             self.r * self.beta * y[2] * y[0] / self.N - self.sigma * y[1],
                             self.sigma * y[1] - self.gamma * y[2],
                             self.gamma * y[2]])

        res = solve_ivp(fun=fun,
                        t_span=(0, np.max(t)),
                        y0=np.array([self.S0, self.E0, self.I0, self.R0]),
                        method='RK45',
                        t_eval=t)
        return res.y

    def show(self, t_begin: float, t_end: float) -> None:
        fig, ax = plt.subplots(1, 1)
        ax.set_title('SEIR Model\n' +
                     r'$r=%d,\,\beta=%.6f,\,\sigma=%.6f,\,\gamma=%.6f$' % (self.r, self.beta, self.sigma, self.gamma))
        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction')
        ax.set_xlim(t_begin, t_end)
        ax.set_ylim(0, 1)

        plot_x = np.linspace(t_begin, t_end, 100)
        plot_S = self.predict(plot_x)
        plot_S, plot_E, plot_I, plot_R = plot_S[0], plot_S[1], plot_S[2], plot_S[3]
        ax.plot(plot_x, plot_I / self.N, label='Infectives')
        ax.plot(plot_x, plot_E / self.N, label='Exposed')
        ax.plot(plot_x, plot_S / self.N, label='Susceptibles')
        ax.plot(plot_x, plot_R / self.N, label='Removed')
        plt.legend()
        plt.show()


def main():
    # model = SI(N=1000, r=120, beta=0.005, I0=1)
    # model = SIS(N=1000, r=100, beta=0.005, gamma=0.1, I0=1)
    # model = SIR(N=1000, r=100, beta=0.003, gamma=0.1, I0=1, R0=0)
    # model.show(t_begin=0, t_end=100)
    # model = SIR(N=1000, r=100, beta=0.003, gamma=0.1, I0=20, R0=400)
    # model.show(t_begin=0, t_end=100)
    # model = SEIR(N=1000, r=100, beta=0.003, sigma=0.3, gamma=0.1, E0=40, I0=20, R0=0)
    # model.show(t_begin=0, t_end=100)
    model = SEIR(N=10000, r=20, beta=0.03, sigma=0.1, gamma=0.1, E0=0, I0=1, R0=0)
    model.show(t_begin=0, t_end=140)


if __name__ == '__main__':
    main()
