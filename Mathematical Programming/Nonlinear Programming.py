import numpy as np
from scipy.optimize import minimize


def Ex1():
    def f(x):
        return 100*(x[1]-x[0]*x[0])**2+(1-x[0])**2

    def grad(x):
        g = np.zeros(2)
        g[0] = -400*(x[1]-x[0]*x[0])*x[0]-2*(1-x[0])
        g[1] = 200*(x[1]-x[0]*x[0])
        return g

    def hessian(x):
        h = np.zeros((2, 2))
        h[0, 0] = -400*(x[1]-x[0]*x[0])+800*x[0]*x[0]+2
        h[0, 1] = -400 * x[0]
        h[1, 0] = -400 * x[0]
        h[1, 1] = 200
        return h

    res = minimize(fun=f,
                   x0=np.zeros(2),
                   method='trust-exact',
                   jac=grad,
                   hess=hessian)
    print(res)


def Ex2():
    def f(x):
        return (x-3)**2-1

    def grad(x):
        return 2*(x-3)

    res = minimize(fun=f,
                   x0=np.array([0]),
                   method='L-BFGS-B',
                   jac=grad,
                   bounds=[(0, 5)])
    print(res)


def Ex3():
    def f(x):
        return np.e ** x[0] * (4 * x[0] * x[0] + 2 * x[1] * x[1] + 4 * x[0] * x[1] + 2 * x[1] + 1)

    def grad(x):
        g = np.zeros(2)
        g[0] = np.e ** x[0] * (4 * x[0] * x[0] + 2 * x[1] * x[1] + 4 * x[0] * x[1] + 8 * x[0] + 6 * x[1] + 1)
        g[1] = np.e ** x[0] * (4 * x[0] + 4 * x[1] + 2)
        return g

    def get_constr():
        def constr_f1(x):
            return x[0] + x[1] - x[0] * x[1] - 1.5

        def constr_grad1(x):
            return np.array([1 - x[1], 1 - x[0]])

        def constr_f2(x):
            return x[0] * x[1] + 10

        def constr_grad2(x):
            return np.array([x[1], x[0]])

        c = [
            dict(type='ineq',
                 fun=constr_f1,
                 jac=constr_grad1),
            dict(type='ineq',
                 fun=constr_f2,
                 jac=constr_grad2)
            ]
        return c

    constr = get_constr()
    res = minimize(fun=f,
                   x0=np.array([-2, 2]),
                   method='SLSQP',
                   jac=grad,
                   constraints=constr)
    print(res)


Ex1()
