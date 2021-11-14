from math import sin, cos, sqrt, fabs
import numpy as np
from scipy import optimize


p_r = 0.01
b_r = 100


def f(x: list):
    return x[0] * x[0] + x[1] * x[1]


def equals_constrains(x: list):
    return [0]


def not_equals_constrains(x: list):
    return [x[0] - 1, 2 - x[0] - x[1]]


def psi_not_equals(f_x: float):
    return pow(min(0, f_x), 2)


def psi_equals(f_x: float):
    return pow(f_x, 2)


def fi(x: float):
    return 1 / x


def help_f(x: list):
    equals, not_equals = 0.0, 0.0
    global p_r

    for constrain in equals_constrains(x):
        equals += psi_equals(constrain)

    for constrain in not_equals_constrains(x):
        not_equals += psi_not_equals(constrain)

    return f(x) + p_r * (not_equals + equals)


def barrier_f(x: list):
    not_equals = 0.0
    global b_r

    for constrain in not_equals_constrains(x):
        not_equals += fi(constrain)

    return f(x) + b_r * not_equals


def barrier(x: list):
    not_equals = 0.0
    global b_r

    for constrain in not_equals_constrains(x):
        not_equals += fi(constrain)

    return b_r * not_equals


def penalty(x: list):
    equals, not_equals = 0.0, 0.0
    global p_r

    for constrain in equals_constrains(x):
        equals += psi_equals(constrain)

    for constrain in not_equals_constrains(x):
        not_equals += psi_not_equals(constrain)

    return p_r * (not_equals + equals)


def fast(x_0: list, eps: float, func, barr: bool):
    k = 0
    x_k = np.array(x_0)

    grad = optimize.approx_fprime(xk=x_k, f=func, epsilon=eps ** 2)

    while sqrt(sum([i * i for i in grad])) > eps:
        alpha = optimize.minimize_scalar(lambda koef: func((x_k - koef * grad).tolist())).x
        if alpha == 0:
            return x_k

        if barr and any(value <= 0 for value in not_equals_constrains(x_k)):
            return x_k
        else:
            x_k -= alpha * grad
            k += 1
            grad = optimize.approx_fprime(xk=x_k, f=func, epsilon=eps ** 2)

    return x_k


def penalty_method(x_0: list, eps: float, z: float):
    k, x = 0, x_0
    global p_r

    x_r_k = fast(x, eps / 10, help_f, False)
    p = penalty(x_r_k)

    while p >= eps:
        p_r, x = z * p_r, x_r_k
        k += 1

        x_r_k = fast(x, eps / 10, help_f, False)
        p = penalty(x_r_k)

    return x_r_k


def barrier_method(x_0: list, eps: float, z: float):
    k, x = 0, x_0
    global b_r

    x_r_k = fast(x, eps / 10, barrier_f, True)
    b = barrier(x_r_k)

    while fabs(b) >= eps:
        b_r, x = z * b_r, x_r_k
        k += 1

        x_r_k = fast(x, eps / 10, barrier_f, True)
        b = barrier(x_r_k)

    return x_r_k


if __name__ == '__main__':
    print(penalty_method([1.0, 1.0], 0.0001, 5))
    print(barrier_method([1.2, 0.0], 0.000001, 1 / 16))
