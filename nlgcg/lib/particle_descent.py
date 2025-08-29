# The Conic Particle Gradient Descent algorithm, as presented in https://arxiv.org/pdf/1907.10300

import numpy as np
from typing import Callable
import logging
from lib.measure import Measure
from lib.ssn import SSN

logging.basicConfig(
    level=logging.DEBUG,
)


class ParticleDescent:
    def __init__(
        self,
        m: int,
        j_N: Callable,  # parameterized objective
        j: Callable,
        p: Callable,  # Dual variable
        grad_p: Callable,
        Omega: np.ndarray,
        a_parameter: float,
        b_parameter: float,
        kernel: Callable,
        constant_dim: int,
        kernel_dim: int,
        alpha: float,
        target: np.ndarray,
        g: Callable,
        f: Callable,
        grad_f: Callable,
        hess_f: Callable,
        ssn_steps: int = 100,
    ):
        self.m = m
        self.j = j
        self.j_N = j_N
        self.p = p
        self.grad_p = grad_p
        self.Omega = Omega
        self.a_parameter = a_parameter
        self.b_parameter = b_parameter
        self.machine_precision = 5e-14
        self.kernel = kernel
        self.constant_dim = constant_dim
        self.kernel_dim = kernel_dim
        self.ssn_steps = ssn_steps
        self.alpha = alpha
        self.target = target
        self.g = g
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.kernel_sign = np.hstack(
            (
                np.ones(self.m ** self.Omega.shape[0]),
                -np.ones(self.m ** self.Omega.shape[0]),
            )
        )

    def parameterize(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        matrix = np.zeros((len(r), 1 + self.Omega.shape[0]))
        matrix[:, 0] = self.kernel_sign * (r**2) / len(r)
        matrix[:, 1:] = theta
        return matrix

    def initial_distribution(self) -> tuple:
        r = np.ones(self.m ** self.Omega.shape[0])
        r = np.hstack((r, -r))
        theta = (
            np.array(
                np.meshgrid(
                    *(
                        np.linspace(bound[0], bound[1], self.m + 2, endpoint=True)[1:-1]
                        for bound in self.Omega
                    )
                )
            )
            .reshape(len(self.Omega), -1)
            .T
        )
        theta = np.vstack((theta, theta))  # Positive and negative support
        c = 0
        return r, theta, c

    def finite_dimensional_step(
        self,
        u: Measure,
        c: float,
    ) -> float:
        K_support = np.hstack(
            (np.ones(self.constant_dim), np.zeros(self.kernel_dim - self.constant_dim))
        ).reshape(-1, 1)
        coefs = np.array([c])
        invariable_kernel = np.zeros((self.kernel_dim))
        if len(u.coefficients):
            measure_K = self.kernel(u.support).T
            invariable_kernel = measure_K @ u.coefficients
        u_0 = coefs.copy()
        ssn = SSN(
            K=K_support,
            alpha=self.alpha,
            target=self.target,
            M=self.M,
            g=self.g,
            f=self.f,
            grad_f=self.grad_f,
            hess_f=self.hess_f,
            invariable_kernel=invariable_kernel,
            mode="unconstrained",
            maximum_iterations=self.ssn_steps,
        )
        ssn_solution = ssn.solve(tol=self.machine_precision, u_0=u_0)
        ssn_normal = ssn_solution.copy()
        c_plus_normal = ssn_normal[0]
        u_plus_normal = u.copy()
        tuples = [
            (u_plus_normal, c_plus_normal),
            (u, c),
        ]
        values = [self.j(u, c) for u, c in tuples]
        best_value = np.argmin(values)
        u_plus, c_plus = tuples[best_value]
        return c_plus

    def retraction(
        self, r: float, del_r: float, theta: np.ndarray, del_theta: np.ndarray
    ) -> tuple:
        r_retraction = r + del_r
        theta_retraction = theta + del_theta
        return r_retraction, theta_retraction

    def solve(
        self,
        max_iters: int = 1000,
    ):
        r, theta, c = self.initial_distribution()
        u = Measure(matrix=self.parameterize(r, theta))
        self.M = float(self.j(u, c) / self.alpha)
        c = self.finite_dimensional_step(u, c)
        self.M = float(self.j(u, c) / self.alpha)
        logging.info(f"0: objective {self.j(u, c):.14E}")
        logging.info(r)
        logging.info(theta)
        for iter in range(max_iters):
            p_u = self.p(u, c)
            grad_p_u = self.grad_p(u, c)
            r_update = (
                -2
                * self.a_parameter
                * r
                * (-self.kernel_sign * p_u(theta) + self.alpha)
            )
            theta_update = (
                -self.b_parameter * np.multiply(self.kernel_sign, grad_p_u(theta).T).T
            )
            r, theta = self.retraction(r, r_update, theta, theta_update)
            theta[:, 0] = np.maximum(theta[:, 0], 1e-5)
            u = Measure(matrix=self.parameterize(r, theta))
            c = self.finite_dimensional_step(u, c)
            self.M = float(self.j(u, c) / self.alpha)
            logging.info(f"{iter + 1}: objective {self.j(u, c):.14E}")
            # logging.info(r)
            # logging.info(theta)
        return u, c
