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

    def parameterize(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        matrix = np.zeros((len(r), 1 + self.Omega.shape[0]))
        matrix[:, 0] = self.kernel_sign * (r**2) / len(r)
        matrix[:, 1:] = theta
        return matrix

    def initial_distribution(self) -> tuple:
        r = np.ones(self.m ** self.Omega.shape[0])
        r = np.hstack((r, -r))
        grids_1d = [np.logspace(-3, 0, self.m + 2)[1:-1]]
        grids_1d += [
            np.linspace(bound[0], bound[1], self.m + 2, endpoint=True)[1:-1]
            for bound in self.Omega[1:]
        ]
        theta = np.array(np.meshgrid(*(grids_1d))).reshape(len(self.Omega), -1).T
        theta = np.vstack((theta, theta * 1.000000001))  # Positive and negative support
        cs = np.array([-1, 1])
        self.kernel_sign = np.sign(r)
        return r, theta, cs

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
            M=float(self.j(u, c) / self.alpha),
            g=self.g,
            f=self.f,
            grad_f=self.grad_f,
            hess_f=self.hess_f,
            invariable_kernel=invariable_kernel,
            mode="unconstrained",
            maximum_iterations=self.ssn_steps,
        )
        ssn_solution = ssn.solve(tol=self.machine_precision, u_0=u_0)
        c_plus = ssn_solution[0]
        cs = [c, c_plus]
        values = [self.j(u, local_c) for local_c in cs]
        best_value = np.argmin(values)
        c_best = cs[best_value]
        return c_best

    def retraction(
        self,
        r: np.ndarray,
        del_r: np.ndarray,
        c: np.ndarray,
        del_c: np.ndarray,
        theta: np.ndarray,
        del_theta: np.ndarray,
        mode: str = "canonical",
    ) -> tuple:
        if mode == "canonical":
            r_retraction = r + del_r
            c_retraction = c + del_c
            theta_retraction = theta + del_theta
        elif mode == "mirror":
            r_retraction = r * np.exp(del_r)  # /r
            c_retraction = c * np.exp(del_c)
            theta_retraction = theta + del_theta
        return r_retraction, c_retraction, theta_retraction

    def solve(
        self,
        max_iters: int = 1000,
        u_0: Measure = Measure(),
        c_0: float = 0,
    ):
        if len(u_0.coefficients):
            self.kernel_sign = np.sign(u_0.coefficients)
            r = self.kernel_sign * np.sqrt(
                self.kernel_sign * u_0.coefficients * len(u_0.coefficients)
            )
            theta = u_0.support
            cs = np.array([c_0])
        else:
            r, theta, cs = self.initial_distribution()
        # params = self.parameterize(r, theta).reshape(-1, 1 + self.Omega.shape[0])
        u = Measure(matrix=self.parameterize(r, theta))
        c = (np.sign(cs[0]) * cs[0] ** 2 + np.sign(cs[1]) * cs[1] ** 2) / len(r)
        # c = self.finite_dimensional_step(u, c)
        logging.info(f"0: objective {self.j(u, c):.14E}")
        objective_values = [self.j(u, c)]
        # logging.info(params)
        # param_update_threshold = 1e-4
        for iter in range(max_iters):
            p_u = self.p(u, c)
            grad_p_u = self.grad_p(u, c)
            inner = c * np.ones(self.constant_dim)
            if len(u.coefficients):
                inner += self.kernel(u.support).T @ u.coefficients
            inner = self.grad_f(inner)
            r_update = (
                -2
                * self.a_parameter
                # * r
                * (-self.kernel_sign * p_u(theta) + self.alpha)
            )
            cs_update = np.array(
                [
                    (self.a_parameter) * inner @ np.ones(self.constant_dim),
                    -(self.a_parameter) * inner @ np.ones(self.constant_dim),
                ]
            )
            # logging.info(cs_update)
            theta_update = (
                self.b_parameter * self.kernel_sign * np.array(grad_p_u(theta)).T
            ).T
            r, cs, theta = self.retraction(
                r, r_update, cs, cs_update, theta, theta_update, mode="mirror"
            )
            keep_indices = np.logical_and(theta[:, 0] > 1e-6, np.abs(r) > 1e-4)
            r = r[keep_indices]
            self.kernel_sign = self.kernel_sign[keep_indices]
            theta = theta[keep_indices]
            # params = self.parameterize(r, theta).reshape(-1, 1 + self.Omega.shape[0])
            u = Measure(matrix=self.parameterize(r, theta))
            c = (np.sign(cs[0]) * cs[0] ** 2 + np.sign(cs[1]) * cs[1] ** 2) / len(r)
            # c = self.finite_dimensional_step(u, c)
            obj = self.j(u, c)
            # if obj > objective_values[-1]:
            #     self.a_parameter *= 0.5
            #     self.b_parameter *= 0.25
            #     logging.info(f"alpha: {self.a_parameter}, beta: {self.b_parameter}")
            # elif objective_values[-1] - obj < param_update_threshold:
            #     self.a_parameter *= 5
            #     self.b_parameter *= 10
            #     param_update_threshold *= 0.01
            #     logging.info(f"alpha: {self.a_parameter}, beta: {self.b_parameter}")
            objective_values.append(obj)
            if (iter + 1) % 100 == 0:
                # logging.info(
                #     f"r_delta: {np.linalg.norm(r_update):.3E}, theta_delta: {np.linalg.norm(theta_update):.3E}, c_update: {np.linalg.norm(cs_update):.3E}"
                # )
                logging.info(
                    f"{iter + 1}: supp: {len(u.coefficients)}, c value: {c:.3E}, objective {obj:.14E}"
                )
                # self.b_parameter = min(self.a_parameter, self.b_parameter * 1.002)
                # self.a_parameter *= 1.001
                # logging.info(f"a: {self.a_parameter}, b:{self.b_parameter}")
                # logging.info(f"min: {np.min(np.abs(r))}, max: {np.max(np.abs(r))}")
                # logging.info(params)
        return u, c, objective_values
