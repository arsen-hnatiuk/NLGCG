# The Conic Particle Gradient Descent algorithm, as presented in https://arxiv.org/pdf/1907.10300

import numpy as np
from typing import Callable
import logging
import os

import jax
import time
from ..lib.measure import Measure
from ..lib.ssn import SSN

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
        do_linesearch: bool = False
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
        self.do_linesearch = do_linesearch

    def parameterize(self, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        matrix = np.zeros((len(r), 1 + self.Omega.shape[0]))
        matrix[:, 0] = self.kernel_sign * (r**2) / len(r)
        matrix[:, 1:] = theta
        return matrix

    def sample_domain(self, size: int, mode: str = "exponential") -> np.ndarray:
        # Generate a uniform sample of shape (size,domain.shape[0]) in the given domain
        columns = []
        for i, bounds in enumerate(self.Omega):
            if i == 0:
                if mode == "exponential":
                    columns.append(
                        np.random.exponential(scale=bounds[1] / 10, size=(size, 1))
                        + bounds[0]
                    )
                elif mode == "uniform":
                    columns.append(
                        np.random.sample((size, 1)) * (bounds[1] - bounds[0])
                        + bounds[0]
                    )
            else:
                columns.append(
                    (np.random.sample((size, 1)) * (bounds[1] - bounds[0]) + bounds[0])
                    * 1.2
                )
        sample = np.concatenate(columns, axis=1)
        return sample

    def initial_distribution(self, mode: str = "exponential") -> tuple:
        r = np.ones(self.m)
        r = np.hstack((r, -r))
        grid_pos = self.sample_domain(self.m, mode=mode)
        grid_neg = self.sample_domain(self.m, mode=mode)
        # grids_1d = [np.logspace(-3, 0, self.m + 2)[1:-1]]
        # grids_1d += [
        #     np.linspace(bound[0], bound[1], self.m + 2, endpoint=True)[1:-1]
        #     for bound in self.Omega[1:]
        # ]
        # theta = np.array(np.meshgrid(*(grids_1d))).reshape(len(self.Omega), -1).T
        theta = np.vstack((grid_pos, grid_neg))  # Positive and negative support
        cs = np.array([-1, 1])
        self.kernel_sign = np.sign(r)
        return r, theta, cs

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
        max_time: int = 1e6,
        u_0: Measure = Measure(),
        c_0: float = 0,
        mode: str = "exponential",
    ):
        t_0 = time.time()
        if len(u_0.coefficients):
            self.kernel_sign = np.sign(u_0.coefficients)
            r = self.kernel_sign * np.sqrt(
                self.kernel_sign * u_0.coefficients * len(u_0.coefficients)
            )
            theta = u_0.support
            cs = np.array([c_0])
        else:
            r, theta, cs = self.initial_distribution(mode=mode)
        u = Measure(matrix=self.parameterize(r, theta))
        c = (np.sign(cs[0]) * cs[0] ** 2 + np.sign(cs[1]) * cs[1] ** 2) / len(r)
        logging.info(f"0: objective {self.j(u, c):.14E}")
        objective_values = [self.j(u, c)]
        times = [time.time() - t_0]
        supports = [len(u.coefficients)]

        p_def = 0
        innr = 0
        update = 0
        retr = 0
        post = 0
        ut = 0

        for it in range(max_iters):
            t = time.time()
            p_u = self.p(u, c)
            grad_p_u = self.grad_p(u, c)
            p_def += time.time() - t

            t = time.time()
            inner = c * np.ones(self.constant_dim)
            if len(u.coefficients):
                inner += self.kernel(u.support).T @ u.coefficients
            inner = self.grad_f(inner)
            innr += time.time() - t

            t = time.time()
            r_update = (
                -2 * self.a_parameter * (-self.kernel_sign * p_u(theta) + self.alpha)
            )
            inner_sum = np.sum(inner)
            cs_update = np.array(
                [
                    (self.a_parameter) * inner_sum,
                    -(self.a_parameter) * inner_sum,
                ]
            )
            theta_update = (
                self.b_parameter * self.kernel_sign * np.array(grad_p_u(theta)).T
            ).T
            update += time.time() - t

            r_old, cs_old, theta_old = r, cs, theta
            decrease = 1.
            while decrease > 0:
                t = time.time()
                r, cs, theta = self.retraction(
                    r_old, r_update, cs_old, cs_update, theta_old, theta_update, mode="mirror"
                )
                retr += time.time() - t

                t = time.time()
                keep_indices = np.logical_and(theta[:, 0] > 1e-6, np.abs(r) > 1e-7)
                r = r[keep_indices]
                self.kernel_sign = self.kernel_sign[keep_indices]
                theta = theta[keep_indices]
                c = (np.sign(cs[0]) * cs[0] ** 2 + np.sign(cs[1]) * cs[1] ** 2) / len(r)
                post += time.time() - t

                t = time.time()
                u = Measure(matrix=self.parameterize(r, theta))
                ut += time.time() - t

                obj = self.j(u, c)

                if self.do_linesearch:
                    decrease = obj - objective_values[-1]
                    ls_fact = 1.
                    #print(f"{it}: {decrease}, {self.b_parameter}, {self.a_parameter}")
                    if decrease > 0:
                        r_update = r_update / (1 + ls_fact)
                        cs_update = cs_update / (1 + ls_fact)
                        theta_update = theta_update / (1 + ls_fact)
                        self.a_parameter = self.a_parameter / (1 + ls_fact)
                        self.b_parameter = self.b_parameter / (1 + ls_fact)
                    else:
                        self.a_parameter = self.a_parameter * (1 + 0.5 * ls_fact)
                        self.b_parameter = self.b_parameter * (1 + 0.5 * ls_fact)
                else:
                    # just accept the step, and do not check for descent
                    decrease = -1.
                
            times.append(time.time() - t_0)
            
            objective_values.append(obj)
            supports.append(len(u.coefficients))
            if np.isnan(obj) or np.isinf(obj):
                logging.info("Divergence")
                return u, c, objective_values, supports, times, False
            elif (
                np.max(np.abs(obj - np.array(objective_values[-101:])))
                < self.machine_precision
            ):
                logging.info("Convergence")
                return u, c, objective_values, supports, times, True
            elif (
                len(objective_values) > 100
                and obj - np.max(objective_values[-101:-1]) > 0
            ):
                logging.info(f"Divergence: {obj}, {np.max(objective_values[-101:-1])}")
                return u, c, objective_values, supports, times, False

            if (it + 1) % 1000 == 0:
                logging.info(obj - np.max(objective_values[-101:-1]))
                #logging.info(np.max(np.abs(obj - np.array(objective_values[-101:]))))
                logging.info(
                    f"r_delta: {np.linalg.norm(r_update):.3E}, theta_delta: {np.linalg.norm(theta_update):.3E}, c_update: {np.linalg.norm(cs_update):.3E}"
                )
                logging.info(
                    f"p: {p_def:.3E}, inner: {innr:.3E}, update: {update:.3E}, retr: {retr:.3E}, post: {post:.3E}, ut: {ut:.3E}"
                )
                logging.info(
                    f"{it + 1}: supp: {len(u.coefficients)}, c value: {c:.3E}, objective {obj:.14E}"
                )
                p_def = 0
                innr = 0
                update = 0
                retr = 0
                post = 0
                ut = 0
            if time.time() - t_0 > max_time:
                logging.info("Max time reached")
                break
        return u, c, objective_values, supports, times, True
