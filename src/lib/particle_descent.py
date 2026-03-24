# The Conic Particle Gradient Descent algorithm, as presented in https://arxiv.org/pdf/1907.10300

import numpy as np
from typing import Callable
import logging
import time
from lib.measure import Measure
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_ = jnp.zeros(0)

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
        do_linesearch: bool = True,
        do_pruning: bool = False,
        residual_tolerance: float = 5e-14,
    ):
        self.m = m
        self.j = j
        self.p = p
        self.grad_p = grad_p
        self.Omega = Omega
        self.a_parameter = a_parameter
        self.b_parameter = b_parameter
        self.residual_tolerance = residual_tolerance
        self.kernel = kernel
        self.constant_dim = constant_dim
        self.kernel_dim = kernel_dim
        self.alpha = alpha
        self.target = target
        self.g = g
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.do_linesearch = do_linesearch
        self.do_pruning = do_pruning
        self.pred_factor = 0.1  # For line search

    def parameterize(
        self, r: np.ndarray, theta: np.ndarray, Nparticle: int = None
    ) -> np.ndarray:
        if Nparticle is None:
            Nparticle = len(r)
        matrix = np.zeros((len(r), 1 + self.Omega.shape[0]))
        matrix[:, 0] = self.kernel_sign * (r**2) / Nparticle
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
        theta = np.vstack((grid_pos, grid_neg))  # Positive and negative support
        cs = np.array([1, -1])
        self.kernel_sign = np.sign(r)
        r = np.abs(r)
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
            r_retraction = r + r * del_r
            c_retraction = c + c * del_c
            theta_retraction = theta + del_theta
        elif mode == "mirror":
            r_retraction = r * np.exp(del_r)
            c_retraction = c * np.exp(del_c)
            theta_retraction = theta + del_theta
        return r_retraction, c_retraction, theta_retraction

    def solve(
        self,
        max_iters: int = 1000000,
        max_time: float = 1e6,
        u_0: Measure = Measure(),
        c_0: float = 0,
        mode: str = "exponential",
        do_logging: bool = True,
    ):
        t_0 = time.perf_counter()
        success = True
        if len(u_0.coefficients):
            self.kernel_sign = np.sign(u_0.coefficients)
            r = np.sqrt(self.kernel_sign * u_0.coefficients * len(u_0.coefficients))
            theta = u_0.support
            cs = np.array([c_0])
        else:
            r, theta, cs = self.initial_distribution(mode=mode)

        # Fix the initial number of particles for the parametrization
        Nparticle = len(r)

        def parameterize(r, theta):
            return self.parameterize(r, theta, Nparticle=Nparticle)

        # Initialize
        u = Measure(matrix=parameterize(r, theta))
        c = np.sum(np.sign(cs) * cs**2) / Nparticle
        if do_logging:
            logging.info(f"0: objective {self.j(u, c):.14E}")
        objective_values = [self.j(u, c)]
        times = [time.perf_counter() - t_0]
        supports = [len(u.coefficients)]

        min_a_parameter = 1e-8

        for it in range(max_iters):
            p_u = self.p(u, c)
            grad_p_u = self.grad_p(u, c)

            inner = c * np.hstack(
                (
                    np.ones(self.constant_dim),
                    np.zeros(self.kernel_dim - self.constant_dim),
                )
            )
            if len(u.coefficients):
                inner += self.kernel(u.support).T @ u.coefficients
            inner = -self.grad_f(inner)
            inner_sum = np.sum(inner[: self.constant_dim])

            inner_r_update = (
                self.kernel_sign * p_u(theta) - self.alpha
            )  # <-- There was an error in sign of alpha
            inner_cs_update = np.sign(cs) * inner_sum
            inner_theta_update = (self.kernel_sign * np.array(grad_p_u(theta)).T).T

            r_update = self.a_parameter * 2.0 * inner_r_update
            cs_update = self.a_parameter * 2.0 * inner_cs_update
            theta_update = self.b_parameter * inner_theta_update

            # Descent, predicted by the gradient: armijo step size
            r_pred = 4 * (r * inner_r_update) @ (r * inner_r_update)
            cs_pred = 4 * (cs * inner_cs_update) @ (cs * inner_cs_update)
            theta_pred = np.sum(
                [
                    r_local**2 * inner_theta_update_local @ inner_theta_update_local
                    for r_local, inner_theta_update_local in zip(r, inner_theta_update)
                ]
            )
            pred_raw = (r_pred + cs_pred + theta_pred) / Nparticle  # the gradient norm
            pred = (
                self.a_parameter * r_pred
                + self.a_parameter * cs_pred
                + self.b_parameter * theta_pred
            ) / Nparticle

            r_old, cs_old, theta_old = r.copy(), cs.copy(), theta.copy()
            decrease = 1.0
            while decrease > 0:
                r, cs, theta = self.retraction(
                    r_old,
                    r_update,
                    cs_old,
                    cs_update,
                    theta_old,
                    theta_update,
                    mode="mirror",
                )

                c = np.sum(np.sign(cs) * cs**2) / Nparticle
                u = Measure(matrix=parameterize(r, theta))
                obj = self.j(u, c)
                if np.isnan(obj):
                    obj = np.inf

                if self.do_linesearch:
                    # # Konstantin's version
                    # r_grad = (1 / Nparticle) * (r_old * r_update) / self.a_parameter
                    # cs_grad = (1 / Nparticle) * (cs_old * cs_update) / self.a_parameter
                    # theta_grad = (
                    #     (1 / Nparticle)
                    #     * (theta_update * r_old[:, None] ** 2)
                    #     / self.b_parameter
                    # )
                    # pred = (
                    #     r_grad.dot(r - r_old)
                    #     + cs_grad.dot(cs - cs_old)
                    #     + (theta_grad.reshape(-1).dot((theta - theta_old).reshape(-1)))
                    # )

                    model = objective_values[-1] - self.pred_factor * pred

                    decrease = obj - model
                    if decrease >= 0:
                        contraction = (
                            max(self.a_parameter / 2, min_a_parameter)
                            / self.a_parameter
                        )
                        self.a_parameter = self.a_parameter * contraction
                        self.b_parameter = self.b_parameter * contraction
                        r_update = r_update * contraction
                        cs_update = cs_update * contraction
                        theta_update = theta_update * contraction
                        pred = pred * contraction

                        if contraction >= 0.99:
                            # exit line-search and accept step
                            logging.warning(
                                f"line-search failed in iteration {it}: value - desired value is {decrease}, reduction is {objective_values[-1] - obj:1.3e}, gradient * step size is {pred:1.3e}, "
                            )
                            # breakpoint()
                            decrease = 0
                    else:
                        if decrease < 0.0:
                            self.a_parameter = self.a_parameter * (1 + 0.25)
                            self.b_parameter = self.b_parameter * (1 + 0.25)
                else:
                    # just accept the step, and do not check for descent
                    decrease = -1.0

            keep_indices = np.logical_and(theta[:, 0] > 1e-6, np.abs(r) > 1e-6)

            if self.do_pruning and not np.all(keep_indices):
                dropped_ind = np.logical_not(keep_indices)
                r_drop = r[dropped_ind]
                theta_drop = theta[dropped_ind]
                r = r[keep_indices]
                theta = theta[keep_indices]
                self.kernel_sign = self.kernel_sign[keep_indices]

                u = Measure(matrix=parameterize(r, theta))
                old_obj = obj
                obj = self.j(u, c)

                if do_logging:
                    logging.info(
                        f"dropped indices {np.where(dropped_ind)[0]} with r={r_drop} and theta={theta_drop}, function change {obj - old_obj}"
                    )
            Nparticle = len(r)

            times.append(time.perf_counter() - t_0)

            objective_values.append(obj)
            supports.append(Nparticle)
            if np.isnan(obj) or np.isinf(obj):
                logging.info("Divergence")
                success = False
                break
            elif pred_raw < 1e-9:
                logging.info(f"Convergence, gradient: {pred_raw:1.3e}")
                success = True
                break
            # elif (
            #     np.max(np.abs(obj - np.array(objective_values[-101:])))
            #     < self.residual_tolerance
            # ):
            #     logging.info("Convergence")
            #     success = True
            #     break
            elif (
                not self.do_linesearch
                and len(objective_values) > 100
                and obj - np.max(objective_values[-101:-1]) > 0
            ):
                logging.info(f"Divergence: {obj}, {np.max(objective_values[-101:-1])}")
                success = False
                break
            if (it + 1) % 1000 == 0 and do_logging:
                logging.info(
                    f"{it + 1}: supp: {Nparticle}, c value: {c:.3E}, a value: {self.a_parameter:.3E}, objective {obj:.14E}"
                )
            if time.perf_counter() - t_0 > max_time:
                logging.info("Max time reached")
                break
            if it + 1 >= max_iters:
                logging.info("Max iterations reached")

        logging.info(
            f"CPG exited after {times[-1]:.3E} seconds with sparsity {len(u.support)} and success {success} to objective value {objective_values[-1]:.14E}"
        )

        return u, c, objective_values, supports, times, success
