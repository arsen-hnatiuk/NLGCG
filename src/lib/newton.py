# Implementation of the various Newton-based inner loop methods of NLGCG

import numpy as np
import logging
import jax
from typing import Callable

jax.config.update("jax_enable_x64", True)

logging.basicConfig(
    level=logging.DEBUG,
)


class Newton:
    def __init__(
        self,
        mode: str,
        Omega: np.ndarray,
        alpha: float,
        j_N: Callable,
        j_N_: Callable,
        f_N: Callable,  # Parameterized diligence function
        grad_f_N: Callable,
        grad_j_N: Callable,
        hess_j_N: Callable,
        grad_f_N_non_reg: Callable,  # gradient of non-regularized variables
        max_inner_loop: int = 10,
        quasi_newton_storage: int = 5,
        lbfgs_c_0: float = 1e-4,
        wolfe_powell_constant: float = 0.9,
        tr_lambda: float = 1.0,
        cg_iterations: int = 100,
        newton_p: float = 1e-1,
        descent_constant: float = 1e-6,
        beta: float = 0.5,
        armijo_constant: float = 1e-4,
        delta_min: float = 1e-8,
    ) -> None:
        self.mode = mode
        self.Omega = Omega
        self.alpha = alpha
        self.j_N = j_N
        self.j_N_ = j_N_
        self.f_N = f_N
        self.grad_f_N = grad_f_N
        self.grad_j_N = grad_j_N
        self.hess_j_N = hess_j_N
        self.grad_f_N_non_reg = grad_f_N_non_reg
        self.max_inner_loop = max_inner_loop
        self.quasi_newton_storage = quasi_newton_storage
        self.lbfgs_c_0 = lbfgs_c_0
        self.lbfgs_c_1 = 1
        self.lbfgs_c_2 = 1 / (2 * self.quasi_newton_storage + 3)
        self.wolfe_powell_constant = wolfe_powell_constant
        self.tr_lambda = tr_lambda
        self.cg_iterations = cg_iterations
        self.beta = beta  # For Armijo rule
        self.armijo_constant = armijo_constant  # For Armijo rule
        self.newton_p = 2 + newton_p  # For Newton step acceptance condition
        self.descent_constant = descent_constant  # For Newton step acceptance condition
        self.delta_min = delta_min
        self.machine_precision = 5e-14
        if mode == "globalized_newton":
            self.solve = self.globalized_newton
        elif mode == "globalized_lbfgs":
            self.solve = self.globalized_lbfgs
        elif mode == "trust_region":
            self.solve = self.trust_region
        elif mode == "trust_region_non_reg":
            self.solve = self.trust_region_non_reg
        elif mode == "trust_region_ssn":
            self.solve = self.trust_region_ssn

    def armijo(
        self,
        full_parameters: np.ndarray,
        direction: np.ndarray,
        gradient: np.ndarray,
        choice: str,
    ) -> tuple:
        sigma = 1
        j_N_init = self.j_N(full_parameters)
        desired_descent = self.armijo_constant * gradient @ direction
        if choice == "Grad" and desired_descent > -self.machine_precision:
            return full_parameters.copy(), -1
        full_parameters_new = full_parameters + (sigma * direction)
        j_N_new = self.j_N(full_parameters_new)
        if np.isnan(j_N_new):
            j_N_new = j_N_init + 1
        while j_N_new - j_N_init > sigma * desired_descent:
            if choice == "Grad" and sigma * desired_descent > -self.machine_precision:
                return full_parameters.copy(), -1
            elif sigma == 0:
                return full_parameters.copy(), -1
            sigma *= self.beta
            full_parameters_new = full_parameters + (sigma * direction)
            j_N_new = self.j_N(full_parameters_new)
            if np.isnan(j_N_new):
                j_N_new = j_N_init + 1
        return full_parameters_new, sigma

    def domain_and_descent_tests(
        self,
        full_parameters: np.ndarray,
        full_parameters_new: np.ndarray,
        local_M: float,
        choice: str,
    ) -> list:
        parameters = full_parameters[:-1].reshape(-1, self.Omega.shape[0] + 1)
        parameters_new = full_parameters_new[:-1].reshape(-1, self.Omega.shape[0] + 1)
        output_bools = []
        points_new = parameters_new[:, 1:]
        coefs_new = parameters_new[:, 0].flatten()
        coefs = parameters[:, 0].flatten()

        # Domain test
        if any(points_new[:, 0] <= 0):
            # Nonpositive variance
            output_bools.append(False)
        else:
            output_bools.append(True)

        # M test
        output_bools.append(bool(np.linalg.norm(coefs_new, ord=1) <= local_M))

        # Sign test
        bad_signs = np.where(np.sign(coefs_new) != np.sign(coefs))[0]
        if not len(bad_signs):
            output_bools.append(True)
        else:
            output_bools.append(False)

        # Absolute descent test
        j_N_diff = self.j_N(full_parameters_new) - self.j_N(full_parameters)
        if choice == "Grad" and j_N_diff >= -self.machine_precision:
            output_bools.append(False)
        elif j_N_diff >= 0:
            output_bools.append(False)
        else:
            output_bools.append(True)

        return output_bools

    # def stationarity_descent_test(
    #     self, parameters: np.ndarray, c: float, epsilon: float, radii: list
    # ) -> tuple:
    #     full_parameters = np.hstack((parameters.flatten(), np.array([c])))
    #     grad_j_N_z = self.grad_j_N(full_parameters)
    #     constant_part = np.abs(grad_j_N_z[-1] * c)
    #     grad_norm = np.linalg.norm(grad_j_N_z)
    #     grad_j_N_z_matrix = grad_j_N_z[:-1].reshape(parameters.shape)
    #     grad_points_matrix = grad_j_N_z_matrix[:, 1:]
    #     points_part = 0
    #     for i, radius in enumerate(radii):
    #         points_part += radius * np.linalg.norm(grad_points_matrix[i])
    #     grad_coefs = grad_j_N_z_matrix[:, 0].flatten()
    #     coefs = parameters[:, 0].flatten()
    #     gap = (
    #         points_part
    #         + self.M * abs(min(0, np.min(np.multiply(grad_coefs, np.sign(coefs)))))
    #         + grad_coefs @ coefs
    #         + constant_part
    #     )
    #     if epsilon <= self.C_raw:
    #         ineq = gap >= 0.5 * epsilon**2 / (self.C_raw * self.M**2)
    #     else:
    #         ineq = gap >= (2 * epsilon - self.C_raw * self.M**2) / 2
    #     return ineq, grad_norm

    def globalized_newton(
        self, k: int, params: np.ndarray, support: float
    ) -> np.ndarray:
        # Geiger/Kanzow
        for s in range(self.max_inner_loop):
            local_M = float(self.j_N(params) / self.alpha)
            grad = self.grad_j_N(params)
            hess = self.hess_j_N(params)
            try:
                update_direction = np.linalg.solve(hess, -grad)
                update_norm = np.linalg.norm(update_direction)
                grad_direction = -np.matmul(grad, update_direction)
                condition = grad_direction >= self.descent_constant * np.power(
                    update_norm, self.newton_p
                )
                if not np.any(condition):
                    raise np.linalg.LinAlgError(
                        "Insufficient descent in Newton direction"
                    )
                choice = "Newt"
            except np.linalg.LinAlgError:
                update_direction = -grad.copy()
                choice = "Grad"
            if any(np.isnan(update_direction)):
                return params
            params_new, sigma = self.armijo(params, update_direction, grad, choice)

            # Check validity of the newton step
            domain_tests = self.domain_and_descent_tests(
                params,
                params_new,
                local_M,
                choice,
            )
            logging.info(
                f"{k}, {s}: Globalized Newton. choice: {choice}, support: {support}, sigma: {sigma:.2E}, objective: {self.j_N(params_new):.14E}"
            )
            if not all(domain_tests):
                logging.info(f"Domain tests: {domain_tests}")
                if not domain_tests[-1]:
                    logging.info(f"grad_norm: {np.linalg.norm(grad):.3E}")
                return params
            params = params_new.copy()
        return params

    def q_function(self, s: np.ndarray, y: np.ndarray) -> float:
        norm_s = s @ s
        norm_y = y @ y
        if norm_s * norm_y:
            scalar_product = s @ y
            return scalar_product * min(1 / norm_s, 1 / norm_y)
        else:
            return 0

    def globalized_lbfgs(
        self, k: int, params: np.ndarray, support: float
    ) -> np.ndarray:
        # https://arxiv.org/pdf/2401.03805

        storage = []
        gamma_minus = 0
        gamma_plus = np.inf
        grad = self.grad_j_N(params)

        for s_iter in range(self.max_inner_loop):
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break

            small_omega = min(
                self.lbfgs_c_0, self.lbfgs_c_1 * grad_norm**self.lbfgs_c_2
            )
            lower_omega = min(small_omega, 1 / small_omega)
            upper_omega = max(small_omega, 1 / small_omega)
            if lower_omega > gamma_plus or upper_omega < gamma_minus:
                gamma = lower_omega
            else:
                gamma = max(gamma_minus, lower_omega)

            quasi_inv_hesse = gamma * np.eye(len(params))

            used = 0
            for s, y, scalar_product in storage:
                if self.q_function(s, y) >= small_omega:
                    used += 1
                    y_s_outer = np.outer(y, s)
                    s_s_outer = np.outer(s, s)
                    v_matrix = np.eye(len(params)) - y_s_outer / scalar_product
                    quasi_inv_hesse = (
                        v_matrix.T @ quasi_inv_hesse @ v_matrix
                        + s_s_outer / scalar_product
                    )
            choice = f"Qua{used}"

            update_direction = -quasi_inv_hesse @ grad
            params_new, sigma = self.armijo(params, update_direction, grad, choice)
            grad_new = self.grad_j_N(params_new)
            s = sigma * update_direction
            y = grad_new - grad
            scalar_product = s @ y

            # Check Wolfe-Powell conditions
            new_grad_direction = grad_new @ update_direction
            grad_direction = grad @ update_direction
            if (
                not np.abs(new_grad_direction)
                <= -self.wolfe_powell_constant * grad_direction
            ):
                logging.info(
                    f"WP: {new_grad_direction >= self.wolfe_powell_constant*grad_direction}, strong WP: {np.abs(new_grad_direction)/grad_direction}"
                )

            if scalar_product > 0:
                if len(storage) == self.quasi_newton_storage:
                    for i, tuple_ in enumerate(storage):
                        if i > 0:
                            storage[i - 1] = tuple_
                    storage[-1] = (s, y, scalar_product)
                else:
                    storage.append((s, y, scalar_product))
                gamma_minus = scalar_product / (y @ y)
                gamma_plus = s @ s / scalar_product
            else:
                gamma_minus = 0
                gamma_plus = np.inf

            logging.info(
                f"{k}, {s_iter}: Globalized LBFGS. choice: {choice}, support: {support}, sigma: {sigma:.2E}, grad: {grad_norm:.2E}, objective: {self.j_N(params_new):.14E}"
            )
            params = params_new.copy()
            grad = grad_new.copy()

        return params

    def steihaug_cg(
        self,
        params_reg: np.ndarray,
        params: np.ndarray,
        hess_grad: np.ndarray,
        grad: np.ndarray,
        eps: float,
        delta: float,
        m: Callable,
        mode: str,
    ) -> tuple:
        r = grad
        r_r = float(r @ r)
        q = np.zeros_like(grad)
        p = -grad
        S_p = -hess_grad
        if np.linalg.norm(r) < eps:
            return q, 0, np.sqrt(r_r)
        for i in range(min(len(grad), self.cg_iterations)):
            p_S_p = S_p @ p
            if p_S_p <= 0:
                a_minus, a_plus = self.constraint_finder(q, p, delta)
                q_minus = q + a_minus * p
                q_plus = q + a_plus * p
                m_q_minus = m(q_minus)
                m_q_plus = m(q_plus)
                if m_q_minus < m_q_plus:
                    return q_minus, i + 1, np.sqrt(r_r)
                else:
                    return q_plus, i + 1, np.sqrt(r_r)
            a = r_r / p_S_p
            q_plus = q + a * p
            if np.linalg.norm(q_plus) > delta:
                a_minus, a_plus = self.constraint_finder(q, p, delta)
                if a_plus >= 0:
                    return q + a_plus * p, i + 1, np.sqrt(r_r)
                elif a_minus >= 0:
                    return q + a_minus * p, i + 1, np.sqrt(r_r)
            r_plus = r + a * S_p
            r_plus_r_plus = float(r_plus @ r_plus)
            if np.linalg.norm(r_plus) < eps:
                return q_plus, i + 1, np.sqrt(r_plus_r_plus)
            beta = r_plus_r_plus / r_r
            p_plus = -r_plus + beta * p

            q = q_plus.copy()
            r = r_plus.copy()
            r_r = r_plus_r_plus
            p = p_plus.copy()
            if mode == "non_reg":
                S_p = jax.jvp(
                    self.grad_f_N_non_reg,
                    (params_reg, params),
                    (np.zeros_like(params_reg), p),
                )[1]
            else:
                S_p = jax.jvp(
                    self.grad_j_N,
                    (params,),
                    (p,),
                )[1]
        return q, i + 1, np.sqrt(r_r)

    def trust_region(
        self,
        k: int,
        params: np.ndarray,
        support: float,
    ) -> np.ndarray:
        # Nocedal/Wright: Numerical Optimization Sect. 7.1
        grad = self.grad_j_N(params)
        grad_norm = np.linalg.norm(grad)
        delta = min(0.01, 0.5 * np.sqrt(grad_norm))
        j_params = self.j_N(params)

        for s_iter in range(self.max_inner_loop):
            if grad_norm < 1e-12:
                break

            hess_grad = jax.jvp(
                self.grad_j_N,
                (params,),
                (grad,),
            )[1]

            def m(q: np.ndarray) -> float:
                hess_q = jax.jvp(
                    self.grad_j_N,
                    (params,),
                    (q,),
                )[1]
                quadratic_part = 0.5 * hess_q @ q
                linear_part = grad @ q
                return linear_part + quadratic_part

            eps = max(
                min((np.sqrt(grad_norm), 0.01)) * grad_norm, self.machine_precision
            )
            dir, steihaug_iters, r_norm = self.steihaug_cg(
                params_reg=np.zeros_like(params),
                params=params,
                hess_grad=hess_grad,
                grad=grad,
                eps=eps,
                delta=delta,
                m=m,
                mode="full",
            )

            params_plus = params + dir
            grad_plus = self.grad_j_N(params_plus)
            grad_norm_plus = np.linalg.norm(grad_plus)
            j_params_plus = self.j_N(params_plus)
            m_dir = m(dir)

            a_red = j_params - j_params_plus
            p_red = -m_dir
            rho = a_red / p_red
            if np.isnan(rho):
                rho = 0

            if rho <= 1e-1 or any(np.isnan(grad_plus)):
                delta = 0.25 * delta
                choice = "Redc"
            else:
                if rho > 0.75 and np.linalg.norm(dir) >= 0.9 * delta:
                    delta = 2 * delta
                    choice = "Incr"
                else:
                    choice = "Keep"
                params = params_plus.copy()
                grad = grad_plus.copy()
                grad_norm = grad_norm_plus
                j_params = j_params_plus

            del params_plus, grad_plus, grad_norm_plus, j_params_plus
            logging.info(
                f"{k}, {s_iter}: Trust Region. choice: {choice}, support: {support}, SteihaugCG iters: {steihaug_iters}, delta: {delta:.2E}, rho: {rho:.2E}, grad: {grad_norm:.2E}, objective {self.j_N(params):.14E}"
            )
        return params

    def trust_region_non_reg(
        self,
        k: int,
        params: np.ndarray,
        support: float,
    ) -> np.ndarray:
        params_non_reg_ = params[:-1].reshape(-1, self.Omega.shape[0] + 1)
        params_reg = params_non_reg_[:, 0].flatten()
        params_non_reg = np.hstack((params_non_reg_[:, 1:].flatten(), params[-1]))

        params = params_non_reg.copy()
        grad = self.grad_f_N_non_reg(params_reg, params)
        grad_norm = np.linalg.norm(grad)
        delta = min(0.01, 0.5 * grad_norm)
        j_params = self.j_N_(params_reg, params)

        for s_iter in range(self.max_inner_loop):
            if grad_norm < 1e-12:
                break

            hess_grad = jax.jvp(
                self.grad_f_N_non_reg,
                (params_reg, params),
                (np.zeros_like(params_reg), grad),
            )[1]

            def m(q: np.ndarray) -> float:
                hess_q = jax.jvp(
                    self.grad_f_N_non_reg,
                    (params_reg, params),
                    (np.zeros_like(params_reg), q),
                )[1]
                quadratic_part = 0.5 * hess_q @ q
                linear_part = grad @ q
                return linear_part + quadratic_part

            eps = max(
                min((np.sqrt(grad_norm), 0.01)) * grad_norm, self.machine_precision
            )
            dir, steihaug_iters, r_norm = self.steihaug_cg(
                params_reg=params_reg,
                params=params,
                hess_grad=hess_grad,
                grad=grad,
                eps=eps,
                delta=delta,
                m=m,
                mode="non_reg",
            )

            params_plus = params + dir
            grad_plus = self.grad_f_N_non_reg(params_reg, params_plus)
            grad_norm_plus = np.linalg.norm(grad_plus)
            j_params_plus = self.j_N_(params_reg, params_plus)
            m_dir = m(dir)

            a_red = j_params - j_params_plus
            p_red = -m_dir
            rho = a_red / p_red
            if np.isnan(rho):
                rho = 0

            if rho <= 1e-1 or any(np.isnan(grad_plus)):
                delta = 0.25 * delta
                choice = "Redc"
            else:
                if rho > 0.75 and np.linalg.norm(dir) >= 0.9 * delta:
                    delta = 2 * delta
                    choice = "Incr"
                else:
                    choice = "Keep"
                params = params_plus.copy()
                grad = grad_plus.copy()
                grad_norm = grad_norm_plus
                j_params = j_params_plus

            del params_plus, grad_plus, grad_norm_plus, j_params_plus
            logging.info(
                f"{k}, {s_iter}: Trust Region non-regularized. choice: {choice}, support: {support}, SteihaugCG iters: {steihaug_iters}, delta: {delta:.2E}, rho: {rho:.2E}, grad: {grad_norm:.2E}, objective {self.j_N_(params_reg, params):.14E}"
            )

        params_non_reg_[:, 1:] = params[:-1].reshape(-1, self.Omega.shape[0]).copy()
        full_params = np.hstack(
            (params_non_reg_.flatten(), np.array([params[-1]]))
        ).copy()
        return full_params

    def H(
        self,
        normal_map_norm: float,
        prox_params: np.ndarray,
        tau: float,
        lbda: float,
    ) -> float:
        return self.j_N(prox_params) + tau * lbda * 0.5 * normal_map_norm**2

    def normal_map(
        self, params: np.ndarray, prox_params: np.ndarray, lbda: float
    ) -> np.ndarray:
        grad_f_prox_params = self.grad_f_N(prox_params)
        return grad_f_prox_params + (params - prox_params) / lbda

    def prox_and_grad(self, params: np.ndarray, lbda: float) -> tuple:
        prox_val = np.zeros(params.shape)
        prox_grad = np.zeros(params.shape)
        for i, val in enumerate(params):
            if i % (self.Omega.shape[0] + 1) or i == (len(params) - 1):
                # Not coefficient: not regularized
                prox_val[i] = val
                prox_grad[i] = 1
            else:
                if np.abs(val) > lbda * self.alpha:
                    prox_val[i] = val - lbda * self.alpha * np.sign(val)
                    prox_grad[i] = 1
        return prox_val, prox_grad

    def constraint_finder(self, q: np.ndarray, p: np.ndarray, delta: float) -> tuple:
        q_q = np.matmul(q, q)
        p_p = np.matmul(p, p)
        q_p = np.matmul(q, p)
        a_negative = (-q_p - np.sqrt((q_p) ** 2 - (q_q - delta**2) * p_p)) / (p_p)
        a_positive = (-q_p + np.sqrt((q_p) ** 2 - (q_q - delta**2) * p_p)) / (p_p)
        return a_negative, a_positive

    def steihaug_cg_ssn(
        self,
        prox_params: np.ndarray,
        D_diagonal: np.ndarray,
        S_g: np.ndarray,
        g_vector: np.ndarray,
        eps: float,
        delta: float,
        m: Callable,
    ) -> tuple:
        r = g_vector
        r_r = r @ r
        q = np.zeros_like(g_vector)
        p = -g_vector
        S_p = -S_g
        if np.sqrt(r_r) < eps:
            return q, 0, np.sqrt(r_r)
        for i in range(min(len(g_vector), self.cg_iterations)):
            p_S_p = np.matmul(S_p, p)
            if p_S_p <= 0:
                if delta > 1e50:
                    return q, i + 1, np.sqrt(r_r)
                a_minus, a_plus = self.constraint_finder(q, p, delta)
                q_minus = q + a_minus * p
                q_plus = q + a_plus * p
                m_q_minus = m(q_minus)
                m_q_plus = m(q_plus)
                if m_q_minus < m_q_plus:
                    return q_minus, i + 1, np.sqrt(r_r)
                else:
                    return q_plus, i + 1, np.sqrt(r_r)
            a = r_r / p_S_p
            q_plus = q + a * p
            if np.linalg.norm(q_plus) > delta:
                a_minus, a_plus = self.constraint_finder(q, p, delta)
                if a_plus >= 0:
                    return q + a_plus * p, i + 1, np.sqrt(r_r)
                elif a_minus >= 0:
                    return q + a_minus * p, i + 1, np.sqrt(r_r)
            r_plus = r + a * S_p
            r_plus_r_plus = r_plus @ r_plus
            if np.sqrt(r_plus_r_plus) < eps:
                return q_plus, i + 1, np.sqrt(r_plus_r_plus)
            beta = r_plus_r_plus / r_r
            p_plus = -r_plus + beta * p

            q = q_plus.copy()
            r = r_plus.copy()
            r_r = r_plus_r_plus
            p = p_plus.copy()
            hess_D_p = jax.jvp(
                self.grad_f_N, (prox_params,), (np.multiply(D_diagonal, p),)
            )[1]
            S_p = np.multiply(D_diagonal, hess_D_p.T).T  # D^T*B*D*p
        return q, i + 1, np.sqrt(r_r)

    def trust_region_ssn(
        self,
        k: int,
        params: np.ndarray,
        support: float,
    ) -> np.ndarray:
        # https://arxiv.org/pdf/2106.09340
        Lipschitz = 1
        # self.tr_lambda = 1/Lipschitz
        tau = 0.1 / (Lipschitz**2 * self.tr_lambda**2 + 2)
        nu = 0.5 * min(
            tau, 0.05 * (1 - 0.5 * tau * (0.5 * self.tr_lambda**2 * Lipschitz**2 + 1))
        )
        n_s = 1

        prox_params, D_diagonal = self.prox_and_grad(params, self.tr_lambda)
        if not self.max_inner_loop:
            return prox_params
        normal_map_vector = self.normal_map(params, prox_params, self.tr_lambda)
        normal_map_norm = np.linalg.norm(normal_map_vector)
        delta = min(0.01, 0.5 * normal_map_norm)
        H_value = self.H(normal_map_norm, prox_params, tau, self.tr_lambda)

        for s_iter in range(self.max_inner_loop):
            if normal_map_norm < 1e-12:
                return prox_params

            opposite_D_diagonal = np.ones_like(D_diagonal) - D_diagonal
            g_vector = np.multiply(D_diagonal, normal_map_vector)
            hess_D_g = jax.jvp(self.grad_f_N, (prox_params,), (g_vector,))[1]
            S_g = np.multiply(D_diagonal, hess_D_g.T).T  # D^T*B*D*g

            def m(q: np.ndarray) -> float:
                D_q = np.multiply(D_diagonal, q)
                B_D_q = jax.jvp(self.grad_f_N, (prox_params,), (D_q,))[1]
                quadratic_part = 0.5 * B_D_q @ D_q
                linear_part = np.matmul(normal_map_vector, D_q)
                return linear_part + quadratic_part

            eps = max(
                min((np.power(normal_map_norm, 2.5), 0.01)), self.machine_precision
            )

            q_bar, steihaug_iters, r_norm = self.steihaug_cg_ssn(
                prox_params, D_diagonal, S_g, g_vector, eps, delta, m
            )
            D_q_bar = np.multiply(D_diagonal, q_bar)
            hess_D_q_bar = jax.jvp(self.grad_f_N, (prox_params,), (D_q_bar,))[
                1
            ]  # B*D*q_bar
            s_bar = (
                q_bar
                - self.tr_lambda * (normal_map_vector + hess_D_q_bar)
                - np.multiply(opposite_D_diagonal, q_bar)
            )
            s = min(1, delta / np.linalg.norm(s_bar)) * s_bar

            params_plus = params + s
            prox_params_plus, D_diagonal_plus = self.prox_and_grad(
                params_plus, self.tr_lambda
            )
            normal_map_vector_plus = self.normal_map(
                params_plus, prox_params_plus, self.tr_lambda
            )
            normal_map_norm_plus = np.linalg.norm(normal_map_vector_plus)
            H_value_plus = self.H(
                normal_map_norm_plus, prox_params_plus, tau, self.tr_lambda
            )
            prox_diff = prox_params_plus - prox_params
            prox_diff_norm = np.linalg.norm(prox_diff)

            a_red = H_value - H_value_plus
            nu_k = min(
                nu,
                0.001 * (n_s * np.log(n_s) ** 2 * prox_diff_norm) ** 0.2,
            )
            p_red = 0.5 * tau * normal_map_norm * min(
                self.tr_lambda, delta, self.tr_lambda * normal_map_norm
            ) + nu_k * normal_map_norm * prox_diff_norm**2 / min(
                delta, self.tr_lambda * normal_map_norm
            )
            rho = a_red / p_red
            if np.isnan(rho):
                rho = 0

            if prox_diff_norm:
                Lipschitz = max(
                    2
                    * (
                        self.f_N(prox_params_plus)
                        - self.f_N(prox_params)
                        - self.grad_f_N(prox_params) @ prox_diff
                    )
                    / prox_diff_norm**2,
                    Lipschitz,
                )
                # self.tr_lambda = 1/Lipschitz
                tau = 0.1 / (Lipschitz**2 * self.tr_lambda**2 + 2)
                nu = 0.5 * min(
                    tau,
                    0.05
                    * (1 - 0.5 * tau * (0.5 * self.tr_lambda**2 * Lipschitz**2 + 1)),
                )

            if rho <= 1e-6 or any(np.isnan(normal_map_vector_plus)):
                delta = max(self.delta_min, 0.25 * delta)
                choice = "Redc"
                if prox_diff_norm:
                    prox_params, D_diagonal = self.prox_and_grad(params, self.tr_lambda)
                    normal_map_vector = self.normal_map(
                        params, prox_params, self.tr_lambda
                    )
                    normal_map_norm = np.linalg.norm(normal_map_vector)
                    H_value = self.H(normal_map_norm, prox_params, tau, self.tr_lambda)
            else:
                n_s += 1
                params = params_plus.copy()
                if rho < 0.75:
                    choice = "Keep"
                else:
                    delta *= 2
                    choice = "Incr"
                if prox_diff_norm:
                    prox_params, D_diagonal = self.prox_and_grad(params, self.tr_lambda)
                    normal_map_vector = self.normal_map(
                        params, prox_params, self.tr_lambda
                    )
                    normal_map_norm = np.linalg.norm(normal_map_vector)
                    H_value = self.H(normal_map_norm, prox_params, tau, self.tr_lambda)
                else:
                    prox_params = prox_params_plus.copy()
                    normal_map_vector = normal_map_vector_plus.copy()
                    normal_map_norm = normal_map_norm_plus
                    D_diagonal = D_diagonal_plus.copy()
                    H_value = H_value_plus
            logging.info(
                f"{k}, {s_iter}: choice: {choice}, support: {support}, SteihaugCG iters: {steihaug_iters}, delta: {delta:.2E}, rho: {rho:.2E}, normal_map: {normal_map_norm:.2E}, objective {self.j_N(prox_params):.14E}"
            )
        return prox_params
