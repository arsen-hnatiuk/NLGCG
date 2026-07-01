import numpy as np
import logging
import time
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp
from typing import Callable
from lib.ssn import SSN
from lib.ssn_rkhs import SSN_RKHS
from lib.newton import Newton
from lib.global_search import GlobalSearch
from lib.measure import Measure

jax.config.update("jax_enable_x64", True)
_ = jnp.zeros(0)

logging.basicConfig(
    level=logging.DEBUG,
)


class NLGCG:
    def __init__(
        self,
        g: Callable,  # Penalty function
        f_N: Callable,  # Parameterized diligence function
        target: np.ndarray,  # In the case of RKHS, sample from true distribution
        grad_f_N: Callable,
        hess_f_N: Callable,
        j: Callable,  # Objective
        j_N: Callable,  # parameterized objective
        p: Callable,  # Dual variable
        grad_p: Callable,
        hess_p: Callable,
        grad_j_N: Callable,
        alpha: float,
        Omega: np.ndarray,
        global_search_resolution: int,  # Only for deterministic grid in global search
        max_radius: float = 1.0,
        kernel: Callable = None,  # Not valid for RKHS
        f: Callable = None,  # Diligence function, not valid for RKHS
        grad_f: Callable = None,  # Not valid for RKHS
        hess_f: Callable = None,  # Not valid for RKHS
        constant_dim: int = 0,  # Not valid for RKHS
        kernel_dim: int = 0,  # Not valid for RKHS
        M: float = 1e6,
        C_0: float = 1,
        dual_variable_goodness: float = 0.5,
        ssn_steps: int = 100,
        newton_tolerance: float = 5e-2,  # Tolerance for Newton steps in Global Search
        experiment_type: float = "Euclidean",  # Else RKHS, no constant c
        kernel_k: Callable = None,  # Only for KRHS experiment
        kernel_s: Callable = None,  # Only for RKHS experiment
        loss_offset: float = 1,  # Only for RKHS experiment
    ) -> None:
        self.target = target
        self.kernel = kernel
        self.p = p
        self.grad_p = grad_p
        self.hess_p = hess_p
        self.alpha = alpha
        self.g = g
        self.f = f
        self.f_N = f_N
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.grad_f_N = grad_f_N
        self.hess_f_N = hess_f_N
        self.Omega = Omega  # Example [[0,1],[1,2]] for [0,1]x[1,2]
        self.max_radius = max_radius
        self.j = j
        self.j_N = j_N
        self.u_0 = Measure()
        self.c_0 = 0
        self.M_0 = float(
            min(M, self.j(self.u_0, self.c_0) / self.alpha)
        )  # Bound on the norm of iterates
        self.M = self.M_0
        self.C_0 = C_0
        self.C_raw = self.C_0  # curvature constant without the M part
        self.global_search_resolution = global_search_resolution
        self.grad_j_N = grad_j_N
        self.constant_dim = constant_dim
        self.kernel_dim = kernel_dim
        self.machine_precision = 5e-14
        self.dual_variable_goodness = dual_variable_goodness
        self.ssn_steps = ssn_steps
        self.newton_tolerance = newton_tolerance
        self.experiment_type = experiment_type
        if experiment_type == "Euclidean":
            self.finite_dimensional_step = self.finite_dimensional_step_euclidean
        else:
            self.finite_dimensional_step = self.finite_dimensional_step_rkhs
        self.kernel_k = kernel_k
        self.kernel_s = kernel_s
        self.loss_offset = loss_offset
        self.initialize_functions()

    def initialize_functions(self):
        if self.experiment_type == "Euclidean":
            full_parameters = np.ones(self.Omega.shape[0] + 2)
            _ = self.kernel(np.ones((1, len(self.Omega))))
            _ = self.f(
                np.hstack(
                    (
                        np.ones(self.constant_dim),
                        np.zeros(self.kernel_dim - self.constant_dim),
                    )
                )
            )
            _ = self.grad_f(
                np.hstack(
                    (
                        np.ones(self.constant_dim),
                        np.zeros(self.kernel_dim - self.constant_dim),
                    )
                )
            )
            _ = self.hess_f(
                np.hstack(
                    (
                        np.ones(self.constant_dim),
                        np.zeros(self.kernel_dim - self.constant_dim),
                    )
                )
            )
        else:
            full_parameters = np.ones(self.Omega.shape[0] + 1)
            _ = self.kernel_k(
                np.ones((1, len(self.Omega))), np.ones((1, len(self.Omega)))
            )
            _ = self.kernel_s(np.ones((1, len(self.Omega))))
        _ = self.p(self.u_0, self.c_0)(np.ones((1, len(self.Omega))))
        _ = self.grad_p(self.u_0, self.c_0)(np.ones((1, len(self.Omega))))
        _ = self.hess_p(self.u_0, self.c_0)(np.ones((1, len(self.Omega))))
        _ = self.g(np.ones(1))
        _ = self.f_N(full_parameters)
        _ = self.grad_f_N(full_parameters)
        _ = self.hess_f_N(full_parameters)
        _ = self.j(self.u_0, self.c_0)
        _ = self.j_N(full_parameters)
        _ = self.grad_j_N(full_parameters)

    def finite_dimensional_step_euclidean(
        self,
        u: Measure,
        c: float,
        Psi: float,
        mode: str = "unconstrained",
        optimization: str = "full",
        log_results: bool = True,
    ) -> tuple:
        K_support = np.hstack(
            (np.ones(self.constant_dim), np.zeros(self.kernel_dim - self.constant_dim))
        ).reshape(-1, 1)
        coefs = np.array([c])
        invariable_kernel = np.zeros((self.kernel_dim))
        if len(u.coefficients):
            measure_K = self.kernel(u.support).T
            if optimization == "full":
                K_support = np.hstack((measure_K, K_support))
                coefs = np.hstack((u.coefficients, coefs))
            elif optimization == "constant":
                invariable_kernel = measure_K @ u.coefficients
        if mode == "positive":
            signs = np.sign(coefs)
            signs[signs == 0] = 1
            K_support = np.multiply(K_support, signs)
            u_0 = np.abs(coefs)
        else:
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
            mode=mode,
            maximum_iterations=self.ssn_steps,
        )
        ssn_solution = ssn.solve(tol=Psi, u_0=u_0, log_results=log_results)
        del ssn
        if mode == "positive":
            ssn_normal = ssn_solution * signs
            ssn_clipped = np.maximum(ssn_solution, np.zeros(len(ssn_solution))) * signs
            # The SSN in positive mode can return a better solution with wrong signs
        else:
            ssn_normal = ssn_solution.copy()
            ssn_clipped = ssn_solution.copy()
        if optimization == "full":
            u_raw_normal = ssn_normal[:-1]
            u_raw_clipped = ssn_clipped[:-1]
            c_plus_normal = ssn_normal[-1]
            c_plus_clipped = ssn_clipped[-1]
            # Reconstruct u
            if len(u_raw_normal):
                u_plus_normal = Measure(
                    support=u.support[u_raw_normal != 0].copy(),
                    coefficients=u_raw_normal[u_raw_normal != 0].copy(),
                )
                u_plus_clipped = Measure(
                    support=u.support[u_raw_clipped != 0].copy(),
                    coefficients=u_raw_clipped[u_raw_clipped != 0].copy(),
                )
            else:
                u_plus_normal = Measure()
                u_plus_clipped = Measure()
        elif optimization == "constant":
            c_plus_normal = ssn_normal[0]
            c_plus_clipped = ssn_clipped[0]
            u_plus_normal = u.copy()
            u_plus_clipped = u.copy()
        tuples = [
            (u_plus_normal, c_plus_normal),
            (u_plus_clipped, c_plus_clipped),
            (u, c),
        ]
        values = [self.j(u, c) for u, c in tuples]
        best_value = np.argmin(values)
        u_plus, c_plus = tuples[best_value]
        return u_plus, c_plus

    def finite_dimensional_step_rkhs(
        self,
        u: Measure,
        c: float,
        Psi: float,
        mode: str = "unconstrained",
        optimization: str = "full",
        log_results: bool = True,
    ) -> Measure:
        if not len(u.coefficients):
            return u, 0
        support = u.support
        K_matrix = self.kernel_k(support, support)
        S_vector = self.kernel_s(support)
        # diff_support = np.subtract.outer(support, support)
        # hessian = self.k_tilde_sigma(diff_support)
        # diff_X = np.subtract.outer(self.X, support)
        # target_part = self.k_tilde(diff_X)
        finite_j = (
            lambda weights: 0.5 * weights.T @ K_matrix @ weights
            - S_vector @ weights
            + self.alpha * np.linalg.norm(weights, ord=1)
            + self.loss_offset
        )
        local_p = lambda weights: S_vector - K_matrix @ weights
        u_0 = u.coefficients.copy()
        ssn = SSN_RKHS(
            alpha=self.alpha,
            M=self.M,
            g=self.g,
            p=local_p,
            hessian=K_matrix,
            j=finite_j,
            maximum_iterations=self.ssn_steps,
            log_results=log_results,
        )
        ssn_solution = ssn.solve(tol=Psi, u_0=u_0)
        del ssn
        u_plus = Measure(
            support=u.support[ssn_solution != 0].copy(),
            coefficients=ssn_solution[ssn_solution != 0].copy(),
        )
        candidates = [u_plus, u]
        values = [self.j(u_, c) for u_ in candidates]
        best_value = np.argmin(values)
        u_plus = candidates[best_value].copy()
        return u_plus, c

    def drop_step(self, u: Measure, c: float) -> tuple:
        if not len(u.coefficients):
            return u, False
        true_j = self.j(u, c)
        p_u = self.p(u, c)
        p_vals = p_u(u.support)
        vals_signs = np.sign(p_vals)
        measure_signs = np.sign(u.coefficients)
        keep_indices = np.where(vals_signs == measure_signs)[0]
        P_vals_unsorted = np.abs(p_vals[keep_indices])
        sorting_indices = np.argsort(P_vals_unsorted)[::-1]
        reduced_support = u.support[keep_indices][sorting_indices]
        reduced_coefficients = u.coefficients[keep_indices][sorting_indices]

        new_support = []
        new_coefficients = []
        for point, coef in zip(reduced_support, reduced_coefficients):
            new_support.append(point)
            new_coefficients.append(coef)
            tentative_u = Measure(support=new_support, coefficients=new_coefficients)
            tentative_j = self.j(tentative_u, c)
            if tentative_j <= true_j:
                if len(new_support) == len(u.support):
                    return u, False
                else:
                    return tentative_u, True
        return u, False

    def local_merging_update_radii(self, u: Measure, c: float) -> tuple:
        if not len(u.coefficients):
            return np.array([[]]), u, [], []
        radii = self.compute_radii(u, c)
        p_u = self.p(u, c)
        p_norm = lambda x: np.abs(p_u(x))
        sorting_indices = np.argsort(p_norm(u.support))[::-1]
        full_set = u.support.copy()[sorting_indices]
        full_coefs = u.coefficients.copy()[sorting_indices]
        full_radii = np.array(radii)[sorting_indices]
        merged = np.array([False] * len(full_set), dtype=bool)
        cluster_points = []
        cluster_coefs = []
        for i, point in enumerate(full_set):
            if not merged[i]:
                local_distances = np.linalg.norm(full_set[~merged] - point, axis=1)
                cluster_indices = local_distances <= 2 * np.maximum(
                    full_radii[~merged], full_radii[i]
                )
                cluster_points.append(point)
                cluster_coefs.append(np.sum(full_coefs[~merged][cluster_indices]))
                merged[~merged] |= cluster_indices
        old_radii = radii.copy()
        new_radii = radii.copy()
        if len(cluster_coefs) == len(u.coefficients):
            # No merging has happened
            return u.to_matrix(), u, new_radii, old_radii
        else:
            u_plus = Measure(support=cluster_points, coefficients=cluster_coefs)
            new_radii = self.compute_radii(u_plus, c)
            return u_plus.to_matrix(), u_plus, new_radii, old_radii

    def compute_radii(self, u: Measure, c: float) -> list:
        # grad_time = 0
        # hess_time = 0
        # t_0 = time.perf_counter()
        radii = []
        if not len(u.coefficients):
            return radii
        # t = time.perf_counter()
        grad_p = self.grad_p(u, c)
        grads = grad_p(u.support)
        # grad_time += time.perf_counter() - t
        # t = time.perf_counter()
        hess_p = self.hess_p(u, c)
        hesses = hess_p(u.support)
        # hess_time += time.perf_counter() - t
        for point_grad, point_hess in zip(grads, hesses):
            grad_norm = np.linalg.norm(point_grad)
            try:
                eigenvalue = np.min(np.abs(np.linalg.eigvals(point_hess)))
                if eigenvalue:
                    local_radius = 4 * grad_norm / eigenvalue
                elif grad_norm:
                    local_radius = self.max_radius
                else:
                    local_radius = 0
                radii.append(min(local_radius, self.max_radius))
            except np.linalg.LinAlgError:
                # If the Hessian contains nan, we cannot compute a radius
                radii.append(self.max_radius)
        # all_time = time.perf_counter() - t_0
        # logging.info(f"grad: {grad_time/all_time:.3E}, hess: {hess_time/all_time:.3E}")
        # logging.info(radii)
        return radii

    def lgcg_step(
        self,
        p_u: Callable,
        u: Measure,
        c: float,
        epsilon: float,
        q_u: float,
        radii: np.ndarray,
        mode: str = "stochastic_adaptive",
        temperature: float = 1.0,
        log_results: bool = True,
    ) -> tuple:
        j_initial = self.j(u, c)
        condition = False
        if len(radii):
            radius = np.max(radii)
        else:
            radius = self.max_radius
        global_search_object = GlobalSearch(
            Omega=self.Omega,
            M=self.M,
            global_search_resolution=self.global_search_resolution,
            dual_variable_goodness=self.dual_variable_goodness,
            alpha=self.alpha,
            len_target=len(self.target),
            grad_p=self.grad_p,
            hess_p=self.hess_p,
            mode=mode,
            newton_tolerance=self.newton_tolerance,
        )
        best_val, found_points, global_valid = global_search_object.solve(
            u, c, epsilon, q_u, p_u, radius, temperature, log_results
        )
        del global_search_object
        phi = self.M * max(best_val - self.alpha, 0) + q_u
        u_norm = np.linalg.norm(u.coefficients, ord=1)
        if u_norm:
            phi_numerical = (
                max(best_val - self.alpha, 0)
                + self.alpha
                - u.duality_pairing(p_u) / u_norm
            )
        else:
            phi_numerical = max(best_val - self.alpha, 0)
        if phi > q_u:
            v = Measure(
                support=found_points, coefficients=self.M * np.sign(p_u(found_points))
            )
        else:
            v = Measure()
        updates = 0
        self.C_raw /= 2
        while not condition:
            # Increase C_raw until it satisfies the descent condition
            updates += 1
            self.C_raw *= 2
            Curv = self.C_raw * self.M**2
            eta = min(1, phi / Curv)
            u_plus = u * (1 - eta) + v * eta
            jdiff = self.j(u_plus, c) - j_initial
            if phi <= Curv:
                expected_decrease = -0.5 * phi**2 / Curv
            else:
                expected_decrease = 0.5 * Curv - phi
            if (
                abs(expected_decrease) < self.machine_precision
                and jdiff < self.machine_precision
            ):
                condition = True
            else:
                condition = jdiff <= expected_decrease
        if updates < 2:
            # There has been no increase of the curvature constant, try a smaller value
            while condition and self.C_raw >= self.C_0:
                previous_u_plus = u_plus.copy()
                self.C_raw /= 2
                Curv = self.C_raw * self.M**2
                eta = min(1, phi / Curv)
                u_plus = u * (1 - eta) + v * eta
                jdiff = self.j(u_plus, c) - j_initial
                if phi <= Curv:
                    expected_decrease = -0.5 * phi**2 / Curv
                else:
                    expected_decrease = 0.5 * Curv - phi
                condition = jdiff <= expected_decrease
            self.C_raw *= 2
            u_plus = previous_u_plus.copy()
        if not global_valid:
            # We have a global maximum x_k
            if mode == "deterministic":
                epsilon = 0.5 * phi
            else:
                epsilon /= 2
        return u_plus, epsilon, global_valid, phi_numerical

    def newton_step(
        self,
        k: int,
        params: np.ndarray,
        support: int,
        mode: str = "trust_region",
        log_results: bool = True,
        full_trace: bool = False,
    ) -> np.ndarray:
        if log_results:
            logging.info(
                f"{k} Newton: mode:{mode}, support: {support}, initial_objective: {self.j_N(params):.14E}"
            )
        newton_method = Newton(
            mode=mode,
            Omega=self.Omega,
            alpha=self.alpha,
            j_N=self.j_N,
            f_N=self.f_N,
            grad_f_N=self.grad_f_N,
            grad_j_N=self.grad_j_N,
        )
        params_new, newton_information = newton_method.solve(
            k=k,
            params=params,
            support=support,
            log_results=log_results,
            full_trace=full_trace,
        )
        del newton_method
        return params_new, newton_information

    def solve(
        self,
        tol: float,
        u_0: Measure = Measure(),
        c_0: float = 0,
        mode: str = "stochastic_adaptive",
        inner_mode: str = "trust_region",
        temperature: float = 1.0,
        log_results: bool = True,
        full_trace: bool = False,
    ) -> tuple:
        all_information = []
        self.M = min(self.M_0, float(self.j(u_0, c_0) / self.alpha))
        self.C_raw = self.C_0
        epsilon = max(1, 0.5 * self.j(u_0, c_0))
        k = 0
        dropped = False
        dropped_tot = 0
        initial_time = time.perf_counter()
        times = [time.perf_counter() - initial_time]
        supports = [0]
        inner_loop = [0]
        objective_values = [self.j(u_0, c_0)]
        epsilons = [epsilon]
        lgcg_lazy = 0
        lgcg_total = 0

        u_plus = u_0.copy()
        c_plus = c_0
        phi_numerical = tol + 1
        while phi_numerical > tol:
            # if k > 5:
            #     break

            global_valid = "N/A"

            t = time.perf_counter()
            if len(u_plus.coefficients):
                u_drop, dropped = self.drop_step(u_plus, c_plus)
                dropped_tot += dropped
            else:
                u_drop = u_plus.copy()
            u_coef, c_coef = self.finite_dimensional_step(
                u_drop,
                c_plus,
                self.machine_precision,
                mode="positive",
                optimization="full",
                log_results=log_results,
            )
            if full_trace and len(u_coef.coefficients):
                all_information.append((u_coef, c_coef))
            self.M = float(self.j(u_coef, c_coef) / self.alpha)
            ssn_1_time = time.perf_counter() - t

            t = time.perf_counter()
            parameters, u_ks, radii, old_radii = self.local_merging_update_radii(
                u_coef, c_coef
            )
            c_ks = c_coef
            u_lm, c_lm = u_ks.copy(), c_ks
            radii_time = time.perf_counter() - t

            t = time.perf_counter()
            if self.experiment_type == "Euclidean":
                full_parameters = np.hstack((parameters.flatten(), c_ks))
            else:
                full_parameters = parameters.flatten()
            if log_results and len(full_parameters):
                e_vals = np.linalg.eigvals(self.hess_f_N(full_parameters))
                logging.info(
                    f"Eigenvalues of Hess f_N. min: {np.min(e_vals):.3E}, max: {np.max(e_vals):.3E}"
                )

            if len(u_ks.coefficients):
                # # Plot the true and predicted sources
                # fig, ax = plt.subplots(figsize=(5, 4))
                # for i, x in enumerate(u_coef.support):
                #     ax.add_patch(
                #         plt.Circle(
                #             (x[0], x[1]),
                #             radius=max(0.5, old_radii[i]),
                #             color="blue",
                #             fill=False,
                #             alpha=0.5,
                #         )
                #     )
                #     plt.plot([x[0]], [x[1]], "o", c="b", markersize=2)
                # for i, x in enumerate(u_ks.support):
                #     plt.plot([x[0]], [x[1]], "o", c="r", markersize=2)
                # ax.set_xlim(self.Omega[0][0], self.Omega[0][1])
                # ax.set_ylim(self.Omega[1][0], self.Omega[1][1])
                # ax.set_xlabel("True and predicted sources")
                # plt.show()
                # if len(u_ks.support) > 1:
                #     for i, x in enumerate(u_ks.support):
                #         if (
                #             np.min(
                #                 np.linalg.norm(
                #                     u_ks.support[
                #                         np.array(
                #                             [
                #                                 _
                #                                 for _ in range(len(u_ks.coefficients))
                #                                 if _ != i
                #                             ]
                #                         )
                #                     ]
                #                     - x,
                #                     axis=1,
                #                 )
                #             )
                #             < 1
                #         ):
                #             # Plot dual variable
                #             p_coef = self.p(u_coef, c_coef)
                #             P = lambda x: np.abs(p_coef(x))
                #             fig, ax = plt.subplots(figsize=(5, 4))
                #             a = np.linspace(x[0] - 1, x[0] + 1, 100)
                #             b = np.linspace(x[1] - 1, x[1] + 1, 100)
                #             B, D = np.meshgrid(a, b)
                #             vals = np.array(
                #                 [
                #                     P(np.array([[x_1, x_2]]))
                #                     for x_1, x_2 in zip(B.flatten(), D.flatten())
                #                 ]
                #             ).reshape((len(a), len(a)))
                #             cb = ax.contourf(B, D, vals, levels=100)
                #             fig.colorbar(cb, ax=ax)
                #             for i, x in enumerate(u_coef.support):
                #                 ax.add_patch(
                #                     plt.Circle(
                #                         (x[0], x[1]),
                #                         radius=old_radii[i],
                #                         color="r",
                #                         fill=False,
                #                         alpha=0.5,
                #                     )
                #                 )
                #                 # plt.plot([x[0]], [x[1]], "o", c="r", markersize=2)
                #             ax.set_xlim(x[0] - 1, x[0] + 1)
                #             ax.set_ylim(x[1] - 1, x[1] + 1)
                #             plt.show()
                #             break
                full_parameters_new, newton_information = self.newton_step(
                    k=k,
                    params=full_parameters,
                    support=len(u_ks.support),
                    mode=inner_mode,
                    log_results=log_results,
                    full_trace=full_trace,
                )
                if full_trace:
                    all_information.append(newton_information)
                if self.experiment_type == "Euclidean":
                    parameters = full_parameters_new[:-1].reshape(parameters.shape)
                    c_ks = full_parameters_new[-1]
                else:
                    parameters = full_parameters_new.reshape(parameters.shape)
                    c_ks = 0
                u_ks = Measure(matrix=parameters)

            newton_time = time.perf_counter() - t
            t = time.perf_counter()
            all_iterates = [
                (u_ks, c_ks),
                (u_coef, c_coef),
                (u_lm, c_lm),
            ]
            iterate_values = [self.j(*iterate) for iterate in all_iterates]
            choice_index = np.nanargmin(iterate_values)
            u, c = all_iterates[choice_index][0].copy(), all_iterates[choice_index][1]
            u, c = self.finite_dimensional_step(
                u,
                c,
                self.machine_precision,
                mode="positive",
                optimization="full",
                log_results=log_results,
            )
            if full_trace:
                all_information.append((u, c))
            p_u = self.p(u, c)
            q_u = self.g(u.coefficients) - u.duality_pairing(p_u)
            ssn_2_time = time.perf_counter() - t

            t = time.perf_counter()
            u_plus, epsilon, global_valid, phi_numerical = self.lgcg_step(
                p_u, u, c, epsilon, q_u, radii, mode, temperature, log_results
            )
            c_plus = c
            lgcg_lazy += int(global_valid)
            lgcg_total += 1
            lgcg_time = time.perf_counter() - t

            # # Plot dual variable
            # P = lambda x: np.abs(p_u(x))
            # a = np.arange(self.Omega[0][0], self.Omega[0][1], 0.5)
            # B, D = np.meshgrid(a, a)
            # vals = np.array(
            #     [
            #         P(np.array([[x_1, x_2]]))
            #         for x_1, x_2 in zip(B.flatten(), D.flatten())
            #     ]
            # ).reshape((len(a), len(a)))
            # plt.contourf(B, D, vals, levels=100)
            # plt.colorbar()
            # # for i, x in enumerate(true_sources):
            # #     if i:
            # #         plt.plot([x[0]], [x[1]], "P", c="r", markersize=10)
            # #     else:
            # #         plt.plot([x[0]], [x[1]], "P", c="r", markersize=10, label="True sources")
            # for i, x in enumerate(u.support):
            #     if i:
            #         plt.plot([x[0]], [x[1]], "o", c="r")
            #     else:
            #         plt.plot([x[0]], [x[1]], "o", c="r", label="Optimal support")
            # # plt.legend()
            # # plt.savefig(results_dir / "optimal_dual_certificate.png", bbox_inches="tight")
            # # plt.close()
            # plt.show()

            times.append(time.perf_counter() - initial_time)
            supports.append(len(u.support))
            inner_loop.append(0)
            objective_values.append(self.j(u, c))
            epsilons.append(epsilon)
            if log_results:
                logging.info(
                    f"{k}: choice: {choice_index}, lazy: {global_valid}, support: {len(u.support)}, epsilon: {epsilon:.3E}, criterion: {phi_numerical:.3E}, c_raw: {self.C_raw}, objective: {self.j(u, c):.14E}"
                )
                logging.info(
                    f"ssn_1: {ssn_1_time:.3E}, radii: {radii_time:.3E}, newton: {newton_time:.3E}, ssn_2: {ssn_2_time:.3E}, lgcg: {lgcg_time:.3E}"
                )
                logging.info(
                    "============================================================================================="
                )
            k += 1

        logging.info(
            f"NLGCG converged in {times[-1]:.3E} seconds with sparsity {len(u.support)} to objective value {objective_values[-1]:.14E}"
        )

        return (
            u,
            c,
            times,
            supports,
            inner_loop,
            lgcg_lazy,
            lgcg_total,
            objective_values,
            dropped_tot,
            epsilons,
            all_information,
        )
