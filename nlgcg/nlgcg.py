import numpy as np
import logging
import time
import jax.numpy as jnp
import jax
import jaxlib
from typing import Callable, Union
from sklearn.utils import gen_batches
from lib.default_values import *
from lib.ssn import SSN
from lib.measure import Measure

jax.config.update("jax_enable_x64", True)

logging.basicConfig(
    level=logging.DEBUG,
)


class NLGCG:
    def __init__(
        self,
        target: np.ndarray,
        kernel: Callable,
        g: Callable,  # Penalty function
        f: Callable,  # Diligence function
        grad_f: Callable,
        hess_f: Callable,
        j: Callable,  # Objective
        j_N: Callable,  # parameterized objective
        p: Callable,  # Dual variable
        grad_p: Callable,
        hess_p: Callable,
        grad_j_N: Callable,
        hess_j_N: Callable,
        alpha: float,
        Omega: np.ndarray,
        global_search_resolution: int,
        constant_dim: int,
        kernel_dim: int,
        M: float = 1e6,
        C_0: float = 1,
        dual_variable_goodness: float = 0.5,
        beta: float = 0.5,
        armijo_constant: float = 1e-4,
        newton_p: float = 1e-1,
        descent_constant: float = 1e-6,
        lazy_sample: int = 1000,
        exact_sample: int = 100000,
        max_inner_loop: int = 20,
        ssn_steps: int = 100,
    ) -> None:
        self.target = target
        self.kernel = kernel
        self.p = p
        self.grad_p = grad_p
        self.hess_p = hess_p
        self.alpha = alpha
        self.g = g
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.Omega = Omega  # Example [[0,1],[1,2]] for [0,1]x[1,2]
        self.max_radius = 1
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
        self.hess_j_N = hess_j_N
        self.constant_dim = constant_dim
        self.kernel_dim = kernel_dim
        self.machine_precision = 1e-12
        self.stop_search = 5
        self.batching_constant = 2e8
        self.dual_variable_goodness = dual_variable_goodness
        self.beta = beta  # For Armijo rule
        self.armijo_constant = armijo_constant  # For Armijo rule
        self.newton_p = 2 + newton_p  # For Newton step acceptance condition
        self.descent_constant = descent_constant  # For Newton step acceptance condition
        self.lazy_sample = lazy_sample
        self.exact_sample = exact_sample
        self.max_inner_loop = max_inner_loop
        self.ssn_steps = ssn_steps

    def project_into_domain(
        self, x: Union[np.ndarray, jaxlib.xla_extension.ArrayImpl]
    ) -> np.ndarray:
        # Project an array into domain, parallelized
        x = np.array(x).copy()
        if x.shape[1] > self.Omega.shape[0]:
            # Also project the weights
            local_Omega = np.vstack((np.array([-self.M, self.M]), self.Omega))
            variance_dimension = 1
        else:
            local_Omega = self.Omega
            variance_dimension = 0
        for i, bounds in zip(range(x.shape[1]), local_Omega):
            column = x[:, i].copy()
            if i != variance_dimension:
                x[:, i] = np.clip(column, bounds[0], bounds[1])
        return x

    def sample_domain(self, size: int, u: Measure = Measure()) -> np.ndarray:
        # Generate a uniform sample of shape (size,domain.shape[0]) in the given domain
        columns = []
        for i, bounds in enumerate(self.Omega):
            if i == 0:
                if len(u.coefficients):
                    distribution_parameter = max(u.support[:, 0].max(), bounds[1])
                else:
                    distribution_parameter = bounds[1]
                columns.append(
                    np.random.exponential(scale=distribution_parameter, size=(size, 1))
                    + bounds[0]
                )
            else:
                columns.append(
                    np.random.sample((size, 1)) * (bounds[1] - bounds[0]) + bounds[0]
                )
        sample = np.concatenate(columns, axis=1)
        return sample

    def get_grid(self, u: Measure) -> np.ndarray:
        grid = (
            np.array(
                np.meshgrid(
                    *(
                        np.linspace(bound[0], bound[1], self.global_search_resolution)[
                            1:
                        ]
                        for bound in self.Omega
                    )
                )
            )
            .reshape(len(self.Omega), -1)
            .T
        )
        if len(u.coefficients):
            grid = np.vstack([grid, u.support])
        return grid

    def global_search(
        self,
        u: Measure,
        c: float,
        epsilon: float,
        q_u: float,
        p_u: Callable,
        radius: float,
        mode: str = "deterministic",
    ) -> tuple:
        p_norm = lambda x: np.abs(np.array(p_u(x)))

        if mode == "stochastic":
            grid = self.sample_domain(self.lazy_sample, u)
            if len(u.support):
                grid = np.vstack((grid, u.support))
            grid_vals = p_norm(grid)
            max_ind = np.argmax(grid_vals)
            best_val = grid_vals[max_ind]
            best_point = grid[max_ind]
            phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
            if phi_val >= self.M * epsilon:
                success = True  # Found a desired point
            else:
                lazy_grid = grid.copy()
                lazy_grid_vals = grid_vals.copy()
                grid = self.sample_domain(self.exact_sample, u)
                grid_vals = p_norm(grid)
                grid = np.vstack((grid, lazy_grid))
                grid_vals = np.hstack((grid_vals, lazy_grid_vals))
                max_ind = np.argmax(grid_vals)
                best_val = grid_vals[max_ind]
                best_point = grid[max_ind]
                phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
                if phi_val >= self.M * epsilon:
                    success = True  # Found a desired point
                else:
                    success = False
        else:
            grad_p = self.grad_p(u, c)
            hess_p = self.hess_p(u, c)
            grid = self.get_grid(u)
            optimize_grid = np.array([True] * grid.shape[0])
            grid_vals = p_norm(grid)
            max_ind = np.argmax(grid_vals)
            best_val = grid_vals[max_ind]
            best_point = grid[max_ind].copy()
            phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
            point_steps = 0
            while point_steps < self.stop_search and phi_val < self.M * epsilon:
                batching_factor = (
                    len(self.target) * self.Omega.shape[0] * (self.Omega.shape[0] + 1)
                    + 2 * self.Omega.shape[0]
                    + 1
                )
                batch_size = int(self.batching_constant // batching_factor)
                for batch in gen_batches(len(grid), batch_size):
                    if phi_val >= self.M * epsilon:
                        break
                    batch_points = grid[batch]
                    optimize_batch = optimize_grid[batch]
                    batch_vals = grid_vals[batch]
                    new_points_plus = np.zeros(batch_points.shape)
                    new_points_minus = np.zeros(batch_points.shape)
                    gradients = grad_p(batch_points)
                    hessians = hess_p(batch_points)
                    for i, (point, gradient, hessian, optimize) in enumerate(
                        zip(batch_points, gradients, hessians, optimize_batch)
                    ):
                        if optimize:
                            try:
                                d = np.linalg.solve(hessian, -gradient)  # Newton step
                            except np.linalg.LinAlgError:
                                d = 0.1 * gradient
                        else:
                            d = np.zeros_like(point)
                        new_points_plus[i] = point + d
                        new_points_minus[i] = point - d
                    # projected_new_points_plus = new_points_plus.copy()
                    projected_new_points_plus = self.project_into_domain(
                        new_points_plus
                    ).copy()
                    # projected_new_points_minus = new_points_minus.copy()
                    projected_new_points_minus = self.project_into_domain(
                        new_points_minus
                    ).copy()

                    p_vals_plus = p_norm(projected_new_points_plus)
                    p_vals_plus_invalid = np.isnan(p_vals_plus)
                    p_vals_plus[p_vals_plus_invalid] = batch_vals[p_vals_plus_invalid]
                    projected_new_points_plus[p_vals_plus_invalid] = batch_points[
                        p_vals_plus_invalid
                    ]

                    p_vals_minus = p_norm(projected_new_points_minus)
                    p_vals_minus_invalid = np.isnan(p_vals_minus)
                    p_vals_minus[p_vals_minus_invalid] = batch_vals[
                        p_vals_minus_invalid
                    ]
                    projected_new_points_minus[p_vals_minus_invalid] = batch_points[
                        p_vals_minus_invalid
                    ]

                    plus_bigges_index = p_vals_plus > p_vals_minus
                    p_vals = np.maximum(p_vals_plus, p_vals_minus)
                    projected_new_points = np.zeros(batch_points.shape)
                    projected_new_points[plus_bigges_index] = projected_new_points_plus[
                        plus_bigges_index
                    ]
                    projected_new_points[~plus_bigges_index] = (
                        projected_new_points_minus[~plus_bigges_index]
                    )
                    grid[batch] = projected_new_points
                    grid_vals[batch] = p_vals
                    optimize_grid[batch] = optimize_batch & ~(
                        p_vals_plus_invalid & p_vals_minus_invalid
                    )

                    max_ind = np.argmax(p_vals)
                    max_val = p_vals[max_ind]
                    if max_val > best_val:
                        best_val = max_val
                        best_point = projected_new_points[max_ind].copy()
                        phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
                        if phi_val >= self.M * epsilon:
                            break

                    del new_points_plus
                    del new_points_minus
                    del projected_new_points
                    del gradients
                    del hessians
                    del p_vals

                point_steps += 1

            if phi_val >= self.M * epsilon:
                success = True  # Found a desired point
            else:
                success = False
        valid_indices = np.where(
            np.logical_and(
                grid_vals >= self.alpha,
                grid_vals
                > self.alpha + (best_val - self.alpha) * self.dual_variable_goodness,
            )
        )[0]
        valid_grid = grid[valid_indices]
        valid_grid_vals = grid_vals[valid_indices]
        order_indices = np.argsort(valid_grid_vals)[::-1]
        order_grid = valid_grid[order_indices][:10]

        if not len(order_grid):
            found_points = np.array([best_point])
        else:
            found_points = np.array([order_grid[0]])
        for point in order_grid[1:]:
            local_distances = np.linalg.norm(found_points - point, axis=1)
            if np.all(local_distances > 2 * radius):
                found_points = np.vstack((found_points, point))
        return (
            best_point,
            found_points,
            success,
        )

    def finite_dimensional_step(
        self,
        u: Measure,
        c: float,
        Psi: float,
        mode: str = "unconstrained",
        optimization: str = "full",
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
        ssn_solution = ssn.solve(tol=Psi, u_0=u_0)
        raw_Psi = ssn.Psi(ssn_solution)
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
        return u_plus, c_plus, raw_Psi

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
            return np.array([[]]), u, []
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
        if len(cluster_coefs) == len(u.coefficients):
            # No merging has happened
            return u.to_matrix(), u, radii
        else:
            u_plus = Measure(support=cluster_points, coefficients=cluster_coefs)
            radii = self.compute_radii(u_plus, c)
            return u_plus.to_matrix(), u_plus, radii

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
        if jnp.isnan(j_N_new):
            j_N_new = j_N_init + 1
        while j_N_new - j_N_init > sigma * desired_descent:
            if choice == "Grad" and sigma * desired_descent > -self.machine_precision:
                return full_parameters.copy(), -1
            elif sigma < self.machine_precision:
                return full_parameters.copy(), -1
            sigma *= self.beta
            full_parameters_new = full_parameters + (sigma * direction)
            j_N_new = self.j_N(full_parameters_new)
            if jnp.isnan(j_N_new):
                j_N_new = j_N_init + 1
        return np.array(full_parameters_new), sigma

    def globalized_newton_step(self, parameters: np.ndarray, c: float) -> tuple:
        full_parameters = np.hstack((parameters.flatten(), np.array([c])))
        grad_j_N_z = self.grad_j_N(full_parameters)
        hess_j_N_z = self.hess_j_N(full_parameters)
        try:
            update_direction = np.linalg.solve(hess_j_N_z, -grad_j_N_z)
            update_norm = np.linalg.norm(update_direction)
            grad_direction = -grad_j_N_z @ update_direction
            condition = (
                grad_direction >= self.descent_constant * update_norm**self.newton_p
            )
            if not condition:
                raise np.linalg.LinAlgError("Insufficient descent in Newton direction")
            choice = "Newt"
        except np.linalg.LinAlgError:
            update_direction = -grad_j_N_z.copy()
            choice = "Grad"
        if any(jnp.isnan(update_direction)):
            return parameters, c, "NAN in update direction", 1
        full_parameters_new, sigma = self.armijo(
            full_parameters, update_direction, grad_j_N_z, choice
        )
        c_new = full_parameters_new[-1]
        # parameters_new = self.project_into_domain(
        #     full_parameters_new[:-1].reshape(parameters.shape)
        # )
        parameters_new = full_parameters_new[:-1].reshape(parameters.shape)
        return parameters_new, c_new, choice, sigma

    def lgcg_step(
        self,
        p_u: Callable,
        u: Measure,
        c: float,
        epsilon: float,
        q_u: float,
        radii: np.ndarray,
        mode: str = "deterministic",
    ) -> tuple:
        j_initial = self.j(u, c)
        condition = False
        if len(radii):
            radius = np.max(radii)
        else:
            radius = self.max_radius
        x_k, found_points, global_valid = self.global_search(
            u, c, epsilon, q_u, p_u, radius, mode
        )
        logging.info(x_k)
        Phi = self.M * max((np.abs(p_u(x_k.reshape(1, -1)))[0] - self.alpha), 0) + q_u
        if Phi > q_u:
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
            eta = min(1, Phi / Curv)
            u_plus = u * (1 - eta) + v * eta
            jdiff = self.j(u_plus, c) - j_initial
            if Phi <= Curv:
                expected_decrease = -0.5 * Phi**2 / Curv
            else:
                expected_decrease = 0.5 * Curv - Phi
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
                eta = min(1, Phi / Curv)
                u_plus = u * (1 - eta) + v * eta
                jdiff = self.j(u_plus, c) - j_initial
                if Phi <= Curv:
                    expected_decrease = -0.5 * Phi**2 / Curv
                else:
                    expected_decrease = 0.5 * Curv - Phi
                condition = jdiff <= expected_decrease
            self.C_raw *= 2
            u_plus = previous_u_plus.copy()
        if not global_valid:
            # We have a global maximum x_k
            epsilon = 0.5 * Phi / self.M
        # support = u_plus.support
        # coefficients = u_plus.coefficients
        # keep_indices = np.abs(coefficients) > self.machine_precision
        # u_plus = Measure(
        #     support=support[keep_indices], coefficients=coefficients[keep_indices]
        # )
        # if self.j(u_plus, c) - j_initial < self.machine_precision:
        return u_plus, epsilon, global_valid
        # else:
        # return u, epsilon, "No Step"

    def domain_and_descent_tests(
        self,
        parameters: np.ndarray,
        c: float,
        parameters_new: np.ndarray,
        c_new: float,
        local_M: float,
        choice: str,
    ) -> list:
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
            # projected_points_new = self.project_into_domain(points_new)
            # projection_distance = np.linalg.norm(points_new - projected_points_new)
            # output_bools.append(projection_distance == 0)

        # M test
        output_bools.append(bool(np.linalg.norm(coefs_new, ord=1) <= local_M))

        # Sign test
        bad_signs = np.where(np.sign(coefs_new) != np.sign(coefs))[0]
        if not len(bad_signs):
            output_bools.append(True)
        else:
            output_bools.append(False)
            # logging.info(parameters[bad_signs])
            # logging.info(parameters_new[bad_signs])

        # Absolute descent test
        full_parameters = np.hstack((parameters.flatten(), np.array([c])))
        full_parameters_new = np.hstack((parameters_new.flatten(), np.array([c_new])))
        j_N_diff = self.j_N(full_parameters_new) - self.j_N(full_parameters)
        if choice == "Grad" and j_N_diff >= -self.machine_precision:
            output_bools.append(False)
        elif choice == "Newt" and j_N_diff >= 0:
            output_bools.append(False)
        else:
            output_bools.append(True)

        return output_bools

    def stationarity_descent_test(
        self, parameters: np.ndarray, c: float, epsilon: float, radii: list
    ) -> tuple:
        full_parameters = np.hstack((parameters.flatten(), np.array([c])))
        grad_j_N_z = self.grad_j_N(full_parameters)
        constant_part = np.abs(grad_j_N_z[-1] * c)
        grad_norm = np.linalg.norm(grad_j_N_z)
        grad_j_N_z_matrix = grad_j_N_z[:-1].reshape(parameters.shape)
        grad_points_matrix = grad_j_N_z_matrix[:, 1:]
        points_part = 0
        for i, radius in enumerate(radii):
            points_part += radius * np.linalg.norm(grad_points_matrix[i])
        grad_coefs = grad_j_N_z_matrix[:, 0].flatten()
        coefs = parameters[:, 0].flatten()
        gap = (
            points_part
            + self.M * abs(min(0, np.min(np.multiply(grad_coefs, np.sign(coefs)))))
            + grad_coefs @ coefs
            + constant_part
        )
        if epsilon <= self.C_raw * self.M:
            ineq = gap >= 0.5 * epsilon**2 / self.C_raw
        else:
            ineq = gap >= (2 * self.M * epsilon - self.C_raw * self.M**2) / 2
        return ineq, grad_norm

    def compute_radii(self, u: Measure, c: float) -> list:
        radii = []
        if not len(u.coefficients):
            return radii
        grad_p = self.grad_p(u, c)
        hess_p = self.hess_p(u, c)
        grads = grad_p(u.support)
        hesses = hess_p(u.support)
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
        return radii

    def nlgcg(
        self,
        tol: float,
        max_radius: float,
        drop_frequency: int = 5,
        u_0: Measure = Measure(),
        c_0: float = 0,
        mode: str = "deterministic",
    ) -> tuple:
        self.max_radius = max_radius
        self.M = self.M_0
        self.C_raw = self.C_0
        epsilon = 0.5 * self.j(u_0, c_0) / self.M
        k = 0
        dropped = False
        optimal = False
        dropped_tot = 0
        initial_time = time.time()
        times = [time.time() - initial_time]
        supports = [0]
        inner_loop = [0]
        objective_values = [self.j(u_0, c_0)]
        epsilons = [epsilon]
        lgcg_lazy = 0
        lgcg_total = 0

        u_plus = u_0.copy()
        c_plus = c_0
        while 2 * self.M * epsilon > tol:
            global_valid = "N/A"

            if len(u_plus.coefficients):
                u_drop, dropped = self.drop_step(u_plus, c_plus)
                dropped_tot += dropped
            else:
                u_drop = u_plus.copy()
            u_coef, c_coef, finite_psi = self.finite_dimensional_step(
                u_drop,
                c_plus,
                self.machine_precision,
                mode="positive",
                optimization="full",
            )
            self.M = float(self.j(u_coef, c_coef) / self.alpha)

            parameters, u_ks, radii = self.local_merging_update_radii(u_coef, c_coef)
            c_ks = c_coef
            local_M = float(self.j(u_ks, c_ks) / self.alpha)
            epsilon_ks = (
                epsilon + 0.5 * (self.j(u_ks, c_ks) - self.j(u_coef, c_coef)) / self.M
            )
            p_u_ks = self.p(u_ks, c_ks)
            q_u_ks = self.g(u_ks.coefficients) - u_ks.duality_pairing(p_u_ks)
            u_ks_gcg, c_ks_gcg = u_ks.copy(), c_ks
            u_ks_new, c_ks_new = u_ks.copy(), c_ks
            u_lm, c_lm = u_ks.copy(), c_ks

            s = 1
            sigma = 0
            if len(u_ks.coefficients):
                logging.info(
                    f"{k}, {0}: Globalization: NotA, support: {len(u_ks.support)}, c_raw: {self.C_raw:.2E}, sigma: {sigma:.2E}, epsilon: {epsilon_ks:.2E}, criterion: {2*self.M*epsilon_ks:.2E}, objective: {self.j_N(np.hstack((parameters.flatten(), np.array([c_ks])))):.12E}"
                )
            while len(u_ks.coefficients):
                # Inner loop

                # Check optimality and stationarity descent
                stationarity_test, grad_norm = self.stationarity_descent_test(
                    parameters, c_ks, epsilon_ks, radii
                )
                if not stationarity_test:
                    u_ks_gcg, epsilon_ks, global_valid = self.lgcg_step(
                        p_u_ks, u_ks, c_ks, epsilon_ks, q_u_ks, radii, mode
                    )
                    c_ks_gcg = c_ks
                    lgcg_lazy += int(global_valid)
                    lgcg_total += 1
                    stationarity_test, grad_norm = self.stationarity_descent_test(
                        parameters, c_ks, epsilon_ks, radii
                    )
                else:
                    global_valid = "N/A"
                if not stationarity_test:
                    times.append(time.time() - initial_time)
                    supports.append(len(u_ks.support))
                    inner_loop.append(1)
                    objective_values.append(self.j(u_ks, c_ks))
                    epsilons.append(epsilon_ks)
                    logging.info(
                        f"{k}, {s}: Globalization: NotA, support: {len(u_ks.support)}, c_raw: {self.C_raw:.2E}, sigma: {sigma:.2E}, epsilon: {epsilon_ks:.2E}, criterion: {2*self.M*epsilon_ks:.2E}, objective: {self.j_N(np.hstack((parameters.flatten(), np.array([c_ks])))):.12E}"
                    )
                    logging.info(
                        f"Stationarity descent test: {stationarity_test}, grad_norm: {grad_norm:.3E}"
                    )
                    break
                if (
                    min(2 * local_M * epsilon_ks, grad_norm) <= tol
                ):  # Optimality reached
                    optimal = True
                    break
                else:
                    optimal = False

                # Newton step
                parameters_new, c_ks_new, newton_choice, sigma = (
                    self.globalized_newton_step(parameters, c_ks)
                )
                u_ks_new = Measure(matrix=parameters_new)

                # Check validity of the newton step
                domain_tests = self.domain_and_descent_tests(
                    parameters, c_ks, parameters_new, c_ks_new, local_M, newton_choice
                )
                if not all(domain_tests):
                    times.append(time.time() - initial_time)
                    supports.append(len(u_ks_new.support))
                    inner_loop.append(1)
                    objective_values.append(self.j(u_ks_new, c_ks_new))
                    epsilons.append(epsilon_ks)
                    logging.info(
                        f"{k}, {s}: Globalization: {newton_choice}, support: {len(u_ks_new.support)}, c_raw: {self.C_raw:.2E}, sigma: {sigma:.2E}, epsilon: {epsilon_ks:.2E}, criterion: {2*self.M*epsilon_ks:.2E}, objective: {self.j_N(np.hstack((parameters_new.flatten(), np.array([c_ks_new])))):.12E}"
                    )
                    logging.info(f"Domain tests: {domain_tests}")
                    break

                # Perform drop and local merging
                if s % drop_frequency == 0:
                    u_ks_drop, dropped = self.drop_step(u_ks_new, c_ks_new)
                    dropped_tot += dropped
                    parameters, u_ks, radii = self.local_merging_update_radii(
                        u_ks_drop, c_ks_new
                    )
                    c_ks = c_ks_new
                    epsilon_ks = (
                        epsilon_ks
                        + 0.5 * (self.j(u_ks, c_ks) - self.j(u_ks_drop, c_ks)) / self.M
                    )
                else:
                    u_ks = u_ks_new.copy()
                    c_ks = c_ks_new
                    parameters = u_ks.to_matrix()
                local_M = float(self.j(u_ks, c_ks) / self.alpha)
                p_u_ks = self.p(u_ks, c_ks)
                q_u_ks = self.g(u_ks.coefficients) - u_ks.duality_pairing(p_u_ks)

                # Monitor statistics
                times.append(time.time() - initial_time)
                supports.append(len(u_ks.support))
                inner_loop.append(1)
                objective_values.append(self.j(u_ks, c_ks))
                epsilons.append(epsilon_ks)
                logging.info(
                    f"{k}, {s}: Globalization: {newton_choice}, support: {len(u_ks.support)}, c_raw: {self.C_raw:.2E}, sigma: {sigma:.2E}, epsilon: {epsilon_ks:.2E}, criterion: {2*self.M*epsilon_ks:.2E}, objective: {self.j_N(np.hstack((parameters.flatten(), np.array([c_ks])))):.12E}"
                )
                s += 1
                if s == self.max_inner_loop:
                    logging.info(
                        "Too many iterations in the inner loop, stopping the process"
                    )
                    break

            if optimal:
                u = u_ks.copy()
                c = c_ks
                times.append(time.time() - initial_time)
                supports.append(len(u.support))
                inner_loop.append(0)
                objective_values.append(self.j(u, c))
                epsilons.append(epsilon_ks)
                logging.info(
                    f"{k} optimal: support: {len(u.support)}, epsilon: {epsilon_ks:.2E}, criterion: {2*self.M*epsilon_ks:.2E}, grad_norm: {grad_norm:.2E} c_raw: {self.C_raw}, objective: {self.j(u, c):.12E}"
                )
                logging.info(
                    "============================================================================================="
                )
                break

            all_iterates = [
                (u_ks, c_ks),
                (u_ks_new, c_ks_new),
                (u_ks_gcg, c_ks_gcg),
                (u_coef, c_coef),
                (u_lm, c_lm),
            ]
            iterate_values = [self.j(*iterate) for iterate in all_iterates]
            choice_index = np.argmin(iterate_values)
            u, c = all_iterates[choice_index][0].copy(), all_iterates[choice_index][1]
            # if True:
            #     # Only for quadratic loss, set optimal constant
            #     c = float(-np.mean(u.duality_pairing(self.kernel) - self.target))
            u, c, finite_psi = self.finite_dimensional_step(
                u,
                c,
                self.machine_precision,
                mode="unconstrained",
                optimization="constant",
            )
            p_u = self.p(u, c)
            q_u = self.g(u.coefficients) - u.duality_pairing(p_u)

            u_plus, epsilon, global_valid = self.lgcg_step(
                p_u, u, c, epsilon, q_u, radii, mode
            )
            c_plus = c
            lgcg_lazy += int(global_valid)
            lgcg_total += 1

            times.append(time.time() - initial_time)
            supports.append(len(u.support))
            inner_loop.append(0)
            objective_values.append(self.j(u, c))
            epsilons.append(epsilon)
            logging.info(
                f"{k}: choice: {choice_index}, lazy: {global_valid}, support: {u.support}, epsilon: {epsilon:.3E}, criterion: {2*self.M*epsilon:.3E}, c_raw: {self.C_raw}, objective: {self.j(u, c):.12E}"
            )
            logging.info(
                "============================================================================================="
            )
            k += 1

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
        )
