# Implementation of global search methods for the solution of the non-convex problem in NLGCG

import numpy as np
import scipy as sp
import logging
import time
import jax
from typing import Callable, Union
from sklearn.utils import gen_batches
from lib.measure import Measure

jax.config.update("jax_enable_x64", True)

logging.basicConfig(
    level=logging.DEBUG,
)


class GlobalSearch:
    def __init__(
        self,
        Omega: np.ndarray,
        M: float,
        global_search_resolution: int,
        dual_variable_goodness: float,
        alpha: float,
        len_target: int,
        grad_p: Callable,
        hess_p: Callable,
        mode: str = "stochastic",
        max_found_points: int = 100,
    ) -> None:
        self.Omega = Omega
        self.M = M
        self.global_search_resolution = global_search_resolution
        self.batching_constant = 2e8
        self.len_target = len_target
        self.dual_variable_goodness = dual_variable_goodness
        self.alpha = alpha
        self.grad_p = grad_p
        self.hess_p = hess_p
        self.mode = mode
        self.max_found_points = max_found_points
        if mode == "deterministic":
            self.get_grid = self.deterministic_grid
            self.stop_search = 5
        elif mode == "stochastic":
            self.get_grid = self.stochastic_grid
            self.stop_search = 3

    def project_into_domain(self, x: np.ndarray) -> np.ndarray:
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
                # # Sample sigma exponentially
                # if len(u.coefficients):
                #     distribution_parameter = max(u.support[:, 0].max(), bounds[1])
                # else:
                #     distribution_parameter = bounds[1]
                # columns.append(
                #     np.random.exponential(scale=distribution_parameter, size=(size, 1))
                #     + bounds[0]
                # )
                columns.append(
                    np.random.sample((size, 1)) * (bounds[1] - bounds[0]) + bounds[0]
                )
            else:
                columns.append(
                    np.random.sample((size, 1)) * (bounds[1] - bounds[0]) + bounds[0]
                )
        sample = np.concatenate(columns, axis=1)
        return sample

    def deterministic_grid(
        self,
        u: Measure,
        p_norm: Callable,
        grad_p_u: Callable,
        q_u: float,
        epsilon: float,
        radius: float,
    ) -> np.ndarray:
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
        grid_vals = p_norm(grid)
        max_ind = np.argmax(grid_vals)
        best_val = grid_vals[max_ind]
        best_point = grid[max_ind].copy()
        phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
        success = phi_val >= epsilon
        return success, grid, grid_vals, best_point, best_val

    def stochastic_grid_old(
        self,
        u: Measure,
        p_norm: Callable,
        grad_p_u: Callable,
        q_u: float,
        epsilon: float,
        radius: float,
    ) -> tuple:
        grid = self.sample_domain(int(1e4), u)
        grid_vals = p_norm(grid)
        first_order_indices = np.argsort(grid_vals)[::-1][:100]
        best_val = grid_vals[first_order_indices[0]]
        best_point = grid[first_order_indices[0]].copy()
        phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
        success = phi_val >= epsilon
        if success:
            return success, grid, grid_vals, best_point, best_val

        refinement_samples = 10
        sample_start_points = np.repeat(
            grid[first_order_indices],
            [refinement_samples] * len(first_order_indices),
            axis=0,
        )
        local_updates = np.tile(
            np.random.multivariate_normal(
                mean=np.zeros(self.Omega.shape[0]),
                cov=radius * np.eye(self.Omega.shape[0]),
                size=refinement_samples,
            ),
            (len(first_order_indices), 1),
        )
        grid_raw = sample_start_points + local_updates
        grid = grid_raw[grid_raw[:, 0] > 0]  # check for pos variance
        grid_vals = p_norm(grid)
        second_order_indices = np.argsort(grid_vals)[::-1][:100]
        grid = grid[second_order_indices]
        grid_vals = grid_vals[second_order_indices]
        if len(u.coefficients):
            grid = np.vstack([grid, u.support])
            grid_vals = np.hstack((grid_vals, p_norm(u.support)))
        best_ind = np.argmax(grid_vals)
        best_val = grid_vals[best_ind]
        best_point = grid[best_ind].copy()
        phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
        success = phi_val >= epsilon
        return success, grid, grid_vals, best_point, best_val

    def determine_step_size_sampling(self, mesh: float) -> float:
        # Determine variance needed for sampled points to be within the mesh at given probability
        probability_interval = [0.79, 0.799]
        incumbent_variance = (mesh**2) / self.Omega.shape[1]  # initial value
        variance_upper = incumbent_variance
        variance_lower = 2 * incumbent_variance
        incumbent_probability = sp.special.gammainc(
            self.Omega.shape[1] / 2, 0.5 * mesh**2 / incumbent_variance
        )
        while (incumbent_probability < probability_interval[0]) or (
            incumbent_probability > probability_interval[1]
        ):
            if incumbent_probability < probability_interval[0]:
                variance_upper = incumbent_variance
            elif incumbent_probability > probability_interval[1]:
                variance_lower = incumbent_variance
            if variance_upper <= variance_lower:
                # Upper bound not yet found
                incumbent_variance /= 2
            else:
                # Binary search
                incumbent_variance = (variance_lower + variance_upper) / 2
            incumbent_probability = sp.special.gammainc(
                self.Omega.shape[1] / 2, 0.5 * mesh**2 / incumbent_variance
            )
        return incumbent_variance

    def separate_points(
        self,
        points: np.ndarray,
        vals: np.ndarray,
        mesh: float,
        best_val: float,
        lipschitzs: float,
    ) -> list:
        # Remove points based on overlap and current best value
        relevant_indices = vals + lipschitzs * mesh > best_val
        filtered_points = points[relevant_indices]
        filtered_vals = vals[relevant_indices]
        filtered_lipschitzs = lipschitzs[relevant_indices]

        sorted_indices = np.argsort(filtered_vals)[::-1]
        sorted_points = filtered_points[sorted_indices]
        sorted_vals = filtered_vals[sorted_indices]
        sorted_lipschitzs = filtered_lipschitzs[sorted_indices]
        separated_points = np.array([sorted_points[0]])
        separated_vals = np.array([sorted_vals[0]])
        separated_lipschitzs = np.array([sorted_lipschitzs[0]])
        for x, val, lipschitz in zip(
            sorted_points[1:], sorted_vals[1:], sorted_lipschitzs[1:]
        ):
            distances = np.linalg.norm(separated_points - x, axis=1)
            if all(distances > mesh):
                separated_points = np.vstack((separated_points, [x]))
                separated_vals = np.append(separated_vals, val)
                separated_lipschitzs = np.append(separated_lipschitzs, lipschitz)
            # if len(filtered_vals) >= self.max_chains:
            #     break
        return separated_points, separated_vals, separated_lipschitzs

    def stochastic_grid(
        self,
        u: Measure,
        p_norm: Callable,
        grad_p_u: Callable,
        q_u: float,
        epsilon: float,
        radius: float,
    ) -> tuple:
        success = False
        N = 90  # sample size 99% confidence
        mesh_reduction_factor = 20
        mesh_reduction = 1 / (mesh_reduction_factor ** (1 / self.Omega.shape[0]))
        mesh = max([bound[1] - bound[0] for bound in self.Omega])

        grid = self.sample_domain(1, u)
        grid_vals = p_norm(grid)
        lipschitzs = np.linalg.norm(grad_p_u(grid), axis=1)
        best_index = np.argmax(grid_vals)
        best_val = grid_vals[best_index].copy()
        best_point = grid[best_index].copy()

        while mesh > 1e-2:
            now = time.time()

            grid, grid_vals, grid_lipschitzs = self.separate_points(
                grid, grid_vals, mesh, best_val, lipschitzs
            )
            step_size = self.determine_step_size_sampling(mesh)
            mesh = mesh_reduction * mesh

            # New points
            sample_updates = np.random.multivariate_normal(
                mean=np.zeros(self.Omega.shape[0]),
                cov=step_size * np.eye(self.Omega.shape[0]),
                size=N * len(grid_vals),
            )
            starting_points = np.repeat(grid, [N] * len(grid_vals), axis=0)
            points_new_raw = starting_points + sample_updates  # new trial points
            points_new = self.project_into_domain(points_new_raw)

            # Compute the P values of trial points
            points_new_vals = p_norm(points_new)
            points_new_lipschitzs = np.linalg.norm(grad_p_u(points_new), axis=1)

            # Update the best value
            best_index = np.argmax(points_new_vals)
            tentative_best_val = points_new_vals[best_index].copy()
            if tentative_best_val > best_val:
                best_val = tentative_best_val
                best_point = points_new[best_index].copy()
                phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
                success = phi_val >= epsilon
                if success:
                    if len(u.coefficients):
                        grid = np.vstack([grid, u.support])
                        grid_vals = np.hstack((grid_vals, p_norm(u.support)))
                    return success, grid, grid_vals, best_point, best_val

            # # Choose promising new points
            # promising_indices = x_new_vals > best_val - lipschitz * mesh

            # Add accepted new points to active tuples
            grid = np.vstack((grid, points_new))
            grid_vals = np.append(grid_vals, points_new_vals)
            lipschitzs = np.append(grid_lipschitzs, points_new_lipschitzs)

            logging.info(
                f"Sample step complete. {len(grid_vals)} points, mesh: {mesh:.3E}, time: {time.time() - now:.3E}"
            )

        if len(u.coefficients):
            grid = np.vstack([grid, u.support])
            grid_vals = np.hstack((grid_vals, p_norm(u.support)))
        return success, grid, grid_vals, best_point, best_val

    def post_process_global_search(
        self,
        success: bool,
        grid: np.ndarray,
        grid_vals: np.ndarray,
        best_point: np.ndarray,
        best_val: float,
        radius: float,
    ) -> tuple:
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
        order_grid = valid_grid[order_indices][: self.max_found_points]

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

    def solve(
        self,
        u: Measure,
        c: float,
        epsilon: float,
        q_u: float,
        p_u: Callable,
        radius: float,
    ) -> tuple:
        p_norm = lambda x: np.abs(np.array(np.nan_to_num(p_u(x))))
        grad_p_u = self.grad_p(u, c)
        success, grid, grid_vals, best_point, best_val = self.get_grid(
            u, p_norm, grad_p_u, q_u, epsilon, radius
        )
        if success:
            return self.post_process_global_search(
                success, grid, grid_vals, best_point, best_val, radius
            )

        # Optimize the grid with Newton steps
        grad_p = self.grad_p(u, c)
        hess_p = self.hess_p(u, c)
        optimize_grid = np.array([True] * grid.shape[0])
        point_steps = 0
        while point_steps < self.stop_search:
            batching_factor = (
                self.len_target * self.Omega.shape[0] * (self.Omega.shape[0] + 1)
                + 2 * self.Omega.shape[0]
                + 1
            )
            batch_size = int(self.batching_constant // batching_factor)
            for batch in gen_batches(len(grid), batch_size):
                optimize_batch = optimize_grid[batch]
                batch_points = grid[batch][optimize_batch]
                batch_vals = grid_vals[batch][optimize_batch]
                new_points_plus = np.zeros(batch_points.shape)
                new_points_minus = np.zeros(batch_points.shape)
                gradients = grad_p(batch_points)
                hessians = hess_p(batch_points)
                for i, (point, gradient, hessian) in enumerate(
                    zip(batch_points, gradients, hessians)
                ):
                    try:
                        d = np.linalg.solve(hessian, -gradient)  # Newton step
                    except np.linalg.LinAlgError:
                        d = 0.1 * gradient
                    new_points_plus[i] = point + d
                    new_points_minus[i] = point - d
                projected_new_points_plus = self.project_into_domain(
                    new_points_plus
                ).copy()
                projected_new_points_minus = self.project_into_domain(
                    new_points_minus
                ).copy()
                p_vals_plus = p_norm(projected_new_points_plus)
                p_vals_minus = p_norm(projected_new_points_minus)
                plus_bigges_index = (p_vals_plus > p_vals_minus) & (
                    p_vals_plus > batch_vals
                )
                minus_bigges_index = (p_vals_minus > p_vals_plus) & (
                    p_vals_minus > batch_vals
                )
                keep_optimizing_index = plus_bigges_index | minus_bigges_index
                p_vals = np.maximum(np.maximum(p_vals_plus, p_vals_minus), batch_vals)
                projected_new_points = batch_points.copy()
                projected_new_points[plus_bigges_index] = projected_new_points_plus[
                    plus_bigges_index
                ]
                projected_new_points[minus_bigges_index] = projected_new_points_minus[
                    minus_bigges_index
                ]
                grid[batch][optimize_batch] = projected_new_points
                grid_vals[batch][optimize_batch] = p_vals
                optimize_batch = optimize_batch & keep_optimizing_index

                max_ind = np.argmax(p_vals)
                max_val = p_vals[max_ind]
                if max_val > best_val:
                    best_val = max_val
                    best_point = projected_new_points[max_ind].copy()
                    phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
                    success = phi_val >= epsilon
                    if success:
                        return self.post_process_global_search(
                            success, grid, grid_vals, best_point, best_val, radius
                        )

                del new_points_plus
                del new_points_minus
                del projected_new_points
                del gradients
                del hessians
                del p_vals

            point_steps += 1

        return self.post_process_global_search(
            success, grid, grid_vals, best_point, best_val, radius
        )
