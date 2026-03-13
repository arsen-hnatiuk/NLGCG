# Implementation of global search methods for the solution of the non-convex problem in NLGCG

import numpy as np
import scipy as sp
import logging
from itertools import product
from typing import Callable
from sklearn.utils import gen_batches

from lib.measure import Measure

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
        mode: str = "stochastic_adaptive",
        max_found_points: int = 100,
        newton_tolerance: float = 5e-2,
        sampling_probability: float = 0.99,
        sample_size=1000,
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
        self.newton_tolerance = newton_tolerance
        self.sampling_probabilty = sampling_probability
        self.sample_size = sample_size
        if mode == "deterministic_adaptive":
            self.get_grid = self.deterministic_grid_adaptive
            self.stop_search = 3
        if mode == "deterministic":
            self.get_grid = self.deterministic_grid
            self.stop_search = 5
        elif mode == "stochastic_adaptive":
            self.get_grid = self.stochastic_grid_adaptive
            self.stop_search = 3
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

    def sample_domain(
        self, size: int, u: Measure = Measure(), mode: str = "uniform"
    ) -> np.ndarray:
        # Generate a uniform sample of shape (size,domain.shape[0]) in the given domain
        columns = []
        for i, bounds in enumerate(self.Omega):
            if i == 0:
                if mode == "exponential":
                    if len(u.coefficients):
                        distribution_parameter = max(u.support[:, 0].max(), bounds[1])
                    else:
                        distribution_parameter = bounds[1]
                    columns.append(
                        np.random.exponential(
                            scale=distribution_parameter, size=(size, 1)
                        )
                        + bounds[0]
                    )
                elif mode == "uniform":
                    columns.append(
                        np.random.sample((size, 1)) * (bounds[1] - bounds[0])
                        + bounds[0]
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
        temperature: float = 1.0,
        do_logging: bool = True,
    ) -> np.ndarray:
        grid = (
            np.array(
                np.meshgrid(
                    *(
                        np.linspace(
                            bound[0], bound[1], self.global_search_resolution + 1
                        )[1:]
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

    def split_point(self, mesh):
        split_configurations = np.array(
            list(product(range(2), repeat=self.Omega.shape[0]))
        )
        split_directions_raw = split_configurations - np.array(
            [0.5] * self.Omega.shape[0]
        )
        split_directions = (
            mesh
            * split_directions_raw
            / (2 * np.linalg.norm(split_directions_raw, axis=1)[:, np.newaxis])
        )
        return split_directions

    def deterministic_grid_adaptive(
        self,
        u: Measure,
        p_norm: Callable,
        grad_p_u: Callable,
        q_u: float,
        epsilon: float,
        radius: float,
        temperature: float = 1.0,
        do_logging: bool = True,
    ) -> tuple:
        lipschitz_function = lambda x: np.linalg.norm(grad_p_u(x), axis=1)
        first_point = np.mean(self.Omega, axis=1)
        mesh = (
            np.sqrt(self.Omega.shape[0])
            * np.max([bound[1] - bound[0] for bound in self.Omega])
            / 2
        )
        grid = np.array([first_point])
        grid_vals = p_norm(grid)
        all_points = grid.copy()
        all_vals = grid_vals.copy()
        if len(u.coefficients):
            all_points = np.vstack([all_points, u.support])
            all_vals = np.hstack((all_vals, p_norm(u.support)))

        best_index = np.argmax(all_vals)
        best_point = all_points[best_index].copy()
        best_val = all_vals[best_index]
        phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
        success = phi_val >= epsilon
        if success:
            grid = np.vstack([grid, np.array([best_point])])
            grid_vals = np.hstack((grid_vals, np.array([best_val])))
            if do_logging:
                logging.info(f"Grid. {len(grid_vals)} points, mesh: {mesh:.3E}")
            return success, grid, grid_vals, best_point, best_val

        while mesh > self.newton_tolerance:
            lipschitzs = self.batch_compute(grid, lipschitz_function)

            relevant_indices = grid_vals + lipschitzs * mesh > best_val
            del lipschitzs
            grid = grid[relevant_indices]

            new_grid = grid.copy()
            for _ in range(3):
                split_directions_deterministic = self.split_point(mesh)
                split_directions = split_directions_deterministic + np.random.normal(
                    0,
                    0.1 * mesh,
                    size=split_directions_deterministic.shape,
                )
                new_grid = np.repeat(new_grid, len(split_directions), axis=0) + np.tile(
                    split_directions, (len(new_grid), 1)
                )
                mesh /= 2
            new_grid = self.project_into_domain(new_grid)
            new_vals = p_norm(new_grid)
            valid_indices = new_vals > 0
            new_grid = new_grid[valid_indices].copy()
            new_vals = new_vals[valid_indices].copy()

            best_index = np.argmax(new_vals)
            tentative_best_val = new_vals[best_index].copy()
            if tentative_best_val > best_val:
                best_val = tentative_best_val
                best_point = new_grid[best_index].copy()
                phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
                success = phi_val >= epsilon
                if success:
                    grid = np.vstack([new_grid, np.array([best_point])])
                    grid_vals = np.hstack((new_vals, np.array([best_val])))
                    if do_logging:
                        logging.info(f"Grid. {len(new_vals)} points, mesh: {mesh:.3E}")
                    return success, grid, grid_vals, best_point, best_val

            grid = new_grid.copy()
            grid_vals = new_vals.copy()
            del new_grid
            del new_vals
            if do_logging:
                logging.info(f"Grid. {len(grid_vals)} points, mesh: {mesh:.3E}")

        lipschitzs = self.batch_compute(grid, lipschitz_function)
        relevant_indices = grid_vals + lipschitzs * mesh > best_val
        del lipschitzs
        grid = grid[relevant_indices]
        grid_vals = grid_vals[relevant_indices]
        grid = np.vstack([grid, np.array([best_point])])
        grid_vals = np.hstack((grid_vals, np.array([best_val])))
        return success, grid, grid_vals, best_point, best_val

    def stochastic_grid(
        self,
        u: Measure,
        p_norm: Callable,
        grad_p_u: Callable,
        q_u: float,
        epsilon: float,
        radius: float,
        temperature: float = 1.0,
        do_logging: bool = True,
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

    def sample_ball(self, size: int, radius: float) -> np.ndarray:
        # Sample uniformly from a ball of given radius in the domain dimension
        dim = self.Omega.shape[0]
        x = np.random.normal(size=(size, dim))
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
        u = np.random.random(size=(size, 1)) ** (1 / dim)
        return radius * (x * u)

    def compute_mesh_reduction(self) -> float:
        dim = self.Omega.shape[0]
        target_value = 1 - (1 - self.sampling_probabilty) ** (1 / self.sample_size)

        def prob_function(value):
            phi_1 = np.arccos(1 - 0.5 * value**2)
            phi_2 = np.arccos(value / 2)
            s_phi_1 = np.sin(phi_1) ** 2
            s_phi_2 = np.sin(phi_2) ** 2
            summand_1 = sp.special.betainc(0.5 * (dim + 1), 0.5, s_phi_1)
            summand_2 = sp.special.betainc(0.5 * (dim + 1), 0.5, s_phi_2)
            return 0.5 * (summand_1 + summand_2 * value**dim)

        value_lower = 0
        value_higher = 1
        value = np.mean([value_lower, value_higher])
        prob_value = prob_function(value)
        while prob_value < 0.99 * target_value or prob_value > 1.01 * target_value:
            if prob_value < target_value:
                value_lower = value
            else:
                value_higher = value
            value = np.mean([value_lower, value_higher])
            prob_value = prob_function(value)
        return value

    def batch_compute(
        self, inputs: np.ndarray, func: Callable, batch_size: int = int(1e4)
    ) -> np.ndarray:
        results = []
        for batch in gen_batches(len(inputs), batch_size):
            batch_results = func(inputs[batch])
            results.append(batch_results)
            del batch_results
        return np.hstack(results).flatten()

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
        for point, val, lipschitz in zip(
            sorted_points[1:], sorted_vals[1:], sorted_lipschitzs[1:]
        ):
            distances = np.linalg.norm(separated_points - point, axis=1)
            if all(distances > max(mesh, self.newton_tolerance)):
                separated_points = np.vstack((separated_points, [point]))
                separated_vals = np.append(separated_vals, val)
                separated_lipschitzs = np.append(separated_lipschitzs, lipschitz)
        return separated_points, separated_vals, separated_lipschitzs

    def stochastic_grid_adaptive(
        self,
        u: Measure,
        p_norm: Callable,
        grad_p_u: Callable,
        q_u: float,
        epsilon: float,
        radius: float,
        temperature: float = 1.0,
        do_logging: bool = True,
    ) -> tuple:
        lipschitz_function = lambda x: np.linalg.norm(grad_p_u(x), axis=1)
        success = False
        sampled_local = False
        mesh_reduction = self.compute_mesh_reduction()
        mesh = (
            max([bound[1] - bound[0] for bound in self.Omega])
            * np.sqrt(self.Omega.shape[0])
            / 2
        )  # radius of sampling domain
        mesh *= temperature  # allow for local sampling from the beginning

        grid = self.sample_domain(self.sample_size, u)
        if len(u.coefficients):
            grid = np.vstack([grid, u.support])
        grid_vals = p_norm(grid)
        lipschitzs = []
        mesh = mesh_reduction * mesh

        best_index = np.argmax(grid_vals)
        best_point = grid[best_index].copy()
        best_val = grid_vals[best_index]
        phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
        success = phi_val >= epsilon
        if do_logging:
            logging.info(f"Global Sampling. {len(grid_vals)} points, mesh: {mesh:.3E}")
        if success:
            return success, grid, grid_vals, best_point, best_val

        while mesh > self.newton_tolerance or not sampled_local:
            if not len(lipschitzs):
                lipschitzs = self.batch_compute(grid, lipschitz_function)
            else:
                points_new_lipschitzs = self.batch_compute(
                    points_new, lipschitz_function
                )
                lipschitzs = np.append(lipschitzs, points_new_lipschitzs)
                del points_new
                del points_new_lipschitzs

            grid, grid_vals, lipschitzs = self.separate_points(
                grid, grid_vals, mesh, best_val, lipschitzs
            )

            # New points
            sample_updates = self.sample_ball(self.sample_size * len(grid_vals), mesh)
            starting_points = np.repeat(
                grid, [self.sample_size] * len(grid_vals), axis=0
            )
            points_new_raw = starting_points + sample_updates  # new trial points
            points_new = self.project_into_domain(points_new_raw)

            # Compute the |p| values of trial points
            points_new_vals = p_norm(
                points_new
            )  # might contain invalid points (0 vals)
            valid_indices = points_new_vals > 0
            points_new = points_new[valid_indices]
            points_new_vals = points_new_vals[valid_indices]

            # Add accepted new points to active tuples
            grid = np.vstack((grid, points_new))
            grid_vals = np.append(grid_vals, points_new_vals)
            mesh = mesh_reduction * mesh
            if do_logging:
                logging.info(
                    f"Local Sampling. {len(grid_vals)} points, mesh: {mesh:.3E}"
                )

            # Update the best value
            best_index = np.argmax(points_new_vals)
            tentative_best_val = points_new_vals[best_index].copy()
            if tentative_best_val > best_val:
                best_val = tentative_best_val
                best_point = points_new[best_index].copy()
                phi_val = max(self.M * (best_val - self.alpha), 0) + q_u
                success = phi_val >= epsilon
                if success:
                    return success, grid, grid_vals, best_point, best_val

            del sample_updates
            del starting_points
            del points_new_raw
            del points_new_vals

            sampled_local = True  # We want to enter the loop at least once

        if not len(lipschitzs):
            lipschitzs = self.batch_compute(grid, lipschitz_function)
        else:
            points_new_lipschitzs = self.batch_compute(points_new, lipschitz_function)
            lipschitzs = np.append(lipschitzs, points_new_lipschitzs)
            del points_new
            del points_new_lipschitzs
        grid, grid_vals, lipschitzs = self.separate_points(
            grid, grid_vals, mesh, best_val, lipschitzs
        )
        return success, grid, grid_vals, best_point, best_val

    def newton_steps(
        self,
        p_norm: Callable,
        grad_p: Callable,
        hess_p: Callable,
        grid: np.ndarray,
        grid_vals: np.ndarray,
        epsilon: float,
        best_point: np.ndarray,
        best_val: float,
        q_u: float,
        do_logging: bool,
    ) -> tuple:
        if do_logging:
            logging.info(f"Newton start points: {len(grid_vals)}")
        success = False
        batching_factor = (
            self.len_target * self.Omega.shape[0] * (self.Omega.shape[0] + 1)
            + 2 * self.Omega.shape[0]
            + 1
        )
        batch_size = int(self.batching_constant // batching_factor)
        optimize_grid = np.array([True] * grid.shape[0])
        point_steps = 0
        while point_steps < self.stop_search:
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
                        return success, grid, grid_vals, best_point, best_val

                del new_points_plus
                del new_points_minus
                del projected_new_points
                del gradients
                del hessians
                del p_vals

            point_steps += 1

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
            best_val,
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
        temperature: float = 1.0,
        do_logging: bool = True,
    ) -> tuple:
        p_norm = lambda x: np.abs(np.array(np.nan_to_num(p_u(x))))
        grad_p = self.grad_p(u, c)
        hess_p = self.hess_p(u, c)

        if len(u.support):
            # Perform Newton on current support points
            grid = u.support.copy()
            grid_vals = p_norm(grid)
            max_ind = np.argmax(grid_vals)
            best_val = grid_vals[max_ind]
            best_point = grid[max_ind].copy()
            success, grid, grid_vals, best_point, best_val = self.newton_steps(
                p_norm,
                grad_p,
                hess_p,
                grid,
                grid_vals,
                epsilon,
                best_point,
                best_val,
                q_u,
                do_logging,
            )
            if success:
                return self.post_process_global_search(
                    success, grid, grid_vals, best_point, best_val, radius
                )

        success, grid, grid_vals, best_point, best_val = self.get_grid(
            u, p_norm, grad_p, q_u, epsilon, radius, temperature, do_logging
        )
        if success:
            return self.post_process_global_search(
                success, grid, grid_vals, best_point, best_val, radius
            )

        # Optimize the grid with Newton steps
        success, grid, grid_vals, best_point, best_val = self.newton_steps(
            p_norm,
            grad_p,
            hess_p,
            grid,
            grid_vals,
            epsilon,
            best_point,
            best_val,
            q_u,
            do_logging,
        )

        return self.post_process_global_search(
            success, grid, grid_vals, best_point, best_val, radius
        )
