# The Adaptive Refinement algorithm, as presented in https://arxiv.org/pdf/2301.07555

import numpy as np
from typing import Callable
import logging
import time
from itertools import product
from lib.measure import Measure
from lib.ssn import SSN

logging.basicConfig(
    level=logging.DEBUG,
)


class AdaptiveRefinement:
    def __init__(
        self,
        observations: np.ndarray,
        variance_exponent: float,
        j: Callable,
        p: Callable,  # Dual variable
        grad_p: Callable,
        hess_p: Callable,
        Omega: np.ndarray,
        kernel: Callable,
        alpha: float,
        target: np.ndarray,
        g: Callable,
        f: Callable,
        grad_f: Callable,
        hess_f: Callable,
        ssn_steps: int = 100,
    ):
        self.observations = observations
        self.variance_exponent = variance_exponent
        self.j = j
        self.p = p
        self.grad_p = grad_p
        self.hess_p = hess_p
        self.Omega = Omega
        self.machine_precision = 5e-14
        self.kernel = kernel
        self.ssn_steps = ssn_steps
        self.alpha = alpha
        self.target = target
        self.g = g
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.split_configurations = np.array(
            list(product(range(2), repeat=self.Omega.shape[0]))
        )
        self.vol_factor = np.sqrt(self.Omega.shape[1])

    def finite_dimensional_step(
        self,
        vertices: np.ndarray,
        coefs: np.ndarray,
    ) -> np.ndarray:
        K_support_raw = self.kernel(vertices).T
        signs = np.sign(coefs)
        signs[signs == 0] = 1
        K_support = np.multiply(K_support_raw, signs)
        u_0 = np.abs(coefs)
        ssn = SSN(
            K=K_support,
            alpha=self.alpha,
            target=self.target,
            M=float((self.f(K_support_raw @ coefs) + self.g(coefs)) / self.alpha),
            g=self.g,
            f=self.f,
            grad_f=self.grad_f,
            hess_f=self.hess_f,
            invariable_kernel=np.zeros(len(K_support)),
            mode="unconstrained",
            maximum_iterations=self.ssn_steps,
            regularization="full",
        )
        ssn_solution = ssn.solve(tol=self.machine_precision, u_0=u_0)
        ssn_normal = ssn_solution * signs
        ssn_clipped = (
            np.maximum(ssn_solution, np.zeros(len(ssn_solution))) * signs
        )  # The SSN in positive mode can return a better solution with wrong signs
        solutions = [ssn_clipped, ssn_normal, coefs]
        values = [self.f(K_support_raw @ cofs) + self.g(cofs) for cofs in solutions]
        best_value = np.argmin(values)
        new_coefs = solutions[best_value]
        return new_coefs

    def split_cell(
        self,
        cell_name: str,
        cells_dict: dict,
        vertices_dict: dict,
        vertices: np.ndarray,
    ) -> tuple:
        cell = cells_dict[cell_name]
        lower = cell[:, 0]
        upper = cell[:, 1]
        mid = (lower + upper) / 2
        diff = mid - lower
        index_offset = len(vertices)
        new_cells_names = []
        for i, config in enumerate(self.split_configurations):
            new_cell_name = cell_name + str(config)
            new_lower = lower + config * diff
            new_upper = new_lower + diff
            new_cell = np.vstack((new_lower, new_upper)).T
            new_cell_vertices = new_cell[
                np.arange(len(cell)), self.split_configurations
            ]
            cells_dict[new_cell_name] = new_cell
            vertices = np.vstack((vertices, new_cell_vertices))
            new_cells_names.append(new_cell_name)
        unique_vertices, inverse_index = np.unique(
            vertices, axis=0, return_inverse=True
        )
        for i, new_cell_name in enumerate(new_cells_names):
            cell_vertex_indices = inverse_index[
                index_offset
                + i * len(self.split_configurations) : index_offset
                + (i + 1) * len(self.split_configurations)
            ]
            vertices_dict[new_cell_name] = cell_vertex_indices
        del cells_dict[cell_name]
        del vertices_dict[cell_name]
        return cells_dict, vertices_dict, unique_vertices, new_cells_names

    def initiate(self) -> tuple:
        first_cell_name = "0"
        first_cell = np.zeros(self.Omega.shape)
        first_cell[:, 0] = self.Omega[:, 0] + 1e-4
        first_cell[:, 1] = self.Omega[:, 1] - 1e-4
        vertices = first_cell[np.arange(len(first_cell)), self.split_configurations]
        cells_dict = {first_cell_name: first_cell}
        vertices_dict = {first_cell_name: np.arange(len(vertices))}
        cells_dict, vertices_dict, vertices, new_cells_names = self.split_cell(
            first_cell_name, cells_dict, vertices_dict, vertices
        )
        return cells_dict, vertices_dict, vertices

    def hess_bound(
        self, q: np.ndarray, cell: np.ndarray, vertices: np.ndarray
    ) -> float:
        kappa = 0
        min_variance = np.min(vertices[:, 0])
        for obs_ind, obs in enumerate(self.observations):
            obs_in_cell = False
            for i, bound in enumerate(cell[1:]):
                if bound[0] <= obs[i] <= bound[1]:
                    obs_in_cell = True
                else:
                    obs_in_cell = False
                    break
            vertex_distances = np.linalg.norm(vertices[:, 1:] - obs, axis=1)
            if obs_in_cell:
                closest_point = obs.copy()
            else:
                closest_point = vertices[np.argmin(vertex_distances)][1:]
            min_dist = np.linalg.norm(obs - closest_point)
            max_dist = np.max(vertex_distances)
            adjusted_closest_point = np.hstack((closest_point, min_variance))

            kernel_factor = (
                self.kernel(adjusted_closest_point.reshape(1, -1))[0][obs_ind]
                / min_variance**2
            )
            summand_1 = max(
                abs(max_dist**2 / min_variance**2 - 1),
                abs(min_dist**2 / min_variance**2 - 1),
            )
            summand_2 = (
                max(
                    abs(2 - self.variance_exponent - max_dist**2 / min_variance**2),
                    abs(2 - self.variance_exponent - min_dist**2 / min_variance**2),
                )
                * 2
                * max_dist
                / min_variance
            )
            summand_31 = abs(
                self.variance_exponent**2
                - self.variance_exponent
                + (2 * self.variance_exponent - 3) * max_dist**2 / min_variance**2
                + max_dist**4 / min_variance**4
            )
            summand_32 = abs(
                self.variance_exponent**2
                - self.variance_exponent
                + (2 * self.variance_exponent - 3) * min_dist**2 / min_variance**2
                + min_dist**4 / min_variance**4
            )
            summand_33 = abs(2 * self.variance_exponent - 9 / 4)
            summand_3 = max(summand_31, summand_32, summand_33)
            hess_bound = kernel_factor * (summand_1 + summand_2 + summand_3)
            kappa += hess_bound * abs(q[obs_ind])
        return kappa

    def solve(
        self,
        max_iters: int = 1000,
        cells_dict: dict = {},
        vertices_dict: dict = {},
        vertices: np.ndarray = np.array([]),
    ) -> tuple:
        if len(cells_dict):
            cells_dict = cells_dict
            vertices_dict = vertices_dict
            vertices = vertices
        else:
            cells_dict, vertices_dict, vertices = self.initiate()
        active_set = vertices.copy()
        coefs = np.zeros(len(active_set))
        for iter in range(max_iters):
            if iter:
                # Determine which cells to subdivide
                subdivide_indices = []
                subdivide_edge = 0
                upper_b_compute = 0
                lower_b_compute = 0

                t = time.time()
                p_vals = p_u(vertices)
                grad_p_vals = grad_p_u(vertices)
                grad_compute = time.time() - t
                t = time.time()
                hess_p_vals = hess_p_u(vertices)
                kappa_compute = time.time() - t
                for cell_name in cells_dict.keys():
                    cell = cells_dict[cell_name]
                    cell_indices = vertices_dict[cell_name]
                    cell_vertices = vertices[cell_indices]
                    cell_edge = cell[0, 1] - cell[0, 0]
                    if cell_edge < subdivide_edge:
                        continue
                    cell_p_vals = p_vals[cell_indices]
                    cell_grad_p_vals = grad_p_vals[cell_indices]
                    kappa = np.max(
                        np.linalg.norm(hess_p_vals[cell_indices], axis=(1, 2))
                    )  # self.hess_bound(q, cell, cell_vertices) # Rough upper bound on the Hessian norm in the cell!!!
                    t = time.time()
                    upper_bound = self.alpha
                    for vertex, p_val, grad_p_val in zip(
                        cell_vertices, cell_p_vals, cell_grad_p_vals
                    ):
                        inner_upper_bound = 0
                        for inner_vertex in cell_vertices:
                            local_bound = (
                                np.abs(p_val + grad_p_val @ (inner_vertex - vertex))
                                + 0.5
                                * kappa
                                * np.linalg.norm(inner_vertex - vertex) ** 2
                            )
                            if local_bound > inner_upper_bound:
                                inner_upper_bound = local_bound
                        if inner_upper_bound < upper_bound:
                            upper_bound = inner_upper_bound
                        if upper_bound < self.alpha:
                            break
                    upper_b_compute += time.time() - t
                    t = time.time()
                    lower_bound = (
                        np.max(np.linalg.norm(cell_grad_p_vals, axis=1))
                        - kappa * cell_edge * self.vol_factor
                    )
                    # lower_bound = 0
                    if upper_bound >= self.alpha and lower_bound <= 0:
                        if cell_edge > subdivide_edge:
                            subdivide_names = [cell_name]
                            subdivide_edge = cell_edge
                        elif cell_edge == subdivide_edge:
                            subdivide_names.append(cell_name)
                            subdivide_edge = cell_edge
                    lower_b_compute += time.time() - t

                del p_vals
                del grad_p_vals
                del hess_p_vals

                # Subdivide cells
                t = time.time()
                new_vertices = []
                new_indices = np.array([])
                for cell_name in subdivide_names:
                    cells_dict, vertices_dict, vertices, new_cells_names = (
                        self.split_cell(cell_name, cells_dict, vertices_dict, vertices)
                    )
                    for name in new_cells_names:
                        local_indices = vertices_dict[name]
                        if not len(new_indices):
                            new_indices = local_indices
                        else:
                            new_indices = np.hstack((new_indices, local_indices))
                unique_new_indices = np.unique(new_indices)
                new_vertices = vertices[unique_new_indices]
                active_set = np.vstack((active_set, new_vertices))
                coefs = np.hstack((coefs, np.zeros(len(new_vertices))))
                provisory_u = Measure(support=active_set, coefficients=coefs)
                active_set = provisory_u.support
                coefs = provisory_u.coefficients
                subdivide = time.time() - t
                logging.info(
                    f"grad: {grad_compute:.3f}, kappa: {kappa_compute:.3f}, upper: {upper_b_compute:.3f}, lower: {lower_b_compute:.3f}, subdivide: {subdivide:.3f}"
                )

            # Determine iterate measure
            t = time.time()
            coefs = self.finite_dimensional_step(active_set, coefs)
            active_set = active_set[coefs != 0].copy()
            coefs = coefs[coefs != 0].copy()
            u = Measure(support=active_set, coefficients=coefs)
            logging.info(
                f"{iter + 1}: cells: {len(cells_dict)}, support: {len(u.coefficients)}, objective: {self.j(u, 0):.14E}"
            )
            q = -self.grad_f(
                np.tensordot(self.kernel(active_set), coefs, axes=([0], [0]))
            )
            p_u = self.p(u, 0)
            grad_p_u = self.grad_p(u, 0)
            hess_p_u = self.hess_p(u, 0)
            logging.info(f"ssn: {time.time() - t:.3f}")
        return cells_dict, vertices_dict, vertices, coefs, u
