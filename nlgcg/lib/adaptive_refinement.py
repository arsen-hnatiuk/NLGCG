# The Adaptive Refinement algorithm, as presented in https://arxiv.org/pdf/2301.07555

import numpy as np
from typing import Callable
import logging
import matplotlib.pyplot as plt
import time
import cvxpy as cp
import random
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
        self.observations = observations
        self.variance_exponent = variance_exponent
        self.j = j
        self.p = p
        self.grad_p = grad_p
        self.hess_p = hess_p
        self.Omega = Omega
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
        self.split_configurations = np.array(
            list(product(range(2), repeat=self.Omega.shape[0]))
        )
        self.vol_factor = np.sqrt(self.Omega.shape[1])

    def finite_dimensional_step(
        self,
        vertices: np.ndarray,
        coefs: np.ndarray,
        c: float,
        q_0: np.ndarray,
    ) -> np.ndarray:
        t = time.time()
        q = self.dual_problem_solver(vertices, q_0)
        p_u = np.abs(self.kernel(vertices) @ q)
        active_indices = p_u >= self.alpha - 1e-6
        vertices = vertices[active_indices].copy()
        coefs = coefs[active_indices].copy()
        cvxpy_time = time.time() - t

        t = time.time()
        K_support = np.hstack(
            (np.ones(self.constant_dim), np.zeros(self.kernel_dim - self.constant_dim))
        ).reshape(-1, 1)
        K_support = np.hstack((self.kernel(vertices).T, K_support))
        u_0 = np.hstack((coefs, c))
        ssn = SSN(
            K=K_support,
            alpha=self.alpha,
            target=self.target,
            M=float((self.f(K_support @ u_0) + self.g(u_0[:-1])) / self.alpha),
            g=self.g,
            f=self.f,
            grad_f=self.grad_f,
            hess_f=self.hess_f,
            invariable_kernel=np.zeros(len(K_support)),
            mode="unconstrained",
            maximum_iterations=self.ssn_steps,
            regularization="mixed",
        )
        ssn_solution = ssn.solve(tol=self.machine_precision, u_0=u_0)
        solutions = [ssn_solution, u_0]
        values = [self.f(K_support @ sol) + self.g(sol[:-1]) for sol in solutions]
        best_value = np.argmin(values)
        best_sol = solutions[best_value]
        new_coefs = best_sol[:-1]
        new_c = best_sol[-1]
        new_vertices = vertices[new_coefs != 0].copy()
        new_coefs = new_coefs[new_coefs != 0].copy()
        ssn_time = time.time() - t
        logging.info(f"cvxpy: {cvxpy_time:.3f}, ssn: {ssn_time:.3f}")
        return new_vertices, new_coefs, new_c

    def dual_problem_solver(self, vertices: np.ndarray, q_0: np.ndarray) -> np.ndarray:
        q = cp.Variable(self.target.shape[0])
        q.value = q_0
        obj = cp.Maximize(-2 * self.target @ q - cp.sum(cp.square(q)))
        kernel_vertices = self.kernel(vertices)
        constraints = [
            kernel_vertices @ q <= self.alpha,
            -kernel_vertices @ q <= self.alpha,
        ]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.SCS, eps=1e-8, max_iters=10000)
        return q.value

    def split_cells(
        self,
        cell_names: list,
        cells_dict: dict,
        vertices_dict: dict,
        vertices: np.ndarray,
    ) -> tuple:
        new_vertices = []
        new_cells_names = []
        index_offset = len(vertices)
        for cell_name in cell_names:
            cell = cells_dict[cell_name]
            lower = cell[:, 0]
            upper = cell[:, 1]
            mid = (lower + upper) / 2
            diff = mid - lower
            for i, config in enumerate(self.split_configurations):
                new_cell_name = cell_name + str(config)
                new_lower = lower + config * diff
                new_upper = new_lower + diff
                new_cell = np.vstack((new_lower, new_upper)).T
                new_cell_vertices = new_cell[
                    np.arange(len(cell)), self.split_configurations
                ]
                if not len(new_vertices):
                    new_vertices = new_cell_vertices
                else:
                    new_vertices = np.vstack((new_vertices, new_cell_vertices))
                cells_dict[new_cell_name] = new_cell
                new_cells_names.append(new_cell_name)
            del cells_dict[cell_name]
            del vertices_dict[cell_name]
        vertices = np.vstack((vertices, new_vertices))
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
        for cell_name in cells_dict.keys():
            if cell_name not in new_cells_names:
                cell_vertex_indices = inverse_index[np.array(vertices_dict[cell_name])]
                vertices_dict[cell_name] = cell_vertex_indices
        return cells_dict, vertices_dict, unique_vertices, new_cells_names

    def initiate(self) -> tuple:
        first_cell_name = "0"
        first_cell = np.zeros(self.Omega.shape)
        first_cell[:, 0] = self.Omega[:, 0] + 1e-3
        first_cell[:, 1] = self.Omega[:, 1] - 1e-3
        vertices = first_cell[np.arange(len(first_cell)), self.split_configurations]
        cells_dict = {first_cell_name: first_cell}
        vertices_dict = {first_cell_name: np.arange(len(vertices))}
        cells_dict, vertices_dict, vertices, new_cells_names = self.split_cells(
            [first_cell_name], cells_dict, vertices_dict, vertices
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
        t_0 = time.time()
        if len(cells_dict):
            cells_dict = cells_dict
            vertices_dict = vertices_dict
            vertices = vertices
        else:
            cells_dict, vertices_dict, vertices = self.initiate()
        active_set = vertices.copy()
        coefs = np.zeros(len(active_set))
        q = np.array(self.grad_f(self.kernel(active_set).T @ coefs))
        u = Measure()
        c = 0
        times = [time.time() - t_0]
        actives = [len(active_set)]
        objective_values = [self.j(u, c)]
        for iter in range(max_iters):
            if iter:
                # Determine which cells to subdivide
                subdivide_names = []
                # subdivide_dict = {}
                # subdivide_cells = {}
                subdivide_edge = 0
                upper_b_compute = 0
                lower_b_compute = 0
                kappa_compute1 = 0
                kappa_compute2 = 0
                kappa_compute3 = 0
                kappa_compute4 = 0
                kappa_compute5 = 0
                kappa_compute6 = 0
                kappa_compute7 = 0
                kappa_compute = 0

                t = time.time()
                p_vals = np.array(p_u(vertices))
                grad_p_vals = np.array(grad_p_u(vertices))
                grad_compute = time.time() - t
                t = time.time()
                hess_p_vals = np.array(hess_p_u(vertices))
                hess_norms = np.linalg.norm(hess_p_vals, axis=(1, 2))
                hess_compute = time.time() - t
                for cell_name in cells_dict.keys():
                    t0 = time.time()
                    cell = cells_dict[cell_name]
                    kappa_compute1 += time.time() - t0
                    t = time.time()
                    cell_indices = vertices_dict[cell_name]
                    kappa_compute2 += time.time() - t
                    t = time.time()
                    cell_vertices = vertices[cell_indices]
                    kappa_compute3 += time.time() - t
                    t = time.time()
                    cell_edge = cell[0, 1] - cell[0, 0]
                    if cell_edge < subdivide_edge:
                        continue
                    kappa_compute4 += time.time() - t
                    t = time.time()
                    cell_p_vals = p_vals[cell_indices]
                    kappa_compute5 += time.time() - t
                    t = time.time()
                    cell_grad_p_vals = grad_p_vals[cell_indices]
                    kappa_compute6 += time.time() - t
                    t = time.time()
                    # kappa = self.hess_bound(
                    #     q,
                    #     cell,
                    #     cell_vertices,
                    # )
                    kappa = np.max(hess_norms[cell_indices])
                    kappa_compute7 += time.time() - t
                    kappa_compute += time.time() - t0
                    t = time.time()
                    upper_bound = self.alpha
                    for vertex, p_val, grad_p_val in zip(
                        cell_vertices, cell_p_vals, cell_grad_p_vals
                    ):
                        inner_upper_bound = 0
                        for inner_vertex in cell_vertices:
                            local_bound = np.abs(
                                p_val + grad_p_val @ (inner_vertex - vertex)
                            ) + 0.5 * kappa * (inner_vertex - vertex) @ (
                                inner_vertex - vertex
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
                    # if upper_bound >= self.alpha and lower_bound <= 0:
                    #     c = "green"
                    # elif upper_bound >= self.alpha:
                    #     c = "blue"
                    # elif lower_bound <= 0:
                    #     c = "purple"
                    # else:
                    #     c = "black"
                    if upper_bound >= self.alpha and lower_bound <= 0:
                        if cell_edge > subdivide_edge:
                            subdivide_names = [cell_name]
                            subdivide_edge = cell_edge
                            # subdivide_dict = {cell_name: c}
                            # subdivide_cells = {cell_name: cell}
                        elif cell_edge == subdivide_edge:
                            subdivide_names.append(cell_name)
                            # subdivide_dict[cell_name] = c
                            # subdivide_cells[cell_name] = cell
                    lower_b_compute += time.time() - t

                # Subdivide cells
                t = time.time()
                new_vertices = []
                new_indices = np.array([])
                cells_dict, vertices_dict, vertices, new_cells_names = self.split_cells(
                    subdivide_names, cells_dict, vertices_dict, vertices
                )
                for name in new_cells_names:
                    local_indices = vertices_dict[name]
                    if not len(new_indices):
                        new_indices = local_indices
                    else:
                        new_indices = np.hstack((new_indices, local_indices))
                unique_new_indices = np.unique(new_indices)
                new_vertices = vertices[unique_new_indices]
                active_set_raw = np.vstack((active_set, new_vertices))
                coefs_raw = np.hstack((coefs, np.zeros(len(new_vertices))))
                active_set, unique_indices = np.unique(
                    active_set_raw, axis=0, return_index=True
                )
                coefs = coefs_raw[unique_indices]
                subdivide = time.time() - t
                # logging.info(
                #     f"kappa1: {kappa_compute1:.3f}, kappa2: {kappa_compute2:.3f}, kappa3: {kappa_compute3:.3f}, kappa4: {kappa_compute4:.3f}, kappa5: {kappa_compute5:.3f}, kappa6: {kappa_compute6:.3f}, kappa7: {kappa_compute7:.3f}"
                # )
                logging.info(
                    f"grad: {grad_compute:.3f}, hess: {hess_compute:.3f}, kappa: {kappa_compute:.3f}, upper: {upper_b_compute:.3f}, lower: {lower_b_compute:.3f}, subdivide: {subdivide:.3f}"
                )

                del p_vals
                del grad_p_vals
                del hess_p_vals
                del active_set_raw

                # # Specific display for the paper
                # n = 1000
                # xx = np.linspace(0, 1, n)
                # yy = np.linspace(0, 1, n)
                # fine_grid = np.zeros((n**2, 2))
                # X_grid, Y_grid = np.meshgrid(xx, yy)
                # fine_grid[:, 0], fine_grid[:, 1] = X_grid.ravel(), Y_grid.ravel()
                # grid_shape = X_grid.shape
                # Aqk = np.abs(p_u(fine_grid))
                # plt.contour(
                #     X_grid,
                #     Y_grid,
                #     Aqk.reshape(grid_shape),
                #     levels=[0.75 * self.alpha, 0.9 * self.alpha],
                #     colors="r",
                #     linestyles=["dotted", "dashed"],
                # )  #    plt.plot(fine_grid, Aqk, 'r--', linewidth=2)
                # plt.contourf(
                #     X_grid,
                #     Y_grid,
                #     Aqk.reshape(grid_shape),
                #     levels=[0.0, self.alpha],
                #     colors=[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.5]],
                #     extend="max",
                # )
                # # for _ in range(5):
                # #     cell_names = list(cells_dict.keys())
                # #     name = random.choice(cell_names)
                # #     cell = cells_dict[name]
                # #     this_vertices = vertices[vertices_dict[name]]
                # #     lower = cell[:, 0]
                # #     plt.plot(lower[0], lower[1], "o", c="red", markersize=5)
                # #     for vertex in this_vertices:
                # #         plt.plot(vertex[0], vertex[1], "o", c="green", markersize=3)
                # for cell_name in cells_dict.keys():
                #     if cell_name in new_cells_names:
                #         continue
                #     cell = cells_dict[cell_name]
                #     lower = cell[:, 0]
                #     c = subdivide_dict.get(cell_name, "black")
                #     plt.plot(lower[0], lower[1], "o", c=c, markersize=4)
                # for cell_name in subdivide_cells.keys():
                #     if cell_name in new_cells_names:
                #         continue
                #     cell = subdivide_cells[cell_name]
                #     lower = cell[:, 0]
                #     c = subdivide_dict.get(cell_name, "black")
                #     plt.plot(lower[0], lower[1], "o", c=c, markersize=4)
                # # for vertex in new_vertices:
                # #     plt.plot(vertex[0], vertex[1], "o", c="green", markersize=2)
                # plt.show()

            # Determine iterate measure
            actives.append(len(active_set))
            active_set, coefs, c = self.finite_dimensional_step(active_set, coefs, c, q)
            q = np.array(
                self.grad_f(
                    self.kernel(active_set).T @ coefs + c * np.ones(len(self.target))
                )
            )
            u = Measure(support=active_set, coefficients=coefs)
            times.append(time.time() - t_0)
            objective_values.append(self.j(u, c))
            logging.info(
                f"{iter + 1}: cells: {len(cells_dict)}, support: {len(u.coefficients)}, objective: {self.j(u, c):.14E}"
            )
            p_u = self.p(u, c)
            grad_p_u = self.grad_p(u, c)
            hess_p_u = self.hess_p(u, c)

        return cells_dict, vertices_dict, vertices, u, objective_values, times, actives
