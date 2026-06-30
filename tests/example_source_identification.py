import numpy as np
import os
import jax
import jax.numpy as jnp
import logging
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

module_path = Path(__file__).resolve().parent.parent
if module_path not in sys.path:
    sys.path.append(str(module_path))
src_path = (module_path / "src").resolve()
if src_path not in sys.path:
    sys.path.append(str(src_path))

from nlgcg import NLGCG
from src.lib.measure import Measure
from src.lib.particle_descent import ParticleDescent
from src.lib.adaptive_refinement import AdaptiveRefinement

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0 xla_python_client_preallocate=false xla_python_client_mem_fraction=0.5"
)
logging.getLogger().setLevel(logging.INFO)
# init jax
jax.config.update("jax_enable_x64", True)
_ = jnp.zeros(0)

# Generate data and define functions

Omega = np.array([[0, 1], [0, 1]])
d = Omega.shape[0]
Omega_size = Omega[0][1] - Omega[0][0]
max_radius = (Omega[0][1] - Omega[0][0]) / 50
alpha = 1e-1
observation_resolution = 4
std_factor = 0.1
true_sources = np.array([[0.28, 0.71], [0.51, 0.27], [0.71, 0.53]])
true_weights = np.array([1, -0.7, 0.8])
optimum = 0.238290673266492

observations = (
    np.array(
        np.meshgrid(
            *(
                np.linspace(bound[0], bound[1], observation_resolution + 2)
                for bound in Omega
            )
        )
    )
    .reshape(len(Omega), -1)
    .T
)
observations = np.array(
    [obs for obs in observations if all(obs != 0) and all(obs != 1)]
)


@jax.jit
def singleton_kernel(omega: np.ndarray):
    outer_factor = np.sqrt(std_factor * np.pi) ** Omega.shape[0]
    inner = -jnp.sum((omega - observations) ** 2, axis=1) / std_factor
    outer = jnp.exp(inner) / outer_factor
    return outer


kernel = jax.vmap(singleton_kernel)

u_hat = Measure(support=true_sources, coefficients=true_weights)
target = u_hat.duality_pairing(kernel)


@jax.jit
def g(w: np.ndarray) -> float:
    return alpha * jnp.linalg.norm(w, ord=1)


@jax.jit
def f(y: np.ndarray) -> float:
    return 0.5 * jnp.sum((y - target) ** 2)


grad_f = jax.jit(jax.grad(f))
hess_f = jax.jit(jax.hessian(f))

j = lambda u, c: f(u.duality_pairing(kernel) + c * np.ones(target.shape)) + g(
    u.coefficients
)


@jax.jit
def p_raw(parameters: np.ndarray, c: float, omega: np.ndarray):
    coefficients = parameters[:, 0]
    support = parameters[:, 1:]
    Ku = jnp.tensordot(kernel(support), coefficients, axes=([0], [0]))
    constant_term = c * jnp.ones(len(target))
    return singleton_kernel(omega) @ -grad_f(Ku + constant_term)


grad_p_raw = jax.jit(jax.grad(p_raw, argnums=2))
hess_p_raw = jax.jit(jax.hessian(p_raw, argnums=2))

p_raw = jax.jit(jax.vmap(p_raw, in_axes=(None, None, 0), out_axes=0))
grad_p_raw = jax.jit(jax.vmap(grad_p_raw, in_axes=(None, None, 0), out_axes=0))
hess_p_raw = jax.jit(jax.vmap(hess_p_raw, in_axes=(None, None, 0), out_axes=0))

p = lambda u, c: lambda omega: p_raw(u.to_matrix(len(Omega)), c, omega)
grad_p = lambda u, c: lambda omega: grad_p_raw(u.to_matrix(len(Omega)), c, omega)
hess_p = lambda u, c: lambda omega: hess_p_raw(u.to_matrix(len(Omega)), c, omega)


# Parameterized versions of f and j
@jax.jit
def f_N(raw_input: np.ndarray) -> float:
    constant = raw_input[-1]
    input = raw_input[:-1].reshape(-1, d + 1)
    weights = input[:, 0]
    omega = input[:, 1:]
    return f(kernel(omega).T @ weights + constant * jnp.ones(target.shape))


@jax.jit
def j_N(raw_input: np.ndarray) -> float:
    input = raw_input[:-1].reshape(-1, d + 1)
    weights = input[:, 0]
    return f_N(raw_input) + g(weights)


grad_f_N = jax.jit(jax.grad(f_N))
hess_f_N = jax.jit(jax.hessian(f_N))
grad_j_N = jax.jit(jax.grad(j_N))


def adapt_time(times, residuals, frame=100, resolution=1):
    to_return = []
    last_pos = 0
    last_res = residuals[0]
    for t in range(int(frame / resolution)):
        minimim_time = t * resolution
        maximum_time = (t + 1) * resolution
        added = False
        for i, (res, tim) in enumerate(zip(residuals[last_pos:], times[last_pos:])):
            if tim < maximum_time and tim >= minimim_time:
                to_return.append(res)
                last_res = res
                last_pos += i + 1
                added = True
                break
        if not added:
            to_return.append(last_res)
        if t * resolution >= times[-1]:
            break
    to_return.append(residuals[-1])
    return to_return


def bring_to_same_length(arrays):
    max_length = max(len(arr) for arr in arrays)
    new_arrays = []
    for arr in arrays:
        if len(arr) < max_length:
            last_val = arr[-1]
            arr = arr + [last_val] * (max_length - len(arr))
        new_arrays.append(arr)
    return new_arrays


def get_grid(size: int) -> np.ndarray:
    grid = (
        np.array(
            np.meshgrid(
                *(np.linspace(bound[0], bound[1], size + 1)[1:] for bound in Omega)
            )
        )
        .reshape(len(Omega), -1)
        .T
    )
    return grid


def create_plots():
    resolution = 0.1  # time resolution for the plots (seconds)
    frame_size = 1000  # Time frame tracked for the residuals (seconds)

    exp = NLGCG(
        target=target,
        kernel=kernel,
        g=g,
        f=f,
        f_N=f_N,
        grad_f=grad_f,
        hess_f=hess_f,
        grad_f_N=grad_f_N,
        hess_f_N=hess_f_N,
        j=j,
        j_N=j_N,
        p=p,
        grad_p=grad_p,
        hess_p=hess_p,
        grad_j_N=grad_j_N,
        alpha=alpha,
        Omega=Omega,
        global_search_resolution=5,
        dual_variable_goodness=0.3,
        constant_dim=len(target),
        kernel_dim=len(target),
        newton_tolerance=2e-2,
        max_radius=max_radius,
    )

    nlgcg_residuals = []
    nlgcg_supports = []
    (
        u_nlgcg,
        c_nlgcg,
        times_nlgcg,
        supports_nlgcg,
        inner_loop,
        lgcg_lazy,
        lgcg_total,
        objective_values_nlgcg,
        dropped_tot,
        epsilons,
        all_information,
    ) = exp.solve(tol=5e-14, temperature=1, log_results=True, full_trace=True)
    local_residuals = adapt_time(
        times_nlgcg,
        [obj - optimum for obj in objective_values_nlgcg],
        frame=frame_size,
        resolution=resolution,
    )
    nlgcg_residuals.append(local_residuals)
    nlgcg_supports.append(supports_nlgcg)

    nlgcg_residuals_mean = np.mean(bring_to_same_length(nlgcg_residuals), axis=0)
    nlgcg_supports_mean = np.mean(bring_to_same_length(nlgcg_supports), axis=0)
    nlgcg_supports_std = np.std(bring_to_same_length(nlgcg_supports), axis=0)

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot residuals
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG"]
    styles = ["-"]
    for array, name, style in zip(
        [
            nlgcg_residuals_mean,
        ],
        names,
        styles,
    ):
        ax.semilogy(resolution * np.arange(len(array)), array, style, label=name)
    ax.fill(
        np.hstack(
            (
                resolution * np.arange(len(nlgcg_residuals_mean)),
                resolution * np.arange(len(nlgcg_residuals_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.max(bring_to_same_length(nlgcg_residuals), axis=0),
                np.min(bring_to_same_length(nlgcg_residuals), axis=0)[::-1],
            )
        ),
        "tab:blue",
        alpha=0.3,
    )
    plt.ylabel("Objective residual")
    plt.xlabel("Time (s)")
    plt.ylim(1e-10, 100)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "residuals.png", bbox_inches="tight")
    plt.close()

    # Plot supports
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG"]
    styles = ["-"]
    for array, name, style in zip(
        [
            nlgcg_supports_mean,
        ],
        names,
        styles,
    ):
        ax.semilogx(np.arange(len(array)), array, style, label=name)
    ax.fill(
        np.hstack(
            (
                np.arange(len(nlgcg_supports_mean)),
                np.arange(len(nlgcg_supports_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.array(nlgcg_supports_mean) - np.array(nlgcg_supports_std),
                np.array(nlgcg_supports_mean)[::-1]
                + np.array(nlgcg_supports_std)[::-1],
            )
        ),
        "tab:blue",
        alpha=0.3,
    )
    plt.ylabel("Support points")
    plt.xlabel("Iterations")
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "supports.png", bbox_inches="tight")
    plt.close()

    # Plot number of coefficients to optimize
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG"]
    styles = ["-"]
    colors = ["tab:blue"]
    for array, name, style, color in zip([nlgcg_supports_mean], names, styles, colors):
        ax.semilogx(np.arange(len(array)), array, style, label=name, c=color)
    ax.fill(
        np.hstack(
            (
                np.arange(len(nlgcg_supports_mean)),
                np.arange(len(nlgcg_supports_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.array(nlgcg_supports_mean) - np.array(nlgcg_supports_std),
                np.array(nlgcg_supports_mean)[::-1]
                + np.array(nlgcg_supports_std)[::-1],
            )
        ),
        "tab:blue",
        alpha=0.3,
    )
    plt.ylabel("Number of coefficients to optimize")
    plt.xlabel("Iterations")
    # plt.ylim(1e-12, 100);
    # plt.xlim(0, 100);
    ax.legend()
    plt.savefig(results_dir / "coefficients.png", bbox_inches="tight")
    plt.close()

    # Plot particles motion
    for ind in range(len(times_nlgcg) - 2):
        u_0, c_0 = all_information[3 * ind]
        u_1, c_1 = all_information[3 * ind + 1]
        newton_part = all_information[3 * ind + 2]
        p_u = p(u_0, c_0)
        old_points = u_0.support.copy()
        all_points = u_1.support.copy()
        if not len(old_points):
            new_points = all_points.copy()
        else:
            new_points = []
            for point in all_points:
                if np.min(np.linalg.norm(old_points - point, axis=1)) > 0:
                    new_points.append(point)
            new_points = np.array(new_points)
        trajectories = defaultdict(list)
        initial_positions = []
        for par_ind, params in enumerate(newton_part):
            reshaped = params[:-1].reshape(-1, Omega.shape[0] + 1)
            points = reshaped[:, 1:]
            for i, point in enumerate(points):
                trajectories[i].append((point[0], point[1]))
                if not par_ind:
                    initial_positions.append(point)
        initial_positions = np.array(initial_positions)
        start_points = []
        for point in all_points:
            if np.min(np.linalg.norm(initial_positions - point, axis=1)) < 1e-10:
                start_points.append(point)
        start_points = np.array(start_points)
        if not len(old_points):
            new_start_points = all_points.copy()
        else:
            new_start_points = []
            for point in start_points:
                if np.min(np.linalg.norm(old_points - point, axis=1)) > 0:
                    new_start_points.append(point)
            new_start_points = np.array(new_start_points)

        u_next, c_next = all_information[3 * ind + 3]
        p_u = p(u_next, c_next)
        P = lambda x: np.abs(p_u(x))
        a = np.arange(0, 1, 0.01)
        B, D = np.meshgrid(a, a)
        vals = np.array(
            [P(np.array([[x_1, x_2]])) for x_1, x_2 in zip(B.flatten(), D.flatten())]
        ).reshape((100, 100))
        plt.contourf(B, D, vals, levels=100)
        for i, trajectory in trajectories.items():
            traj_x = [position[0] for position in trajectory]
            traj_y = [position[1] for position in trajectory]
            if i:
                (line,) = plt.plot(traj_x, traj_y, c="black")
                line.set_path_effects(
                    [pe.Stroke(linewidth=2, foreground="white"), pe.Normal()]
                )
            else:
                (line,) = plt.plot(traj_x, traj_y, c="black", label="Position sliding")
                line.set_path_effects(
                    [pe.Stroke(linewidth=2, foreground="white"), pe.Normal()]
                )
        # for i, x in enumerate(all_points):
        #     if i:
        #         plt.plot([x[0]], [x[1]], "o", c="r", markersize=5)
        #     else:
        #         plt.plot(
        #             [x[0]], [x[1]], "o", c="r", markersize=5, label="Old support points"
        #         )
        # for i, x in enumerate(new_points):
        #     if i:
        #         plt.plot([x[0]], [x[1]], "o", c="g", markersize=5)
        #     else:
        #         plt.plot(
        #             [x[0]], [x[1]], "o", c="g", markersize=5, label="New support points"
        #         )
        for i, x in enumerate(start_points):
            if i:
                plt.plot([x[0]], [x[1]], "o", c="r", markersize=5)
            else:
                plt.plot(
                    [x[0]], [x[1]], "o", c="r", markersize=5, label="Start position"
                )
        for i, x in enumerate(new_start_points):
            if i:
                plt.plot([x[0]], [x[1]], "o", c="g", markersize=5)
            else:
                plt.plot(
                    [x[0]],
                    [x[1]],
                    "o",
                    c="g",
                    markersize=5,
                    label="New start positions",
                )
        plt.legend()
        plt.savefig(results_dir / f"position_sliding_{ind}.png", bbox_inches="tight")
        plt.close()

    logging.getLogger().setLevel(logging.INFO)  # Reinstate logging


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Nrun", type=int, default=10, help="Number runs of stochastic methods"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/source_identification",
        help="Directory for plots",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    create_plots()
