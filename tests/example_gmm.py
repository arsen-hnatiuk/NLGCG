import numpy as np
import os
import torch
import jax
import math
import jax.numpy as jnp
import logging
import sys
import joblib
import matplotlib.pyplot as plt
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

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0 xla_python_client_preallocate=false xla_python_client_mem_fraction=0.5"
)
logging.getLogger().setLevel(logging.INFO)
# init jax
jax.config.update("jax_enable_x64", True)
_ = jnp.zeros(0)

# Generate Data and Define Functions

Omega = np.array([[-35, 35], [-35, 35]])
d = Omega.shape[0]
source_number = 25
circle_radius = 30
sample_size_raw = 30000
sample_size = 24000
alpha = 1e-4
tau = 0.1
endpoint = False
max_radius = (Omega[0][1] - Omega[0][0]) / 100
optimum = 0.000218755398904130

# Draw sample from true distribution
angles = torch.linspace(0, 2 * math.pi, source_number + 1)[:-1]
true_means = circle_radius * torch.stack([torch.cos(angles), torch.sin(angles)], 1)
torch.manual_seed(31415)
true_weights = torch.distributions.Dirichlet(torch.ones(source_number)).sample()
torch.manual_seed(42)
probs = true_weights / true_weights.sum()
component_idx = torch.multinomial(probs, sample_size_raw, replacement=True)
X_all = true_means[component_idx] + torch.randn(sample_size_raw, 2)
torch.manual_seed(0)
n_test = sample_size_raw // 5
perm = torch.randperm(sample_size_raw)
X = np.array(X_all[perm[n_test:]])  # must be shape=(sample_size, d)


def define_experiment():
    @jax.jit
    def singleton_kernel_k(omega_1: np.ndarray, omega_2: np.ndarray):
        # omega_1 is 1D, omega_2 is 2D, output vector
        diff = omega_2 - omega_1
        inner = -jnp.sum(diff**2, axis=1) / (4 * (1 + tau**2))
        outer = jnp.exp(inner) / (jnp.sqrt(4 * (1 + tau**2) * jnp.pi) ** d)
        return outer

    @jax.jit
    def kernel_k(omega_1: np.ndarray, omega_2: np.ndarray):
        # Both arguments are 2D, output matrix
        diff = omega_1[:, None, :] - omega_2[None, :, :]
        inner = -jnp.sum(diff**2, axis=-1) / (4 * (1 + tau**2))
        outer = jnp.exp(inner) / (jnp.sqrt(4 * (1 + tau**2) * jnp.pi) ** d)
        return outer

    @jax.jit
    def singleton_kernel_s(omega: np.ndarray):
        diff = X - omega
        inner = -jnp.sum(diff**2, axis=1) / (2 * (1 + 2 * tau**2))
        outer = jnp.exp(inner) / (jnp.sqrt(2 * (1 + 2 * tau**2) * jnp.pi) ** d)
        return jnp.mean(outer)

    kernel_s = jax.vmap(singleton_kernel_s)

    loss_offset = 0.003756  # float(1.0 / (2 * sample_size * (4 * math.pi * tau**2)))

    @jax.jit
    def g(w: np.ndarray) -> float:
        return alpha * jnp.linalg.norm(w, ord=1)

    @jax.jit
    def f_N(raw_input: np.ndarray) -> float:
        input = raw_input.reshape(-1, d + 1)
        weights = input[:, 0]
        positions = input[:, 1:]
        K_matrix = kernel_k(positions, positions)
        S_vector = kernel_s(positions)
        return 0.5 * weights @ K_matrix @ weights - S_vector @ weights + loss_offset

    @jax.jit
    def j_N(raw_input: np.ndarray) -> float:
        input = raw_input.reshape(-1, d + 1)
        weights = input[:, 0]
        return f_N(raw_input) + g(weights)

    def j(u: Measure, c: float) -> float:
        u_matrix = u.to_matrix(param_dimension=2)
        return j_N(u_matrix)

    @jax.jit
    def p_raw(parameters: np.ndarray, omega: np.ndarray):
        weights = parameters[:, 0]
        positions = parameters[:, 1:]
        K_vector = singleton_kernel_k(omega, positions)
        S_value = singleton_kernel_s(omega)
        to_return = -K_vector @ weights + S_value
        return to_return

    grad_p_raw = jax.jit(jax.grad(p_raw, argnums=1))
    hess_p_raw = jax.jit(jax.hessian(p_raw, argnums=1))

    p_raw = jax.jit(jax.vmap(p_raw, in_axes=(None, 0), out_axes=0))
    grad_p_raw = jax.jit(jax.vmap(grad_p_raw, in_axes=(None, 0), out_axes=0))
    hess_p_raw = jax.jit(jax.vmap(hess_p_raw, in_axes=(None, 0), out_axes=0))

    p = lambda u, c: jax.jit(lambda omega: p_raw(u.to_matrix(len(Omega)), omega))
    grad_p = lambda u, c: jax.jit(
        lambda omega: grad_p_raw(u.to_matrix(len(Omega)), omega)
    )
    hess_p = lambda u, c: jax.jit(
        lambda omega: hess_p_raw(u.to_matrix(len(Omega)), omega)
    )

    grad_f_N = jax.jit(jax.grad(f_N))
    hess_f_N = jax.jit(jax.hessian(f_N))
    grad_j_N = jax.jit(jax.grad(j_N))

    return (
        kernel_k,
        kernel_s,
        loss_offset,
        g,
        f_N,
        grad_f_N,
        hess_f_N,
        j,
        j_N,
        p,
        grad_p,
        hess_p,
        grad_j_N,
    )


def define_nlgcg_experiment():
    (
        kernel_k,
        kernel_s,
        loss_offset,
        g,
        f_N,
        grad_f_N,
        hess_f_N,
        j,
        j_N,
        p,
        grad_p,
        hess_p,
        grad_j_N,
    ) = define_experiment()
    exp = NLGCG(
        g=g,
        kernel_k=kernel_k,
        kernel_s=kernel_s,
        loss_offset=loss_offset,
        f_N=f_N,
        target=X,
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
        newton_tolerance=1,
        max_radius=max_radius,
        experiment_type="RKHS",
    )
    return exp


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


def create_plots(Nrun: int = 10):
    resolution = 1  # time resolution for the plots (seconds)
    frame_size = 1000  # Time frame tracked for the residuals (seconds)

    # NLGCG
    nlgcg_residuals = []
    nlgcg_supports = []
    nlgcg_converged = 0
    for i in range(Nrun):
        logging.info(f"Running NLGCG (trial {i+1})")
        exp_nlgcg = define_nlgcg_experiment()
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
        ) = exp_nlgcg.solve(tol=5e-14, temperature=1, log_results=False)
        local_residuals = adapt_time(
            times_nlgcg,
            [obj - optimum for obj in objective_values_nlgcg],
            frame=frame_size,
            resolution=resolution,
        )
        if local_residuals[-1] < 1e-5:
            nlgcg_converged += 1
        nlgcg_residuals.append(local_residuals)
        nlgcg_supports.append(supports_nlgcg)
        del exp_nlgcg
    logging.info(f"NLGCG converged in {(nlgcg_converged/Nrun)*100}% of cases.")

    nlgcg_residuals_mean = np.mean(bring_to_same_length(nlgcg_residuals), axis=0)
    nlgcg_supports_mean = np.mean(bring_to_same_length(nlgcg_supports), axis=0)
    nlgcg_supports_std = np.std(bring_to_same_length(nlgcg_supports), axis=0)

    # Particle descent metrics from cluster
    particle_descent_times = np.array(joblib.load(module_path / "tests/times.joblib"))
    particle_descent_supports = np.array(
        joblib.load(module_path / "tests/supports.joblib")
    )
    particle_descent_objectives = np.array(
        joblib.load(module_path / "tests/objectives.joblib")
    )
    particle_descent_residuals = adapt_time(
        particle_descent_times,
        [obj - optimum for obj in particle_descent_objectives],
        frame=frame_size,
        resolution=resolution,
    )

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot residuals
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "FS&P"]
    styles = ["-", "--"]
    colors = ["tab:blue", "tab:orange"]
    for array, name, style, color in zip(
        [nlgcg_residuals_mean, particle_descent_residuals],
        names,
        styles,
        colors,
    ):
        ax.semilogy(
            resolution * np.arange(len(array)), array, style, label=name, color=color
        )
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
    plt.ylim(1e-10, 1e-2)
    plt.xlim(0, 400)
    ax.legend()
    plt.savefig(results_dir / "residuals.png", bbox_inches="tight")
    plt.close()

    # Plot supports
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "FS&P"]
    styles = ["-", "--"]
    for array, name, style, color in zip(
        [
            nlgcg_supports_mean,
            particle_descent_supports,
        ],
        names,
        styles,
        colors,
    ):
        ax.semilogx(np.arange(len(array)), array, style, label=name, color=color)
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
    # plt.ylim(1e-12, 100)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "supports.png", bbox_inches="tight")
    plt.close()

    # Plot the true and predicted sources
    fig, ax = plt.subplots(figsize=(5, 4))
    predicted_weights = u_nlgcg.coefficients / np.linalg.norm(u_nlgcg.coefficients)
    for i, x in enumerate(u_nlgcg.support):
        if not i:
            ax.add_patch(
                plt.Circle(
                    (x[0], x[1]),
                    radius=max(1, predicted_weights[i] * 10),
                    color="blue",
                    fill=False,
                    alpha=0.5,
                    label="Predicted source",
                )
            )
        else:
            ax.add_patch(
                plt.Circle(
                    (x[0], x[1]),
                    radius=max(1, predicted_weights[i] * 10),
                    color="blue",
                    fill=False,
                    alpha=0.5,
                )
            )
    for i, (x, true_weight) in enumerate(zip(true_means, probs)):
        if not i:
            ax.add_patch(
                plt.Circle(
                    (x[0], x[1]),
                    radius=max(1, true_weight * 10),
                    color="red",
                    fill=False,
                    alpha=0.5,
                    label="True source",
                )
            )
        else:
            ax.add_patch(
                plt.Circle(
                    (x[0], x[1]),
                    radius=max(1, true_weight * 10),
                    color="red",
                    fill=False,
                    alpha=0.5,
                )
            )
    ax.set_xlim(Omega[0][0], Omega[0][1])
    ax.set_ylim(Omega[1][0], Omega[1][1])
    ax.set_xlabel("True and predicted sources")
    ax.legend()
    plt.savefig(results_dir / "sources.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Nrun", type=int, default=10, help="Number runs of stochastic methods"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/gmm",
        help="Directory for plots",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    create_plots()
