#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0"
)
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# init jax
_ = jnp.zeros(0)

import logging
import sys
import matplotlib.pyplot as plt
from pathlib import Path

module_path = Path(__file__).resolve().parent.parent
if module_path not in sys.path:
    sys.path.append(str(module_path))
src_path = (module_path / "src").resolve()
if src_path not in sys.path:
    sys.path.append(str(src_path))

from nlgcg import NLGCG
from src.lib.particle_descent import ParticleDescent
from src.lib.adaptive_refinement import AdaptiveRefinement

logging.getLogger().setLevel(logging.INFO)

# Generate Data and Define Functions

sigma_space = np.array([0, 2])
omega_space = np.array([[-1, 1], [-1, 1]])
Omega = np.vstack((sigma_space, omega_space))
d = omega_space.shape[0]
variance_exponent = 0.1
alpha = 1e-3
observation_resolution = 20
endpoint = False
max_radius = 1e-2

true_function = lambda x: np.minimum(1 - np.abs(x[:, 0]), 1 - np.abs(x[:, 1]))

max_radius = (Omega[0][1] - Omega[0][0]) / 100

observations = (
    np.array(
        np.meshgrid(
            *(
                np.linspace(
                    bound[0] + 1e-5,
                    bound[1] - np.sqrt(3) * 1e-5,
                    observation_resolution,
                    endpoint=endpoint,
                )
                for bound in omega_space
            )
        )
    )
    .reshape(len(omega_space), -1)
    .T
)


@jax.jit
def singleton_kernel(omega: np.ndarray):
    sigma = omega[0]
    x = omega[1:]
    inner = -jnp.sum((x - observations) ** 2, axis=1) / (2 * sigma**2)
    outer = (jnp.exp(inner) * sigma ** (variance_exponent)) / (
        jnp.sqrt(2 * jnp.pi) ** d
    )
    return outer


kernel = jax.vmap(singleton_kernel)

target = true_function(observations)


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
    input = raw_input[:-1].reshape(-1, d + 2)
    weights = input[:, 0]
    omega = input[:, 1:]
    return f(kernel(omega).T @ weights + constant * jnp.ones(target.shape))


@jax.jit
def j_N(raw_input: np.ndarray) -> float:
    input = raw_input[:-1].reshape(-1, d + 2)
    weights = input[:, 0]
    return f_N(raw_input) + g(weights)


grad_f_N = jax.jit(jax.grad(f_N))
grad_j_N = jax.jit(jax.grad(j_N))

optimum = 0.03394931259295045


def create_plots(Nrun: int = 10):

    exp_nlgcg = NLGCG(
        target=target,
        kernel=kernel,
        g=g,
        f=f,
        f_N=f_N,
        grad_f=grad_f,
        hess_f=hess_f,
        grad_f_N=grad_f_N,
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
    )

    exp_particle = ParticleDescent(
        m=250,
        j=j,
        p=p,
        grad_p=grad_p,
        Omega=Omega,
        a_parameter=0.001,
        b_parameter=0.001 / 2,
        kernel=kernel,
        constant_dim=len(target),
        kernel_dim=len(target),
        alpha=alpha,
        target=target,
        g=g,
        f=f,
        grad_f=grad_f,
        hess_f=hess_f,
        residual_tolerance=5e-14,
    )

    exp_adaptive = AdaptiveRefinement(
        observations=observations,
        j=j,
        p=p,
        grad_p=grad_p,
        hess_p=hess_p,
        Omega=Omega,
        kernel=kernel,
        constant_dim=len(target),
        kernel_dim=len(target),
        alpha=alpha,
        target=target,
        g=g,
        f=f,
        grad_f=grad_f,
        hess_f=hess_f,
        ssn_steps=1000,
    )

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

    resolution = 1  # time resolution for the plots (seconds)
    frame_size = 1000  # Time frame tracked for the residuals (seconds)

    # deterministic NLGCG
    logging.info("Running deterministic NLGCG")
    (
        u_opt,
        c_opt,
        times_det_nlgcg,
        supports_det_nlgcg,
        inner_loop,
        lgcg_lazy,
        lgcg_total,
        objective_values_det_nlgcg,
        dropped_tot,
        epsilons,
    ) = exp_nlgcg.solve(
        tol=5e-14,
        max_radius=max_radius,
        temperature=0.1,
        mode="deterministic",
        do_logging=False,
    )
    residuals_det_nlgcg = adapt_time(
        times_det_nlgcg,
        [obj - optimum for obj in objective_values_det_nlgcg],
        frame=frame_size,
        resolution=resolution,
    )

    # Adaptive refinement
    logging.info("Running adaptive refinement")
    (
        cells_dict,
        vertices_dict,
        vertices,
        u,
        objective_values_adaptive,
        times_adaptive,
        actives,
        supports_adaptive,
    ) = exp_adaptive.solve(max_iters=50, do_logging=False)
    residuals_adaptive = adapt_time(
        times_adaptive,
        [obj - optimum for obj in objective_values_adaptive],
        frame=frame_size,
        resolution=resolution,
    )

    # NLGCG stochastic
    nlgcg_residuals = []
    nlgcg_supports = []
    for i in range(Nrun):
        logging.info(f"Running stochastic NLGCG (trial {i})")
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
        ) = exp_nlgcg.solve(
            tol=5e-14, max_radius=max_radius, temperature=0.1, do_logging=False
        )
        local_residuals = adapt_time(
            times_nlgcg,
            [obj - optimum for obj in objective_values_nlgcg],
            frame=frame_size,
            resolution=resolution,
        )
        nlgcg_residuals.append(local_residuals)
        nlgcg_supports.append(supports_nlgcg)

    nlgcg_residuals_mean = np.mean(bring_to_same_length(nlgcg_residuals), axis=0)
    nlgcg_residuals_std = np.std(bring_to_same_length(nlgcg_residuals), axis=0)
    nlgcg_supports_mean = np.mean(bring_to_same_length(nlgcg_supports), axis=0)
    nlgcg_supports_std = np.std(bring_to_same_length(nlgcg_supports), axis=0)

    # Particle Descent Stochastic
    particle_residuals = []
    particle_supports = []
    for i in range(Nrun):
        logging.info(f"Running Particle descent (trial {i})")
        max_iter = int(1e4)
        u, c, objective_values_particle, supports_particle, times_particle, success = (
            exp_particle.solve(max_iters=max_iter, mode="exponential", do_logging=False)
        )
        local_residuals = adapt_time(
            times_particle,
            [obj - optimum for obj in objective_values_particle],
            frame=frame_size,
            resolution=resolution,
        )
        particle_residuals.append(local_residuals)
        particle_supports.append(supports_particle)

    particle_residuals_filtered = []
    converged_frac = 0
    for i in range(len(particle_residuals)):
        if particle_residuals[i][-1] < 1e-5:
            particle_residuals_filtered.append(particle_residuals[i])
            converged_frac += 1 / Nrun

    if converged_frac > 1.9 / Nrun:
        particle_residuals_mean = np.mean(
            bring_to_same_length(particle_residuals_filtered), axis=0
        )
        particle_residuals_std = np.std(
            bring_to_same_length(particle_residuals_filtered), axis=0
        )
    else:
        particle_residuals_mean = np.mean(
            bring_to_same_length(particle_residuals), axis=0
        )
        particle_residuals_std = np.std(
            bring_to_same_length(particle_residuals), axis=0
        )

    particle_supports_mean = np.mean(bring_to_same_length(particle_supports), axis=0)
    particle_supports_std = np.std(bring_to_same_length(particle_supports), axis=0)
    logging.info(f"Particle descent converged in {converged_frac*100}% of cases.")

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot residuals
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "Adaptive Refinement", "RNLGCG", "Particle Descent"]
    styles = ["-", "-.", "--", ":"]
    for array, name, style in zip(
        [
            residuals_det_nlgcg,
            residuals_adaptive,
            nlgcg_residuals_mean,
            particle_residuals_mean,
        ],
        names,
        styles,
    ):
        ax.semilogy(resolution * np.arange(len(array)), array, style, label=name)

    ax.fill(
        np.hstack(
            (
                resolution * np.arange(len(particle_residuals_mean)),
                resolution * np.arange(len(particle_residuals_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.array(particle_residuals_mean) - np.array(particle_residuals_std),
                np.array(particle_residuals_mean)[::-1]
                + np.array(particle_residuals_std)[::-1],
            )
        ),
        "red",
        alpha=0.3,
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
                np.array(nlgcg_residuals_mean) - np.array(nlgcg_residuals_std),
                np.array(nlgcg_residuals_mean)[::-1]
                + np.array(nlgcg_residuals_std)[::-1],
            )
        ),
        "green",
        alpha=0.3,
    )

    plt.ylabel("Objective residual")
    plt.xlabel("Time (s)")
    plt.ylim(1e-12, 100)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "residuals.png", bbox_inches="tight")
    plt.close()

    # Plot supports
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "Adaptive Refinement", "RNLGCG", "Particle Descent"]
    styles = ["-", "-.", "--", ":"]
    for array, name, style in zip(
        [
            supports_det_nlgcg,
            supports_adaptive,
            nlgcg_supports_mean,
            particle_supports_mean,
        ],
        names,
        styles,
    ):
        ax.semilogx(np.arange(len(array)), array, style, label=name)

    ax.fill(
        np.hstack(
            (
                np.arange(len(particle_supports_mean)),
                np.arange(len(particle_supports_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.array(particle_supports_mean) - np.array(particle_supports_std),
                np.array(particle_supports_mean)[::-1]
                + np.array(particle_supports_std)[::-1],
            )
        ),
        "red",
        alpha=0.3,
    )
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
        "green",
        alpha=0.3,
    )
    plt.ylabel("Support points")
    plt.xlabel("Iterations")
    # plt.ylim(1e-12, 100)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "support_size.png", bbox_inches="tight")
    plt.close()

    # Plot number of coefficients to optimize
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "Adaptive Refinement", "RNLGCG"]
    styles = ["-", "-.", "--"]
    for array, name, style in zip(
        [supports_det_nlgcg, actives, nlgcg_supports_mean], names, styles
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
        "green",
        alpha=0.3,
    )
    plt.ylabel("Number of coefficients to optimize")
    plt.xlabel("Iterations")
    # plt.ylim(1e-12, 100);
    # plt.xlim(0, 100);
    ax.legend()
    plt.savefig(results_dir / "support_size_2.png", bbox_inches="tight")
    plt.close()

    # Plot the function with reconstruction error
    resolution = 100
    a_1 = np.linspace(omega_space[0][0], omega_space[0][1], resolution, endpoint=False)
    a_2 = np.linspace(omega_space[1][0], omega_space[1][1], resolution, endpoint=False)
    x, y = np.meshgrid(a_1, a_2)
    points = np.array(list(zip(x.flatten(), y.flatten())))
    vals = true_function(points).reshape((resolution, resolution))

    def plot_kernel(omega: np.ndarray):
        sigma = omega[0]
        x_omega = omega[1:]
        inner = -(jnp.linalg.norm(x_omega - points, axis=1) ** 2) / (2 * sigma**2)
        outer = (jnp.exp(inner) * sigma**variance_exponent) / (np.sqrt(2 * np.pi) ** d)
        return outer

    pred_vals = u_nlgcg.duality_pairing(jax.vmap(plot_kernel)).reshape(
        (resolution, resolution)
    ) + c_nlgcg * np.ones((resolution, resolution))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    contour1 = ax1.contourf(x, y, vals, levels=100)
    fig.colorbar(contour1, ax=ax1)
    contour2 = ax2.contourf(x, y, pred_vals, levels=100)
    fig.colorbar(contour2, ax=ax2)
    contour3 = ax3.contourf(x, y, np.abs(pred_vals - vals), levels=100)
    fig.colorbar(contour3, ax=ax3)
    for i, x in enumerate(u_nlgcg.support):
        if u_nlgcg.coefficients[i] < 0:
            color = "b"
        else:
            color = "r"
        ax2.add_patch(
            plt.Circle((x[1], x[2]), radius=x[0], color=color, fill=False, alpha=0.5)
        )
        ax3.add_patch(
            plt.Circle((x[1], x[2]), radius=x[0], color=color, fill=False, alpha=0.5)
        )
    ax1.set_xlim(omega_space[0][0], omega_space[0][1])
    ax1.set_ylim(omega_space[1][0], omega_space[1][1])
    ax2.set_xlim(omega_space[0][0], omega_space[0][1])
    ax2.set_ylim(omega_space[1][0], omega_space[1][1])
    ax3.set_xlim(omega_space[0][0], omega_space[0][1])
    ax3.set_ylim(omega_space[1][0], omega_space[1][1])
    ax1.set_xlabel("True function")
    ax2.set_xlabel("Predicted function")
    ax3.set_xlabel("Absolute error in prediction")
    plt.savefig(results_dir / "reconstruction_error.png", bbox_inches="tight")
    plt.close()
    logging.error(
        f"Function reconstruction error L2: {np.linalg.norm(pred_vals-vals)/len(vals)}, Linf: {np.max(np.abs(pred_vals-vals))}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Nrun", type=int, default=10, help="Number runs of stochastic methods"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/function_approximation",
        help="Directory for plots",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    create_plots(Nrun=args.Nrun)
