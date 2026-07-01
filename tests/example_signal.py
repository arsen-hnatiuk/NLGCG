import numpy as np
import os
import jax
import jax.numpy as jnp
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

# Generate Data and Define Functions

observation_resolution = 120
Omega = np.array([[0, observation_resolution // 2]])
alpha = 1e-1
true_sources = np.array([[3.125], [7.0], [np.sqrt(179.0)]])
true_weights = np.array([-1.0, 0.7, 0.5])
true_measure = Measure(support=true_sources, coefficients=true_weights)
max_radius = (Omega[0][1] - Omega[0][0]) / 100.0
optimum = 0.21975385192787872


def define_experiment():
    observations = np.arange(0, 1, 1 / observation_resolution)

    @jax.jit
    def singleton_kernel(omega: float):
        return jnp.sin(2 * jnp.pi * omega * observations)

    kernel = jax.vmap(singleton_kernel)

    target = true_measure.duality_pairing(kernel)

    @jax.jit
    def g(w: np.ndarray) -> float:
        return alpha * jnp.linalg.norm(w, ord=1)

    @jax.jit
    def f(y: np.ndarray) -> float:
        return 0.5 * jnp.sum((y - target) ** 2)

    grad_f = jax.jit(jax.grad(f))
    hess_f = jax.jit(jax.hessian(f))

    @jax.jit
    def _j(coefficients, support, c):
        if support.shape[0] > 0:
            values = kernel(support)
            y = values.T @ coefficients
        else:
            y = 0
        return f(y + c * jnp.ones(target.shape)) + g(coefficients)

    def j(u: Measure, c: float):
        return _j(u.coefficients, u.support, c)

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
        input = raw_input[:-1].reshape(-1, Omega.shape[0] + 1)
        weights = input[:, 0]
        omega = input[:, 1:]
        return f(kernel(omega).T @ weights + constant * jnp.ones(target.shape))

    @jax.jit
    def j_N(raw_input: np.ndarray) -> float:
        input = raw_input[:-1].reshape(-1, Omega.shape[0] + 1)
        weights = input[:, 0]
        return f_N(raw_input) + g(weights)

    grad_f_N = jax.jit(jax.grad(f_N))
    hess_f_N = jax.jit(jax.hessian(f_N))
    grad_j_N = jax.jit(jax.grad(j_N))

    return (
        observations,
        target,
        kernel,
        g,
        f,
        f_N,
        grad_f,
        hess_f,
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
        observations,
        target,
        kernel,
        g,
        f,
        f_N,
        grad_f,
        hess_f,
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
        global_search_resolution=100,
        dual_variable_goodness=0.3,
        constant_dim=len(target),
        kernel_dim=len(target),
        newton_tolerance=2e-2,
        max_radius=max_radius,
    )
    return exp, p


def define_particle_descent_experiment(m=50, a_parameter=0.001, b_parameter=0.0005):
    (
        observations,
        target,
        kernel,
        g,
        f,
        f_N,
        grad_f,
        hess_f,
        grad_f_N,
        hess_f_N,
        j,
        j_N,
        p,
        grad_p,
        hess_p,
        grad_j_N,
    ) = define_experiment()
    exp = ParticleDescent(
        m=m,
        j=j,
        p=p,
        grad_p=grad_p,
        Omega=Omega,
        a_parameter=a_parameter,
        b_parameter=b_parameter,
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
    return exp


def define_adaptive_refinement_experiment():
    (
        observations,
        target,
        kernel,
        g,
        f,
        f_N,
        grad_f,
        hess_f,
        grad_f_N,
        hess_f_N,
        j,
        j_N,
        p,
        grad_p,
        hess_p,
        grad_j_N,
    ) = define_experiment()
    exp = AdaptiveRefinement(
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
    return exp


def create_particle_matrix():

    # NLGCG
    exp, p = define_nlgcg_experiment()
    (
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
    ) = exp.solve(tol=5e-14, temperature=0.1)

    print(
        f"found optimimum with value {objective_values[-1]} (difference to ref {optimum} is {objective_values[-1] - optimum})"
    )
    print(f"times: {np.array(times)}")
    print(f"dropped: {dropped_tot}")
    print(f"constant {c}")
    print(f"solution {u}")

    # logging.getLogger().setLevel(logging.WARN)

    runs_per_Nparticle = {12: 50, 24: 50, 50: 20, 100: 10, 200: 5}
    success_per_Nparticle = {key: 0 for key in runs_per_Nparticle.keys()}

    for Nparticle, Nruns in runs_per_Nparticle.items():
        # Particle Gradient Descent (a_parameter does not matter so much because of linesearch)
        print(f"running trials with {Nparticle*2} particles")
        a_parameter = 0.0000001
        b_parameter_factor = 0.1
        b_parameter = b_parameter_factor * a_parameter
        exp = define_particle_descent_experiment(
            m=Nparticle, a_parameter=a_parameter, b_parameter=b_parameter
        )

        # run Nruns to determine success probability
        for it in range(Nruns):
            u, c, objective_values, supports, times, success = exp.solve(
                max_iters=int(1e6), max_time=5 * 60, mode="uniform"
            )

            print(
                f"\t({it+1} of {Nruns}): found optimum with value {objective_values[-1]} (residual {objective_values[-1] - optimum})"
            )
            # print(u.to_matrix()[np.abs(u.coefficients) > 1e-3, :])

            success = objective_values[-1] - optimum < 1e-3
            success_per_Nparticle[Nparticle] += int(success)

            if success:
                plt.semilogy(
                    np.arange(len(objective_values)),
                    np.array(objective_values) - optimum,
                )
                plt.show()
                plt.close()

        print(
            {
                Npart: success_per_Nparticle[Npart] / Nrun
                for Npart, Nrun in runs_per_Nparticle.items()
            }
        )
        del exp


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
    resolution = 0.1  # time resolution for the plots (seconds)
    frame_size = 100  # Time frame tracked for the residuals (seconds)

    # NLGCG
    nlgcg_residuals = []
    nlgcg_supports = []
    nlgcg_converged = 0
    for i in range(Nrun):
        logging.info(f"Running NLGCG (trial {i+1})")
        exp_nlgcg, p = define_nlgcg_experiment()
        (
            u_opt,
            c_opt,
            times_nlgcg,
            supports_nlgcg,
            inner_loop,
            lgcg_lazy,
            lgcg_total,
            objective_values_nlgcg,
            dropped_tot,
            epsilons,
            all_information,
        ) = exp_nlgcg.solve(tol=5e-14, temperature=0.1, log_results=False)
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

    # Particle Descent Stochastic
    particle_residuals = []
    particle_supports = []
    particle_converged = 0
    for i in range(Nrun):
        success = False
        while not success:
            logging.info(f"Running Particle descent (trial {i+1})")
            exp_particle = define_particle_descent_experiment()
            (
                u,
                c,
                objective_values_particle,
                supports_particle,
                times_particle,
                success,
            ) = exp_particle.solve(
                max_time=frame_size, mode="uniform", log_results=False
            )
        local_residuals = adapt_time(
            times_particle,
            [obj - optimum for obj in objective_values_particle],
            frame=frame_size,
            resolution=resolution,
        )
        if local_residuals[-1] < 1e-5:
            particle_converged += 1
        particle_residuals.append(local_residuals)
        particle_supports.append(supports_particle)
        del exp_particle
    logging.info(
        f"Particle descent converged in {(particle_converged/Nrun)*100}% of cases."
    )

    particle_residuals_mean = np.mean(bring_to_same_length(particle_residuals), axis=0)
    particle_supports_mean = np.mean(bring_to_same_length(particle_supports), axis=0)
    particle_supports_std = np.std(bring_to_same_length(particle_supports), axis=0)

    # Adaptive refinement
    logging.info("Running adaptive refinement")
    exp_adaptive = define_adaptive_refinement_experiment()
    (
        cells_dict,
        vertices_dict,
        vertices,
        u,
        objective_values_adaptive,
        times_adaptive,
        actives,
        supports_adaptive,
    ) = exp_adaptive.solve(max_time=frame_size, log_results=False)
    residuals_adaptive = adapt_time(
        times_adaptive,
        [obj - optimum for obj in objective_values_adaptive],
        frame=frame_size,
        resolution=resolution,
    )
    del exp_adaptive

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot residuals
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "Particle Descent", "Adaptive Refinement"]
    styles = ["-", "--", ":"]
    for array, name, style in zip(
        [
            nlgcg_residuals_mean,
            particle_residuals_mean,
            residuals_adaptive,
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
                np.max(bring_to_same_length(particle_residuals), axis=0),
                np.min(bring_to_same_length(particle_residuals), axis=0)[::-1],
            )
        ),
        "tab:orange",
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
    names = ["NLGCG", "Particle Descent", "Adaptive Refinement"]
    styles = ["-", "--", ":"]
    for array, name, style in zip(
        [
            nlgcg_supports_mean,
            particle_supports_mean,
            supports_adaptive,
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
        "tab:orange",
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
    names = ["NLGCG", "Adaptive Refinement"]
    styles = ["-", ":"]
    colors = ["tab:blue", "tab:green"]
    for array, name, style, color in zip(
        [nlgcg_supports_mean, actives], names, styles, colors
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
    plt.ylabel("Number of coefficients to optimize")
    plt.xlabel("Iterations")
    # plt.ylim(1e-12, 100);
    # plt.xlim(0, 100);
    ax.legend()
    plt.savefig(results_dir / "coefficients.png", bbox_inches="tight")
    plt.close()

    # Plot optimal/true support and dual variable
    a = np.arange(Omega[0][0], Omega[0][1], 0.001)
    p_u = p(u_opt, c_opt)
    vals = p_u(a)
    plt.figure(figsize=(5, 4))
    plt.plot(a, vals)
    plt.axvline(x=-1, linestyle="-", c="r", label="Support")
    plt.axvline(x=-1, linestyle="-", c="g", label="Truth")
    for i, pos in enumerate(u_opt.support):
        ymax = 0.5 + (
            u_opt.coefficients[i] * alpha * 0.25 + 0.05 * np.sign(u_opt.coefficients[i])
        )
        plt.axvline(x=pos, ymin=0.5, ymax=ymax, linestyle="-", c="r")
    for point, weight in zip(true_sources, true_weights):
        ymax = 0.5 + (weight * alpha * 0.25 + 0.05 * np.sign(weight))
        plt.axvline(x=point, ymin=0.5, ymax=ymax, alpha=0.5, linestyle="-", c="g")
    plt.xlabel("Frequency")
    plt.ylim(-alpha * 1.1, alpha * 1.1)
    plt.xlim(0, 20)
    plt.legend()
    plt.savefig(results_dir / "sources.png", bbox_inches="tight")
    plt.close()

    def signal(x, sources, weights, c):
        # Input is 2D array of shape (number of points, Omega dimension)
        weighted_signal = c * np.ones(x.shape)
        for point, weight in zip(sources, weights):
            weighted_signal += weight * np.sin(2 * np.pi * point * x).flatten()
        return weighted_signal

    # Plot the measured signal
    a = np.arange(0, 1, 0.001)
    true_signal = signal(a, true_sources, true_weights, 0.0)
    plt.figure(figsize=(5, 4))
    plt.plot(a, true_signal)
    plt.xlabel("Time")
    plt.savefig(results_dir / "signal.png", bbox_inches="tight")
    plt.close()

    # Plot reconstruction error
    a = np.arange(0, 1, 0.001)
    predicted_signal = signal(a, u_opt.support, u_opt.coefficients, c_opt)
    error = true_signal - predicted_signal
    logging.error(
        f"Signal reconstruction error L2: {np.linalg.norm(error)/np.sqrt(len(a))}, Linf: {np.max(np.abs(error))}"
    )
    plt.figure(figsize=(5, 4))
    plt.plot(a, np.abs(true_signal - predicted_signal))
    plt.xlabel("Time")
    plt.savefig(results_dir / "signal_error.png", bbox_inches="tight")
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
        default="results/signal_processing",
        help="Directory for plots",
    )
    parser.add_argument(
        "--particle-matrix",
        action="store_true",
        help="Generate a particle matrix plot instead of algorithm comparison",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.particle_matrix:
        create_particle_matrix()
    else:
        create_plots(Nrun=args.Nrun)
