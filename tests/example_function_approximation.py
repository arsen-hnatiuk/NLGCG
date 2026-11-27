import numpy as np
import os
import jax
import jax.numpy as jnp
import logging
import pickle
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

module_path = Path(__file__).resolve().parent.parent
if module_path not in sys.path:
    sys.path.append(str(module_path))
src_path = (module_path / "src").resolve()
if src_path not in sys.path:
    sys.path.append(str(src_path))

from src.lib.measure import Measure

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=0 xla_python_client_preallocate=false xla_python_client_mem_fraction=0.5"
)
logging.getLogger().setLevel(logging.INFO)
# init jax
jax.config.update("jax_enable_x64", True)
_ = jnp.zeros(0)

# Generate Data and Define Functions

sigma_space = np.array([0, 2])
omega_space = np.array([[-1, 1], [-1, 1]])
corners = np.array(list(product(*omega_space)))
larger_omega_space = np.array([[-1.1, 1.1], [-1.1, 1.1]])
Omega = np.vstack((sigma_space, larger_omega_space))
d = omega_space.shape[0]
variance_exponent = 0.1
alpha = 1e-5
observation_resolution = 20
test_resolution = 100
endpoint = False
max_radius = (Omega[0][1] - Omega[0][0]) / 100
true_function = lambda x: np.minimum(1 - np.abs(x[:, 0]), 1 - np.abs(x[:, 1]))
optimum = 0.0003470297509151517
resolution = 1  # time resolution for the plots (seconds)
frame_size = 1000  # Time frame tracked for the residuals (seconds)
tol = 1e-12
temperature = 1
newton_tolerance = 2e-1


def define_experiment():
    observations = (
        np.array(
            np.meshgrid(
                *(
                    np.linspace(
                        bound[0], bound[1], observation_resolution, endpoint=True
                    )
                    for bound in omega_space
                )
            )
        )
        .reshape(len(omega_space), -1)
        .T
    )
    train_edge_indices = np.array(
        [
            any(obs == omega_space[0][0]) or any(obs == omega_space[0][1])
            for obs in observations
        ]
    )
    train_corner_indices = np.array(
        [not all(np.linalg.norm(corners - obs, axis=1)) for obs in observations]
    )
    train_mesh = (
        (omega_space[0][1] - omega_space[0][0]) / (observation_resolution - 1)
    ) ** 2  # Assume square domain
    train_quadrature = train_mesh * np.ones(len(observations))  # Trapezoid rule
    train_quadrature[train_edge_indices] /= 2  # boundary points
    train_quadrature[train_corner_indices] /= 2  # corner points
    train_quadrature = np.diag(train_quadrature)
    inverse_train_quadrature = np.linalg.inv(train_quadrature)

    test_observations = (
        np.array(
            np.meshgrid(
                *(
                    np.linspace(bound[0], bound[1], test_resolution, endpoint=True)
                    for bound in omega_space
                )
            )
        )
        .reshape(len(omega_space), -1)
        .T
    )
    test_edge_indices = np.array(
        [
            any(obs == omega_space[0][0]) or any(obs == omega_space[0][1])
            for obs in test_observations
        ]
    )
    test_corner_indices = np.array(
        [not all(np.linalg.norm(corners - obs, axis=1)) for obs in test_observations]
    )
    test_mesh = (
        (omega_space[0][1] - omega_space[0][0]) / (test_resolution - 1)
    ) ** 2  # Assume square domain
    test_quadrature = test_mesh * np.ones(len(test_observations))  # Trapezoid rule
    test_quadrature[test_edge_indices] /= 2  # boundary points
    test_quadrature[test_corner_indices] /= 2  # corner points
    test_quadrature = np.diag(test_quadrature)

    @jax.jit
    def singleton_kernel(omega: np.ndarray):
        sigma = omega[0]
        x = omega[1:]
        inner = -jnp.sum((x - observations) ** 2, axis=1) / (2 * sigma**2)
        outer = (jnp.exp(inner) * sigma ** (variance_exponent)) / (
            jnp.sqrt(2 * jnp.pi) ** d
        )
        return outer

    @jax.jit
    def test_singleton_kernel(omega: np.ndarray):
        sigma = omega[0]
        x = omega[1:]
        inner = -jnp.sum((x - test_observations) ** 2, axis=1) / (2 * sigma**2)
        outer = (jnp.exp(inner) * sigma ** (variance_exponent)) / (
            jnp.sqrt(2 * jnp.pi) ** d
        )
        return outer

    kernel = jax.vmap(singleton_kernel)
    test_kernel = jax.vmap(test_singleton_kernel)

    target = true_function(observations)
    test_target = true_function(test_observations)

    @jax.jit
    def g(w: np.ndarray) -> float:
        return alpha * jnp.linalg.norm(w, ord=1)

    @jax.jit
    def f(y):
        residuum = y - target
        return 0.5 * residuum @ train_quadrature @ residuum

    @jax.jit
    def test_f(y):
        residuum = y - test_target
        return 0.5 * residuum @ test_quadrature @ residuum

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
        test_f,
        test_kernel,
        test_target,
        test_observations,
        inverse_train_quadrature,
    )


def define_nlgcg_experiment():
    from nlgcg import NLGCG

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
        test_f,
        test_kernel,
        test_target,
        test_observations,
        inverse_train_quadrature,
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
        global_search_resolution=5,
        dual_variable_goodness=0.3,
        constant_dim=len(target),
        kernel_dim=len(target),
        newton_tolerance=newton_tolerance,
        max_radius=max_radius,
    )
    return exp, test_f, test_kernel, test_target, test_observations


def define_particle_descent_experiment(m=250, a_parameter=1e-5, b_parameter=5e-6):
    from src.lib.particle_descent import ParticleDescent

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
        test_f,
        test_kernel,
        test_target,
        test_observations,
        inverse_train_quadrature,
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
        grad_f=grad_f,
    )
    return exp, test_f, test_kernel, test_target, test_observations


def define_adaptive_refinement_experiment():
    from src.lib.adaptive_refinement import AdaptiveRefinement

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
        test_f,
        test_kernel,
        test_target,
        test_observations,
        inverse_train_quadrature,
    ) = define_experiment()
    exp = AdaptiveRefinement(
        observations=observations,
        j=j,
        p=p,
        grad_p=grad_p,
        hess_p=hess_p,
        inverse_quadrature=inverse_train_quadrature,
        Omega=Omega + 1e-5,
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
    return exp, test_f, test_kernel, test_target,test_observations


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


def compute_experiments(Nrun, log_results):
    # NLGCG
    for i in range(Nrun):
        logging.info(f"Running NLGCG (trial {i+1})")
        (
            exp_nlgcg,
            test_f,
            test_kernel,
            test_target,
            test_observations,
        ) = define_nlgcg_experiment()
        (
            us_nlgcg,
            cs_nlgcg,
            times_nlgcg,
            supports_nlgcg,
            inner_loop,
            lgcg_lazy,
            lgcg_total,
            objective_values_nlgcg,
            dropped_tot,
            epsilons,
            all_information,
        ) = exp_nlgcg.solve(
            tol=tol,
            temperature=temperature,
            log_results=log_results,
            optimum=optimum,
            max_time=frame_size,
        )
        with open(f"{results_dir}/nlgcg_{i}.pkl", "wb") as file:
            pickle.dump((us_nlgcg, cs_nlgcg, times_nlgcg), file)
        del exp_nlgcg
        jax.clear_caches()

    # Particle Descent Stochastic
    particle_converged = 0
    for i in range(Nrun):
        diverged = True
        while diverged:
            logging.info(f"Running Particle descent (trial {i+1})")
            (
                exp_particle,
                test_f,
                test_kernel,
                test_target,
                test_observations,
            ) = define_particle_descent_experiment()
            (
                us_particle,
                cs_particle,
                objective_values_particle,
                supports_particle,
                times_particle,
                success,
                diverged,
            ) = exp_particle.solve(
                max_time=frame_size, mode="exponential", log_results=log_results
            )
            del exp_particle
            jax.clear_caches()
        with open(f"{results_dir}/particle_{i}.pkl", "wb") as file:
            pickle.dump((us_particle, cs_particle, times_particle), file)
        if objective_values_particle[-1] - optimum < 1e-5:
            particle_converged += 1
    logging.info(
        f"Particle descent converged in {(particle_converged/Nrun)*100}% of cases."
    )

    # Adaptive Refinement
    logging.info("Running adaptive refinement")
    exp_adaptive, test_f, test_kernel, test_target, test_observations = (
        define_adaptive_refinement_experiment()
    )
    (
        cells_dict,
        vertices_dict,
        vertices,
        us_adaptive,
        cs_adaptive,
        objective_values_adaptive,
        times_adaptive,
        actives,
        supports_adaptive,
    ) = exp_adaptive.solve(max_time=frame_size, log_results=log_results)
    del exp_adaptive
    jax.clear_caches()
    with open(f"{results_dir}/adaptive_0.pkl", "wb") as file:
        pickle.dump((us_adaptive, cs_adaptive, times_adaptive, actives), file)


def create_plots(Nrun, log_results, run_experiments):
    if run_experiments:
        logging.info("Running all experiments")
        compute_experiments(Nrun, log_results)
    else:
        logging.info("Skipping experiments, processing results")

    (
        exp_nlgcg,
        test_f,
        test_kernel,
        test_target,
        test_observations,
    ) = define_nlgcg_experiment()

    # NLGCG
    nlgcg_residuals = []
    nlgcg_tests = []
    nlgcg_supports = []
    nlgcg_supports_time = []
    best_test_nlgcg = np.inf
    nlgcg_converged = 0
    for i in range(Nrun):
        logging.info(f"Processing NLGCG (trial {i+1})")

        try:
            filename = f"{results_dir}/nlgcg_{i}.pkl"
            with open(filename, "rb") as file:
                us_nlgcg, cs_nlgcg, times_nlgcg = pickle.load(file)
        except FileNotFoundError:
            text_bit = filename.split("/")[-1].split(".")[0]
            algo_name = text_bit.split("_")[0]
            run_number = text_bit.split("_")[1]
            logging.error(
                f"Run {run_number} of algorithm {algo_name} not found. Use --run_experiments to generate results."
            )
            return

        supports_nlgcg = [len(u.coefficients) for u in us_nlgcg]
        nlgcg_supports.append(supports_nlgcg.copy())
        objective_values_nlgcg = [exp_nlgcg.j(u, c) for u, c in zip(us_nlgcg, cs_nlgcg)]
        if objective_values_nlgcg[-1] - optimum < 1e-5:
            nlgcg_converged += 1
        test_values_nlgcg = [
            test_f(u.duality_pairing(test_kernel) + c * np.ones(test_target.shape))
            for u, c, in zip(us_nlgcg, cs_nlgcg)
        ]
        best_test_iteration = np.argmin(test_values_nlgcg)
        if test_values_nlgcg[best_test_iteration] < best_test_nlgcg:
            best_test_nlgcg = test_values_nlgcg[best_test_iteration]
            u_nlgcg = us_nlgcg[best_test_iteration]
            c_nlgcg = cs_nlgcg[best_test_iteration]
        local_residuals = adapt_time(
            times_nlgcg,
            [obj - optimum for obj in objective_values_nlgcg],
            frame=frame_size,
            resolution=resolution,
        )
        local_test_values = adapt_time(
            times_nlgcg, test_values_nlgcg, frame=frame_size, resolution=resolution
        )
        local_supports = adapt_time(
            times_nlgcg, supports_nlgcg, frame=frame_size, resolution=resolution
        )
        nlgcg_residuals.append(local_residuals)
        nlgcg_tests.append(local_test_values)
        nlgcg_supports_time.append(local_supports)

    logging.info(f"NLGCG converged in {(nlgcg_converged/Nrun)*100}% of cases.")

    nlgcg_residuals_mean = np.mean(bring_to_same_length(nlgcg_residuals), axis=0)
    nlgcg_tests_mean = np.mean(bring_to_same_length(nlgcg_tests), axis=0)
    nlgcg_supports_time_mean = np.mean(bring_to_same_length(nlgcg_supports_time), axis=0)
    nlgcg_supports_mean = np.mean(bring_to_same_length(nlgcg_supports), axis=0)
    nlgcg_supports_std = np.std(bring_to_same_length(nlgcg_supports), axis=0)

    # Particle Descent Stochastic
    particle_residuals = []
    particle_tests = []
    particle_supports = []
    best_test_particle = np.inf
    particle_converged = 0
    for i in range(Nrun):
        logging.info(f"Processing Particle descent (trial {i+1})")

        try:
            filename = f"{results_dir}/particle_{i}.pkl"
            with open(filename, "rb") as file:
                us_particle, cs_particle, times_particle = pickle.load(file)
        except FileNotFoundError:
            text_bit = filename.split("/")[-1].split(".")[0]
            algo_name = text_bit.split("_")[0]
            run_number = text_bit.split("_")[1]
            logging.error(
                f"Run {run_number} of algorithm {algo_name} not found. Use --run_experiments to generate results."
            )
            return

        supports_particle = [len(u.coefficients) for u in us_particle]
        objective_values_particle = [
            exp_nlgcg.j(u, c) for u, c in zip(us_particle, cs_particle)
        ]
        if objective_values_particle[-1] - optimum < 1e-5:
            particle_converged += 1
        test_values_particle = [
            test_f(u.duality_pairing(test_kernel) + c * np.ones(test_target.shape))
            for u, c, in zip(us_particle, cs_particle)
        ]
        best_test_iteration = np.argmin(test_values_particle)
        if test_values_particle[best_test_iteration] < best_test_particle:
            best_test_particle = test_values_particle[best_test_iteration]
            u_particle = us_particle[best_test_iteration]
            c_particle = cs_particle[best_test_iteration]

        local_residuals = adapt_time(
            times_particle,
            [obj - optimum for obj in objective_values_particle],
            frame=frame_size,
            resolution=resolution,
        )
        test_values_particle = test_values_particle[: len(times_particle)]
        local_test_values = adapt_time(
            times_particle,
            test_values_particle,
            frame=frame_size,
            resolution=resolution,
        )
        local_supports = adapt_time(
            times_particle, supports_particle, frame=frame_size, resolution=resolution
        )
        particle_residuals.append(local_residuals)
        particle_tests.append(local_test_values)
        particle_supports.append(local_supports)

    particle_residuals_mean = np.mean(bring_to_same_length(particle_residuals), axis=0)
    particle_tests_mean = np.mean(bring_to_same_length(particle_tests), axis=0)
    particle_supports_mean = np.mean(bring_to_same_length(particle_supports), axis=0)
    particle_supports_std = np.std(bring_to_same_length(particle_supports), axis=0)

    logging.info(f"Particle descent converged in {(particle_converged/Nrun)*100}% of cases.")

    # Adaptive refinement
    logging.info("Processing Adaptive refinement")

    try:
        filename = f"{results_dir}/adaptive_0.pkl"
        with open(filename, "rb") as file:
            us_adaptive, cs_adaptive, times_adaptive, actives = pickle.load(file)
    except FileNotFoundError:
        text_bit = filename.split("/")[-1].split(".")[0]
        algo_name = text_bit.split("_")[0]
        run_number = text_bit.split("_")[1]
        logging.error(
            f"Run {run_number} of algorithm {algo_name} not found. Use --run_experiments to generate results."
        )
        return

    supports_adaptive = [len(u.coefficients) for u in us_adaptive]
    objective_values_adaptive = [
        exp_nlgcg.j(u, c) for u, c in zip(us_adaptive, cs_adaptive)
    ]
    test_values_adaptive = [
        test_f(u.duality_pairing(test_kernel) + c * np.ones(test_target.shape))
        for u, c, in zip(us_adaptive, cs_adaptive)
    ]
    best_test_iteration = np.argmin(
        np.array(test_values_adaptive)[np.array(times_adaptive) < frame_size]
    )
    best_test_adaptive = test_values_adaptive[best_test_iteration]
    u_adaptive = us_adaptive[best_test_iteration]
    c_adaptive = cs_adaptive[best_test_iteration]
    residuals_adaptive = adapt_time(
        times_adaptive,
        [obj - optimum for obj in objective_values_adaptive],
        frame=frame_size,
        resolution=resolution,
    )
    test_values_adaptive = adapt_time(
        times_adaptive,
        test_values_adaptive,
        frame=frame_size,
        resolution=resolution,
    )
    supports_adaptive = adapt_time(
        times_adaptive, supports_adaptive, frame=frame_size, resolution=resolution
    )

    adaptive_converged = objective_values_adaptive[-1] - optimum < 1e-5
    logging.info(f"Adaptive refinement converged: {adaptive_converged}")

    logging.getLogger().setLevel(logging.WARNING)  # Supress logging

    # Plot residuals
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "Particle Descent", "Adaptive Refinement"]
    styles = ["-", "--", ":"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for array, name, style, color in zip(
        [nlgcg_residuals_mean, particle_residuals_mean, residuals_adaptive],
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
    plt.ylim(1e-10, 1)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "residuals.png", bbox_inches="tight")
    plt.close()

    # Plot test scores
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "Particle Descent", "Adaptive Refinement"]
    styles = ["-", "--", ":"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for array, name, style, color in zip(
        [nlgcg_tests_mean, particle_tests_mean, test_values_adaptive],
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
                resolution * np.arange(len(particle_tests_mean)),
                resolution * np.arange(len(particle_tests_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.max(bring_to_same_length(particle_tests), axis=0),
                np.min(bring_to_same_length(particle_tests), axis=0)[::-1],
            )
        ),
        "tab:orange",
        alpha=0.3,
    )
    ax.fill(
        np.hstack(
            (
                resolution * np.arange(len(nlgcg_tests_mean)),
                resolution * np.arange(len(nlgcg_tests_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.max(bring_to_same_length(nlgcg_tests), axis=0),
                np.min(bring_to_same_length(nlgcg_tests), axis=0)[::-1],
            )
        ),
        "tab:blue",
        alpha=0.3,
    )
    plt.ylabel("MSE on test set")
    plt.xlabel("Time (s)")
    # plt.ylim(1e-5, 1e-0)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "test_values.png", bbox_inches="tight")
    plt.close()

    # Plot supports
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "Particle Descent", "Adaptive Refinement"]
    styles = ["-", "--", ":"]
    for array, name, style, color in zip(
        [nlgcg_supports_time_mean, particle_supports_mean, supports_adaptive],
        names,
        styles,
        colors,
    ):
        ax.plot(resolution*np.arange(len(array)), array, style, label=name, color=color)
    ax.fill(
        np.hstack(
            (
                np.arange(len(particle_supports_mean)),
                np.arange(len(particle_supports_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.max(bring_to_same_length(particle_supports), axis=0),
                np.min(bring_to_same_length(particle_supports), axis=0)[::-1],
            )
        ),
        "tab:orange",
        alpha=0.3,
    )
    ax.fill(
        np.hstack(
            (
                np.arange(len(nlgcg_supports_time_mean)),
                np.arange(len(nlgcg_supports_time_mean))[::-1],
            )
        ),
        np.hstack(
            (
                np.max(bring_to_same_length(nlgcg_supports_time), axis=0),
                np.min(bring_to_same_length(nlgcg_supports_time), axis=0)[::-1],
            )
        ),
        "tab:blue",
        alpha=0.3,
    )
    plt.ylabel("Support points")
    plt.xlabel("Time (s)")
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

    # Plot pruning of Particle Descent
    thresholds = np.linspace(0, 0.01, 26)
    max_coef = np.max(np.abs(u_particle.coefficients))
    pruned_test_vals = []
    pruned_support_sizes = []
    for threshold in thresholds:
        pruned_indices = np.abs(u_particle.coefficients) > threshold * max_coef
        pruned_support_sizes.append(np.sum(pruned_indices))
        pruned_u = Measure(
            support=u_particle.support[pruned_indices],
            coefficients=u_particle.coefficients[pruned_indices],
        )
        pruned_test_vals.append(
            test_f(
                pruned_u.duality_pairing(test_kernel)
                + c_particle * np.ones(test_target.shape)
            )
        )
    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax2 = ax1.twinx()
    (p1,) = ax1.semilogy(
        thresholds, pruned_test_vals, "-", label="MSE on test set", color="darkred"
    )
    (p2,) = ax2.plot(
        thresholds, pruned_support_sizes, "-", label="Support size", color="lightpink"
    )
    ax1.set_ylabel("MSE on test set")
    ax2.set_ylabel("Support size")
    ax1.set_xlabel("Pruning threshold")
    ax1.legend(handles=[p1, p2], loc="upper center")
    plt.savefig(results_dir / "particle_pruning.png", bbox_inches="tight")
    plt.close()

    # Plot the function with reconstruction error
    x = test_observations[:, 0]
    y = test_observations[:, 1]
    true_vals = true_function(test_observations)
    pred_vals_nlgcg = u_nlgcg.duality_pairing(test_kernel) + c_nlgcg * np.ones(test_target.shape)
    pred_vals_particle = u_particle.duality_pairing(test_kernel) + c_particle * np.ones(test_target.shape)
    pred_vals_adaptive = u_adaptive.duality_pairing(test_kernel) + c_adaptive * np.ones(test_target.shape)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    contour1 = ax1.tricontourf(x, y, true_vals, levels=100)
    fig.colorbar(contour1, ax=ax1)
    ax1.set_xlim(omega_space[0][0], omega_space[0][1])
    ax1.set_ylim(omega_space[1][0], omega_space[1][1])
    ax1.set_xlabel("True function")
    for pred_vals, name, u, ax in zip(
        [pred_vals_nlgcg, pred_vals_particle, pred_vals_adaptive],
        ["NLGCG", "Particle Descent", "Adaptive Refinement"],
        [u_nlgcg, u_particle, u_adaptive],
        [ax2, ax3, ax4],
    ):
        contour = ax.tricontourf(x, y, pred_vals, levels=100)
        fig.colorbar(contour, ax=ax)
        for i, point in enumerate(u.support):
            if u.coefficients[i] < 0:
                color = "b"
            else:
                color = "r"
            ax.add_patch(
                plt.Circle(
                    (point[1], point[2]),
                    radius=point[0],
                    color=color,
                    fill=False,
                    alpha=0.5,
                )
            )
        ax.set_xlim(omega_space[0][0], omega_space[0][1])
        ax.set_ylim(omega_space[1][0], omega_space[1][1])
        ax.set_xlabel(f"Predicted function {name}")
    plt.savefig(results_dir / f"predictions.png", bbox_inches="tight")
    plt.close()

    logging.getLogger().setLevel(logging.INFO)  # Reinstate logging

    logging.info(
        f"Best test value NLGCG: {best_test_nlgcg:.3E}, Particle Descent: {best_test_particle:.3E}, Adaptive Refinement: {best_test_adaptive:.3E}"
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
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        help="Extensive logging",
    )
    parser.add_argument(
        "--run_experiments",
        action=argparse.BooleanOptionalAction,
        help="Run all experiments",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    create_plots(
        Nrun=args.Nrun, log_results=args.verbose, run_experiments=args.run_experiments
    )
