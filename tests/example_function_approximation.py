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

sigma_space = np.array([0, 2])
omega_space = np.array([[-1, 1], [-1, 1]])
Omega = np.vstack((sigma_space, omega_space))
d = omega_space.shape[0]
variance_exponent = 0.1
alpha = 1e-3
observation_resolution = 20
test_resolution = 100
endpoint = False
max_radius = (Omega[0][1] - Omega[0][0]) / 100
true_function = lambda x: np.minimum(1 - np.abs(x[:, 0]), 1 - np.abs(x[:, 1]))
optimum = 0.0339493125929431


def define_experiment():
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
    test_observations = (
        np.array(
            np.meshgrid(
                *(
                    np.linspace(
                        bound[0] + 1e-5,
                        bound[1] - np.sqrt(3) * 1e-5,
                        test_resolution,
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
    def f(y: np.ndarray) -> float:
        return 0.5 * jnp.sum((y - target) ** 2)

    def test_f(y: np.ndarray) -> float:
        return 0.5 * jnp.sum((y - test_target) ** 2)

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
        test_f,
        test_kernel,
        test_target,
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
        newton_tolerance=2e-2,
        max_radius=max_radius,
    )
    return exp, test_f, test_kernel, test_target


def define_particle_descent_experiment(m=250, a_parameter=1e-5, b_parameter=5e-6):
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
    return exp, test_f, test_kernel, test_target


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
        test_f,
        test_kernel,
        test_target,
    ) = define_experiment()
    exp = AdaptiveRefinement(
        observations=observations,
        j=j,
        p=p,
        grad_p=grad_p,
        hess_p=hess_p,
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
    return exp, test_f, test_kernel, test_target


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


def create_plots(Nrun: int = 10):
    resolution = 1  # time resolution for the plots (seconds)
    frame_size = 1000  # Time frame tracked for the residuals (seconds)

    # NLGCG
    nlgcg_residuals = []
    nlgcg_tests = []
    nlgcg_supports = []
    nlgcg_converged = 0
    for i in range(Nrun):
        logging.info(f"Running NLGCG (trial {i+1})")
        exp_nlgcg, test_f, test_kernel, test_target = define_nlgcg_experiment()
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
            all_inormation,
        ) = exp_nlgcg.solve(
            tol=5e-14, temperature=1, log_results=False, optimum=optimum
        )
        test_values_nlgcg = [
            test_f(u.duality_pairing(test_kernel) + c * np.ones(test_target.shape))
            for u, c, in zip(us_nlgcg, cs_nlgcg)
        ]
        best_test_iteration = np.argmin(test_values_nlgcg)
        u_nlgcg = us_nlgcg[best_test_iteration]
        c_nlgcg = cs_nlgcg[best_test_iteration]
        del exp_nlgcg, us_nlgcg, cs_nlgcg
        jax.clear_caches()
        local_residuals = adapt_time(
            times_nlgcg,
            [obj - optimum for obj in objective_values_nlgcg],
            frame=frame_size,
            resolution=resolution,
        )
        local_test_values = adapt_time(
            times_nlgcg, test_values_nlgcg, frame=frame_size, resolution=resolution
        )
        if local_residuals[-1] < 1e-5:
            nlgcg_converged += 1
        nlgcg_residuals.append(local_residuals)
        nlgcg_tests.append(local_test_values)
        nlgcg_supports.append(supports_nlgcg)
    logging.info(f"NLGCG converged in {(nlgcg_converged/Nrun)*100}% of cases.")

    nlgcg_residuals_mean = np.mean(bring_to_same_length(nlgcg_residuals), axis=0)
    nlgcg_tests_mean = np.mean(bring_to_same_length(nlgcg_tests), axis=0)
    nlgcg_supports_mean = np.mean(bring_to_same_length(nlgcg_supports), axis=0)
    nlgcg_supports_std = np.std(bring_to_same_length(nlgcg_supports), axis=0)

    # Particle Descent Stochastic
    particle_residuals = []
    particle_tests = []
    particle_supports = []
    particle_converged = 0
    for i in range(Nrun):
        success = False
        while not success:
            logging.info(f"Running Particle descent (trial {i+1})")
            exp_particle, test_f, test_kernel, test_target = (
                define_particle_descent_experiment()
            )
            (
                us_particle,
                cs_particle,
                objective_values_particle,
                supports_particle,
                times_particle,
                success,
            ) = exp_particle.solve(
                max_time=frame_size, mode="exponential", log_results=False
            )
            test_values_particle_raw = [
                test_f(u.duality_pairing(test_kernel) + c * np.ones(test_target.shape))
                for u, c, in zip(us_particle, cs_particle)
            ]
            best_test_iteration = np.argmin(test_values_particle_raw)
            u_particle = us_particle[best_test_iteration]
            c_particle = cs_particle[best_test_iteration]
            del exp_particle, us_particle, cs_particle
            jax.clear_caches()
        local_residuals = adapt_time(
            times_particle,
            [obj - optimum for obj in objective_values_particle],
            frame=frame_size,
            resolution=resolution,
        )
        # For particle descent, the test values are recorded every 100 iterations
        test_values_particle = []
        for test_val in test_values_particle_raw:
            test_values_particle += [test_val] * 100
        test_values_particle = test_values_particle[: len(times_particle)]
        local_test_values = adapt_time(
            times_particle,
            test_values_particle,
            frame=frame_size,
            resolution=resolution,
        )
        if local_residuals[-1] < 1e-5:
            particle_converged += 1
        particle_residuals.append(local_residuals)
        particle_tests.append(local_test_values)
        particle_supports.append(supports_particle)
    logging.info(
        f"Particle descent converged in {(particle_converged/Nrun)*100}% of cases."
    )

    particle_residuals_mean = np.mean(bring_to_same_length(particle_residuals), axis=0)
    particle_tests_mean = np.mean(bring_to_same_length(particle_tests), axis=0)
    particle_supports_mean = np.mean(bring_to_same_length(particle_supports), axis=0)
    particle_supports_std = np.std(bring_to_same_length(particle_supports), axis=0)

    # Adaptive refinement
    logging.info("Running adaptive refinement")
    exp_adaptive, test_f, test_kernel, test_target = (
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
    ) = exp_adaptive.solve(max_time=frame_size, log_results=False)
    test_values_adaptive = [
        test_f(u.duality_pairing(test_kernel) + c * np.ones(test_target.shape))
        for u, c, in zip(us_adaptive, cs_adaptive)
    ]
    best_test_iteration = np.argmin(
        np.array(test_values_adaptive)[np.array(times_adaptive) < frame_size]
    )
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
    del exp_adaptive, us_adaptive, cs_adaptive

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
    plt.ylim(1e-10, 100)
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
    plt.ylim(1e-1, 1e0)
    # plt.xlim(0, 100)
    ax.legend()
    plt.savefig(results_dir / "test_values.png", bbox_inches="tight")
    plt.close()

    # Plot supports
    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["NLGCG", "Particle Descent", "Adaptive Refinement"]
    styles = ["-", "--", ":"]
    for array, name, style, color in zip(
        [nlgcg_supports_mean, particle_supports_mean, supports_adaptive],
        names,
        styles,
        colors,
    ):
        ax.semilogx(np.arange(len(array)), array, style, label=name, color=color)
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

    pred_vals_nlgcg = u_nlgcg.duality_pairing(jax.vmap(plot_kernel)).reshape(
        (resolution, resolution)
    ) + c_nlgcg * np.ones((resolution, resolution))
    pred_vals_particle = u_particle.duality_pairing(jax.vmap(plot_kernel)).reshape(
        (resolution, resolution)
    ) + c_particle * np.ones((resolution, resolution))
    pred_vals_adaptive = u_adaptive.duality_pairing(jax.vmap(plot_kernel)).reshape(
        (resolution, resolution)
    ) + c_adaptive * np.ones((resolution, resolution))

    for pred_vals, name, u in zip(
        [pred_vals_nlgcg, pred_vals_particle, pred_vals_adaptive],
        ["NLGCG", "Particle Descent", "Adaptive Refinement"],
        [u_nlgcg, u_particle, u_adaptive],
    ):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        contour1 = ax1.contourf(x, y, vals, levels=100)
        fig.colorbar(contour1, ax=ax1)
        contour2 = ax2.contourf(x, y, pred_vals, levels=100)
        fig.colorbar(contour2, ax=ax2)
        contour3 = ax3.contourf(x, y, np.abs(pred_vals - vals), levels=100)
        fig.colorbar(contour3, ax=ax3)
        for i, point in enumerate(u.support):
            if u.coefficients[i] < 0:
                color = "b"
            else:
                color = "r"
            ax2.add_patch(
                plt.Circle(
                    (point[1], point[2]),
                    radius=point[0],
                    color=color,
                    fill=False,
                    alpha=0.5,
                )
            )
        ax1.set_xlim(omega_space[0][0], omega_space[0][1])
        ax1.set_ylim(omega_space[1][0], omega_space[1][1])
        ax2.set_xlim(omega_space[0][0], omega_space[0][1])
        ax2.set_ylim(omega_space[1][0], omega_space[1][1])
        ax3.set_xlim(omega_space[0][0], omega_space[0][1])
        ax3.set_ylim(omega_space[1][0], omega_space[1][1])
        ax1.set_xlabel("True function")
        ax2.set_xlabel(f"Predicted function {name}")
        ax3.set_xlabel("Absolute error in prediction")
        plt.savefig(
            results_dir / f"reconstruction_error_{name}.png", bbox_inches="tight"
        )
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
        default="results/function_approximation",
        help="Directory for plots",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    create_plots(Nrun=args.Nrun)
