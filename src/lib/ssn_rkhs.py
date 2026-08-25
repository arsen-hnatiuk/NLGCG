import numpy as np
import logging
from typing import Callable

logging.basicConfig(
    level=logging.DEBUG,
)


class SSN_RKHS:
    def __init__(
        self,
        alpha: float,
        M: float,
        g: Callable,
        hessian: np.ndarray,
        j: Callable,
        p: Callable,
        maximum_iterations: int = 1000,
        log_results: bool = True,
    ) -> None:
        self.machine_precision = 1e-12
        self.alpha = alpha
        self.g = g
        self.p = p  # -f'
        self.hessian = hessian
        self.j = j
        self.M = M
        self.maximum_iterations = maximum_iterations
        self.log_results = log_results

    def Psi(self, u: np.ndarray) -> np.ndarray:
        # sup_v <p(u),v-u>+g(u)-g(v)
        u = u.copy()
        p = self.p(u)
        constant_part = -np.matmul(p, u) + self.g(u)
        variable_part = max(0, self.M * (np.max(np.absolute(p)) - self.alpha))
        return constant_part + variable_part

    def prox(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        to_return = np.zeros(q.shape)
        for i, val in enumerate(q):
            if np.abs(val) > self.alpha:
                to_return[i] = val - self.alpha * np.sign(val)
        return to_return

    def grad_prox(self, q: np.ndarray) -> np.ndarray:
        q = q.copy()
        return np.diag(np.where(np.abs(q) > self.alpha, 1, 0))

    def solve(self, tol: float, u_0: np.ndarray) -> np.ndarray:
        # Semismooth Newton method (globalized via line search)
        theta = tol  # Set initial value for the step length parameter
        Id = np.identity(len(u_0))
        initial_j = self.j(u_0)
        q = u_0  # + self.p(u_0)
        prox_q = self.prox(q)  # The actual iterate
        psi_val = min(self.Psi(prox_q), self.Psi(q))
        k = 0
        while psi_val > tol:
            if k > self.maximum_iterations:
                if self.log_results:
                    logging.warning(
                        f"SSN in {len(prox_q)} dimensions and tolerance {tol:.3E}: MAX ITERATIONS REACHED, {psi_val:.3E} achieved"
                    )
                if self.j(prox_q) <= initial_j:
                    return prox_q
                else:
                    return u_0
            right_hand = q - prox_q - self.p(prox_q)
            left_hand = Id + (self.hessian - Id) @ self.grad_prox(q)
            theta = theta / 10
            qdiff = tol + 1
            while qdiff >= tol:
                theta = 2 * theta
                try:
                    direction = np.linalg.solve(left_hand + theta * Id, right_hand)
                except np.linalg.LinAlgError:
                    if self.log_results:
                        logging.warning(
                            f"SSN in {len(prox_q)} dimensions and tolerance {tol:.3E}: LINEAR SYSTEM NOT SOLVABLE, {psi_val:.3E} achieved"
                        )
                    if self.j(prox_q) <= initial_j:
                        return prox_q
                    else:
                        return u_0
                qnew = q - direction
                prox_qnew = self.prox(qnew)
                qdiff = self.j(prox_qnew) - self.j(prox_q)
            q = qnew
            prox_q = prox_qnew
            psi_val = self.Psi(prox_q)
            k += 1

        # if self.log_results:
        #     logging.info(
        #         f"SSN in {len(prox_q)} dimensions converged in {k} iterations to tolerance {tol:.3E}"
        #     )
        if self.j(prox_q) <= initial_j:
            return prox_q
        else:
            return u_0
