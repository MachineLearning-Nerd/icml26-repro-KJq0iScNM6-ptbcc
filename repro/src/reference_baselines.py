"""Faithful CPU implementations of the paper's two key reference baselines.

BWA follows the 50-line appendix and MIT-licensed reference implementation at
yuan-li/truth-inference-at-scale (commit 621789b2d57324d3559dc973b2613d2296d73f55).
FGBCC follows the authors' public implementation at JuJuCHEN-HHU/CodeForFGBCC
(commit e2ca2b8a876bf9cceb871e8cec9081870a30aab4).  Its scalar independent
variance updates are vectorized without changing their equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as ssp
from scipy.optimize import minimize
from scipy.special import digamma
from scipy.stats import dirichlet, entropy


@dataclass
class BaselineFit:
    scores: np.ndarray
    iterations: int
    converged: bool
    objective: float | None = None


def _sparse_labels(dataset: object) -> tuple[ssp.csr_matrix, list[ssp.csr_matrix]]:
    shape = (dataset.n_tasks, dataset.n_workers)
    coordinates = np.vstack((dataset.task, dataset.worker))
    exists = ssp.coo_matrix(
        (np.ones(dataset.n_labels), coordinates), shape=shape, dtype=np.float64
    ).tocsr()
    is_one = []
    for label in range(dataset.n_classes):
        selected = dataset.label == label
        is_one.append(
            ssp.coo_matrix(
                (
                    np.ones(int(selected.sum())),
                    np.vstack((dataset.task[selected], dataset.worker[selected])),
                ),
                shape=shape,
                dtype=np.float64,
            ).tocsr()
        )
    return exists, is_one


def fit_bwa(
    dataset: object,
    *,
    a_v: float = 15.0,
    lambda_: float = 1.0,
    prior_correction: bool = True,
    max_iterations: int = 500,
) -> BaselineFit:
    """Bayesian Weighted Average, matching Li et al. (2019) Appendix A."""

    exists, is_one = _sparse_labels(dataset)
    labels_per_item = np.bincount(dataset.task, minlength=dataset.n_tasks)
    labels_per_worker = np.asarray(exists.sum(axis=0)).ravel()
    adjustment = 4.0 * (1.0 - 1.0 / dataset.n_classes) if prior_correction else 1.0
    scores = np.empty((dataset.n_tasks, dataset.n_classes), dtype=np.float64)
    max_used = 0
    all_converged = True

    for label in range(dataset.n_classes):
        z = np.asarray(is_one[label].sum(axis=1)).ravel() / labels_per_item
        b_v = (
            a_v
            * float(labels_per_item.dot(z * (1.0 - z)))
            / dataset.n_labels
            * adjustment
        )
        converged = False
        for iteration in range(1, max_iterations + 1):
            previous = z.copy()
            mu = float(z.mean())
            squared_error = np.asarray(
                (exists.multiply(z[:, None]) - is_one[label]).power(2).sum(axis=0)
            ).ravel()
            worker_precision = (a_v + labels_per_worker) / (b_v + squared_error)
            z = np.asarray(
                (
                    lambda_ * mu
                    + is_one[label].dot(worker_precision)
                )
                / (
                    lambda_
                    + exists.dot(worker_precision)
                )
            ).ravel()
            if np.allclose(previous, z, rtol=1e-3):
                converged = True
                break
        scores[:, label] = z
        max_used = max(max_used, iteration)
        all_converged = all_converged and converged
    return BaselineFit(scores, max_used, all_converged)


class _VectorizedFGBCC:
    """Equation-preserving vectorization of the FGBCC authors' Python code."""

    def __init__(self, dataset: object) -> None:
        self.dataset = dataset
        self.num_items = dataset.n_tasks
        self.num_workers = dataset.n_workers
        self.num_classes = dataset.n_classes
        self.exists, self.is_one_ij = _sparse_labels(dataset)
        self.is_one_ji = [matrix.T.tocsr() for matrix in self.is_one_ij]

        shape = (self.num_workers, self.num_classes, self.num_classes)
        self.lambda_jkl = np.zeros(shape, dtype=np.float64)
        self.eta2_jkl = np.zeros(shape, dtype=np.float64)
        self.phi_ik = np.zeros((self.num_items, self.num_classes), dtype=np.float64)
        for label in range(self.num_classes):
            self.phi_ik[:, label] += np.asarray(
                self.is_one_ij[label].sum(axis=1)
            ).ravel()
        self.phi_ik /= self.phi_ik.sum(axis=1, keepdims=True)
        self.alpha = self.phi_ik.sum(axis=0)

        labels_per_worker = np.bincount(
            dataset.worker, minlength=self.num_workers
        )
        share_long_workers = float(np.mean(labels_per_worker > 30))
        if self.num_workers < 100 and self.num_items >= 1000:
            self.eta2_jkl.fill(30.0)
            self.lambda_jkl.fill(2.0)
            diagonal = np.arange(self.num_classes)
            self.lambda_jkl[:, diagonal, diagonal] = 6.0
        elif (
            500 <= self.num_items < 1000 and self.num_workers >= 150
        ) or (
            self.num_items >= 1000
            and self.num_workers >= 100
            and share_long_workers > 0.5
        ):
            self.eta2_jkl.fill(15.0)
        else:
            self.eta2_jkl.fill(4.0)
            diagonal = np.arange(self.num_classes)
            self.lambda_jkl[:, diagonal, diagonal] = 2.0
        self.mu_jkl = self.lambda_jkl.copy()
        self.sigma_jkl = self.eta2_jkl.copy()
        self.counts_jkl = np.zeros(shape, dtype=np.float64)
        self.zeta_jk = np.exp(self.lambda_jkl + self.eta2_jkl / 2.0).sum(axis=2)

    def _refresh_counts(self) -> None:
        self.counts_jkl.fill(0.0)
        for observed in range(self.num_classes):
            for truth in range(self.num_classes):
                self.counts_jkl[:, truth, observed] = np.asarray(
                    self.is_one_ji[observed].dot(self.phi_ik[:, truth])
                ).ravel()

    def _lambda_objective(self, flattened: np.ndarray) -> float:
        candidate = flattened.reshape(self.lambda_jkl.shape)
        exponentials = np.exp(candidate + self.eta2_jkl / 2.0)
        total_counts = self.counts_jkl.sum(axis=2)
        value = float((self.counts_jkl * candidate).sum())
        value -= float(
            (
                total_counts
                * exponentials.sum(axis=2)
                / self.zeta_jk
            ).sum()
        )
        value -= 0.5 * float(
            (np.square(candidate - self.mu_jkl) / self.sigma_jkl).sum()
        )
        return -value

    def _lambda_gradient(self, flattened: np.ndarray) -> np.ndarray:
        candidate = flattened.reshape(self.lambda_jkl.shape)
        exponentials = np.exp(candidate + self.eta2_jkl / 2.0)
        gradient = self.counts_jkl - (
            candidate - self.mu_jkl
        ) / self.sigma_jkl
        gradient -= (
            self.counts_jkl.sum(axis=2) / self.zeta_jk
        )[:, :, None] * exponentials
        return -gradient.ravel()

    def _update_lambda(self) -> None:
        result = minimize(
            self._lambda_objective,
            self.lambda_jkl.ravel(),
            method="L-BFGS-B",
            jac=self._lambda_gradient,
            tol=1e-5,
            options={"disp": False, "maxiter": 35},
        )
        if not np.isfinite(result.x).all():
            raise FloatingPointError("FGBCC lambda optimization produced non-finite values")
        self.lambda_jkl = result.x.reshape(self.lambda_jkl.shape)

    def _update_eta2(self, max_newton_iterations: int = 1000) -> None:
        log_eta = np.log(self.eta2_jkl)
        total_counts = self.counts_jkl.sum(axis=2)[:, :, None]
        for _ in range(max_newton_iterations):
            eta = np.exp(log_eta)
            exponentials = np.exp(self.lambda_jkl + eta / 2.0)
            first = (
                -0.5
                * total_counts
                / self.zeta_jk[:, :, None]
                * exponentials
                - 0.5 / self.sigma_jkl
                + 0.5 / eta
            )
            active = np.abs(first) > 1e-5
            if not active.any():
                self.eta2_jkl = eta
                return
            second = (
                -0.25
                * total_counts
                / self.zeta_jk[:, :, None]
                * exponentials
                - 0.5 / np.square(eta)
            )
            step = (first * eta) / (first * eta + second * np.square(eta))
            log_eta[active] -= step[active]
            if not np.isfinite(log_eta).all():
                raise FloatingPointError("FGBCC variance update produced non-finite values")
        raise RuntimeError("FGBCC variance Newton update did not converge")

    def _elbo(self, gamma: np.ndarray, elog_tau: np.ndarray, elog_v: np.ndarray) -> float:
        value = float(((gamma - 1.0) * elog_tau).sum())
        value += float((self.counts_jkl * elog_v).sum())
        value -= 0.5 * float(np.log(self.sigma_jkl).sum())
        value -= 0.5 * float(
            (
                (
                    np.square(self.lambda_jkl - self.mu_jkl)
                    + self.eta2_jkl
                )
                / self.sigma_jkl
            ).sum()
        )
        value += float(dirichlet.entropy(gamma))
        value += float(entropy(self.phi_ik.T).sum())
        value += 0.5 * float(np.log(self.eta2_jkl).sum())
        return value

    def run(self, max_iterations: int = 1000) -> BaselineFit:
        elbo = 1.0
        converged = False
        for iteration in range(1, max_iterations + 1):
            self._refresh_counts()
            gamma = self.alpha + self.phi_ik.sum(axis=0)
            elog_tau = digamma(gamma) - digamma(gamma.sum())

            self.zeta_jk = np.exp(
                self.lambda_jkl + self.eta2_jkl / 2.0
            ).sum(axis=2)
            self._update_lambda()
            self.zeta_jk = np.exp(
                self.lambda_jkl + self.eta2_jkl / 2.0
            ).sum(axis=2)
            self._update_eta2()
            exponentials = np.exp(self.lambda_jkl + self.eta2_jkl / 2.0)
            self.zeta_jk = exponentials.sum(axis=2)
            elog_v = (
                self.lambda_jkl
                - exponentials.sum(axis=2)[:, :, None]
                / self.zeta_jk[:, :, None]
                - np.log(self.zeta_jk)[:, :, None]
                + 1.0
            )

            scores = np.broadcast_to(
                elog_tau - 1.0, (self.num_items, self.num_classes)
            ).copy()
            for observed in range(self.num_classes):
                for truth in range(self.num_classes):
                    scores[:, truth] += self.is_one_ij[observed].dot(
                        elog_v[:, truth, observed]
                    )
            scores -= scores.max(axis=1, keepdims=True)
            self.phi_ik = np.exp(scores)
            self.phi_ik /= self.phi_ik.sum(axis=1, keepdims=True)

            previous_elbo = elbo
            elbo = self._elbo(gamma, elog_tau, elog_v)
            self.mu_jkl = self.lambda_jkl.copy()
            self.sigma_jkl = self.eta2_jkl.copy()
            if abs((elbo - previous_elbo) / previous_elbo) <= 1e-6:
                converged = True
                break
        return BaselineFit(self.phi_ik, iteration, converged, elbo)


def fit_fgbcc(dataset: object, *, max_iterations: int = 1000) -> BaselineFit:
    return _VectorizedFGBCC(dataset).run(max_iterations=max_iterations)
