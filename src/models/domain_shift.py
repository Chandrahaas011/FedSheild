"""Source-calibrated batch domain-shift monitoring for lightweight IDS deployment."""
from __future__ import annotations

import math

import numpy as np


BENIGN = "BENIGN"
MALICIOUS = "MALICIOUS"
UNKNOWN = "UNKNOWN"
DOMAIN_SHIFT_DETECTED = "DOMAIN_SHIFT_DETECTED"


def _finite_sample_upper_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("at least one one-dimensional calibration score is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    rank = math.ceil((len(values) + 1) * (1.0 - alpha)) - 1
    return float(values[min(max(rank, 0), len(values) - 1)])


class QuantileJSDShiftSentinel:
    """Detect batch covariate shift with source-fitted marginal histograms.

    The sentinel is deliberately separate from the attack classifier. It fits feature
    bins and reference probabilities on source training features, then calibrates one
    batch-level threshold on a disjoint source-validation role. Labels are neither
    accepted nor required by this estimator.
    """

    def __init__(
        self,
        window_size: int = 512,
        quantile_bins: int = 10,
        calibration_alpha: float = 0.05,
        aggregation: str = "mean",
        maximum_reference_rows: int = 100_000,
        minimum_calibration_windows: int = 10,
        seed: int = 0,
    ):
        if window_size < 32:
            raise ValueError("window_size must be at least 32")
        if quantile_bins < 2:
            raise ValueError("quantile_bins must be at least two")
        if aggregation not in {"mean", "median"}:
            raise ValueError("aggregation must be 'mean' or 'median'")
        if maximum_reference_rows < window_size:
            raise ValueError("maximum_reference_rows must be at least window_size")
        if minimum_calibration_windows < 2:
            raise ValueError("minimum_calibration_windows must be at least two")
        if not 0.0 < calibration_alpha < 1.0:
            raise ValueError("calibration_alpha must lie strictly between zero and one")
        self.window_size = int(window_size)
        self.quantile_bins = int(quantile_bins)
        self.calibration_alpha = float(calibration_alpha)
        self.aggregation = aggregation
        self.maximum_reference_rows = int(maximum_reference_rows)
        self.minimum_calibration_windows = int(minimum_calibration_windows)
        self.seed = int(seed)

    @staticmethod
    def _validate_X(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional matrix")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        return X

    @staticmethod
    def _probabilities(column: np.ndarray, edges: np.ndarray) -> np.ndarray:
        bins = np.searchsorted(edges, column, side="right")
        counts = np.bincount(bins, minlength=len(edges) + 1).astype(np.float64)
        # A tiny fixed pseudocount keeps logarithms finite without materially changing
        # the empirical distribution for 512-row deployment windows.
        counts += 1.0e-12
        return counts / counts.sum()

    @staticmethod
    def _js_divergence(reference: np.ndarray, observed: np.ndarray) -> float:
        midpoint = 0.5 * (reference + observed)
        divergence = 0.5 * np.sum(reference * np.log2(reference / midpoint))
        divergence += 0.5 * np.sum(observed * np.log2(observed / midpoint))
        return float(divergence)

    def fit(self, X_reference: np.ndarray):
        X_reference = self._validate_X(X_reference)
        rng = np.random.default_rng(self.seed + 7001)
        if len(X_reference) > self.maximum_reference_rows:
            selected = np.sort(
                rng.choice(
                    len(X_reference), self.maximum_reference_rows, replace=False
                )
            )
            reference = X_reference[selected]
        else:
            reference = X_reference

        quantiles = np.linspace(0.0, 1.0, self.quantile_bins + 1)[1:-1]
        self.bin_edges_ = []
        self.reference_probabilities_ = []
        for feature_index in range(reference.shape[1]):
            edges = np.unique(
                np.quantile(reference[:, feature_index], quantiles)
            ).astype(np.float32)
            self.bin_edges_.append(edges)
            self.reference_probabilities_.append(
                self._probabilities(reference[:, feature_index], edges)
            )
        self.n_features_in_ = int(reference.shape[1])
        self.reference_rows_ = int(len(reference))
        return self

    def feature_scores(self, X_window: np.ndarray) -> np.ndarray:
        X_window = self._validate_X(X_window)
        if not hasattr(self, "bin_edges_"):
            raise ValueError("sentinel is not fitted")
        if X_window.shape[1] != self.n_features_in_:
            raise ValueError("feature count does not match the fitted sentinel")
        scores = np.empty(self.n_features_in_, dtype=np.float64)
        for feature_index, edges in enumerate(self.bin_edges_):
            observed = self._probabilities(X_window[:, feature_index], edges)
            scores[feature_index] = self._js_divergence(
                self.reference_probabilities_[feature_index], observed
            )
        return scores

    def score_window(self, X_window: np.ndarray) -> float:
        scores = self.feature_scores(X_window)
        if self.aggregation == "mean":
            return float(scores.mean())
        return float(np.median(scores))

    def calibrate(self, X_source_validation: np.ndarray):
        """Calibrate on alternating windows and retain disjoint audit-window scores."""
        X_source_validation = self._validate_X(X_source_validation)
        if not hasattr(self, "bin_edges_"):
            raise ValueError("fit must be called before calibrate")
        if X_source_validation.shape[1] != self.n_features_in_:
            raise ValueError("feature count does not match the fitted sentinel")
        total_windows = len(X_source_validation) // self.window_size
        required = 2 * self.minimum_calibration_windows
        if total_windows < required:
            raise ValueError(
                f"source validation provides {total_windows} complete windows; "
                f"at least {required} are required"
            )
        rng = np.random.default_rng(self.seed + 7019)
        selected = rng.permutation(len(X_source_validation))[
            : total_windows * self.window_size
        ]
        windows = selected.reshape(total_windows, self.window_size)
        scores = np.asarray(
            [self.score_window(X_source_validation[index]) for index in windows],
            dtype=np.float64,
        )
        self.calibration_window_indices_ = windows[::2]
        self.audit_window_indices_ = windows[1::2]
        self.calibration_scores_ = scores[::2]
        self.audit_scores_ = scores[1::2]
        self.threshold_ = _finite_sample_upper_quantile(
            self.calibration_scores_, self.calibration_alpha
        )
        self.validation_rows_used_ = int(len(selected))
        self.validation_rows_omitted_ = int(len(X_source_validation) - len(selected))
        return self

    def assess_window(self, X_window: np.ndarray) -> dict:
        if not hasattr(self, "threshold_"):
            raise ValueError("sentinel is not calibrated")
        if len(X_window) != self.window_size:
            raise ValueError(
                f"operational assessment requires exactly {self.window_size} rows"
            )
        score = self.score_window(X_window)
        shifted = bool(score > self.threshold_)
        return {
            "status": DOMAIN_SHIFT_DETECTED if shifted else "IN_DOMAIN",
            "shift_detected": shifted,
            "score": score,
            "threshold": float(self.threshold_),
            "rows": int(len(X_window)),
        }


class MarginalShiftSentinel(QuantileJSDShiftSentinel):
    """Comparable source-only marginal shift detector for JSD, PSI, or KS.

    This class is used only for a frozen method comparison. Each method receives the
    same source reference rows, calibration windows, audit windows, and target windows.
    """

    def __init__(self, *args, method: str = "jsd_mean", maximum_ks_reference_rows=4096, **kwargs):
        super().__init__(*args, **kwargs)
        if method not in {"jsd_mean", "psi_mean", "ks_mean"}:
            raise ValueError("method must be jsd_mean, psi_mean, or ks_mean")
        if maximum_ks_reference_rows < self.window_size:
            raise ValueError("maximum_ks_reference_rows must be at least window_size")
        self.method = method
        self.maximum_ks_reference_rows = int(maximum_ks_reference_rows)

    def fit(self, X_reference: np.ndarray):
        super().fit(X_reference)
        if self.method == "ks_mean":
            X_reference = self._validate_X(X_reference)
            rng = np.random.default_rng(self.seed + 7027)
            if len(X_reference) > self.maximum_ks_reference_rows:
                rows = np.sort(rng.choice(
                    len(X_reference), self.maximum_ks_reference_rows, replace=False
                ))
                X_reference = X_reference[rows]
            self.ks_reference_sorted_ = np.sort(X_reference, axis=0).astype(np.float32)
        return self

    @staticmethod
    def _psi(reference: np.ndarray, observed: np.ndarray) -> float:
        return float(np.sum((observed - reference) * np.log(observed / reference)))

    @staticmethod
    def _ks(reference_sorted: np.ndarray, observed: np.ndarray) -> float:
        observed = np.sort(np.asarray(observed, dtype=np.float32))
        n_observed = len(observed)
        reference_cdf = np.searchsorted(reference_sorted, observed, side="right") / len(reference_sorted)
        upper_observed_cdf = np.arange(1, n_observed + 1) / n_observed
        lower_observed_cdf = np.arange(0, n_observed) / n_observed
        return float(max(
            np.max(upper_observed_cdf - reference_cdf),
            np.max(reference_cdf - lower_observed_cdf),
        ))

    def feature_scores(self, X_window: np.ndarray) -> np.ndarray:
        X_window = self._validate_X(X_window)
        if X_window.shape[1] != self.n_features_in_:
            raise ValueError("feature count does not match the fitted sentinel")
        scores = np.empty(self.n_features_in_, dtype=np.float64)
        for feature_index, edges in enumerate(self.bin_edges_):
            if self.method == "ks_mean":
                scores[feature_index] = self._ks(
                    self.ks_reference_sorted_[:, feature_index], X_window[:, feature_index]
                )
            else:
                observed = self._probabilities(X_window[:, feature_index], edges)
                reference = self.reference_probabilities_[feature_index]
                scores[feature_index] = (
                    self._js_divergence(reference, observed)
                    if self.method == "jsd_mean"
                    else self._psi(reference, observed)
                )
        return scores

    def calibrate(self, X_calibration: np.ndarray, X_audit: np.ndarray | None = None):
        """Fit a threshold; optional audit windows are fully disjoint from calibration."""
        if X_audit is None:
            return super().calibrate(X_calibration)
        X_calibration = self._validate_X(X_calibration)
        X_audit = self._validate_X(X_audit)
        if X_calibration.shape[1] != self.n_features_in_ or X_audit.shape[1] != self.n_features_in_:
            raise ValueError("feature count does not match the fitted sentinel")
        calibration_windows = len(X_calibration) // self.window_size
        audit_windows = len(X_audit) // self.window_size
        if calibration_windows < self.minimum_calibration_windows or audit_windows < 1:
            raise ValueError("insufficient complete source calibration or audit windows")
        calibration_rows = calibration_windows * self.window_size
        audit_rows = audit_windows * self.window_size
        calibration_rng = np.random.default_rng(self.seed + 7019)
        audit_rng = np.random.default_rng(self.seed + 7021)
        self.calibration_window_indices_ = calibration_rng.permutation(len(X_calibration))[
            :calibration_rows
        ].reshape(calibration_windows, self.window_size)
        self.audit_window_indices_ = audit_rng.permutation(len(X_audit))[:audit_rows].reshape(
            audit_windows, self.window_size
        )
        self.calibration_scores_ = np.asarray(
            [self.score_window(X_calibration[index]) for index in self.calibration_window_indices_],
            dtype=np.float64,
        )
        self.audit_scores_ = np.asarray(
            [self.score_window(X_audit[index]) for index in self.audit_window_indices_],
            dtype=np.float64,
        )
        self.threshold_ = _finite_sample_upper_quantile(
            self.calibration_scores_, self.calibration_alpha
        )
        self.validation_rows_used_ = int(calibration_rows + audit_rows)
        self.validation_rows_omitted_ = int(
            len(X_calibration) - calibration_rows + len(X_audit) - audit_rows
        )
        return self


class ShiftGuardedClassifier:
    """Serializable four-state operational wrapper for one deployment batch."""

    def __init__(self, abstaining_classifier, shift_sentinel: QuantileJSDShiftSentinel):
        self.abstaining_classifier = abstaining_classifier
        self.shift_sentinel = shift_sentinel

    def predict_proba(self, X):
        return self.abstaining_classifier.predict_proba(X)

    def predict_operational(self, X) -> np.ndarray:
        X = np.asarray(X)
        assessment = self.shift_sentinel.assess_window(X)
        if assessment["shift_detected"]:
            return np.full(len(X), DOMAIN_SHIFT_DETECTED, dtype="<U24")
        predictions = self.abstaining_classifier.predict_with_rejection(X)
        output = np.full(len(X), UNKNOWN, dtype="<U24")
        output[predictions == 0] = BENIGN
        output[predictions == 1] = MALICIOUS
        return output
