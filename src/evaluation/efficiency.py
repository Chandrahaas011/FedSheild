from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import platform
import time
from pathlib import Path

import joblib
import numpy as np


def process_memory_mb() -> dict[str, float | None]:
    """Return current and peak process working set without an extra dependency."""
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if ok:
            return {
                "working_set_mb": float(counters.WorkingSetSize / 2**20),
                "peak_working_set_mb": float(counters.PeakWorkingSetSize / 2**20),
            }
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1024 if platform.system() != "Darwin" else 2**20
        return {"working_set_mb": None, "peak_working_set_mb": float(peak / divisor)}
    except (ImportError, AttributeError):
        return {"working_set_mb": None, "peak_working_set_mb": None}


def benchmark_model(
    model,
    X: np.ndarray,
    artifact_path: Path,
    single_repeats: int = 200,
    batch_size: int = 1024,
    batch_repeats: int = 30,
) -> dict:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path, compress=3)

    sample_count = min(len(X), max(batch_size, single_repeats))
    sample = np.asarray(X[:sample_count])
    # One inference thread avoids measuring OpenMP thread-launch overhead and makes
    # comparisons repeatable across estimators. Training may still use available CPUs.
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        model.predict_proba(sample[: min(32, sample_count)])
        single_times = []
        for index in range(single_repeats):
            row = sample[index % sample_count : index % sample_count + 1]
            start = time.perf_counter_ns()
            model.predict_proba(row)
            single_times.append((time.perf_counter_ns() - start) / 1e6)

        batch = sample[: min(batch_size, sample_count)]
        batch_times = []
        for _ in range(batch_repeats):
            start = time.perf_counter_ns()
            model.predict_proba(batch)
            batch_times.append((time.perf_counter_ns() - start) / 1e9)
    total_rows = len(batch) * batch_repeats
    total_seconds = sum(batch_times)

    memory = process_memory_mb()
    return {
        "serialized_model_bytes": int(artifact_path.stat().st_size),
        "single_record_latency_ms_median": float(np.median(single_times)),
        "single_record_latency_ms_p95": float(np.percentile(single_times, 95)),
        "batch_size": int(len(batch)),
        "batch_latency_ms_median": float(np.median(batch_times) * 1000),
        "batch_latency_ms_p95": float(np.percentile(batch_times, 95) * 1000),
        "throughput_records_per_second": float(total_rows / total_seconds),
        **memory,
    }


def benchmark_abstaining_model(
    model,
    X: np.ndarray,
    artifact_path: Path,
    single_repeats: int = 200,
    batch_size: int = 1024,
    batch_repeats: int = 30,
) -> dict:
    """Measure end-to-end probability plus rejection-decision inference."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path, compress=3)
    sample_count = min(len(X), max(batch_size, single_repeats))
    sample = np.asarray(X[:sample_count])
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        model.predict_with_rejection(sample[: min(32, sample_count)])
        single_times = []
        for index in range(single_repeats):
            row = sample[index % sample_count : index % sample_count + 1]
            start = time.perf_counter_ns()
            model.predict_with_rejection(row)
            single_times.append((time.perf_counter_ns() - start) / 1e6)
        batch = sample[: min(batch_size, sample_count)]
        batch_times = []
        for _ in range(batch_repeats):
            start = time.perf_counter_ns()
            model.predict_with_rejection(batch)
            batch_times.append((time.perf_counter_ns() - start) / 1e9)
    memory = process_memory_mb()
    return {
        "measurement": "end_to_end_probability_and_rejection",
        "serialized_model_bytes": int(artifact_path.stat().st_size),
        "single_record_latency_ms_median": float(np.median(single_times)),
        "single_record_latency_ms_p95": float(np.percentile(single_times, 95)),
        "batch_size": int(len(batch)),
        "batch_latency_ms_median": float(np.median(batch_times) * 1000),
        "batch_latency_ms_p95": float(np.percentile(batch_times, 95) * 1000),
        "throughput_records_per_second": float(
            len(batch) * batch_repeats / sum(batch_times)
        ),
        **memory,
    }
