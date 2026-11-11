# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Opt-in, read-only CPU diagnostics for non-finite gradient bucket norms."""

import json
import math
import os
import socket
import time
from pathlib import Path

import torch

_CHUNK_NUMEL = 1024 * 1024


def _stats(tensor: torch.Tensor) -> dict:
    """Inspect bounded CPU copies; never cast to float64 on the accelerator."""
    flat = tensor.detach().view(-1)
    nan_count = posinf_count = neginf_count = 0
    max_abs_finite = 0.0
    partial_sums = []
    for start in range(0, flat.numel(), _CHUNK_NUMEL):
        values = flat[start : start + _CHUNK_NUMEL].cpu().to(dtype=torch.float64)
        nan_count += int(torch.isnan(values).sum().item())
        posinf_count += int(torch.isposinf(values).sum().item())
        neginf_count += int(torch.isneginf(values).sum().item())
        finite_values = values[torch.isfinite(values)]
        if finite_values.numel():
            max_abs_finite = max(max_abs_finite, float(finite_values.abs().max().item()))
            partial_sums.append(float(finite_values.square().sum().item()))
    finite_sum_sq = math.fsum(partial_sums)
    finite_l2 = math.sqrt(finite_sum_sq)
    nonfinite_count = nan_count + posinf_count + neginf_count
    return {
        "numel": flat.numel(),
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "nonfinite_count": nonfinite_count,
        "max_abs_finite": max_abs_finite,
        "finite_sum_sq": finite_sum_sq,
        "finite_l2": finite_l2,
        "norm64": "nan" if nan_count else "inf" if posinf_count + neginf_count else finite_l2,
    }


def _merge_stats(items: list[dict]) -> dict:
    totals = {
        key: sum(item[key] for item in items)
        for key in ("numel", "nan_count", "posinf_count", "neginf_count", "nonfinite_count")
    }
    totals["max_abs_finite"] = max((item["max_abs_finite"] for item in items), default=0.0)
    totals["finite_sum_sq"] = math.fsum(item["finite_sum_sq"] for item in items)
    totals["finite_l2"] = math.sqrt(totals["finite_sum_sq"])
    totals["norm64"] = (
        "nan"
        if totals["nan_count"]
        else "inf" if totals["nonfinite_count"] else totals["finite_l2"]
    )
    return totals


@torch.no_grad()
def write_bucket_diagnostics(bucket, group_index: int, grad_norm: torch.Tensor) -> Path:
    """Write a rank-local JSON report without collectives or gradient mutations.

    Args:
        bucket: Megatron bucket containing parameter-to-gradient slice mappings.
        group_index: Bucket index within the current communication bucket group.
        grad_norm: Original scalar norm that triggered this diagnostic.

    Returns:
        Path of the atomically written JSON report.
    """
    flat = bucket.grad_data.detach().view(-1)
    bucket_stats = _stats(flat)
    parameters = []
    ranges = []
    for index, param in enumerate(bucket.params_list):
        start, end = bucket.param_to_index[param]
        if not 0 <= start <= end <= flat.numel():
            raise ValueError(f"Invalid parameter gradient range: {(start, end)}")
        stats = _stats(flat[start:end])
        parameters.append(
            {
                "name": getattr(param, "_diag_name", f"unnamed_parameter_{index}"),
                "shape": list(param.shape),
                "parameter_dtype": str(param.dtype),
                "start": start,
                "end": end,
                **stats,
            }
        )
        ranges.append((start, end))
    padding = []
    cursor = 0
    for start, end in sorted(ranges):
        if start < cursor:
            raise ValueError("Overlapping parameter gradient ranges")
        if start > cursor:
            padding.append({"start": cursor, "end": start, **_stats(flat[cursor:start])})
        cursor = end
    if cursor < flat.numel():
        padding.append({"start": cursor, "end": flat.numel(), **_stats(flat[cursor:])})
    parameter_totals = _merge_stats(parameters)
    padding_totals = _merge_stats(padding)
    combined = _merge_stats([parameter_totals, padding_totals])
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    top20 = sorted(parameters, key=lambda item: item["finite_l2"], reverse=True)[:20]
    largest = max(parameters, key=lambda item: item["max_abs_finite"], default=None)
    observed_norm = float(grad_norm.detach().cpu().item())
    report = {
        "schema_version": 1,
        "rank": rank,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "timestamp_ns": time.time_ns(),
        "group_bucket_index": group_index,
        "bucket_id": bucket.bucket_id,
        "bucket_offset": bucket.offset,
        "bucket_numel_unpadded": bucket.numel_unpadded,
        "grad_dtype": str(flat.dtype),
        "grad_device": str(flat.device),
        "observed_norm": str(observed_norm) if not math.isfinite(observed_norm) else observed_norm,
        "chunk_numel": _CHUNK_NUMEL,
        "bucket": bucket_stats,
        "parameter_totals": parameter_totals,
        "padding_totals": padding_totals,
        "padding_ranges": padding,
        "coverage": {
            "numel_matches": combined["numel"] == bucket_stats["numel"],
            "nonfinite_count_matches": combined["nonfinite_count"] == bucket_stats["nonfinite_count"],
            "finite_sum_sq_matches": math.isclose(
                combined["finite_sum_sq"], bucket_stats["finite_sum_sq"], rel_tol=1e-12, abs_tol=0.0
            ),
        },
        "all_parameters": parameters,
        "nonfinite_parameters": [item for item in parameters if item["nonfinite_count"]],
        "top20_by_finite_l2": top20,
        "largest_finite_parameter": largest,
    }
    directory = Path(os.environ.get("GLM53_GRAD_DIAG_DIR", "/tmp/glm53-grad-diagnostics"))
    directory.mkdir(parents=True, exist_ok=True)
    filename = (
        f"rank{rank}-bucket{bucket.bucket_id}-group{group_index}-offset{bucket.offset}"
        f"-pid{os.getpid()}-{report['timestamp_ns']}.json"
    )
    path = directory / filename
    temporary = path.with_suffix(".json.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(
        f"[GLM53_GRAD_DIAG] rank={rank} bucket={bucket.bucket_id} group={group_index} "
        f"nonfinite={bucket_stats['nonfinite_count']} maxabs_finite={bucket_stats['max_abs_finite']:.6g} "
        f"norm64={bucket_stats['norm64']} largest_parameter={largest['name'] if largest else None} "
        f"padding_nonfinite={padding_totals['nonfinite_count']} report={path}",
        flush=True,
    )
    return path


def maybe_dump_grad_diagnostics(bucket, group_index: int, grad_norm: torch.Tensor) -> Path | None:
    """Run optional failure diagnostics while preserving the original fatal check."""
    if os.environ.get("GLM53_GRAD_DIAG") != "1":
        return None
    try:
        if bool(torch.isfinite(grad_norm).item()):
            return None
        return write_bucket_diagnostics(bucket, group_index, grad_norm)
    except Exception as exc:
        print(f"[GLM53_GRAD_DIAG] diagnostic failed: {type(exc).__name__}: {exc}", flush=True)
        return None
