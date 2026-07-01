# Instance_Reader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import re


@dataclass
class InstanceData:
    n_jobs_including_dummy: int
    horizon: Optional[int]
    n_renewable_resources: int

    successors: Dict[int, List[int]]
    predecessors: Dict[int, List[int]]

    duration: Dict[int, int]
    requests_R: Dict[int, List[int]]
    expected_workload: Dict[int, List[int]]

    lR: Dict[int, List[int]]
    uR: Dict[int, List[int]]

    es: Dict[int, int]
    ls: Dict[int, int]

    scenarios: Dict[int, Dict[int, List[int]]]
    resource_capacities: Dict[int, int]


def _is_separator_line(line: str) -> bool:
    s = line.strip()
    return (len(s) >= 3) and (set(s) <= {"*"} or set(s) <= {"-"})


def _parse_ints_from_line(line: str) -> List[int]:
    return [int(x) for x in re.findall(r"-?\d+", line)]


def _parse_n_renewable_resources_from_file(lines: List[str]) -> int:
    in_resources = False
    for ln in lines:
        if ln.strip() == "RESOURCES":
            in_resources = True
            continue
        if in_resources:
            if ln.strip().startswith("*"):
                break
            if "renewable" in ln:
                nums = _parse_ints_from_line(ln)
                if nums:
                    return int(nums[0])
    return 0


def _infer_k_from_path(path: str) -> Optional[int]:
    # Supports: k_1, k_2, ... (underscore format)
    m = re.search(r"(?:^|[\\/])k_(\d+)(?=[_\\/]|$)", path)
    if m:
        return int(m.group(1))
    # Supports: K=4, k=4, k = 4, K = 4 (equals format, with or without spaces)
    m = re.search(r"[\\/]k\s*=\s*(\d+)[\\/]", path, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _parse_es_ls(lines: List[str], n_jobs: int) -> tuple[Dict[int, int], Dict[int, int]]:
    es: Dict[int, int] = {i: 0 for i in range(n_jobs)}
    ls: Dict[int, int] = {i: 0 for i in range(n_jobs)}

    start = None
    for idx, ln in enumerate(lines):
        if ln.strip().startswith("ES/LS-VALUES:"):
            start = idx
            break
    if start is None:
        raise ValueError("Could not find 'ES/LS-VALUES:' block.")

    i = start + 1
    while i < len(lines) and (lines[i].strip() == "" or _is_separator_line(lines[i]) or "jobnr" in lines[i]):
        i += 1

    while i < len(lines):
        ln = lines[i].strip()
        if ln == "" or _is_separator_line(ln):
            i += 1
            continue
        if ln.startswith("BIG-M-VALUES:") or ln.startswith("*"):
            break

        parts = ln.split()
        if len(parts) >= 3 and parts[0].isdigit():
            job = int(parts[0])
            es[job] = int(parts[1])
            ls[job] = int(parts[2])

        i += 1

    return es, ls


def _parse_resource_limits(lines: List[str], n_jobs: int, n_R: int) -> tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    lR: Dict[int, List[int]] = {i: [0] * n_R for i in range(n_jobs)}
    uR: Dict[int, List[int]] = {i: [0] * n_R for i in range(n_jobs)}

    start = None
    for idx, ln in enumerate(lines):
        if ln.strip().startswith("RESOURCE LIMITS:"):
            start = idx
            break
    if start is None:
        raise ValueError("Could not find 'RESOURCE LIMITS:' block.")

    i = start + 1
    while i < len(lines):
        s = lines[i].strip()
        if s == "" or _is_separator_line(s) or "jobnr" in s:
            i += 1
            continue
        break

    while i < len(lines):
        s = lines[i].strip()
        if s == "" or _is_separator_line(s):
            i += 1
            continue
        if s.startswith("SCENARIOS:") or s.startswith("ES/LS-VALUES:") or s.startswith("BIG-M-VALUES:") or s.startswith("*"):
            break

        parts = s.split()
        if len(parts) >= 1 and parts[0].isdigit():
            job = int(parts[0])
            for k in range(n_R):
                li = 1 + 2 * k
                ui = 1 + 2 * k + 1
                if ui < len(parts):
                    lR[job][k] = int(parts[li])
                    uR[job][k] = int(parts[ui])
        i += 1

    return lR, uR


def _parse_resource_capacities(lines: List[str], n_R: int) -> Dict[int, int]:
    """
    Tolerante Variante
    Wenn weniger als n_R Kapazitaeten in der Datei stehen, wird der Rest mit 0 aufgefuellt.
    """
    start = None
    for idx, ln in enumerate(lines):
        if ln.strip().startswith("RESOURCEAVAILABILITIES:"):
            start = idx
            break
    if start is None:
        raise ValueError("Could not find 'RESOURCEAVAILABILITIES:' block.")

    i = start + 1

    def first_numeric_line(from_idx: int) -> Optional[List[int]]:
        j = from_idx
        while j < len(lines):
            s = lines[j].strip()
            if s == "" or _is_separator_line(s) or s.startswith("*"):
                j += 1
                continue
            nums = _parse_ints_from_line(s)
            if not nums:
                j += 1
                continue
            if "R" in s:
                j += 1
                continue
            return nums
        return None

    nums = first_numeric_line(i)
    if nums is None:
        raise ValueError("Could not parse resource capacities after 'RESOURCEAVAILABILITIES:'.")

    # build caps from what we have, pad with 0 if needed
    caps: Dict[int, int] = {}
    for k in range(1, n_R + 1):
        if k - 1 < len(nums):
            caps[k] = int(nums[k - 1])
        else:
            caps[k] = 0

    return caps


def _parse_scenarios(lines: List[str], n_jobs: int, n_R: int, number_scen: int) -> Dict[int, Dict[int, List[int]]]:
    scenarios: Dict[int, Dict[int, List[int]]] = {}

    start = None
    for idx, ln in enumerate(lines):
        if ln.strip().startswith("SCENARIOS:"):
            start = idx
            break
    if start is None:
        raise ValueError("Could not find 'SCENARIOS:' block.")

    i = start + 1
    current_k: Optional[int] = None

    while i < len(lines) and (lines[i].strip() == "" or _is_separator_line(lines[i])):
        i += 1

    re_resource = re.compile(r"RESOURCE\s+k\s*=\s*(\d+)", re.IGNORECASE)
    re_s_line = re.compile(r"^\s*s\s*=\s*(\d+)\s*:\s*\((.*)\)\s*$", re.IGNORECASE)

    while i < len(lines):
        ln = lines[i].strip()

        if ln == "" or _is_separator_line(ln):
            i += 1
            continue

        if ln.startswith("ES/LS-VALUES:") or ln.startswith("BIG-M-VALUES:") or ln.startswith("*"):
            break

        m_k = re_resource.search(ln)
        if m_k:
            current_k = int(m_k.group(1))
            i += 1
            continue

        m_s = re_s_line.match(ln)
        if m_s:
            if current_k is None:
                raise ValueError("Found scenario line before 'RESOURCE k = ...' header.")

            s_idx = int(m_s.group(1))
            if 1 <= s_idx <= number_scen and 1 <= current_k <= n_R:
                inner = m_s.group(2)
                values = [v.strip() for v in inner.split(",") if v.strip() != ""]
                workloads = [int(v) for v in values]

                if len(workloads) != n_jobs:
                    print(
                        f"[warn] scenario s={s_idx}, k={current_k}: "
                        f"expected {n_jobs} workloads, got {len(workloads)}"
                    )

                scenarios.setdefault(s_idx, {})[current_k] = workloads

            i += 1
            continue

        i += 1

    for s in range(1, number_scen + 1):
        if s not in scenarios:
            print(f"[warn] scenario s={s} not found at all.")
            continue

    return scenarios


def read_instance(path: str, number_scen: int, k_override: Optional[int] = None) -> InstanceData:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # jobs
    n_jobs = None
    for ln in lines:
        if "jobs (incl. supersource/sink )" in ln:
            nums = _parse_ints_from_line(ln)
            if nums:
                n_jobs = int(nums[-1])
            break
    if n_jobs is None:
        raise ValueError("Could not find 'jobs (incl. supersource/sink )' line.")

    # horizon
    horizon = None
    for ln in lines:
        if ln.strip().startswith("horizon"):
            nums = _parse_ints_from_line(ln)
            if nums:
                horizon = int(nums[-1])
            break

    # header info only for warning
    n_R_file = _parse_n_renewable_resources_from_file(lines)

    # Decide n_R
    if k_override is not None:
        n_R = int(k_override)
    else:
        k_from_path = _infer_k_from_path(path)
        if k_from_path is not None:
            n_R = int(k_from_path)
        else:
            # default to 4 if nothing is specified
            n_R = 4

    if n_R <= 0:
        raise ValueError("n_R must be >= 1")

    if n_R_file > 0 and n_R != n_R_file:
        print(f"[warn] File declares {n_R_file} renewable resources, but reader uses n_R={n_R} as requested.")

    # precedence
    successors: Dict[int, List[int]] = {i: [] for i in range(n_jobs)}

    prec_start = None
    for idx, ln in enumerate(lines):
        if ln.strip().startswith("PRECEDENCE RELATIONS:"):
            prec_start = idx
            break
    if prec_start is None:
        raise ValueError("Could not find 'PRECEDENCE RELATIONS:' block.")

    i = prec_start + 1
    while i < len(lines) and (lines[i].strip() == "" or "jobnr" in lines[i]):
        i += 1

    while i < len(lines):
        ln = lines[i].strip()
        if ln == "" or _is_separator_line(ln):
            i += 1
            continue
        if ln.startswith("REQUESTS/DURATIONS:"):
            break
        parts = ln.split()
        if len(parts) >= 3 and parts[0].isdigit():
            job = int(parts[0])
            declared_n_succ = int(parts[2])
            succs = [int(tok) for tok in parts[3:] if tok.isdigit()]
            if declared_n_succ != len(succs):
                print(f"[warn] job {job}: declared #succ={declared_n_succ}, parsed={len(succs)} -> {succs}")
            successors[job] = succs
        i += 1

    predecessors_sets: Dict[int, Set[int]] = {i: set() for i in range(n_jobs)}
    for u, succs in successors.items():
        for v in succs:
            predecessors_sets[v].add(u)
    predecessors: Dict[int, List[int]] = {j: sorted(list(preds)) for j, preds in predecessors_sets.items()}

    # requests durations
    req_start = None
    for idx, ln in enumerate(lines):
        if ln.strip().startswith("REQUESTS/DURATIONS:"):
            req_start = idx
            break
    if req_start is None:
        raise ValueError("Could not find 'REQUESTS/DURATIONS:' block.")

    h = req_start + 1
    while h < len(lines) and not ("jobnr" in lines[h] and "duration" in lines[h]):
        h += 1
    if h >= len(lines):
        raise ValueError("Could not find REQUESTS/DURATIONS header line.")

    header_tokens = lines[h].split()
    ew_indices = [idx for idx, tok in enumerate(header_tokens) if tok == "E(W"]
    n_EW = len(ew_indices)

    duration: Dict[int, int] = {i: 0 for i in range(n_jobs)}
    requests_R: Dict[int, List[int]] = {i: [0] * n_R for i in range(n_jobs)}
    expected_workload: Dict[int, List[int]] = {i: [0] * n_EW for i in range(n_jobs)}

    r0 = h + 1
    while r0 < len(lines) and (lines[r0].strip() == "" or _is_separator_line(lines[r0])):
        r0 += 1

    r = r0
    while r < len(lines):
        ln = lines[r].strip()
        if ln == "" or _is_separator_line(ln):
            r += 1
            continue
        if ln.startswith("RESOURCEAVAILABILITIES:") or ln.startswith("RESOURCE LIMITS:") or ln.startswith("SCENARIOS:"):
            break
        if ln.startswith("*"):
            break

        parts = ln.split()
        if len(parts) < 3 or not parts[0].isdigit():
            r += 1
            continue

        job = int(parts[0])
        duration[job] = int(parts[2])

        for kk in range(n_R):
            idx_val = 3 + kk
            if idx_val < len(parts):
                requests_R[job][kk] = int(parts[idx_val])

        if n_EW > 0 and len(parts) >= 3 + n_R + n_EW:
            ew_start = len(parts) - n_EW
            for kk in range(n_EW):
                expected_workload[job][kk] = int(parts[ew_start + kk])

        r += 1

    resource_capacities = _parse_resource_capacities(lines, n_R)
    lR, uR = _parse_resource_limits(lines, n_jobs, n_R)
    es, ls = _parse_es_ls(lines, n_jobs)
    scenarios = _parse_scenarios(lines, n_jobs, n_R, number_scen=number_scen)

    return InstanceData(
        n_jobs_including_dummy=n_jobs,
        horizon=horizon,
        n_renewable_resources=n_R,
        successors=successors,
        predecessors=predecessors,
        duration=duration,
        requests_R=requests_R,
        expected_workload=expected_workload,
        lR=lR,
        uR=uR,
        es=es,
        ls=ls,
        scenarios=scenarios,
        resource_capacities=resource_capacities,
    )