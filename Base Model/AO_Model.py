from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_FLOOR

import math

import gurobipy as gp
from gurobipy import GRB


# -------------------------------------------------
# Model containers
# -------------------------------------------------
@dataclass
class AO_Model:
    model: gp.Model

    activities: List[int]
    scenarios: List[int]
    resources: List[int]
    time_points: List[int]

    xB: gp.tupledict
    y: gp.tupledict
    roh: gp.tupledict

    d: gp.tupledict
    r: gp.tupledict

    xC: gp.tupledict
    xEff: gp.tupledict


def buildModel(
    data,
    number_scen: int,
    epsilon,
    threads: int = 4,
    mip_gap: float = 1e-5,
    time_limit_seconds: Optional[float] = None,
    seed: int = 42,
) -> AO_Model:

    # -------------------------------------------------
    # Epsilon exakt behandeln
    # -------------------------------------------------
    epsilon_dec = epsilon if isinstance(epsilon, Decimal) else Decimal(str(epsilon))
    allowed_violations = int(
        (Decimal(str(number_scen)) * epsilon_dec).to_integral_value(rounding=ROUND_FLOOR)
    )

    # -------------------------------------------------
    # Sets from Instance_Reader.py
    # -------------------------------------------------
    activities = list(range(int(data.n_jobs_including_dummy)))  # V
    scenarios = list(range(1, int(number_scen) + 1))  # S
    resources = list(range(1, int(data.n_renewable_resources) + 1))  # K

    if data.horizon is not None:
        horizon = int(data.horizon)
    else:
        horizon = int(max(data.ls.values())) if data.ls else 0
    time_points = list(range(0, horizon + 1))  # T

    precedence_constr = [(i, j) for i in activities for j in data.successors.get(i, [])]  # E

    # -------------------------------------------------
    # Build model
    # -------------------------------------------------
    model = gp.Model("AO_FSRCPSP")
    model.Params.Threads = int(threads)
    model.Params.MIPGap = float(mip_gap)
    model.Params.Seed = int(seed)
    if time_limit_seconds is not None:
        model.Params.TimeLimit = float(time_limit_seconds)

    # -------------------------------------------------
    # Binary decision variables (xB, y, rho)
    # -------------------------------------------------

    start_index = [(i, t) for i in activities for t in time_points]
    xB = model.addVars(start_index, vtype=GRB.BINARY, name="xB")

    y_index = [(i, t, pi) for pi in scenarios for i in activities for t in time_points]
    y = model.addVars(y_index, vtype=GRB.BINARY, name="y")

    roh = model.addVars(scenarios, vtype=GRB.BINARY, name="roh")

    # -------------------------------------------------
    # Real-valued decision variables (d, r)
    # -------------------------------------------------

    d = model.addVars(
        [(i, pi) for pi in scenarios for i in activities],
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="d",
    )

    r_index = [(i, k, t, pi) for pi in scenarios for i in activities for k in resources for t in time_points]
    r = model.addVars(r_index, vtype=GRB.CONTINUOUS, lb=0.0, name="r")

    # -------------------------------------------------
    # Continuous decision variables (xC, xEff)
    # -------------------------------------------------

    xC = model.addVars(start_index, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="xC")
    xEff = model.addVars(start_index, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="xEff")

    model.update()

    # -------------------------------------------------
    # Fix invalid start times to 0
    # -------------------------------------------------
    for i in activities:
        es_i = int(data.es[i])
        ls_i = int(data.ls[i])

        for t in time_points:
            if t < es_i or t > ls_i:
                xB[i, t].LB = 0.0
                xB[i, t].UB = 0.0
                xC[i, t].LB = 0.0
                xC[i, t].UB = 0.0

    model.update()

    # -------------------------------------------------
    # Big-M Value -> only for real activities 1,...,n
    # -------------------------------------------------
    M_prec = {}
    real_activities = activities[1:-1]

    for i in real_activities:
        for pi in scenarios:
            m_val = 0
            for k_idx, k in enumerate(resources):
                l_resources = int(data.lR[i][k_idx])
                if l_resources > 0 and k in data.scenarios[pi]:
                    workload = int(data.scenarios[pi][k][i])
                    m_val = max(m_val, int(math.ceil(workload / l_resources)))
            M_prec[(i, pi)] = m_val

    # -------------------------------------------------
    # Objective function
    # -------------------------------------------------
    model.setObjective(
        gp.quicksum(t * xEff[activities[-1], t] for t in time_points),
        GRB.MINIMIZE
    )

    # -------------------------------------------------
    # Restrictions
    # -------------------------------------------------

    model.addConstrs(
        (xEff[i, t] == xB[i, t] + xC[i, t] for i in activities for t in time_points),
        name="Def_xEff"
    )

    model.addConstrs(
        (gp.quicksum(xEff[i, t] for t in range(int(data.es[i]), int(data.ls[i]) + 1)) == 1 for i in activities),
        name="Start"
    )

    model.addConstrs(
        (
            gp.quicksum(t * xEff[j, t] for t in range(int(data.es[j]), int(data.ls[j]) + 1))
            - gp.quicksum(t * xEff[i, t] for t in range(int(data.es[i]), int(data.ls[i]) + 1))
            >= d[i, pi] - (len(time_points) + M_prec[(i, pi)]) * roh[pi]
            for (i, j) in precedence_constr if i in real_activities for pi in scenarios
        ),
        name="End_to_Start"
    )

    model.addConstrs(
        (
            gp.quicksum(r[i, k, t, pi] for i in activities)
            - (gp.quicksum(int(data.uR[i][k - 1]) for i in activities) - int(data.resource_capacities[k])) * roh[pi]
            <= int(data.resource_capacities[k])
            for k in resources for t in time_points for pi in scenarios
        ),
        name="Ressourcenkapazit"
    )

    model.addConstrs(
        (
            gp.quicksum(r[i, k, t, pi] for t in range(int(data.es[i]), int(data.ls[i]) + 1))
            == int(data.scenarios[pi][k][i])
            for pi in scenarios for k in resources for i in activities
            if k in data.scenarios[pi]
        ),
        name="Deckung_Gesamtressourcenbedarf"
    )

    model.addConstrs(
        (
            y[i, t, pi] <= gp.quicksum(r[i, k, tau, pi] for k in resources for tau in time_points if tau >= t)
            for i in activities for t in time_points for pi in scenarios
        ),
        name="Big_M_Methode_1"
    )

    model.addConstrs(
        (
            gp.quicksum(r[i, k, tau, pi] for k in resources for tau in time_points if tau >= t)
            <= (gp.quicksum(int(data.scenarios[pi][k][i]) for k in resources if k in data.scenarios[pi])) * y[i, t, pi]
            for i in activities for t in time_points for pi in scenarios
        ),
        name="Big_M_Methode_2"
    )

    model.addConstrs(
        (
            gp.quicksum(
                (y[i, t, pi] + gp.quicksum(xEff[i, tau] for tau in time_points if tau <= t) - 1)
                for t in time_points
            ) == d[i, pi]
            for i in real_activities for pi in scenarios
        ),
        name="Identifizierung_der_Dauer"
    )

    model.addConstrs(
        (
            int(data.lR[i][k - 1]) * (y[i, t, pi] + gp.quicksum(xEff[i, tau] for tau in time_points if tau <= t) - 1)
            <= r[i, k, t, pi]
            for i in activities for k in resources for t in time_points for pi in scenarios
        ),
        name="Variable_Ressourcenzuordnung_1"
    )

    model.addConstrs(
        (
            r[i, k, t, pi]
            <= int(data.uR[i][k - 1]) * (y[i, t, pi] + gp.quicksum(xEff[i, tau] for tau in time_points if tau <= t) - 1)
            for i in activities for k in resources for t in time_points for pi in scenarios
        ),
        name="Variable_Ressourcenzuordnung_2"
    )

    # -------------------------------------------------
    # (9) Exact scenario violation limit from epsilon
    # -------------------------------------------------
    model.addConstr(
        gp.quicksum(roh[pi] for pi in scenarios) <= allowed_violations,
        name="Zulaessigkeit_Vorrangbeziehungen_Ressourcenkapazit"
    )

    model.addConstrs(
        (gp.quicksum(y[i, t, pi] for t in time_points) >= d[i, pi] for i in activities for pi in scenarios),
        name="Test_1"
    )

    model.addConstrs(
        (
            y[i, t, pi] == 1
            for i in real_activities for t in time_points if t <= int(data.es[i]) for pi in scenarios
        ),
        name="Test_2"
    )

    model.addConstrs(
        (
            y[i, t, pi] >= xEff[i, t]
            for i in real_activities for t in time_points if t <= int(data.es[i]) for pi in scenarios
        ),
        name="Test_3"
    )

    model.update()

    return AO_Model(
        model=model,
        activities=activities,
        scenarios=scenarios,
        resources=resources,
        time_points=time_points,
        xB=xB,
        y=y,
        roh=roh,
        d=d,
        r=r,
        xC=xC,
        xEff=xEff,
    )


# -------------------------------------------------
# AO-Relax-and-Fix Bounds for subproblem M_z
# -------------------------------------------------

def reset_bounds_for_subproblem_Mz(
    ao: AO_Model,
    data,
) -> None:
    for i in ao.activities:
        es_i = int(data.es[i])
        ls_i = int(data.ls[i])

        for t in ao.time_points:
            if t < es_i or t > ls_i:
                ao.xB[i, t].LB = 0.0
                ao.xB[i, t].UB = 0.0
                ao.xC[i, t].LB = 0.0
                ao.xC[i, t].UB = 0.0
            else:
                ao.xB[i, t].LB = 0.0
                ao.xB[i, t].UB = 1.0
                ao.xC[i, t].LB = 0.0
                ao.xC[i, t].UB = 1.0

    for _, var in ao.y.items():
        var.LB = 0.0
        var.UB = 1.0

    ao.model.update()


def set_bounds_for_subproblem_Mz(
    ao: AO_Model,
    data,
    i_curr: int,
    t_left: int,
    t_right: int,
    fixed_start_times: Dict[int, int],
) -> None:
    fixed_activities = set(fixed_start_times.keys())

    for i in ao.activities:
        es_i = int(data.es[i])
        ls_i = int(data.ls[i])

        for t in range(es_i, ls_i + 1):
            if i in fixed_activities:
                s_fix = int(fixed_start_times[i])

                ao.xC[i, t].LB = 0.0
                ao.xC[i, t].UB = 0.0

                if t == s_fix:
                    ao.xB[i, t].LB = 1.0
                    ao.xB[i, t].UB = 1.0
                else:
                    ao.xB[i, t].LB = 0.0
                    ao.xB[i, t].UB = 0.0

            elif i == i_curr:
                if t < t_left and t >= es_i:
                    ao.xB[i, t].LB = 0.0
                    ao.xB[i, t].UB = 0.0
                elif t > t_right and t <= ls_i:
                    ao.xB[i, t].LB = 0.0
                    ao.xB[i, t].UB = 0.0
                else:
                    ao.xB[i, t].LB = 0.0
                    ao.xB[i, t].UB = 1.0

                if t >= es_i and t <= ls_i:
                    ao.xC[i, t].LB = 0.0
                    ao.xC[i, t].UB = 0.0

            else:
                ao.xB[i, t].LB = 0.0
                ao.xB[i, t].UB = 0.0

                ao.xC[i, t].LB = 0.0
                ao.xC[i, t].UB = 1.0

    ao.model.update()


# -------------------------------------------------
# Warm start (from SSGS schedule)
# -------------------------------------------------

def set_warmstart_for_subproblem_Mz(
    ao: AO_Model,
    data,
    S_warm: Dict[int, int],
    i_curr: int,
    t_left: int,
    t_right: int,
    fixed_start_times: Dict[int, int],
) -> None:
    fixed_activities = set(fixed_start_times.keys())

    for _, var in ao.xB.items():
        var.Start = 0.0

    for _, var in ao.xC.items():
        var.Start = 0.0

    for _, var in ao.xEff.items():
        var.Start = 0.0

    for i in ao.activities:
        es_i = int(data.es[i])
        ls_i = int(data.ls[i])

        if i in fixed_activities:
            s_i = int(fixed_start_times[i])
        elif i == i_curr:
            s_i = int(S_warm.get(i, es_i))
            if s_i < t_left:
                s_i = t_left
            if s_i > t_right:
                s_i = t_right
        else:
            s_i = int(S_warm.get(i, es_i))

        if i in fixed_activities or i == i_curr:
            ao.xB[i, s_i].Start = 1.0
            ao.xC[i, s_i].Start = 0.0
            ao.xEff[i, s_i].Start = 1.0
        else:
            ao.xB[i, s_i].Start = 0.0
            ao.xC[i, s_i].Start = 1.0
            ao.xEff[i, s_i].Start = 1.0

    ao.model.update()

# -------------------------------------------------
# Read solution
# -------------------------------------------------