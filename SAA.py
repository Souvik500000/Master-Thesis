"""
SAA_With_Criticality.py
Main SAA algorithm with integrated criticality scoring.
"""

from ortools.sat.python import cp_model
from Input_data_reading import inputs_data_read_optimized
from Criticality import CriticalityScorer, is_instance_critical
import os
import time
import pandas as pd
import numpy as np

# ===================================================
# MODIFIED SAA FUNCTIONS WITH CRITICALITY INTEGRATION
# ===================================================

def solve_instance_saa_refined(folder_path, num_scenarios_used=None):
    """Refined SAA solver with integrated criticality scoring."""
    folder_name = os.path.basename(folder_path)
    file_name = folder_name + ".xlsx"
    file_path = os.path.join(folder_path, file_name)
    
    # Initialize result values
    makespan = None
    solve_time = None
    approach_used = None
    refinement_improvement = 0
    criticality_info = {}
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return makespan, solve_time, approach_used, refinement_improvement, criticality_info
        
    # Read input data
    data = inputs_data_read_optimized(file_name, folder_name, folder_path)
    if data is None:
        print(f"Failed to load data for {folder_name}")
        return makespan, solve_time, approach_used, refinement_improvement, criticality_info
    
    # Extract data for criticality scoring
    instance_data = {
        'activities': data['activities'],
        'unified_preced_array': data['unified_preced_array'],
        'l': data['l'],
        'u': data.get('u', data['l']),  # Use lower bounds if upper not available
        'R': data['R'],
        'es_ls_combined_list': data['es_ls_combined_list'],
        'work_load_all_scenarios_list': data['work_load_all_scenarios_list'],
        'total_time_horizon_time_units_value': data['total_time_horizon_time_units_value']
    }
    
    # Extract data for SAA solving
    activities = data['activities']
    es_ls = data['es_ls_combined_list']
    precedence = data['unified_preced_array']
    l = data['l']  # Resource demands
    R = data['R']  # Resource availability
    resources = list(R.keys())
    time_horizon = data['total_time_horizon_time_units_value']
    workload_scenarios = data['work_load_all_scenarios_list']
    
    # Use all scenarios if not specified, otherwise use subset
    if num_scenarios_used is None or num_scenarios_used > len(workload_scenarios):
        num_scenarios_used = len(workload_scenarios)
    
    selected_scenarios = workload_scenarios[:num_scenarios_used]
    
    print(f"  Using {num_scenarios_used} scenarios out of {len(workload_scenarios)} available")
    
    # Earliest/latest start times
    earliest_start = {act: es_ls[i][0] for i, act in enumerate(activities)}
    latest_start = {act: es_ls[i][1] for i, act in enumerate(activities)}
    
    # === NEW: Compute criticality BEFORE solving ===
    print(f"  Computing criticality scores...")
    is_critical_initial, crit_scores, crit_reasons = is_instance_critical(instance_data)
    
    print(f"  Criticality Analysis:")
    print(f"    Hybrid Score: {crit_scores['hybrid_score']:.3f}")
    print(f"    Resource: {crit_scores['resource_constrainedness']:.3f}")
    print(f"    Time: {crit_scores['time_pressure']:.3f}")
    print(f"    Network: {crit_scores['network_density']:.3f}")
    print(f"    Variability: {crit_scores['workload_variability']:.3f}")
    print(f"    Size: {crit_scores['size_complexity']:.3f}")
    
    if is_critical_initial:
        print(f"    Initial Decision: CRITICAL - May apply refinement")
        for reason in crit_reasons:
            print(f"      Reason: {reason}")
    else:
        print(f"    Initial Decision: NON-CRITICAL - Likely Fast SAA only")
    
    # Analyze scenarios to find worst-case
    worst_case_scenario, worst_scenario_idx = find_worst_case_scenario_composite(
        selected_scenarios, activities, l, resources
    )
    print(f"  Worst-case scenario: #{worst_scenario_idx}")
    
    start_time = time.time()
    
    # Step 1: Get initial solution using Fast SAA
    initial_solution, initial_makespan = fast_saa_approach(
        activities, precedence, earliest_start, latest_start, 
        l, R, resources, time_horizon, selected_scenarios
    )
    
    if initial_solution is None:
        # Fallback to original approach if Fast SAA fails
        solution, makespan, approach_used = try_multiple_approaches_fallback(
            activities, precedence, earliest_start, latest_start, 
            l, R, resources, time_horizon, selected_scenarios
        )
        solve_time = time.time() - start_time
        criticality_info = {
            'is_critical': is_critical_initial,
            'hybrid_score': crit_scores['hybrid_score'],
            'resource_score': crit_scores['resource_constrainedness'],
            'time_score': crit_scores['time_pressure'],
            'network_score': crit_scores['network_density'],
            'variability_score': crit_scores['workload_variability'],
            'size_score': crit_scores['size_complexity']
        }
        return makespan, solve_time, approach_used, 0, criticality_info
    
    # === NEW: Final criticality decision (with initial solution info) ===
    # Re-evaluate with initial solution information
    is_critical_final, final_scores, final_reasons = is_instance_critical(
        instance_data, initial_makespan, initial_solution
    )
    
    print(f"  Final Criticality Decision:")
    print(f"    Initial makespan: {initial_makespan}")
    print(f"    Activities: {len(activities)}")
    print(f"    Stretch: {initial_makespan/len(activities):.2f}")
    print(f"    Decision: {'CRITICAL' if is_critical_final else 'NON-CRITICAL'}")
    
    # Apply refinement only if critical
    if is_critical_final:
        print(f"  Applying Conservative SAA refinement...")
        refined_solution, refined_makespan = conservative_saa_refinement(
            activities, precedence, earliest_start, latest_start, 
            l, R, resources, time_horizon, selected_scenarios, initial_solution
        )
        
        if refined_solution is not None and refined_makespan < initial_makespan:
            improvement = initial_makespan - refined_makespan
            improvement_percent = (improvement / initial_makespan) * 100
            print(f"  Refinement improved makespan by {improvement} units ({improvement_percent:.1f}%)")
            solution = refined_solution
            makespan = refined_makespan
            approach_used = "Fast SAA + Conservative Refinement"
            refinement_improvement = improvement
        else:
            solution = initial_solution
            makespan = initial_makespan
            approach_used = "Fast SAA (refinement failed or not beneficial)"
            refinement_improvement = 0
    else:
        solution = initial_solution
        makespan = initial_makespan
        approach_used = "Fast SAA only"
        refinement_improvement = 0
    
    solve_time = time.time() - start_time
    
    # Store criticality information
    criticality_info = {
        'is_critical': is_critical_final,
        'hybrid_score': crit_scores['hybrid_score'],
        'resource_score': crit_scores['resource_constrainedness'],
        'time_score': crit_scores['time_pressure'],
        'network_score': crit_scores['network_density'],
        'variability_score': crit_scores['workload_variability'],
        'size_score': crit_scores['size_complexity'],
        'initial_decision': is_critical_initial,
        'final_decision': is_critical_final,
        'improvement': refinement_improvement
    }
    
    return makespan, solve_time, approach_used, refinement_improvement, criticality_info


# ===================================================
# ORIGINAL SAA FUNCTIONS (KEEPING AS IS)
# ===================================================

def calculate_scenario_composite_score(scenario, activities, l, resources):
    """Calculate composite score for a scenario (higher = worse)."""
    if not scenario or not activities:
        return 0
    
    total_demand = 0
    resource_demands = {}
    
    # Initialize resource demands dictionary
    for res in resources:
        resource_demands[res] = 0
    
    # Calculate total demand and per-resource demand
    for i, act in enumerate(activities):
        if i < len(scenario):
            scenario_factor = scenario[i]
        else:
            scenario_factor = 1.0  # Default if scenario doesn't have this activity
        
        for res in resources:
            demand = l[act][res] * scenario_factor
            total_demand += demand
            resource_demands[res] += demand
    
    # 1. Total Demand Score (40%)
    total_score = total_demand
    
    # 2. Peak Demand Score (30%)
    if resource_demands:
        peak_demand = max(resource_demands.values())
    else:
        peak_demand = 0
    peak_score = peak_demand
    
    # 3. Imbalance Score (30%)
    if resource_demands and len(resource_demands) > 1:
        demand_values = list(resource_demands.values())
        mean_demand = np.mean(demand_values)
        if mean_demand > 0:
            cv = np.std(demand_values) / mean_demand  # Coefficient of variation
        else:
            cv = 0
    else:
        cv = 0
    imbalance_score = cv * total_demand  # Weight by total demand
    
    # Composite Score: Weighted combination
    composite_score = (0.4 * total_score) + (0.3 * peak_score) + (0.3 * imbalance_score)
    
    return composite_score

def find_worst_case_scenario_composite(scenarios, activities, l, resources):
    """Find worst-case scenario using composite scoring."""
    if not scenarios:
        return None, -1
    
    worst_scenario = None
    worst_score = -float('inf')
    worst_idx = 0
    
    for idx, scenario in enumerate(scenarios):
        score = calculate_scenario_composite_score(scenario, activities, l, resources)
        
        if score > worst_score:
            worst_score = score
            worst_scenario = scenario
            worst_idx = idx
    
    return worst_scenario, worst_idx

def conservative_saa_refinement(activities, precedence, earliest_start, latest_start, 
                             l, R, resources, time_horizon, scenarios, initial_solution):
    """Conservative SAA refinement using initial solution as starting point."""
    if len(scenarios) == 0:
        return None, None
    
    # Find worst-case scenario using composite scoring
    worst_case_scenario, worst_idx = find_worst_case_scenario_composite(
        scenarios, activities, l, resources
    )
    print(f"    Refinement using worst-case scenario #{worst_idx}")
        
    model = cp_model.CpModel()
    start_vars = {}
    
    # Initialize start variables with tightened bounds around initial solution
    refinement_window = 10  # Allow ±10 time units adjustment
    
    for act in activities:
        initial_start = initial_solution[act]
        lb = max(earliest_start[act], initial_start - refinement_window)
        ub = min(latest_start[act], initial_start + refinement_window)
        start_vars[act] = model.NewIntVar(int(lb), int(ub), f"start_{act}")
    
    max_makespan = max(latest_start.values()) + 20
    makespan = model.NewIntVar(0, max_makespan, "makespan")
    
    # Precedence constraints
    for pred_idx, succ_idx in precedence:
        pred = activities[pred_idx]
        succ = activities[succ_idx]
        model.Add(start_vars[pred] + 1 <= start_vars[succ])
    
    # Conservative resource constraints using worst-case scenario
    for res in resources:
        for t in range(time_horizon):
            demands = []
            for i, act in enumerate(activities):
                if earliest_start[act] <= t <= latest_start[act]:
                    is_start = model.NewBoolVar(f"start_{act}_{t}")
                    model.Add(start_vars[act] == t).OnlyEnforceIf(is_start)
                    model.Add(start_vars[act] != t).OnlyEnforceIf(is_start.Not())
                    
                    # Use scenario-influenced demand
                    if i < len(worst_case_scenario):
                        scenario_factor = worst_case_scenario[i]
                        adjusted_demand = min(l[act][res] * scenario_factor, R[res][t])
                    else:
                        adjusted_demand = l[act][res]
                        
                    demands.append(is_start * int(adjusted_demand))
            
            if demands:
                model.Add(sum(demands) <= R[res][t])
    
    # Makespan constraint
    completion_times = [start_vars[act] + 1 for act in activities]
    model.AddMaxEquality(makespan, completion_times)
    
    # Objective: minimize makespan
    model.Minimize(makespan)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0  # Limited time for refinement
    solver.parameters.num_search_workers = 4
    
    status = solver.Solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution = {act: solver.Value(start_vars[act]) for act in activities}
        return solution, solver.Value(makespan)
    
    return None, None

def fast_saa_approach(activities, precedence, earliest_start, latest_start, 
                    l, R, resources, time_horizon, scenarios):
    """Fast SAA that uses scenario data efficiently."""
    if not scenarios:
        return None, None
        
    model = cp_model.CpModel()
    start_vars = {}
    
    # Initialize start variables
    for act in activities:
        start_vars[act] = model.NewIntVar(
            int(earliest_start[act]), 
            int(latest_start[act]), 
            f"start_{act}"
        )
    
    max_makespan = max(latest_start.values()) + 30
    makespan = model.NewIntVar(0, max_makespan, "makespan")
    
    # Precedence constraints
    for pred_idx, succ_idx in precedence:
        pred = activities[pred_idx]
        succ = activities[succ_idx]
        model.Add(start_vars[pred] + 1 <= start_vars[succ])
    
    # Scenario usage: weighted average for resource demands
    scenario_weights = [1.0 / len(scenarios)] * len(scenarios)
    
    for res in resources:
        for t in range(min(60, time_horizon)):  # Limited horizon for speed
            demands = []
            for i, act in enumerate(activities):
                if earliest_start[act] <= t <= latest_start[act]:
                    is_start = model.NewBoolVar(f"start_{act}_{t}")
                    model.Add(start_vars[act] == t).OnlyEnforceIf(is_start)
                    model.Add(start_vars[act] != t).OnlyEnforceIf(is_start.Not())
                    
                    # Weighted average demand across scenarios
                    avg_demand = 0
                    for s, scenario in enumerate(scenarios):
                        if i < len(scenario):
                            scenario_demand = l[act][res] * scenario[i]
                        else:
                            scenario_demand = l[act][res]
                        avg_demand += scenario_demand * scenario_weights[s]
                    
                    demands.append(is_start * int(avg_demand))
            
            if demands:
                model.Add(sum(demands) <= R[res][t])
    
    # Makespan constraint
    completion_times = [start_vars[act] + 1 for act in activities]
    model.AddMaxEquality(makespan, completion_times)
    model.Minimize(makespan)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0
    solver.parameters.num_search_workers = 4
    
    status = solver.Solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution = {act: solver.Value(start_vars[act]) for act in activities}
        return solution, solver.Value(makespan)
    
    return None, None

def try_multiple_approaches_fallback(activities, precedence, earliest_start, latest_start, 
                                  l, R, resources, time_horizon, scenarios):
    """Fallback approach if Fast SAA fails."""
    approaches = [
        ("Conservative SAA", conservative_saa_approach),
        ("Robust Deterministic", robust_deterministic_approach)
    ]
    
    for approach_name, approach_func in approaches:
        print(f"    Fallback: Trying {approach_name}...")
        solution, makespan = approach_func(
            activities, precedence, earliest_start, latest_start, 
            l, R, resources, time_horizon, scenarios
        )
        
        if solution is not None:
            if validate_solution(activities, solution, precedence, l, R, resources, time_horizon):
                print(f"    ✓ {approach_name} succeeded with makespan: {makespan}")
                return solution, makespan, approach_name
    
    print("    ✗ All fallback approaches failed")
    return None, None, "Failed"

def conservative_saa_approach(activities, precedence, earliest_start, latest_start, 
                            l, R, resources, time_horizon, scenarios):
    """Conservative SAA approach using composite scoring for worst-case scenario."""
    if len(scenarios) == 0:
        return None, None
    
    # Find worst-case scenario using composite scoring
    worst_case_scenario, worst_idx = find_worst_case_scenario_composite(
        scenarios, activities, l, resources
    )
    print(f"    Using worst-case scenario #{worst_idx} (composite method)")
        
    model = cp_model.CpModel()
    start_vars = {}
    
    # Initialize start variables
    for act in activities:
        start_vars[act] = model.NewIntVar(
            int(earliest_start[act]), 
            int(latest_start[act]), 
            f"start_{act}"
        )
    
    max_makespan = max(latest_start.values()) + 40
    makespan = model.NewIntVar(0, max_makespan, "makespan")
    
    # Precedence constraints
    for pred_idx, succ_idx in precedence:
        pred = activities[pred_idx]
        succ = activities[succ_idx]
        model.Add(start_vars[pred] + 1 <= start_vars[succ])
    
    # Conservative resource constraints using worst-case scenario
    for res in resources:
        for t in range(time_horizon):
            demands = []
            for i, act in enumerate(activities):
                if earliest_start[act] <= t <= latest_start[act]:
                    is_start = model.NewBoolVar(f"start_{act}_{t}")
                    model.Add(start_vars[act] == t).OnlyEnforceIf(is_start)
                    model.Add(start_vars[act] != t).OnlyEnforceIf(is_start.Not())
                    
                    if i < len(worst_case_scenario):
                        scenario_factor = worst_case_scenario[i]
                        adjusted_demand = min(l[act][res] * scenario_factor, R[res][t])
                    else:
                        adjusted_demand = l[act][res]
                        
                    demands.append(is_start * int(adjusted_demand))
            
            if demands:
                model.Add(sum(demands) <= R[res][t])
    
    # Makespan constraint
    completion_times = [start_vars[act] + 1 for act in activities]
    model.AddMaxEquality(makespan, completion_times)
    model.Minimize(makespan)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.num_search_workers = 4
    
    status = solver.Solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution = {act: solver.Value(start_vars[act]) for act in activities}
        return solution, solver.Value(makespan)
    
    return None, None

def robust_deterministic_approach(activities, precedence, earliest_start, latest_start, 
                                l, R, resources, time_horizon, scenarios):
    """Robust deterministic approach."""
    model = cp_model.CpModel()
    start_vars = {}
    
    for act in activities:
        start_vars[act] = model.NewIntVar(
            int(earliest_start[act]), 
            int(latest_start[act]), 
            f"start_{act}"
        )
    
    max_makespan = max(latest_start.values()) + 20
    makespan = model.NewIntVar(0, max_makespan, "makespan")
    
    # Precedence constraints
    for pred_idx, succ_idx in precedence:
        pred = activities[pred_idx]
        succ = activities[succ_idx]
        model.Add(start_vars[pred] + 1 <= start_vars[succ])
    
    # Resource constraints
    for res in resources:
        for t in range(time_horizon):
            demands = []
            for act in activities:
                if earliest_start[act] <= t <= latest_start[act]:
                    is_start = model.NewBoolVar(f"start_{act}_{t}")
                    model.Add(start_vars[act] == t).OnlyEnforceIf(is_start)
                    model.Add(start_vars[act] != t).OnlyEnforceIf(is_start.Not())
                    demands.append(is_start * l[act][res])
            
            if demands:
                model.Add(sum(demands) <= R[res][t])
    
    model.AddMaxEquality(makespan, [start_vars[act] + 1 for act in activities])
    model.Minimize(makespan)
    
    # Try multiple seeds
    best_solution = None
    best_makespan = float('inf')
    seeds = [42, 123, 456, 789, 111]
    
    for seed in seeds:
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = seed
        solver.parameters.max_time_in_seconds = 15.0
        solver.parameters.num_search_workers = 4
        
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            current_makespan = solver.Value(makespan)
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_solution = {act: solver.Value(start_vars[act]) for act in activities}
    
    if best_solution is not None:
        return best_solution, best_makespan
    
    return None, None

def validate_solution(activities, solution, precedence, l, R, resources, time_horizon):
    """Validate solution feasibility."""
    if solution is None:
        return False
    
    # Check precedence
    for pred_idx, succ_idx in precedence:
        pred = activities[pred_idx]
        succ = activities[succ_idx]
        if solution[pred] + 1 > solution[succ]:
            return False
    
    # Check resources
    for res in resources:
        for t in range(time_horizon):
            total_demand = 0
            for act in activities:
                if solution[act] == t:
                    total_demand += l[act][res]
            if total_demand > R[res][t]:
                return False
    
    return True

# ===================================================
# MODIFIED MAIN EXECUTION WITH CRITICALITY ANALYSIS
# ===================================================

if __name__ == "__main__":
    # Configure input folder
    base_folder = "/Users/souvikchakraborty/Downloads/Activity-Oriented fixed and Optimise procedure/FSRCPSP_Instanzen/j10/k = 2"
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "output_saa_with_criticality")
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "saa_results_with_criticality.csv")
    
    # SAA parameters
    num_scenarios_used = 200
    
    # Prepare results table
    results = []
    
    # Find all subfolders
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    total_instances = len(subfolders)
    print(f"Found {total_instances} instances to process")
    print(f"Using refined SAA approach with integrated criticality scoring")
    print(f"Using composite scoring for worst-case scenario identification")
    
    # Process instances
    successful_solves = 0
    failed_solves = 0
    total_improvement = 0
    refined_instances = 0
    critical_instances = 0
    
    for idx, folder_path in enumerate(subfolders, 1):
        folder_name = os.path.basename(folder_path)
        print(f"\n[{idx}/{total_instances}] Processing: {folder_name}")
        print("-" * 60)
        
        try:
            makespan, solve_time, approach_used, improvement, criticality_info = solve_instance_saa_refined(
                folder_path, num_scenarios_used
            )
            
            if makespan is not None:
                status_msg = f"✓ Solved in {solve_time:.2f}s, Makespan: {makespan}"
                if improvement > 0:
                    status_msg += f" (Improved by {improvement})"
                print(f"  {status_msg}")
                print(f"  Method: {approach_used}")
                print(f"  Criticality: {'CRITICAL' if criticality_info.get('is_critical', False) else 'NON-CRITICAL'}")
                print(f"  Hybrid Score: {criticality_info.get('hybrid_score', 0):.3f}")
                
                # Collect results
                result_row = {
                    'Instance': folder_name,
                    'Makespan': makespan,
                    'SolveTime': solve_time,
                    'Approach_Used': approach_used,
                    'Improvement': improvement,
                    'Is_Critical': criticality_info.get('is_critical', False),
                    'Hybrid_Score': criticality_info.get('hybrid_score', 0),
                    'Resource_Score': criticality_info.get('resource_score', 0),
                    'Time_Score': criticality_info.get('time_score', 0),
                    'Network_Score': criticality_info.get('network_score', 0),
                    'Variability_Score': criticality_info.get('variability_score', 0),
                    'Size_Score': criticality_info.get('size_score', 0),
                    'Scenarios_Used': num_scenarios_used,
                    'Status': 'Success'
                }
                results.append(result_row)
                
                successful_solves += 1
                total_improvement += improvement
                
                if criticality_info.get('is_critical', False):
                    critical_instances += 1
                if improvement > 0:
                    refined_instances += 1
            else:
                print("  ✗ Failed to solve")
                results.append({
                    'Instance': folder_name,
                    'Makespan': 'Failed',
                    'SolveTime': 'Failed',
                    'Approach_Used': 'Failed',
                    'Improvement': 0,
                    'Is_Critical': 'Failed',
                    'Hybrid_Score': 'Failed',
                    'Scenarios_Used': num_scenarios_used,
                    'Status': 'Failed'
                })
                failed_solves += 1
            
            # Progress update
            success_rate = (successful_solves / idx) * 100
            print(f"\n  Progress: {successful_solves}/{idx} successful ({success_rate:.1f}%)")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                'Instance': folder_name,
                'Makespan': 'Error',
                'SolveTime': 'Error',
                'Approach_Used': 'Error',
                'Improvement': 0,
                'Is_Critical': 'Error',
                'Hybrid_Score': 'Error',
                'Scenarios_Used': num_scenarios_used,
                'Status': 'Error'
            })
            failed_solves += 1
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Summary statistics
    success_df = df[df['Makespan'].apply(lambda x: isinstance(x, (int, float)))]
    
    print("\n" + "="*80)
    print("SAA OPTIMIZATION WITH INTEGRATED CRITICALITY SCORING - SUMMARY")
    print("="*80)
    print(f"Total instances: {total_instances}")
    print(f"Successfully solved: {successful_solves}")
    print(f"Failed: {failed_solves}")
    print(f"Success rate: {(successful_solves/total_instances)*100:.1f}%")
    
    if not success_df.empty:
        print(f"\nPerformance Metrics:")
        print(f"  Average makespan: {success_df['Makespan'].mean():.2f}")
        print(f"  Average solve time: {success_df['SolveTime'].mean():.2f}s")
        print(f"  Best makespan: {success_df['Makespan'].min()}")
        print(f"  Worst makespan: {success_df['Makespan'].max()}")
        
        print(f"\nCriticality Analysis:")
        if 'Is_Critical' in success_df.columns:
            critical_count = success_df['Is_Critical'].sum()
            non_critical_count = len(success_df) - critical_count
            print(f"  Critical instances: {critical_count} ({critical_count/len(success_df)*100:.1f}%)")
            print(f"  Non-critical instances: {non_critical_count} ({non_critical_count/len(success_df)*100:.1f}%)")
        
        if 'Hybrid_Score' in success_df.columns:
            print(f"  Average hybrid score: {success_df['Hybrid_Score'].mean():.3f}")
            print(f"  Min hybrid score: {success_df['Hybrid_Score'].min():.3f}")
            print(f"  Max hybrid score: {success_df['Hybrid_Score'].max():.3f}")
        
        print(f"\nRefinement Analysis:")
        print(f"  Instances refined: {refined_instances} ({refined_instances/successful_solves*100:.1f}% of successful solves)")
        print(f"  Total improvement: {total_improvement} units")
        print(f"  Average improvement per refined instance: {total_improvement/refined_instances if refined_instances > 0 else 0:.2f}")
        
        print(f"\nApproach Distribution:")
        approach_counts = success_df['Approach_Used'].value_counts()
        for approach, count in approach_counts.items():
            percentage = (count / len(success_df)) * 100
            print(f"  {approach}: {count} instances ({percentage:.1f}%)")
        
        # Analyze criticality effectiveness
        if 'Is_Critical' in success_df.columns and 'Improvement' in success_df.columns:
            print(f"\nCriticality Decision Effectiveness:")
            # True Positive: Critical AND improvement > 0
            true_positives = success_df[(success_df['Is_Critical'] == True) & (success_df['Improvement'] > 0)].shape[0]
            # False Positive: Critical BUT improvement = 0
            false_positives = success_df[(success_df['Is_Critical'] == True) & (success_df['Improvement'] == 0)].shape[0]
            # False Negative: Not critical BUT improvement > 0 (missed opportunity)
            false_negatives = success_df[(success_df['Is_Critical'] == False) & (success_df['Improvement'] > 0)].shape[0]
            
            print(f"  True Positives (correct critical calls): {true_positives}")
            print(f"  False Positives (unnecessary refinements): {false_positives}")
            print(f"  False Negatives (missed refinements): {false_negatives}")
            
            if (true_positives + false_positives) > 0:
                precision = true_positives / (true_positives + false_positives)
                print(f"  Precision (correct critical decisions): {precision:.2%}")
        
        # Save detailed solution report
        solution_report = os.path.join(output_dir, "detailed_saa_analysis.csv")
        success_df.to_csv(solution_report, index=False)
        print(f"\nDetailed solution report: {solution_report}")
        
        # Save criticality analysis report
        if 'Hybrid_Score' in success_df.columns:
            crit_report = os.path.join(output_dir, "criticality_analysis.csv")
            crit_df = success_df[['Instance', 'Is_Critical', 'Hybrid_Score', 
                                 'Resource_Score', 'Time_Score', 'Network_Score',
                                 'Variability_Score', 'Size_Score', 'Improvement']]
            crit_df.to_csv(crit_report, index=False)
            print(f"Criticality analysis report: {crit_report}")