from ortools.sat.python import cp_model
from Input_data_reading import inputs_data_read_optimized
import os
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def solve_instance(folder_path):
    """Solves a single instance and returns makespan and solve time."""
    folder_name = os.path.basename(folder_path)
    file_name = folder_name + ".xlsx"
    file_path = os.path.join(folder_path, file_name)
    
    # Initialize result values
    makespan = None
    solve_time = None
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return makespan, solve_time
        
    # Read input data
    data = inputs_data_read_optimized(file_name, folder_name, folder_path)
    if data is None:
        print(f"Failed to load data for {folder_name}")
        return makespan, solve_time
    
    # Extract data
    activities = data['activities']
    es_ls = data['es_ls_combined_list']
    precedence = data['unified_preced_array']
    l = data['l']  # Resource demands
    R = data['R']  # Resource availability
    time_periods = list(data['t'])
    resources = list(R.keys())  # Dynamically get resources
    time_horizon = data['total_time_horizon_time_units_value']
    
    # Earliest/latest start times
    earliest_start = {act: es_ls[i][0] for i, act in enumerate(activities)}
    latest_start = {act: es_ls[i][1] for i, act in enumerate(activities)}
    
    # Calculate total float for criticality
    total_float = {act: latest_start[act] - earliest_start[act] for act in activities}
    
    # Activity-Oriented Fix-and-Optimize with time tracking
    start_time = time.time()
    solution, makespan = activity_oriented_fao(
        activities, precedence, earliest_start, latest_start, 
        l, R, resources, time_horizon, total_float
    )
    solve_time = time.time() - start_time
    
    return makespan, solve_time

def activity_oriented_fao(activities, precedence, earliest_start, latest_start, 
                         l, R, resources, time_horizon, total_float):
    """Optimized Fix-and-Optimize procedure with makespan minimization."""
    # Base model setup
    model = cp_model.CpModel()
    start_vars = {}
    
    # Initialize start variables
    for act in activities:
        start_vars[act] = model.NewIntVar(
            int(earliest_start[act]), 
            int(latest_start[act]), 
            f"start_{act}"
        )
    
    # Create makespan variable
    makespan = model.NewIntVar(
        min(earliest_start.values()), 
        max(latest_start.values()) + 100,  # More conservative upper bound
        "makespan"
    )
    model.AddMaxEquality(makespan, [start_vars[act] + 1 for act in activities])
    
    # Precedence constraints
    for pred_idx, succ_idx in precedence:
        pred = activities[pred_idx]
        succ = activities[succ_idx]
        model.Add(start_vars[pred] + 1 <= start_vars[succ])
    
    # Resource constraints - compatible with all OR-Tools versions
    for res in resources:
        for t in range(time_horizon):
            # Get activities that could potentially start at time t
            possible_activities = [
                act for act in activities
                if earliest_start[act] <= t <= latest_start[act]
            ]
            
            if not possible_activities:
                continue
                
            demands = []
            for act in possible_activities:
                # Create indicator variable for start time = t
                is_start = model.NewBoolVar(f"start_{act}_{t}_{res}")
                model.Add(start_vars[act] == t).OnlyEnforceIf(is_start)
                model.Add(start_vars[act] != t).OnlyEnforceIf(is_start.Not())
                # Add resource demand if starting at t
                demands.append(is_start * l[act][res])
            
            # Add resource constraint for this time period
            if demands:  # Ensure we have demands to sum
                model.Add(sum(demands) <= R[res][t])
    
    # Objective: Minimize makespan
    model.Minimize(makespan)
    
    # Generate multiple initial solutions
    best_solution = None
    best_makespan = float('inf')
    seeds = [42, 123, 789, 555, 999]  # Different random seeds
    
    for seed in seeds:
        solver = cp_model.CpSolver()
        solver.parameters.random_seed = seed
        solver.parameters.max_time_in_seconds = 30.0
        status = solver.Solve(model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            current_makespan = solver.Value(makespan)
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_solution = {act: solver.Value(start_vars[act]) for act in activities}
    
    if best_solution is None:
        return None, None
    
    # Fix-and-Optimize iterations
    fixed_activities = set()
    solution = best_solution
    current_makespan = best_makespan
    
    # Order activities by criticality (lowest float first)
    activity_order = sorted(
        activities, 
        key=lambda act: (total_float[act], act)
    )
    
    for idx, act_to_fix in enumerate(activity_order):
        # Dynamic window sizing based on progress
        progress = idx / len(activities)
        if progress < 0.3:  # Early phase
            window_size = 3
        elif progress < 0.7:  # Middle phase
            window_size = 2
        else:  # Final phase
            window_size = 1
        
        new_model = cp_model.CpModel()
        new_start_vars = {}
        new_makespan = new_model.NewIntVar(0, current_makespan + 10, "makespan")
        
        # Create variables with tightened windows
        for act in activities:
            if act in fixed_activities:
                new_start_vars[act] = new_model.NewIntVar(
                    solution[act], solution[act], f"start_{act}"
                )
            else:
                lb = max(earliest_start[act], solution[act] - window_size)
                ub = min(latest_start[act], solution[act] + window_size)
                new_start_vars[act] = new_model.NewIntVar(
                    lb, ub, f"start_{act}"
                )
        
        # Makespan constraint
        new_model.AddMaxEquality(
            new_makespan, 
            [new_start_vars[act] + 1 for act in activities]
        )
        new_model.Add(new_makespan <= current_makespan)
        
        # Precedence constraints
        for pred_idx, succ_idx in precedence:
            pred = activities[pred_idx]
            succ = activities[succ_idx]
            new_model.Add(new_start_vars[pred] + 1 <= new_start_vars[succ])
        
        # Resource constraints (optimized for current makespan)
        for res in resources:
            # Only consider time periods up to current makespan
            for t in range(min(current_makespan + 1, time_horizon)):
                # Get activities that could start at time t
                possible_activities = [
                    act for act in activities
                    if (act not in fixed_activities and
                        max(earliest_start[act], solution[act] - window_size) <= t <= 
                        min(latest_start[act], solution[act] + window_size))
                ]
                
                if not possible_activities:
                    continue
                
                demand = []
                for act in possible_activities:
                    is_start = new_model.NewBoolVar(f"is_start_{act}_{t}_{res}")
                    new_model.Add(new_start_vars[act] == t).OnlyEnforceIf(is_start)
                    new_model.Add(new_start_vars[act] != t).OnlyEnforceIf(is_start.Not())
                    demand.append(is_start * l[act][res])
                
                # Add fixed activity demands
                fixed_demand = sum(
                    l[act][res] for act in fixed_activities 
                    if solution[act] == t
                )
                new_model.Add(sum(demand) + fixed_demand <= R[res][t])
        
        # Solve updated model
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30.0
        solver.parameters.num_search_workers = 8  # Parallel search
        status = solver.Solve(new_model)
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution = {act: solver.Value(new_start_vars[act]) for act in activities}
            current_makespan = solver.Value(new_makespan)
            fixed_activities.add(act_to_fix)
        else:
            # If no solution found, fix with current solution
            fixed_activities.add(act_to_fix)
    
    # Calculate final makespan
    final_makespan = max(solution.values())
    return solution, final_makespan

# ===== Main Execution =====
if __name__ == "__main__":
    # Configure input folder
    base_folder = "/Users/souvikchakraborty/Downloads/Activity-Oriented fixed and Optimise procedure/FSRCPSP_Instanzen/j120/k = 1"
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "compatible_results_summary_j120_1.csv")
    
    # Prepare results table
    results = []
    
    # Find all subfolders (each containing an instance)
    subfolders = [f.path for f in os.scandir(base_folder) if f.is_dir()]
    print(f"Found {len(subfolders)} instances to process")
    
    # Process instances
    for folder_path in subfolders:
        folder_name = os.path.basename(folder_path)
        print(f"\nProcessing: {folder_name}")
        
        start_time = time.time()
        try:
            makespan, solve_time = solve_instance(folder_path)
            elapsed = time.time() - start_time
            
            if makespan is not None:
                print(f"  Solved in {solve_time:.2f}s, Makespan: {makespan}")
                results.append({
                    'Instance': folder_name,
                    'Makespan': makespan,
                    'SolveTime': solve_time
                })
            else:
                print("  Failed to solve")
                results.append({
                    'Instance': folder_name,
                    'Makespan': 'Failed',
                    'SolveTime': 'Failed'
                })
        except Exception as e:
            print(f"  Error: {str(e)}")
            results.append({
                'Instance': folder_name,
                'Makespan': 'Error',
                'SolveTime': 'Error'
            })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Summary statistics
    success_df = df[df['Makespan'].apply(lambda x: isinstance(x, (int, float)))]
    
    print("\n===== Optimization Summary =====")
    print(f"Total instances: {len(subfolders)}")
    print(f"Successfully solved: {len(success_df)}")
    print(f"Failed: {len(df) - len(success_df)}")
    
    if not success_df.empty:
        print(f"\nAverage makespan: {success_df['Makespan'].mean():.2f}")
        print(f"Average solve time: {success_df['SolveTime'].mean():.2f}s")
        print(f"Best makespan: {success_df['Makespan'].min()}")
        print(f"Worst makespan: {success_df['Makespan'].max()}")
        
        # Save detailed solution report
        solution_report = os.path.join(output_dir, "solution_analysis.csv")
        success_df.to_csv(solution_report, index=False)
        print(f"Detailed solution report: {solution_report}")