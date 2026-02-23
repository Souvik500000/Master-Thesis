from pandas import read_excel, ExcelFile
import os
import numpy as np

def inputs_data_read_optimized(file_name, file_name_without_extension, folder_path):
    file_path = os.path.join(folder_path, file_name)
    print("%%\n" + file_path)

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    try:
        with ExcelFile(file_path) as xls:
            # Precedence matrix
            precedence_activities = xls.parse("a(i,j)", index_col=0).fillna(0).astype(int)
            rows, cols = np.where(precedence_activities == 1)
            unified_preced_array = [[int(r), int(c)] for r, c in zip(rows, cols)]

            # ES and LS values
            es_df = xls.parse("ES", header=None, names=["Activity", "ES"])
            ls_df = xls.parse("LS", header=None, names=["Activity", "LS"])
            early_start_dictionary = es_df.set_index("Activity")["ES"].to_dict()
            latest_start_dictionary = ls_df.set_index("Activity")["LS"].to_dict()
            activities = es_df["Activity"].tolist()
            es_ls_combined_list = [
                [early_start_dictionary[act], latest_start_dictionary[act]]
                for act in activities
            ]

            # Resource bounds
            lr_df = xls.parse("l(i,k)", index_col=0)
            lower_bound_resource_dict = lr_df.to_dict(orient="index")

            ur_df = xls.parse("u(i,k)", index_col=0)
            upper_bound_resource_dict = ur_df.to_dict(orient="index")

            # Constants M1 and M2
            m1_val = xls.parse("M1", header=None).iloc[0, 0]
            m2_val = xls.parse("M2", header=None).iloc[0, 0]

            # Time horizon
            time_horizons = xls.parse("t", header=None)[0].tolist()
            total_time_horizon_time_units_value = len(time_horizons)

            # Resource availability over time - DYNAMIC READING
            rk_df = xls.parse("R(k,t)", index_col=0)
            
            # Get all resource columns (k1, k2, k3, etc.)
            resource_columns = [col for col in rk_df.columns if col.startswith('k')]
            
            # Create R_dict dynamically
            R_dict = {}
            for resource in resource_columns:
                R_dict[resource] = rk_df[resource].tolist()

            # Workload (w(i,k,pi))
            workload_df = xls.parse("w(i,k,pi)").fillna(0)
            workload_matrix = workload_df.values
            num_scenarios = workload_matrix.shape[1] - 2
            work_load_all_scenarios_list = [
                workload_matrix[:, col].tolist()
                for col in range(2, 2 + num_scenarios)
            ]
            workload_dictionary = {row[0]: row[2] for row in workload_matrix}

    except Exception as e:
        print(f"Error while reading Excel file: {e}")
        return None

    return {
        'activities': activities,
        'unified_preced_array': unified_preced_array,
        'es_ls_combined_list': es_ls_combined_list,
        'l': lower_bound_resource_dict,
        'u': upper_bound_resource_dict,
        'm1_val': m1_val,
        'm2_val': m2_val,
        'total_time_horizon_time_units_value': total_time_horizon_time_units_value,
        'R': R_dict,
        'workload_dictionary': workload_dictionary,
        'work_load_all_scenarios_list': work_load_all_scenarios_list,
        't': list(range(len(R_dict[resource_columns[0]]))) if resource_columns else []
    }

# Optional: Test if running standalone
if __name__ == "__main__":
    result = inputs_data_read_optimized(
        "j102_2.xlsx",
        "j102_2",
        "/Users/souvikchakraborty/Downloads/Activity-Oriented fixed and Optimise procedure/FSRCPSP_Instanzen/j10/k = 3/j102_2"
    )
    
    if result:
        print(f"Found {len(result['R'])} resources: {list(result['R'].keys())}")
        print(f"Time horizon: {result['total_time_horizon_time_units_value']}")