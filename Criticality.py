"""
CriticalityScorer.py
A standalone module for computing criticality scores for FSRCPSP instances.
"""

import numpy as np
from typing import Dict, List, Optional

class CriticalityScorer:
    """
    Optimized hybrid criticality scorer for FSRCPSP instances.
    Computes composite difficulty scores without visualization overhead.
    """
    
    def __init__(self, weights: Optional[Dict] = None):
        """
        Initialize the criticality scorer.
        
        Args:
            weights: Optional custom weights for components
        """
        # Default balanced weights
        if weights is None:
            self.weights = {
                'network': 0.20,      # Precedence complexity
                'resource': 0.25,     # Resource constraints
                'time': 0.25,         # Schedule tightness
                'variability': 0.15,  # Scenario uncertainty
                'size': 0.15          # Problem scale
            }
        else:
            self.weights = weights
    
    def compute_criticality_scores(self, instance_data: Dict) -> Dict:
        """
        Main function to compute all criticality scores.
        
        Args:
            instance_data: Dictionary containing instance data with keys:
                - 'activities': List of activity names/IDs
                - 'unified_preced_array': Precedence relationships
                - 'l': Lower resource bounds
                - 'u': Upper resource bounds (use 'l' if not available)
                - 'R': Resource availability
                - 'es_ls_combined_list': Earliest/latest start times
                - 'work_load_all_scenarios_list': Workload scenarios
                - 'total_time_horizon_time_units_value': Time horizon
        
        Returns:
            Dictionary with all scores and metadata
        """
        scores = {}
        
        # Compute component scores
        scores['network_density'] = self._compute_network_density(
            instance_data['unified_preced_array'],
            len(instance_data['activities'])
        )
        
        # Use lower bounds as upper bounds if 'u' not provided
        upper_bounds = instance_data.get('u', instance_data['l'])
        
        scores['resource_constrainedness'] = self._compute_resource_constrainedness(
            instance_data['l'],
            upper_bounds,
            instance_data['R']
        )
        
        scores['time_pressure'] = self._compute_time_pressure(
            instance_data['es_ls_combined_list']
        )
        
        scores['workload_variability'] = self._compute_workload_variability(
            instance_data['work_load_all_scenarios_list']
        )
        
        scores['size_complexity'] = self._compute_instance_size_complexity(
            len(instance_data['activities']),
            len(instance_data['R']),
            instance_data['total_time_horizon_time_units_value']
        )
        
        # Compute weighted hybrid score
        components = ['network_density', 'resource_constrainedness', 
                     'time_pressure', 'workload_variability', 'size_complexity']
        component_weights = [self.weights['network'], self.weights['resource'], 
                           self.weights['time'], self.weights['variability'], self.weights['size']]
        
        scores['hybrid_score'] = np.average(
            [scores[comp] for comp in components],
            weights=component_weights
        )
        
        # Add basic instance info
        scores['num_activities'] = len(instance_data['activities'])
        scores['num_resources'] = len(instance_data['R'])
        scores['num_precedence'] = len(instance_data['unified_preced_array'])
        scores['time_horizon'] = instance_data['total_time_horizon_time_units_value']
        
        return scores
    
    def _compute_network_density(self, precedence_array: List[List[int]], 
                                num_activities: int) -> float:
        """Compute normalized network density (0-1)."""
        if num_activities <= 1:
            return 0.0
        
        max_edges = num_activities * (num_activities - 1) / 2
        actual_edges = len(precedence_array)
        density = actual_edges / max_edges
        
        # Sigmoid transformation for normalization
        return 1 / (1 + np.exp(-5 * (density - 0.3)))
    
    def _compute_resource_constrainedness(self, lower_bounds: Dict, 
                                         upper_bounds: Dict,
                                         resource_availability: Dict) -> float:
        """Compute resource utilization pressure (0-1)."""
        scores = []
        
        for resource, availability in resource_availability.items():
            total_demand = 0
            count = 0
            
            for activity in lower_bounds:
                if resource in lower_bounds[activity]:
                    lower = lower_bounds[activity].get(resource, 0)
                    upper = upper_bounds.get(activity, {}).get(resource, lower)
                    total_demand += (lower + upper) / 2
                    count += 1
            
            if count > 0 and availability:
                avg_demand = total_demand / count
                avg_availability = np.mean(availability)
                
                if avg_availability > 0:
                    utilization = avg_demand / avg_availability
                    # Constrainedness increases with utilization
                    score = 1 / (1 + np.exp(-4 * (utilization - 1)))
                    scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    def _compute_time_pressure(self, es_ls_list: List[List[int]]) -> float:
        """Compute schedule tightness (0-1)."""
        if not es_ls_list:
            return 0.5
        
        slacks = [ls - es for es, ls in es_ls_list if ls >= es]
        
        if not slacks:
            return 0.5
        
        avg_slack = np.mean(slacks)
        max_slack = max(slacks)
        
        if max_slack > 0:
            normalized_slack = avg_slack / max_slack
            time_pressure = 1 - normalized_slack
            
            # Adjust for variance (mixed criticality)
            std_slack = np.std(slacks) if len(slacks) > 1 else 0
            if avg_slack > 0:
                variance_factor = min(1, std_slack / avg_slack)
                time_pressure *= (1 + 0.3 * variance_factor)
            
            return min(1, max(0, time_pressure))
        
        return 0.9  # Very tight schedule if no slack
    
    def _compute_workload_variability(self, workload_scenarios: List[List[float]]) -> float:
        """Compute scenario uncertainty (0-1)."""
        if not workload_scenarios or len(workload_scenarios) < 2:
            return 0.5
        
        scenario_means = [np.mean(scenario) for scenario in workload_scenarios if scenario]
        
        if len(scenario_means) < 2:
            return 0.5
        
        cv = np.std(scenario_means) / (np.mean(scenario_means) + 1e-10)
        return min(1, cv / 0.5)
    
    def _compute_instance_size_complexity(self, num_activities: int, 
                                         num_resources: int,
                                         time_horizon: int) -> float:
        """Compute normalized size complexity (0-1)."""
        # Normalize with reasonable upper bounds
        act_score = min(1, num_activities / 100)
        res_score = min(1, num_resources / 10)
        time_score = min(1, time_horizon / 200)
        
        # Weighted combination favoring activities
        return 0.5 * act_score + 0.3 * res_score + 0.2 * time_score
    
    def identify_critical_instance(self, instance_data: Dict, 
                                 current_makespan: Optional[float] = None,
                                 initial_solution: Optional[Dict] = None,
                                 thresholds: Optional[Dict] = None) -> tuple:
        """
        Determine if an instance is critical based on computed scores.
        
        Args:
            instance_data: Raw instance data
            current_makespan: Current makespan (if available)
            initial_solution: Initial solution (if available)
            thresholds: Custom decision thresholds
        
        Returns:
            Tuple: (is_critical, scores, reasons)
        """
        # Default thresholds
        if thresholds is None:
            thresholds = {
                'hybrid': 0.7,
                'resource': 0.8,
                'variability': 0.75,
                'network': 0.6,
                'time': 0.7,
                'stretch': 2.5
            }
        
        # Compute scores
        scores = self.compute_criticality_scores(instance_data)
        
        is_critical = False
        reasons = []
        
        # Rule 1: High overall difficulty
        if scores['hybrid_score'] > thresholds['hybrid']:
            is_critical = True
            reasons.append(f"High hybrid score: {scores['hybrid_score']:.3f}")
        
        # Rule 2: Severe resource constraints
        elif scores['resource_constrainedness'] > thresholds['resource']:
            is_critical = True
            reasons.append(f"High resource pressure: {scores['resource_constrainedness']:.3f}")
        
        # Rule 3: High stochastic variability
        elif scores['workload_variability'] > thresholds['variability']:
            is_critical = True
            reasons.append(f"High workload variability: {scores['workload_variability']:.3f}")
        
        # Rule 4: Complex AND tight instances
        elif (scores['network_density'] > thresholds['network'] and 
              scores['time_pressure'] > thresholds['time']):
            is_critical = True
            reasons.append(f"Complex network ({scores['network_density']:.3f}) + tight schedule ({scores['time_pressure']:.3f})")
        
        # Rule 5: Large instance with challenges
        elif (scores['size_complexity'] > 0.7 and
              (scores['resource_constrainedness'] > 0.6 or 
               scores['time_pressure'] > 0.6)):
            is_critical = True
            reasons.append(f"Large instance with challenges")
        
        # Rule 6: If we have initial solution and it's poor
        if initial_solution is not None and current_makespan is not None:
            activities = instance_data['activities']
            stretch = current_makespan / len(activities) if len(activities) > 0 else 0
            if stretch > thresholds['stretch']:
                is_critical = True
                reasons.append(f"Poor initial solution (stretch: {stretch:.2f})")
        
        return is_critical, scores, reasons


# ============================================================================
# UTILITY FUNCTIONS (for direct use without class instantiation)
# ============================================================================

def compute_criticality_scores(instance_data: Dict, weights: Optional[Dict] = None) -> Dict:
    """
    Utility function to compute criticality scores without creating class instance.
    
    Args:
        instance_data: Instance data dictionary
        weights: Optional custom weights
        
    Returns:
        Dictionary of criticality scores
    """
    scorer = CriticalityScorer(weights)
    return scorer.compute_criticality_scores(instance_data)

def is_instance_critical(instance_data: Dict, 
                        current_makespan: Optional[float] = None,
                        initial_solution: Optional[Dict] = None,
                        thresholds: Optional[Dict] = None) -> tuple:
    """
    Utility function to determine if instance is critical.
    
    Args:
        instance_data: Instance data dictionary
        current_makespan: Current makespan
        initial_solution: Initial solution
        thresholds: Decision thresholds
        
    Returns:
        Tuple: (is_critical, scores, reasons)
    """
    scorer = CriticalityScorer()
    return scorer.identify_critical_instance(instance_data, current_makespan, 
                                            initial_solution, thresholds)


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_criticality_scorer():
    """Test function for the criticality scorer."""
    # Sample test data structure
    test_data = {
        'activities': [f'A{i}' for i in range(10)],
        'unified_preced_array': [[0, 1], [1, 2], [2, 3], [3, 4]],
        'l': {f'A{i}': {'R1': i+1, 'R2': i+2} for i in range(10)},
        'R': {'R1': [10, 10, 10, 10], 'R2': [15, 15, 15, 15]},
        'es_ls_combined_list': [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9], 
                               [5, 10], [6, 11], [7, 12], [8, 13], [9, 14]],
        'work_load_all_scenarios_list': [
            [1.0, 1.1, 0.9, 1.2, 0.8, 1.0, 1.1, 0.9, 1.2, 0.8],
            [1.2, 1.3, 1.1, 1.4, 1.0, 1.2, 1.3, 1.1, 1.4, 1.0]
        ],
        'total_time_horizon_time_units_value': 50
    }
    
    scorer = CriticalityScorer()
    scores = scorer.compute_criticality_scores(test_data)
    
    print("Criticality Scores Test:")
    print("=" * 50)
    for key, value in scores.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Test criticality decision
    is_critical, scores, reasons = scorer.identify_critical_instance(test_data)
    print(f"\nCriticality Decision: {is_critical}")
    if reasons:
        print("Reasons:")
        for reason in reasons:
            print(f"  - {reason}")


if __name__ == "__main__":
    test_criticality_scorer()