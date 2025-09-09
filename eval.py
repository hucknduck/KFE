
import subprocess
import argparse
import os
from typing import Dict, Any, Optional
from itertools import combinations_with_replacement


class Eval:
    def __init__(self, args: Optional[Dict[str, Any]] = None):
        self.default_args = {
            'dataset': 'squirrel',
            'epochs': 1000,
            'patience': 200,
            'hidden': 512,
            'layers': 2,
            'device': 0,
            'runs': 1,
            'optimizer': 'Adam',
            'hop_lp': 2,
            'hop_hp': 5,
            'pro_dropout': 0.6,
            'lin_dropout': 0.0,
            'eta': -0.5,
            'bands': 3,
            'lr_adaptive': 0.1,
            'wd_adaptive': 0.05,
            'lr_adaptive2': 0.0,
            'wd_adaptive2': 0.0,
            'lr_lin': 0.005,
            'wd_lin': 0.0,
            'gf': 'sym',
            'activation': True,
            'full': True,
            'random_split': True,
            'combine': 'sum',
            'seed': 42,
            'hops': '5,5,5',
            'bandwidths': '1,3'
        }
        
        self.args = self.default_args.copy()
        if args:
            self.args.update(args)
    
    def build_sbatch_command(self) -> list:
        cmd = ['sbatch', 'slurm/array.sbatch']
        
        for key, value in self.args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
            else:
                cmd.extend([f'--{key}', str(value)])
        
        return cmd
    
    def execute_evaluation(self) -> subprocess.CompletedProcess:

        cmd = self.build_sbatch_command()
        
        print(f"Executing command: {' '.join(cmd)}")
        
        try:
            os.makedirs('logs', exist_ok=True)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"Job submitted successfully!")
            print(f"STDOUT: {result.stdout}")
            if result.stderr:
                print(f"STDERR: {result.stderr}")
            
            return result
            
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
            raise
    
    def update_args(self, new_args: Dict[str, Any]) -> None:

        self.args.update(new_args)
        print(f"Updated arguments: {new_args}")
    
    def get_args(self) -> Dict[str, Any]:

        return self.args.copy()
    
    def print_args(self) -> None:

        print("Current evaluation arguments:")
        for key, value in self.args.items():
            print(f"  {key}: {value}")

    def eval1(self):
        bandwidth_list = [1, 3, 5, 10]
        band_list = [2, 3, 4, 5]

        def compute_all_combinations(bands, bandwidth_list):
            midbands = bands - 2
            if midbands <= 0:
                return []
            
            all_pairs = []
            for bandwidth1 in bandwidth_list:
                for bandwidth2 in bandwidth_list:
                    all_pairs.append((bandwidth1, bandwidth2))
            
            all_combinations = list(combinations_with_replacement(all_pairs, midbands))
            
            return all_combinations
        
        for bands in band_list:
            combinations = compute_all_combinations(bands, bandwidth_list)
            



def main():
    evaluator = Eval()
    
    # Test the combinations generation
    print("=== Testing All Possible Sets of Pairs with Size of 'midbands' ===")
    compute_func = evaluator.eval1()
    
    print("\n=== Additional Examples ===")
    # Test with different bandwidth lists
    test_bandwidths = [1, 2, 3]
    test_bands = [3, 4, 5]
    
    for bands in test_bands:
        combinations = compute_func(bands, test_bandwidths)
        print(f"Bands: {bands}, Midbands: {bands-2}, Bandwidth options: {test_bandwidths}")
        print(f"Total combinations: {len(combinations)}")
        if len(combinations) <= 5:
            for i, combo in enumerate(combinations):
                print(f"  {i+1}: {combo}")
        print()
    
    print("=== Mathematical Analysis ===")
    print("For n bandwidth options, we have n² possible pairs.")
    print("For k midbands, we have C(n² + k - 1, k) combinations with replacement.")
    print("This is the number of ways to choose k items from n² items with replacement.")


if __name__ == "__main__":
    main()
