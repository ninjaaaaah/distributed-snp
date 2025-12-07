"""
Example usage of the standalone SNP simulator.
This script demonstrates how to use the standalone simulator with various input methods.
"""

import sys
from pathlib import Path

# Add parent directory to path to import standalone module
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.simulate import simulate_all, simulate_all_last_only, simulate_step
import json


def example_with_json_file():
    """Example: Simulate using a JSON file."""
    print("=" * 60)
    print("Example 1: Simulating from JSON file")
    print("=" * 60)
    
    # Create a temporary test file
    test_data = {
        "neurons": [
            {"id": "in1", "type": "input", "content": "10"},
            {"id": "reg1", "type": "regular", "content": 0, "rules": ["a/a \\to a;0"]},
            {"id": "out", "type": "output", "content": "0"}
        ],
        "synapses": [
            {"from": "in1", "to": "reg1", "weight": 1.0},
            {"from": "reg1", "to": "out", "weight": 1.0}
        ]
    }
    
    test_file = Path("/tmp/example_snp_system.json")
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    try:
        result = simulate_all(test_file)
        print(f"Neuron keys: {result['keys']}")
        print(f"Number of steps: {len(result['states'])}")
        print(f"Final configuration: {result['configurations'][-1]}")
    finally:
        if test_file.exists():
            test_file.unlink()
    print()


def example_with_dictionary():
    """Example: Simulate using a Python dictionary."""
    print("=" * 60)
    print("Example 2: Simulating from Python dictionary")
    print("=" * 60)
    
    # Simple example system
    system_data = {
        "neurons": [
            {
                "id": "n1",
                "type": "input",
                "content": "10"
            },
            {
                "id": "n2",
                "type": "regular",
                "content": 2,
                "rules": ["a^2/a \\to a;0"]
            },
            {
                "id": "n3",
                "type": "output",
                "content": "0"
            }
        ],
        "synapses": [
            {"from": "n1", "to": "n2", "weight": 1.0},
            {"from": "n2", "to": "n3", "weight": 1.0}
        ]
    }
    
    result = simulate_all(system_data)
    print(f"Neuron keys: {result['keys']}")
    print(f"States:\n{result['states']}")
    print(f"Configurations:\n{result['configurations']}")
    print()


def example_last_only():
    """Example: Get only the final configuration."""
    print("=" * 60)
    print("Example 3: Getting only final configuration")
    print("=" * 60)
    
    system_data = {
        "neurons": [
            {"id": "n1", "type": "input", "content": "10"},
            {"id": "n2", "type": "regular", "content": 1, "rules": ["a/a \\to a;0"]},
            {"id": "n3", "type": "output", "content": "0"}
        ],
        "synapses": [
            {"from": "n1", "to": "n2", "weight": 1.0},
            {"from": "n2", "to": "n3", "weight": 1.0}
        ]
    }
    
    result = simulate_all_last_only(system_data)
    print(f"Final contents: {result['contents']}")
    print()


def example_single_step():
    """Example: Simulate a single step."""
    print("=" * 60)
    print("Example 4: Single step simulation")
    print("=" * 60)
    
    system_data = {
        "neurons": [
            {
                "id": "n1",
                "type": "input",
                "content": "10"
            },
            {
                "id": "n2",
                "type": "regular",
                "content": 2,
                "rules": ["a^2/a \\to a;0"]
            },
            {
                "id": "n3",
                "type": "output",
                "content": "0"
            }
        ],
        "synapses": [
            {"from": "n1", "to": "n2", "weight": 1.0},
            {"from": "n2", "to": "n3", "weight": 1.0}
        ]
    }
    
    result = simulate_step(system_data)
    print(f"Neuron keys: {result['keys']}")
    print(f"States after one step: {result['states']}")
    print(f"Configurations after one step: {result['configurations']}")
    print(f"System halted: {result['halted']}")
    print()


def example_with_json_string():
    """Example: Simulate using a JSON string."""
    print("=" * 60)
    print("Example 5: Simulating from JSON string")
    print("=" * 60)
    
    system_data = {
        "neurons": [
            {
                "id": "n1",
                "type": "input",
                "content": "101"
            },
            {
                "id": "n2",
                "type": "regular",
                "content": 1,
                "rules": ["a/a \\to a;0"]
            },
            {
                "id": "n3",
                "type": "output",
                "content": "0"
            }
        ],
        "synapses": [
            {"from": "n1", "to": "n2", "weight": 1.0},
            {"from": "n2", "to": "n3", "weight": 1.0}
        ]
    }
    
    json_string = json.dumps(system_data)
    result = simulate_all(json_string)
    print(f"Neuron keys: {result['keys']}")
    print(f"Number of steps: {len(result['states'])}")
    print()


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("SNP System Standalone Simulator - Examples")
    print("*" * 60)
    print("\n")
    
    example_with_dictionary()
    example_with_json_string()
    example_single_step()
    example_last_only()
    example_with_json_file()
    
    print("*" * 60)
    print("All examples completed!")
    print("*" * 60)
