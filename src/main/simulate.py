"""
Standalone simulation function for SNP Systems.
This module provides a standalone function to simulate SNP systems
without requiring FastAPI or web server dependencies.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Union

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.models import SNPSystem, MatrixSNPSystem


def simulate_all(input_data: Union[Dict, str, Path]) -> Dict:
    """
    Simulate an SNP system and return all states and configurations.
    
    Args:
        input_data: Can be one of:
            - Dict: A dictionary containing the SNP system specification
            - str: A JSON string containing the SNP system specification
            - Path: Path to a JSON file containing the SNP system specification
    
    Returns:
        Dict containing:
            - states: List of states for each timestep
            - configurations: List of configurations for each timestep
            - keys: List of neuron identifiers
    
    Example:
        >>> system_data = {
        ...     "neurons": [...],
        ...     "synapses": [...]
        ... }
        >>> result = simulate_all(system_data)
        >>> print(result['states'])
    """
    # Parse input data
    if isinstance(input_data, (str, Path)):
        if isinstance(input_data, str):
            # Try to parse as JSON string first
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                # If that fails, treat it as a file path
                with open(input_data, 'r') as f:
                    data = json.load(f)
        else:
            # It's a Path object
            with open(input_data, 'r') as f:
                data = json.load(f)
    else:
        # It's already a dictionary
        data = input_data
    
    # Create SNPSystem from the data
    system = SNPSystem(**data)
    
    # Create MatrixSNPSystem and simulate
    matrixSNP = MatrixSNPSystem(system)
    matrixSNP.pseudorandom_simulate_all()
    
    # Return results
    return {
        "states": matrixSNP.states.tolist(),
        "configurations": matrixSNP.contents.tolist(),
        "keys": matrixSNP.neuron_keys,
    }


def simulate_all_last_only(input_data: Union[Dict, str, Path]) -> Dict:
    """
    Simulate an SNP system and return only the final configuration.
    
    Args:
        input_data: Can be one of:
            - Dict: A dictionary containing the SNP system specification
            - str: A JSON string containing the SNP system specification
            - Path: Path to a JSON file containing the SNP system specification
    
    Returns:
        Dict containing:
            - contents: Final configuration of the system
    
    Example:
        >>> system_data = {
        ...     "neurons": [...],
        ...     "synapses": [...]
        ... }
        >>> result = simulate_all_last_only(system_data)
        >>> print(result['contents'])
    """
    # Parse input data
    if isinstance(input_data, (str, Path)):
        if isinstance(input_data, str):
            # Try to parse as JSON string first
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                # If that fails, treat it as a file path
                with open(input_data, 'r') as f:
                    data = json.load(f)
        else:
            # It's a Path object
            with open(input_data, 'r') as f:
                data = json.load(f)
    else:
        # It's already a dictionary
        data = input_data
    
    # Create SNPSystem from the data
    system = SNPSystem(**data)
    
    # Create MatrixSNPSystem and simulate
    matrixSNP = MatrixSNPSystem(system)
    matrixSNP.pseudorandom_simulate_all()
    
    # Return results
    return {
        "contents": matrixSNP.content.tolist(),
    }


def simulate_step(input_data: Union[Dict, str, Path]) -> Dict:
    """
    Simulate a single step of an SNP system.
    
    Args:
        input_data: Can be one of:
            - Dict: A dictionary containing the SNP system specification
            - str: A JSON string containing the SNP system specification
            - Path: Path to a JSON file containing the SNP system specification
    
    Returns:
        Dict containing:
            - states: Current states after one step
            - configurations: Current configurations after one step
            - keys: List of neuron identifiers
            - halted: Boolean indicating if the system has halted
    
    Example:
        >>> system_data = {
        ...     "neurons": [...],
        ...     "synapses": [...]
        ... }
        >>> result = simulate_step(system_data)
        >>> print(result['halted'])
    """
    # Parse input data
    if isinstance(input_data, (str, Path)):
        if isinstance(input_data, str):
            # Try to parse as JSON string first
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                # If that fails, treat it as a file path
                with open(input_data, 'r') as f:
                    data = json.load(f)
        else:
            # It's a Path object
            with open(input_data, 'r') as f:
                data = json.load(f)
    else:
        # It's already a dictionary
        data = input_data
    
    # Create SNPSystem from the data
    system = SNPSystem(**data)
    
    # Create MatrixSNPSystem and simulate one step using pseudorandom_simulate_next
    matrixSNP = MatrixSNPSystem(system)
    matrixSNP.pseudorandom_simulate_next()
    
    # Return results
    return {
        "states": matrixSNP.state.tolist(),
        "configurations": matrixSNP.content.tolist(),
        "keys": matrixSNP.neuron_keys,
        "halted": bool(matrixSNP.halted),
    }


if __name__ == "__main__":
    # CLI interface for standalone usage
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simulate SNP systems from JSON input"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to JSON file containing SNP system specification"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "last", "step"],
        default="all",
        help="Simulation mode: 'all' for full simulation, 'last' for final config only, 'step' for single step"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (optional, defaults to stdout)"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print JSON output"
    )
    
    args = parser.parse_args()
    
    # Run simulation based on mode
    if args.mode == "all":
        result = simulate_all(args.input_file)
    elif args.mode == "last":
        result = simulate_all_last_only(args.input_file)
    else:  # step
        result = simulate_step(args.input_file)
    
    # Format output
    indent = 2 if args.pretty else None
    output_json = json.dumps(result, indent=indent)
    
    # Write to file or stdout
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
        print(f"Results written to {args.output}")
    else:
        print(output_json)
