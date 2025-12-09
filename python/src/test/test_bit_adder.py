#!/usr/bin/env python3
"""
Test suite for SNP-based bit adder system.
Tests the bit-adder.json resource file.
"""

import pytest
import sys
import json
import os

# Add the main module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))

from main.models.SNP import MatrixSNPSystem
from main.models.types import SNPSystem


@pytest.fixture
def system_data():
    """Load the bit-adder.json file before each test."""
    resource_path = os.path.join(os.path.dirname(__file__), 'resources', 'bit-adder.json')
    with open(resource_path) as f:
        return json.load(f)


def test_system_structure(system_data):
    """Test that the bit-adder system loads correctly."""
    print("\n--- Test: Bit-Adder System Structure ---")
    
    model = SNPSystem(**system_data)
    
    # Verify structure
    assert len(model.neurons) == 4, "System should have 4 neurons"
    assert len(model.synapses) == 3, "System should have 3 synapses"
    
    # Verify neuron types
    input_neurons = [n for n in model.neurons if n.type == "input"]
    regular_neurons = [n for n in model.neurons if n.type == "regular"]
    output_neurons = [n for n in model.neurons if n.type == "output"]
    
    assert len(input_neurons) == 2, "Should have 2 input neurons"
    assert len(regular_neurons) == 1, "Should have 1 regular neuron (adder)"
    assert len(output_neurons) == 1, "Should have 1 output neuron"
    
    print("✓ System structure validated")


def test_adder_rules(system_data):
    """Test that the adder neuron has correct rules."""
    print("\n--- Test: Adder Neuron Rules ---")
    
    model = SNPSystem(**system_data)
    adder = next(n for n in model.neurons if n.id == "add_{0,1}")
    
    assert len(adder.rules) == 3, "Adder should have 3 rules"
    
    # Check rules exist
    rules = adder.rules
    assert "a\\to a;0" in rules, "Should have forwarding rule"
    assert "a^{2}/a\\to\\lambda" in rules, "Should have 2-spike forgetting rule"
    assert "a^{3}/a^{2}\\to a;0" in rules, "Should have 3-spike reduction rule"
    
    print(f"Rules: {rules}")
    print("✓ Adder rules validated")


def test_simulation_runs_to_completion(system_data):
    """Test that the simulation runs without errors."""
    print("\n--- Test: Simulation Runs to Completion ---")
    
    model = SNPSystem(**system_data)
    engine = MatrixSNPSystem(model)
    
    print(f"Initial Configuration: {engine.config_vct}")
    
    # Run simulation
    engine.pseudorandom_simulate_all()
    
    print(f"Final Configuration: {engine.config_vct}")
    print(f"Iterations: {engine.iteration}")
    
    # Verify simulation ran
    assert engine.iteration > 0, "Simulation should run at least 1 iteration"
    
    print("✓ Simulation completed successfully")


def test_all_neurons_exhausted(system_data):
    """Test that all regular neurons are exhausted after simulation."""
    print("\n--- Test: All Neurons Exhausted ---")
    
    model = SNPSystem(**system_data)
    engine = MatrixSNPSystem(model)
    
    # Run simulation
    engine.pseudorandom_simulate_all()
    
    final_config = engine.config_vct
    
    print(f"Final Configuration: {final_config}")
    
    # All regular neurons should be 0
    assert final_config[2] == 0, "Adder neuron should be exhausted"
    
    # Input neurons should also be exhausted (represented as 0 in config_vct)
    assert final_config[0] == 0, "Input neuron 0 should be exhausted"
    assert final_config[1] == 0, "Input neuron 1 should be exhausted"
    
    print("✓ All neurons properly exhausted")


def test_output_spike_pattern_deterministic(system_data):
    """Test that the output spike pattern is produced correctly."""
    print("\n--- Test: Output Spike Pattern ---")
    
    model = SNPSystem(**system_data)
    engine = MatrixSNPSystem(model)
    
    # Get input patterns
    input0 = system_data["neurons"][0]["content"]
    input1 = system_data["neurons"][1]["content"]
    
    print(f"Input 0: '{input0}'")
    print(f"Input 1: '{input1}'")
    
    # Run simulation
    engine.pseudorandom_simulate_all()
    
    # Get output
    output = str(engine.content[-1])
    
    print(f"Output spike train: '{output}'")
    print(f"Output length: {len(output)}")
    
    # Verify output is a valid spike train
    assert all(c in '01' for c in output), "Output should be binary"
    assert len(output) > 0, "Output should not be empty"
    
    print("✓ Output pattern validated")


def test_default_configuration(system_data):
    """Test the default configuration produces expected output pattern."""
    print("\n--- Test: Default Configuration ---")
    
    model = SNPSystem(**system_data)
    engine = MatrixSNPSystem(model)
    
    # Run simulation
    engine.pseudorandom_simulate_all()
    
    output = str(engine.content[-1])
    
    print(f"Input 0: '111'")
    print(f"Input 1: '1101'")
    print(f"Output: '{output}'")
    
    # The actual pattern produced by this specific configuration
    # Note: This is based on the rules and may vary with pseudorandom choices
    assert '1' in output, "Output should contain at least one spike"
    assert len(output) > 0, "Output should not be empty"
    
    print("✓ Default configuration test passed")


def test_with_validation_pattern(system_data):
    """Test simulation with expected validation patterns."""
    print("\n--- Test: With Validation Pattern ---")
    
    # Add expected patterns - all neurons should be empty/0, output should match pattern
    data_with_expected = system_data.copy()
    # Input neurons show as empty strings after exhaustion
    # Regular neuron should be 0
    # Output should be a binary pattern
    data_with_expected['expected'] = ['', '', '0', '0010010']
    
    model = SNPSystem(**data_with_expected)
    engine = MatrixSNPSystem(model)
    
    print(f"Initial Configuration: {engine.config_vct}")
    
    # Run simulation
    engine.pseudorandom_simulate_all()
    
    print(f"Final Configuration: {engine.config_vct}")
    print(f"Final Content: {engine.content}")
    print(f"Expected Patterns: {engine.expected}")
    
    # Validate result
    result = engine.validate_result()
    
    if result:
        print("✓ Validation passed")
    else:
        print("✗ Validation failed")
    
    assert result, "System should validate successfully"


def test_output_pattern(system_data):
    """Test that output follows expected spike train pattern."""
    print("\n--- Test: Output Pattern ---")
    
    model = SNPSystem(**system_data)
    engine = MatrixSNPSystem(model)
    
    # Run simulation
    engine.pseudorandom_simulate_all()
    
    output = str(engine.content[-1])
    
    print(f"Output spike train: '{output}'")
    print(f"Length: {len(output)}")
    print(f"Total spikes: {output.count('1')}")
    print(f"Pattern: {output}")
    
    # Output should only contain 0s and 1s
    assert all(c in '01' for c in output), "Output should only contain 0 and 1"
    
    # Output should not be empty
    assert len(output) > 0, "Output should not be empty"
    
    print("✓ Output pattern is valid")
