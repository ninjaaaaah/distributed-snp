#!/usr/bin/env python3
"""
Test suite for complex SNP system with multiple regular neurons and bidirectional connections.
Tests the system.json file from Downloads folder.
"""

import pytest
import sys
import json
import os

# Add the main module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main.models.SNP import MatrixSNPSystem
from main.models.types import SNPSystem


@pytest.fixture
def system_data():
    """Load the system.json file before each test."""
    system_path = "src/test/resources/2n-generator.json"
    with open(system_path) as f:
        return json.load(f)


def test_system_loads_correctly(system_data):
    """Test that the system JSON loads and validates correctly."""
    print("\n--- Test: System Loads Correctly ---")

    # Should not raise any validation errors
    model = SNPSystem(**system_data)

    # Verify structure
    assert len(model.neurons) == 4, "System should have 4 neurons"
    assert len(model.synapses) == 5, "System should have 5 synapses"

    # Verify neuron types
    regular_neurons = [n for n in model.neurons if n.type == "regular"]
    output_neurons = [n for n in model.neurons if n.type == "output"]

    assert len(regular_neurons) == 3, "System should have 3 regular neurons"
    assert len(output_neurons) == 1, "System should have 1 output neuron"

    print("✓ System structure validated")


def test_initial_configuration(system_data):
    """Test that initial configuration is set correctly."""
    print("\n--- Test: Initial Configuration ---")

    model = SNPSystem(**system_data)
    engine = MatrixSNPSystem(model)

    expected_initial = [2, 1, 3, 0]  # neuron 1: 2, neuron 2: 1, neuron 3: 3, output: 0

    print(f"Initial Configuration: {engine.config_vct}")
    print(f"Expected: {expected_initial}")

    assert engine.config_vct.tolist() == expected_initial, \
        "Initial configuration should match neuron contents"

    print("✓ Initial configuration correct")


def test_simulation_runs_to_completion(system_data):
    """Test that the simulation runs without errors."""
    print("\n--- Test: Simulation Runs to Completion ---")

    model = SNPSystem(**system_data)
    engine = MatrixSNPSystem(model)

    print(f"Initial: {engine.config_vct}")

    # Run simulation - should not raise any errors
    engine.pseudorandom_simulate_all()

    print(f"Final: {engine.config_vct}")
    print(f"Iterations: {engine.iteration}")

    # Verify simulation ran
    assert engine.iteration > 0, "Simulation should run at least 1 iteration"

    print("✓ Simulation completed successfully")


def test_final_state_matches_expected(system_data):
    """Test that the final state matches expected values."""
    print("\n--- Test: Final State Matches Expected ---")

    model = SNPSystem(**system_data)
    engine = MatrixSNPSystem(model)

    print(f"Initial Configuration: {engine.config_vct}")

    # Run simulation
    engine.pseudorandom_simulate_all()

    # All regular neurons should be exhausted (0), output neuron has spike pattern
    expected_final = [0, 0, 0, 0]  # All neurons should have 0 spikes at end

    print(f"Final Configuration: {engine.config_vct}")
    print(f"Expected: {expected_final}")
    print(f"Total Iterations: {engine.iteration}")

    assert engine.config_vct.tolist() == expected_final, \
        "Final configuration should have all neurons with 0 spikes"

    print("✓ Final state matches expected")


def test_output_neuron_receives_spikes(system_data):
    """Test that the output neuron receives spikes from neuron 3."""
    print("\n--- Test: Output Neuron Receives Spikes ---")

    model = SNPSystem(**system_data)
    engine = MatrixSNPSystem(model)

    # Output neuron is at index 3
    initial_output = engine.config_vct[3]

    print(f"Initial output neuron content: {initial_output}")

    # Run simulation
    engine.pseudorandom_simulate_all()

    # Check the spike train in the final content
    final_output_content = engine.content[3]

    print(f"Final output neuron spike train: {final_output_content}")

    # Output should have received spikes (non-empty string)
    assert len(str(final_output_content)) > 0, \
        "Output neuron should receive spikes during simulation"

    print("✓ Output neuron received spikes")


def test_bidirectional_connections_work(system_data):
    """Test that bidirectional connections between neurons 1 and 2 work."""
    print("\n--- Test: Bidirectional Connections Work ---")

    model = SNPSystem(**system_data)
    engine = MatrixSNPSystem(model)

    initial_state = engine.config_vct.copy()

    print(f"Initial: Neuron 1={initial_state[0]}, Neuron 2={initial_state[1]}")

    # Run simulation
    engine.pseudorandom_simulate_all()

    final_state = engine.config_vct

    print(f"Final: Neuron 1={final_state[0]}, Neuron 2={final_state[1]}")
    print(f"Changes: Neuron 1 {initial_state[0]} → {final_state[0]}, "
          f"Neuron 2 {initial_state[1]} → {final_state[1]}")

    # Both neurons should be exhausted (0 spikes)
    assert final_state[0] == 0, "Neuron 1 should be exhausted at end"
    assert final_state[1] == 0, "Neuron 2 should be exhausted at end"

    print("✓ Bidirectional connections functioning")


def test_with_validation_pattern(system_data):
    """Test simulation with expected validation patterns."""
    print("\n--- Test: With Validation Pattern ---")

    # Add expected patterns to the system
    data_with_expected = system_data.copy()
    data_with_expected['expected'] = ['0', '0', '0', '^10(00)*10$']

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

    assert result, "System should validate successfully with pattern"
