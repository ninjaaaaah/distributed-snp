import unittest
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.models import SNPSystem, MatrixSNPSystem

class TestRegexValidation(unittest.TestCase):
    
    def test_input_to_output_transfer(self):
        """
        Test input->regular->output spike transfer with validation.
        Input "101" (2 spikes) transfers through regular to output.
        Final state: input exhausted (2), regular has spikes (2), output has received spikes (1).
        """
        print("\n--- Test: Input to Output Transfer ---")
        
        data = {
            "neurons": [
                {"id": "in1", "type": "input", "content": "101"},
                {"id": "reg1", "type": "regular", "content": 1, "rules": ["a/a \\to a;0"]},
                {"id": "out1", "type": "output", "content": "0"}
            ],
            "synapses": [
                {"from": "in1", "to": "reg1", "weight": 1},
                {"from": "reg1", "to": "out1", "weight": 1}
            ],
            "expected": ["2", "2", "1"]  # Based on test_system.json behavior
        }
        
        model = SNPSystem(**data)
        engine = MatrixSNPSystem(model)
        
        print(f"Initial Configuration: {engine.config_vct}")
        
        # Run simulation to completion
        engine.pseudorandom_simulate_all()
        
        print(f"Final Configuration: {engine.content}")
        print(f"Expected Patterns: {engine.expected}")
        print(f"Total iterations: {engine.iteration}")
        
        # Verify the result matches expected pattern
        result = engine.validate_result()
        if result:
            print("✓ Validation passed")
        self.assertTrue(result, "System should validate successfully after simulation.")

    def test_simple_spike_processing(self):
        """
        Test basic spike processing with realistic expectations.
        Input "10" (1 spike) should transfer through system.
        """
        print("\n--- Test: Simple Spike Processing ---")
        
        data = {
            "neurons": [
                {"id": "in1", "type": "input", "content": "10"},
                {"id": "reg1", "type": "regular", "content": 0, "rules": ["a/a \\to a;0"]},
                {"id": "out1", "type": "output", "content": "0"}
            ],
            "synapses": [
                {"from": "in1", "to": "reg1", "weight": 1},
                {"from": "reg1", "to": "out1", "weight": 1}
            ],
            "expected": ["1", "0", "0"]  # Input exhausted (1), regular passed through (0), output didn't receive yet (0)
        }
        
        model = SNPSystem(**data)
        engine = MatrixSNPSystem(model)
        
        print(f"Initial Configuration: {engine.config_vct}")
        
        # Run simulation
        engine.pseudorandom_simulate_all()
        
        print(f"Final Configuration: {engine.content}")
        print(f"Expected Patterns: {engine.expected}")
        print(f"Total iterations: {engine.iteration}")
        
        result = engine.validate_result()
        if result:
            print("✓ Validation passed")
        self.assertTrue(result, "System should process and validate input correctly.")

    def test_multiple_spikes_transfer(self):
        """
        Test transfer of multiple spikes with pattern matching.
        Input "111" (3 spikes) with initial regular content of 2.
        """
        print("\n--- Test: Multiple Spikes Transfer ---")
        
        data = {
            "neurons": [
                {"id": "in1", "type": "input", "content": "111"},
                {"id": "reg1", "type": "regular", "content": 2, "rules": ["a/a \\to a;0"]},
                {"id": "out1", "type": "output", "content": "0"}
            ],
            "synapses": [
                {"from": "in1", "to": "reg1", "weight": 1},
                {"from": "reg1", "to": "out1", "weight": 1}
            ],
            "expected": ["3", "2", "0"]  # Based on similar patterns
        }
        
        model = SNPSystem(**data)
        engine = MatrixSNPSystem(model)
        
        print(f"Initial Configuration: {engine.config_vct}")
        
        # Run simulation
        engine.pseudorandom_simulate_all()
        
        print(f"Final Configuration: {engine.content}")
        print(f"Expected Patterns: {engine.expected}")
        print(f"Total iterations: {engine.iteration}")
        
        result = engine.validate_result()
        if result:
            print("✓ Validation passed")
        self.assertTrue(result, "System should transfer all spikes and validate.")

    def test_negative_validation_wrong_output(self):
        """
        Test that validation fails when output doesn't match expected pattern.
        """
        print("\n--- Test: Negative Validation (Wrong Expected) ---")
        
        data = {
            "neurons": [
                {"id": "in1", "type": "input", "content": "10"},
                {"id": "reg1", "type": "regular", "content": 0, "rules": ["a/a \\to a;0"]},
                {"id": "out1", "type": "output", "content": "0"}
            ],
            "synapses": [
                {"from": "in1", "to": "reg1", "weight": 1},
                {"from": "reg1", "to": "out1", "weight": 1}
            ],
            "expected": [".", ".", "999"]  # Wrong: expecting "999" but will get "10"
        }
        
        model = SNPSystem(**data)
        engine = MatrixSNPSystem(model)
        
        # Run simulation
        engine.pseudorandom_simulate_all()
        
        print(f"Final Configuration: {engine.content}")
        print(f"Expected (WRONG): {engine.expected}")
        
        result = engine.validate_result()
        if not result:
            print("✓ Correctly detected mismatch")
        self.assertFalse(result, "Validation should fail when expected doesn't match actual.")

    def test_regex_pattern_validation(self):
        """
        Test that regex patterns work correctly for validation.
        Pattern [0-9]+ should match any sequence of digits.
        """
        print("\n--- Test: Regex Pattern Validation ---")
        
        data = {
            "neurons": [
                {"id": "in1", "type": "input", "content": "12345"},
                {"id": "reg1", "type": "regular", "content": 0, "rules": ["a/a \\to a;0"]},
                {"id": "out1", "type": "output", "content": "0"}
            ],
            "synapses": [
                {"from": "in1", "to": "reg1", "weight": 1},
                {"from": "reg1", "to": "out1", "weight": 1}
            ],
            "expected": ["[0-9]+", "[0-9]+", "[0-9]+"]  # Pattern: one or more digits for all neurons
        }
        
        model = SNPSystem(**data)
        engine = MatrixSNPSystem(model)
        
        # Run simulation
        engine.pseudorandom_simulate_all()
        
        print(f"Final Configuration: {engine.content}")
        print(f"Expected Pattern: {engine.expected}")
        
        result = engine.validate_result()
        if result:
            print("✓ Pattern [0-9]+ matched the output")
        self.assertTrue(result, "Regex pattern [0-9]+ should match digit sequences.")

    def test_strict_fullmatch_behavior(self):
        """
        Test that validation uses fullmatch (not match).
        Pattern "1" should NOT match "10".
        """
        print("\n--- Test: Strict fullmatch Behavior ---")
        
        data = {
            "neurons": [
                {"id": "in1", "type": "input", "content": "10"},
                {"id": "reg1", "type": "regular", "content": 0, "rules": ["a/a \\to a;0"]},
                {"id": "out1", "type": "output", "content": "0"}
            ],
            "synapses": [
                {"from": "in1", "to": "reg1", "weight": 1},
                {"from": "reg1", "to": "out1", "weight": 1}
            ],
            "expected": [".", ".", "1"]  # Expecting exactly "1", but will get "10"
        }
        
        model = SNPSystem(**data)
        engine = MatrixSNPSystem(model)
        
        # Run simulation
        engine.pseudorandom_simulate_all()
        
        print(f"Final Configuration: {engine.content}")
        print(f"Expected Pattern: {engine.expected}")
        
        result = engine.validate_result()
        
        if not result:
            print("✓ Correctly rejected '10' against pattern '1' (fullmatch works)")
        else:
            print("✗ INCORRECTLY accepted '10' against pattern '1'")
        
        # If using match(), "1" would match "10" (prefix match)
        # If using fullmatch(), "1" will NOT match "10" (exact match)
        self.assertFalse(result, 
                        "Pattern '1' should NOT match '10' when using fullmatch.")

if __name__ == "__main__":
    unittest.main()