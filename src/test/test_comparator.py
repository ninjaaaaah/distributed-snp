#!/usr/bin/env python3
"""
Test suite for SNP-based comparator system.
Tests the comparator.json resource file which computes min/max of two inputs.
"""

import unittest
import sys
import json
import os

# Add the main module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))

from models.SNP import MatrixSNPSystem
from models.types import SNPSystem


class TestComparator(unittest.TestCase):
    """Test suite for SNP comparator system."""
    
    def setUp(self):
        """Load the comparator.json file before each test."""
        resource_path = os.path.join(os.path.dirname(__file__), 'resources', 'comparator.json')
        with open(resource_path) as f:
            self.data = json.load(f)
    
    def test_system_structure(self):
        """Test that the comparator system loads correctly."""
        print("\n--- Test: Comparator System Structure ---")
        
        model = SNPSystem(**self.data)
        
        # Verify structure
        self.assertEqual(len(model.neurons), 6, "System should have 6 neurons")
        self.assertEqual(len(model.synapses), 7, "System should have 7 synapses")
        
        # Verify neuron types
        input_neurons = [n for n in model.neurons if n.type == "input"]
        regular_neurons = [n for n in model.neurons if n.type == "regular"]
        output_neurons = [n for n in model.neurons if n.type == "output"]
        
        self.assertEqual(len(input_neurons), 2, "Should have 2 input neurons (a, b)")
        self.assertEqual(len(regular_neurons), 2, "Should have 2 regular neurons (both, one)")
        self.assertEqual(len(output_neurons), 2, "Should have 2 output neurons (min, max)")
        
        print("✓ System structure validated")
    
    def test_neuron_ids(self):
        """Test that neurons have correct IDs."""
        print("\n--- Test: Neuron IDs ---")
        
        model = SNPSystem(**self.data)
        neuron_ids = [n.id for n in model.neurons]
        
        self.assertIn("a", neuron_ids, "Should have input neuron 'a'")
        self.assertIn("b", neuron_ids, "Should have input neuron 'b'")
        self.assertIn("both", neuron_ids, "Should have regular neuron 'both'")
        self.assertIn("one", neuron_ids, "Should have regular neuron 'one'")
        self.assertIn("min", neuron_ids, "Should have output neuron 'min'")
        self.assertIn("max", neuron_ids, "Should have output neuron 'max'")
        
        print("✓ Neuron IDs validated")
    
    def test_regular_neuron_rules(self):
        """Test that regular neurons have correct rules."""
        print("\n--- Test: Regular Neuron Rules ---")
        
        model = SNPSystem(**self.data)
        
        both = next(n for n in model.neurons if n.id == "both")
        one = next(n for n in model.neurons if n.id == "one")
        
        # 'both' neuron: forwards pairs, forgets singles
        self.assertEqual(len(both.rules), 2, "'both' should have 2 rules")
        self.assertIn("a^{2}\\to a;0", both.rules, "'both' should forward pairs")
        self.assertIn("a\\to\\lambda", both.rules, "'both' should forget singles")
        
        # 'one' neuron: forgets pairs, forwards singles
        self.assertEqual(len(one.rules), 2, "'one' should have 2 rules")
        self.assertIn("a^{2}\\to\\lambda", one.rules, "'one' should forget pairs")
        self.assertIn("a\\to a;0", one.rules, "'one' should forward singles")
        
        print("✓ Regular neuron rules validated")
    
    def test_simulation_runs_to_completion(self):
        """Test that the simulation runs without errors."""
        print("\n--- Test: Simulation Runs to Completion ---")
        
        # Set test inputs
        self.data['neurons'][0]['content'] = '111'  # a = 3
        self.data['neurons'][1]['content'] = '11'   # b = 2
        self.data['neurons'][4]['content'] = ''     # Clear min
        self.data['neurons'][5]['content'] = ''     # Clear max
        
        model = SNPSystem(**self.data)
        engine = MatrixSNPSystem(model)
        
        print(f"Initial Configuration: {engine.config_vct}")
        
        # Run simulation
        engine.pseudorandom_simulate_all()
        
        print(f"Final Configuration: {engine.config_vct}")
        print(f"Iterations: {engine.iteration}")
        
        # Verify simulation ran
        self.assertGreater(engine.iteration, 0, "Simulation should run at least 1 iteration")
        
        print("✓ Simulation completed successfully")
    
    def test_all_neurons_exhausted(self):
        """Test that all regular and input neurons are exhausted after simulation."""
        print("\n--- Test: All Neurons Exhausted ---")
        
        self.data['neurons'][0]['content'] = '111'
        self.data['neurons'][1]['content'] = '11'
        self.data['neurons'][4]['content'] = ''
        self.data['neurons'][5]['content'] = ''
        
        model = SNPSystem(**self.data)
        engine = MatrixSNPSystem(model)
        
        # Run simulation
        engine.pseudorandom_simulate_all()
        
        final_config = engine.config_vct
        
        print(f"Final Configuration: {final_config}")
        
        # All input and regular neurons should be 0
        self.assertEqual(final_config[0], 0, "Input 'a' should be exhausted")
        self.assertEqual(final_config[1], 0, "Input 'b' should be exhausted")
        self.assertEqual(final_config[2], 0, "Regular 'both' should be exhausted")
        self.assertEqual(final_config[3], 0, "Regular 'one' should be exhausted")
        
        print("✓ All neurons properly exhausted")
    
    def test_comparator_with_different_inputs(self):
        """Test comparator with a > b."""
        print("\n--- Test: Comparator with a > b ---")
        
        self.data['neurons'][0]['content'] = '111'  # a = 3
        self.data['neurons'][1]['content'] = '11'   # b = 2
        self.data['neurons'][4]['content'] = ''
        self.data['neurons'][5]['content'] = ''
        
        model = SNPSystem(**self.data)
        engine = MatrixSNPSystem(model)
        
        engine.pseudorandom_simulate_all()
        
        min_output = str(engine.content[4])
        max_output = str(engine.content[5])
        
        print(f"a = 3, b = 2")
        print(f"min output: '{min_output}' ({min_output.count('1')} spikes)")
        print(f"max output: '{max_output}' ({max_output.count('1')} spikes)")
        
        # Min should have fewer spikes than max
        self.assertLess(min_output.count('1'), max_output.count('1'),
                       "min should have fewer spikes than max")
        
        print("✓ Comparator correctly identifies min and max")
    
    def test_comparator_with_equal_inputs(self):
        """Test comparator with a = b."""
        print("\n--- Test: Comparator with a = b ---")
        
        self.data['neurons'][0]['content'] = '111'  # a = 3
        self.data['neurons'][1]['content'] = '111'  # b = 3
        self.data['neurons'][4]['content'] = ''
        self.data['neurons'][5]['content'] = ''
        
        model = SNPSystem(**self.data)
        engine = MatrixSNPSystem(model)
        
        engine.pseudorandom_simulate_all()
        
        min_output = str(engine.content[4])
        max_output = str(engine.content[5])
        
        print(f"a = 3, b = 3")
        print(f"min output: '{min_output}' ({min_output.count('1')} spikes)")
        print(f"max output: '{max_output}' ({max_output.count('1')} spikes)")
        
        # When equal, both outputs should reflect the same value
        # The 'both' neuron processes pairs, 'one' processes singles
        self.assertGreater(len(min_output), 0, "min output should not be empty")
        self.assertGreater(len(max_output), 0, "max output should not be empty")
        
        print("✓ Comparator handles equal inputs")
    
    def test_comparator_with_zero_input(self):
        """Test comparator with one input being zero."""
        print("\n--- Test: Comparator with Zero Input ---")
        
        self.data['neurons'][0]['content'] = '11'   # a = 2
        self.data['neurons'][1]['content'] = ''     # b = 0
        self.data['neurons'][4]['content'] = ''
        self.data['neurons'][5]['content'] = ''
        
        model = SNPSystem(**self.data)
        engine = MatrixSNPSystem(model)
        
        engine.pseudorandom_simulate_all()
        
        min_output = str(engine.content[4])
        max_output = str(engine.content[5])
        
        print(f"a = 2, b = 0")
        print(f"min output: '{min_output}' ({min_output.count('1')} spikes)")
        print(f"max output: '{max_output}' ({max_output.count('1')} spikes)")
        
        # Max should have spikes, min should have fewer or none
        self.assertGreaterEqual(max_output.count('1'), min_output.count('1'),
                               "max should have at least as many spikes as min")
        
        print("✓ Comparator handles zero input")
    
    def test_with_validation_pattern(self):
        """Test simulation with expected validation patterns."""
        print("\n--- Test: With Validation Pattern ---")
        
        # Add expected patterns
        data_with_expected = self.data.copy()
        data_with_expected['expected'] = ['', '', '0', '0', '01{2}0{4}', '01{4}0{2}']
        
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
        
        self.assertTrue(result, "System should validate successfully")


def run_tests():
    """Run all tests with verbose output."""
    print("\n" + "="*60)
    print("Testing SNP Comparator System")
    print("="*60)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestComparator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print(f"Test Results: {result.testsRun - len(result.failures) - len(result.errors)} passed, "
          f"{len(result.failures) + len(result.errors)} failed")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
