"""
Test script for the standalone simulator.
Validates that the standalone functions work correctly.
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.simulate import simulate_all, simulate_all_last_only, simulate_step


def test_simulate_all_with_dict():
    """Test simulate_all with dictionary input."""
    print("Testing simulate_all with dictionary input...")
    
    system_data = {
        "neurons": [
            {"id": "n1", "type": "input", "content": "10"},
            {"id": "n2", "type": "regular", "content": 2, "rules": ["a^2/a \\to a;0"]},
            {"id": "n3", "type": "output", "content": "0"}
        ],
        "synapses": [
            {"from": "n1", "to": "n2", "weight": 1.0},
            {"from": "n2", "to": "n3", "weight": 1.0}
        ]
    }
    
    result = simulate_all(system_data)
    
    assert "states" in result, "Result should contain 'states'"
    assert "configurations" in result, "Result should contain 'configurations'"
    assert "keys" in result, "Result should contain 'keys'"
    assert isinstance(result["states"], list), "States should be a list"
    assert isinstance(result["configurations"], list), "Configurations should be a list"
    assert isinstance(result["keys"], list), "Keys should be a list"
    
    print("✓ simulate_all with dictionary works correctly")
    return True


def test_simulate_all_with_json_string():
    """Test simulate_all with JSON string input."""
    print("Testing simulate_all with JSON string input...")
    
    system_data = {
        "neurons": [
            {"id": "n1", "type": "input", "content": "10"},
            {"id": "n2", "type": "regular", "content": 2, "rules": ["a^2/a \\to a;0"]},
            {"id": "n3", "type": "output", "content": "0"}
        ],
        "synapses": [
            {"from": "n1", "to": "n2", "weight": 1.0},
            {"from": "n2", "to": "n3", "weight": 1.0}
        ]
    }
    
    json_string = json.dumps(system_data)
    result = simulate_all(json_string)
    
    assert "states" in result, "Result should contain 'states'"
    assert "configurations" in result, "Result should contain 'configurations'"
    assert "keys" in result, "Result should contain 'keys'"
    
    print("✓ simulate_all with JSON string works correctly")
    return True


def test_simulate_all_with_file():
    """Test simulate_all with file path input."""
    print("Testing simulate_all with file path input...")
    
    # Create a test file with proper format
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
    
    # Write test file temporarily
    test_file = Path("/tmp/test_snp_system.json")
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    try:
        result = simulate_all(test_file)
        
        assert "states" in result, "Result should contain 'states'"
        assert "configurations" in result, "Result should contain 'configurations'"
        assert "keys" in result, "Result should contain 'keys'"
        assert len(result["states"]) > 0, "Should have at least one state"
        
        print("✓ simulate_all with file path works correctly")
        return True
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()


def test_simulate_all_last_only():
    """Test simulate_all_last_only function."""
    print("Testing simulate_all_last_only...")
    
    system_data = {
        "neurons": [
            {"id": "n1", "type": "input", "content": "10"},
            {"id": "n2", "type": "regular", "content": 2, "rules": ["a^2/a \\to a;0"]},
            {"id": "n3", "type": "output", "content": "0"}
        ],
        "synapses": [
            {"from": "n1", "to": "n2", "weight": 1.0},
            {"from": "n2", "to": "n3", "weight": 1.0}
        ]
    }
    
    result = simulate_all_last_only(system_data)
    
    assert "contents" in result, "Result should contain 'contents'"
    assert isinstance(result["contents"], list), "Contents should be a list"
    
    print("✓ simulate_all_last_only works correctly")
    return True


def test_simulate_step():
    """Test simulate_step function."""
    print("Testing simulate_step...")
    
    system_data = {
        "neurons": [
            {"id": "n1", "type": "input", "content": "10"},
            {"id": "n2", "type": "regular", "content": 2, "rules": ["a^2/a \\to a;0"]},
            {"id": "n3", "type": "output", "content": "0"}
        ],
        "synapses": [
            {"from": "n1", "to": "n2", "weight": 1.0},
            {"from": "n2", "to": "n3", "weight": 1.0}
        ]
    }
    
    result = simulate_step(system_data)
    
    assert "states" in result, "Result should contain 'states'"
    assert "configurations" in result, "Result should contain 'configurations'"
    assert "keys" in result, "Result should contain 'keys'"
    assert "halted" in result, "Result should contain 'halted'"
    assert isinstance(result["halted"], bool), "Halted should be a boolean"
    
    print("✓ simulate_step works correctly")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Standalone Simulator Tests")
    print("=" * 60 + "\n")
    
    tests = [
        test_simulate_all_with_dict,
        test_simulate_all_with_json_string,
        test_simulate_all_with_file,
        test_simulate_all_last_only,
        test_simulate_step,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
