# Standalone SNP System Simulator

This directory contains a standalone version of the SNP system simulator that can be used independently of the FastAPI web server.


## Features

- **No web server required**: Use the simulator directly from Python code or command line
- **Multiple input formats**: Accept JSON files, JSON strings, or Python dictionaries
- **Three simulation modes**:
  - `all`: Complete simulation returning all states and configurations
  - `last`: Returns only the final configuration
  - `step`: Simulates a single step

## Usage

### As a Python Module

```python
from standalone.simulate import simulate_all

# From a dictionary
system_data = {
    "neurons": [...],
    "synapses": [...]
}
result = simulate_all(system_data)
print(result['states'])
print(result['configurations'])
print(result['keys'])

# From a JSON file
result = simulate_all("path/to/system.json")

# From a JSON string
import json
json_string = json.dumps(system_data)
result = simulate_all(json_string)
```

### From Command Line

```bash
# Full simulation
python standalone/simulate.py tests/bit-adder/input-bit-adder.json

# Full simulation with pretty output
python standalone/simulate.py tests/bit-adder/input-bit-adder.json --pretty

# Only final configuration
python standalone/simulate.py tests/bit-adder/input-bit-adder.json --mode last

# Single step simulation
python standalone/simulate.py tests/bit-adder/input-bit-adder.json --mode step

# Save output to file
python standalone/simulate.py tests/bit-adder/input-bit-adder.json --output result.json --pretty
```

## API Reference

### `simulate_all(input_data)`

Simulate an SNP system and return all states and configurations.

**Parameters:**
- `input_data`: Dict, str (JSON string), or Path (file path)

**Returns:**
- Dictionary with:
  - `states`: List of states for each timestep
  - `configurations`: List of configurations for each timestep
  - `keys`: List of neuron identifiers

### `simulate_all_last_only(input_data)`

Simulate an SNP system and return only the final configuration.

**Parameters:**
- `input_data`: Dict, str (JSON string), or Path (file path)

**Returns:**
- Dictionary with:
  - `contents`: Final configuration of the system

### `simulate_step(input_data)`

Simulate a single step of an SNP system.

**Parameters:**
- `input_data`: Dict, str (JSON string), or Path (file path)

**Returns:**
- Dictionary with:
  - `states`: Current states after one step
  - `configurations`: Current configurations after one step
  - `keys`: List of neuron identifiers
  - `halted`: Boolean indicating if the system has halted

## Example

```python
from standalone.simulate import simulate_all
import json

# Load test data
with open('tests/bit-adder/input-bit-adder.json', 'r') as f:
    system_data = json.load(f)

# Run simulation
result = simulate_all(system_data)

# Access results
for i, (state, config) in enumerate(zip(result['states'], result['configurations'])):
    print(f"Step {i}:")
    print(f"  States: {state}")
    print(f"  Configurations: {config}")
```

## Dependencies

This module requires the same dependencies as the main application:
- numpy
- pydantic

These dependencies are already listed in the project's `requirements.txt`.
