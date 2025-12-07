import unittest
from main.models.SNP import MatrixSNPSystem
from main.models.types import SNPSystem
import json


class TestSystem(unittest.TestCase):
    def test_something(self):
        pass


class TestSuite(unittest.TestSuite):
    def __init__(self, json_files):
        super().__init__()
        self.load_tests(json_files)

    def load_tests(self, json_files):
        for file_name in json_files:
            with open(file_name) as f:
                data = json.load(f)

            print(f"Test: {file_name}".ljust(68), end="")

            def test_system(self):
                system = SNPSystem(**data)
                matrixSNP = MatrixSNPSystem(system)
                matrixSNP.pseudorandom_simulate_all()
                self.assertTrue(matrixSNP.validate_result())

            setattr(TestSystem, f"test_{file_name}", test_system)
            self.addTest(TestSystem(f"test_{file_name}"))

            print("OK")


if __name__ == "__main__":
    systems = [
        "tests/kn-generator/2n-generator.json",
        "tests/kn-generator/3n-generator.json",
        "tests/bit-adder/input-bit-adder.json",
        "tests/bit-adder/reg-bit-adder.json",
        "tests/comparator/increasing-comparator-input.json",
        "tests/comparator/increasing-comparator-regular.json",
        "tests/input-output/binary-input.json",
        "tests/input-output/combined-output.json",
        "tests/input-output/knary-input.json",
    ]
    test_suite = TestSuite(systems)
    unittest.TextTestRunner().run(test_suite)
