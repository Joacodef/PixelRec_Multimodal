#!/usr/bin/env python
"""
A command-line test runner for the multimodal recommender system.

This script utilizes Python's native unittest framework to discover and execute
tests. It provides options to run the entire test suite, only the unit tests,
or a specific test module, making it a flexible tool for development and
continuous integration.
"""
import sys
import unittest
from pathlib import Path

# Adds the project root directory to the system path. This is necessary to ensure
# that modules from the 'src' directory can be correctly imported by the test files
# located in the 'tests/' directory.
sys.path.append(str(Path(__file__).resolve().parent.parent))


def run_all_tests():
    """
    Discovers and runs all tests located in the 'tests/' directory.

    This function searches recursively through the 'tests/' directory for any
    files that match the pattern 'test_*.py'. It aggregates all found tests
    into a single test suite and executes them.

    Returns:
        int: An exit code, 0 if all tests pass successfully, and 1 otherwise.
    """
    # Initializes the test loader, which is responsible for finding tests.
    loader = unittest.TestLoader()
    # Specifies the top-level directory where test discovery will begin.
    test_dir = Path(__file__).resolve().parent.parent / 'tests'
    
    # Discovers all test cases from modules matching the specified pattern.
    suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    # Initializes a test runner to execute the discovered test suite.
    # Verbosity is set to 2 for detailed output.
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Returns an exit code based on the success or failure of the test run.
    return 0 if result.wasSuccessful() else 1


def run_unit_tests():
    """
    Discovers and runs only the unit tests from the 'tests/unit/' directory.

    This function specifically targets the unit test suite, allowing for a faster
    and more focused test run that excludes longer-running integration tests.

    Returns:
        int: An exit code, 0 if all unit tests pass, and 1 otherwise.
    """
    loader = unittest.TestLoader()
    # Specifies the directory containing only the unit tests.
    test_dir = Path(__file__).resolve().parent.parent / 'tests' / 'unit'
    
    suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


def run_specific_test(test_path):
    """
    Runs a single, specified test module or class.

    This function is useful for debugging, allowing a developer to execute
    only the tests relevant to a specific piece of functionality.

    Args:
        test_path (str): The dot-notation path to the test module.
                         For example: 'tests.unit.test_data_filter'.

    Returns:
        int: An exit code, 0 if the specified tests pass, and 1 otherwise.
    """
    loader = unittest.TestLoader()
    # Loads tests directly from the specified module name.
    suite = loader.loadTestsFromName(test_path)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    # This block enables the script to be executed from the command line.
    import argparse
    
    # Initializes the argument parser to handle command-line options.
    parser = argparse.ArgumentParser(description='Run tests for the multimodal recommender system.')
    # Defines a flag to run only the unit tests.
    parser.add_argument('--unit', action='store_true', help='Run only unit tests.')
    # Defines an option to run a specific test file by its path.
    parser.add_argument('--test', type=str, help='Run a specific test module (e.g., tests.unit.test_data_filter).')
    
    args = parser.parse_args()
    
    # Executes the appropriate test-running function based on the provided arguments.
    # The script prioritizes running a specific test, then the unit test suite,
    # and defaults to running all tests if no arguments are given.
    if args.test:
        exit_code = run_specific_test(args.test)
    elif args.unit:
        exit_code = run_unit_tests()
    else:
        exit_code = run_all_tests()
    
    # Exits the script with the corresponding exit code from the test run.
    sys.exit(exit_code)