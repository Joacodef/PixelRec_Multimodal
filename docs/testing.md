# Testing Guide

This project uses the **`pytest`** framework for all testing. A robust suite of tests is included to ensure the reliability and correctness of the code. All tests are located in the **`tests/`** directory.

The CI pipeline, defined in **`.github/workflows/ci-tests.yaml`**, automatically runs the entire test suite on every push and pull request to the `main` branch.

---

### Test Structure

The tests are organized into two primary categories:

* **Unit Tests (`tests/unit`)**: These tests focus on individual components (e.g., a single function or class) in isolation. They ensure that the core logic of each module in the `src/` directory behaves as expected.

* **Integration Tests (`tests/integration`)**: These tests verify the end-to-end functionality of the system. They primarily focus on the command-line scripts in the `scripts/` directory, ensuring that they work together correctly and handle data and configurations as intended.

---

### How to Run Tests

There are two primary ways to run the tests: using the dedicated script or running `pytest` directly.

#### 1. Using the `run_tests.py` Script (Recommended)

The most convenient way to run all tests is by executing the provided script. It automatically discovers and runs the entire test suite.

* **Command**:
    ```bash
    python scripts/run_tests.py
    ```

#### 2. Using `pytest` Directly

You can also invoke `pytest` from the command line for more granular control.

* **To run all tests**:
    ```bash
    pytest
    ```

* **To run only unit tests**:
    ```bash
    pytest tests/unit/
    ```

* **To run only integration tests**:
    ```bash
    pytest tests/integration/
    ```

* **To run a specific test file**:
    ```bash
    pytest tests/unit/src/models/test_layers.py
    ```

* **To run tests with verbose output**:
    ```bash
    pytest -v
    ```