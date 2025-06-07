# Testing Guide

This directory contains tests for the Multimodal Recommender System.

## Directory Structure

```
tests/
├── __init__.py
├── unit/                 # Unit tests for individual components
│   ├── __init__.py
│   └── test_data_filter.py
├── integration/          # Integration tests (placeholder)
│   └── __init__.py
└── conftest.py          # Shared test configuration
```

## Running Tests

### Run all tests:
```bash
python scripts/run_tests.py
```

### Run only unit tests:
```bash
python scripts/run_tests.py --unit
```

### Run a specific test file:
```bash
python scripts/run_tests.py --test tests.unit.test_data_filter
```

### Using pytest (if installed):
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_data_filter.py -v
```

### Using unittest directly:
```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python -m unittest tests.unit.test_data_filter.TestDataFilter
```

## Test Coverage

Currently implemented tests:
- **DataFilter**: Tests for data filtering and preprocessing operations
  - Filter interactions by valid items
  - Filter by user/item activity thresholds
  - Align item info with interactions
  - Calculate filtering statistics

## Adding New Tests

1. Create a new test file in the appropriate directory (`unit/` or `integration/`)
2. Name it `test_<component>.py`
3. Import the component to test
4. Create a test class inheriting from `unittest.TestCase`
5. Write test methods starting with `test_`

Example:
```python
import unittest
from src.your_module import YourClass

class TestYourClass(unittest.TestCase):
    def setUp(self):
        # Setup code run before each test
        self.instance = YourClass()
    
    def test_specific_functionality(self):
        # Test code
        result = self.instance.method()
        self.assertEqual(result, expected_value)
```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Clear naming**: Test names should describe what they test
3. **Single purpose**: Each test should verify one thing
4. **Fast execution**: Unit tests should run quickly
5. **Reproducible**: Tests should produce consistent results