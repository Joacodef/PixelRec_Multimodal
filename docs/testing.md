# Testing Guide

A good testing suite is crucial for ensuring the reliability and correctness of the multimodal recommender system. This guide outlines the testing philosophy, structure, and procedures for running and adding tests.

The framework uses `pytest` for test discovery and execution, and `pytest-cov` for coverage analysis.

## Test Philosophy

The testing strategy is divided into two main categories:

* **Unit Tests (`tests/unit/`)**: These tests focus on individual components (classes and functions) in isolation. They are located in the `src/` directory and are designed to be fast and specific. Each module in `src` should have a corresponding test file in `tests/unit/src/`.
* **Integration Tests (`tests/integration/`)**: These tests verify the end-to-end functionality of the executable scripts found in the `scripts/` directory. They ensure that all components work together correctly, from loading configurations and data to producing the final output (e.g., model checkpoints, evaluation results, recommendations).

## Running Tests

### Prerequisites

First, install the necessary testing libraries:
```bash
pip install pytest pytest-cov
```

### Recommended Commands

The following commands are based on the continuous integration setup (see `.github/workflows/ci-tests.yaml`) and are the recommended way to run the test suite.

**1. Run All Tests:**
To execute the complete test suite (both unit and integration tests):
```bash
pytest tests/
```

**2. Run a Specific Test Suite:**
You can run the unit or integration tests separately.

* **Unit tests only:**
    ```bash
    pytest tests/unit/
    ```
* **Integration tests only:**
    ```bash
    pytest tests/integration/
    ```

**3. Run Tests with Coverage Report:**
To measure how much of the source code is covered by tests, run `pytest` with `pytest-cov`. The `--cov-report=term-missing` flag will provide a summary in the terminal and list any lines that are not covered.
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

**4. Run a Specific Test File:**
To run a single test file, provide its path:
```bash
pytest tests/unit/src/data/test_dataset.py
```

**5. Run a Specific Test or Class:**
To run a specific test class or function, use the `::` syntax.
```bash
# Run all tests in the TestDataFilter class
pytest tests/unit/src/data/processors/test_data_filter.py::TestDataFilter

# Run a single test method
pytest tests/unit/src/data/processors/test_data_filter.py::TestDataFilter::test_filter_by_activity_combined
```

## Test Structure

The `tests/` directory mirrors the project's structure to make locating tests intuitive.

```
tests/
├── __init__.py
├── integration/
│   └── scripts/
│       ├── test_checkpoint_manager.py
│       ├── test_create_splits.py
│       ├── test_evaluate.py
│       ├── test_extract_encoders.py
│       ├── test_generate_recommendations.py
│       ├── test_precompute_cache.py
│       ├── test_preprocess_data.py
│       └── test_train.py
└── unit/
    └── src/
        ├── data/
        │   ├── processors/
        │   │   ├── test_data_filter.py
        │   │   ├── test_feature_cache_processor.py
        │   │   ├── test_image_processor.py
        │   │   ├── test_numerical_processor.py
        │   │   └── test_text_processor.py
        │   ├── test_cache_utils.py
        │   ├── test_dataset.py
        │   ├── test_simple_cache.py
        │   └── test_splitting.py
        ├── evaluation/
        │   ├── test_advanced_metrics.py
        │   ├── test_metrics.py
        │   ├── test_novelty.py
        │   └── test_tasks.py
        ├── inference/
        │   ├── test_baseline_recommenders.py
        │   └── test_recommender.py
        ├── models/
        │   ├── test_layers.py
        │   ├── test_losses.py
        │   └── test_multimodal.py
        └── training/
            └── test_trainer.py
```

## Current Test Coverage

The project has extensive test coverage across its components:

* **Data Processing & Loading (`src/data`)**:
    * **Processors**: `DataFilter`, `ImageProcessor`, `NumericalProcessor`, `TextProcessor`, `FeatureCacheProcessor`.
    * **Core Data Classes**: `MultimodalDataset`, `DataSplitter`, `SimpleFeatureCache`.
    * **Utilities**: `cache_utils`.
* **Evaluation (`src/evaluation`)**:
    * Standard metrics (Precision, Recall, NDCG, MAP).
    * Advanced metrics (MRR, Hit Rate, Gini, Serendipity).
    * Novelty and Diversity metrics.
    * The evaluation task framework (`TopKRetrievalEvaluator`, `TopKRankingEvaluator`).
* **Inference (`src/inference`)**:
    * The main `Recommender` class for the multimodal model.
    * All baseline recommenders (`Random`, `Popularity`, `ItemKNN`, `UserKNN`).
* **Models (`src/models`)**:
    * The `MultimodalRecommender` architecture, including its forward pass, feature fusion, and gradient flow.
    * Custom layers like `CrossModalAttention`.
    * All loss functions (`ContrastiveLoss`, `BPRLoss`, `MultimodalRecommenderLoss`).
* **Training (`src/training`)**:
    * The `Trainer` class, including checkpointing, early stopping, and metric logging.
* **End-to-End Scripts (`scripts/`)**:
    * The entire preprocessing pipeline (`test_preprocess_data.py`).
    * Data splitting (`test_create_splits.py`).
    * Feature cache pre-computation (`test_precompute_cache.py`).
    * Model training (`test_train.py`).
    * Model evaluation (`test_evaluate.py`).
    * Recommendation generation (`test_generate_recommendations.py`).
    * Utilities for managing checkpoints (`test_checkpoint_manager.py`) and encoders (`test_extract_encoders.py`).

## Adding New Tests

When adding new functionality, please include corresponding tests to maintain coverage and ensure robustness.

1.  **Create a New Test File**: Place the file in the appropriate directory within `tests/`, mirroring the location of the module or script you are testing.
2.  **Name it Conventionally**: The test file should be named `test_<component>.py`.
3.  **Import `unittest` and Your Component**:
    ```python
    import unittest
    from src.your_module import YourClass
    ```
4.  **Create a Test Class**: The class should inherit from `unittest.TestCase`.
5.  **Write Test Methods**: Each test method must start with `test_`. Use descriptive names for your test methods.
6.  **Use Assertions**: Use `self.assertEqual()`, `self.assertTrue()`, `self.assertAlmostEqual()`, etc., to verify the behavior of your code.

### Example Template (`tests/unit/src/test_your_module.py`):
```python
import unittest
import pandas as pd
from pathlib import Path
import sys

# Add project root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.your_module import YourClass

class TestYourClass(unittest.TestCase):

    def setUp(self):
        """Code to set up test fixtures, run before each test method."""
        self.instance = YourClass()
        self.sample_data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    def test_some_functionality(self):
        """A descriptive name for what this test verifies."""
        # Call the method you want to test
        result = self.instance.process(self.sample_data)
        
        # Assert the expected outcome
        expected_result = 42
        self.assertEqual(result, expected_result)
        
    def test_another_edge_case(self):
        """Test how the component handles a specific edge case."""
        # ... your test logic ...
        self.assertTrue(self.instance.is_valid())

if __name__ == '__main__':
    unittest.main()
