Basándome en el código del repositorio, aquí está el nuevo README propuesto:

# Multimodal Recommender System

A PyTorch-based framework for building multimodal recommendation systems that integrate visual, textual, and numerical features to generate personalized recommendations.

## Overview

This system implements a neural recommender that combines multiple data modalities:
- **Visual features**: Extracted from item images using pre-trained vision models (ResNet, CLIP, DINO, ConvNeXT)
- **Textual features**: Processed from item descriptions using language models (BERT, RoBERTa, Sentence-BERT, MPNet)
- **Numerical features**: Item metadata and interaction statistics

The architecture uses attention mechanisms to fuse multimodal representations and supports contrastive learning for vision-text alignment.

## Key Features

### Architecture
- Configurable fusion of user embeddings, item embeddings, and multimodal features
- Multi-head self-attention for feature integration
- Support for various pre-trained backbone models
- Optional CLIP-style contrastive learning

### Data Processing
- Modular preprocessing pipeline with dedicated processors for each data type
- Automatic image validation and compression
- Text cleaning and augmentation capabilities
- Flexible data splitting strategies (stratified, temporal, user-based, item-based)

### Training
- Configurable optimizers (AdamW, Adam, SGD) and learning rate schedulers
- Early stopping with model checkpointing
- Gradient clipping and dropout regularization
- Optional Weights & Biases integration for experiment tracking

### Evaluation
- Standard recommendation metrics: Precision@K, Recall@K, NDCG@K, MRR, Hit Rate
- Multiple evaluation tasks: Top-K retrieval and ranking
- Baseline comparisons: Random, Popularity, ItemKNN, UserKNN
- Efficient negative sampling for large-scale evaluation

### Performance Optimization
- Feature caching system with LRU memory management
- Support for pre-computing all item features before training
- Batch processing for efficient inference
- Model-specific cache directories for different configurations

## Installation

1. Clone the repository
2. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Prepare your data in the following format:
- `item_info.csv`: Item metadata with columns for item_id, title, description, and numerical features
- `interactions.csv`: User-item interactions with user_id and item_id columns
- `images/`: Directory containing item images named as `{item_id}.jpg`

### 2. Configuration

Edit `configs/simple_config.yaml` to specify:
- Model architecture choices
- Data paths and preprocessing options
- Training hyperparameters
- Evaluation settings

### 3. Preprocessing

```bash
python scripts/preprocess_data.py --config configs/simple_config.yaml
```

### 4. Data Splitting

```bash
python scripts/create_splits.py --config configs/simple_config.yaml
```

### 5. Training

```bash
python scripts/train.py --config configs/simple_config.yaml --device cuda
```

### 6. Evaluation

```bash
python scripts/evaluate.py --config configs/simple_config.yaml \
    --test_data data/splits/test.csv \
    --eval_task retrieval \
    --recommender_type multimodal
```

### 7. Generate Recommendations

```bash
python scripts/generate_recommendations.py --config configs/simple_config.yaml \
    --users user_123 user_456 \
    --output recommendations.json
```

## Project Structure

```
multimodal-recommender/
├── configs/              # Configuration files
├── scripts/              # Executable scripts for training, evaluation, etc.
├── src/                  # Source code
│   ├── config.py        # Configuration management
│   ├── data/            # Dataset and preprocessing modules
│   ├── models/          # Model architectures
│   ├── training/        # Training logic
│   ├── evaluation/      # Evaluation framework
│   └── inference/       # Recommendation generation
├── data/                # Data directories (created during setup)
├── models/              # Model checkpoints
└── results/             # Evaluation results and figures
```

## Configuration Options

The system supports extensive configuration through YAML files:

- **Model parameters**: Architecture choices, embedding dimensions, fusion strategies
- **Training settings**: Batch size, learning rate, optimizer configuration
- **Data processing**: Feature normalization, text augmentation, image compression
- **Evaluation options**: Metrics, negative sampling strategies, baseline comparisons

See `configs/advanced_config.yaml` for all available options.

## Performance Considerations

- Use `scripts/precompute_cache.py` to pre-compute features before training
- Enable feature caching to speed up data loading
- Adjust batch size based on GPU memory
- Use negative sampling for efficient evaluation on large datasets

## Requirements

- Python 3.7+
- PyTorch 2.2.1+
- Transformers 4.47.1+
- CUDA-capable GPU (recommended)

See `requirements.txt` for complete dependencies.

## License

This project is provided as-is for research and educational purposes.
