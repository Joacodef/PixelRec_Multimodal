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
- Dynamic numerical feature validation

### Training
- Configurable optimizers (AdamW, Adam, SGD) and learning rate schedulers
- Early stopping with model checkpointing
- Gradient clipping and dropout regularization
- Optional Weights & Biases integration for experiment tracking
- Model-specific checkpoint organization

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

### 3. Data Preprocessing

Process raw data into the format required by the system:

```bash
python scripts/preprocess_data.py --config configs/simple_config.yaml
```

Optional preprocessing arguments:
```bash
python scripts/preprocess_data.py --config configs/simple_config.yaml \
    --skip-caching \
    --force-reprocess
```

### 4. Data Splitting

Create standardized train/validation/test splits:

```bash
python scripts/create_splits.py --config configs/simple_config.yaml
```

Optional splitting arguments:
```bash
python scripts/create_splits.py --config configs/simple_config.yaml \
    --sample_n 10000
```

### 5. Feature Caching (Optional but Recommended)

Pre-compute multimodal features to accelerate training:

```bash
# Full feature caching
python scripts/precompute_cache.py --config configs/simple_config.yaml

# Test mode (process only 100 items)
python scripts/precompute_cache.py --config configs/simple_config.yaml --test

# Force recomputation of existing cache
python scripts/precompute_cache.py --config configs/simple_config.yaml --force

# Limit number of items processed
python scripts/precompute_cache.py --config configs/simple_config.yaml --max_items 5000
```

### 6. Cache Management

Manage feature caches for different model combinations:

```bash
# List available caches
python scripts/cache.py list

# Show statistics for specific model combination
python scripts/cache.py stats resnet_sentence-bert

# Clear cache for specific model combination
python scripts/cache.py clear resnet_sentence-bert

# Clear all caches (with confirmation)
python scripts/cache.py clear --all
```

### 7. Training

Train the multimodal recommender system:

```bash
python scripts/train.py --config configs/simple_config.yaml --device cuda
```

Advanced training options:
```bash
python scripts/train.py --config configs/simple_config.yaml \
    --device cuda \
    --use_wandb \
    --wandb_project "MyProject" \
    --wandb_run_name "experiment_1" \
    --resume path/to/checkpoint.pth \
    --verbose
```

### 8. Checkpoint Management

Manage model-specific checkpoint organization:

```bash
# List all checkpoints and their organization status
python scripts/checkpoint_manager.py list --checkpoint-dir models/checkpoints

# Automatically organize existing checkpoints by model combination
python scripts/checkpoint_manager.py organize --checkpoint-dir models/checkpoints

# Preview organization changes without moving files
python scripts/checkpoint_manager.py organize --checkpoint-dir models/checkpoints --dry-run

# Manually organize checkpoints with unknown model combinations
python scripts/checkpoint_manager.py organize-manual --checkpoint-dir models/checkpoints

# Create JSON summary of checkpoint organization
python scripts/checkpoint_manager.py info --checkpoint-dir models/checkpoints
```

### 9. Evaluation

Evaluate trained models on test data:

```bash
python scripts/evaluate.py --config configs/simple_config.yaml \
    --test_data data/splits/test.csv \
    --eval_task retrieval \
    --recommender_type multimodal
```

Advanced evaluation options:
```bash
python scripts/evaluate.py --config configs/simple_config.yaml \
    --test_data data/splits/test.csv \
    --train_data data/splits/train.csv \
    --eval_task retrieval \
    --recommender_type multimodal \
    --checkpoint_name best_model.pth \
    --use_sampling \
    --num_negatives 100 \
    --sampling_strategy random \
    --save_predictions user_predictions.json \
    --warmup_recommender_cache \
    --use_parallel \
    --num_workers 4
```

Evaluation task options:
- `retrieval`: Top-K retrieval evaluation
- `ranking`: Ranking quality evaluation

Recommender types:
- `multimodal`: Trained multimodal model
- `random`: Random baseline
- `popularity`: Popularity-based baseline
- `item_knn`: Item-based collaborative filtering
- `user_knn`: User-based collaborative filtering

### 10. Generate Recommendations

Generate recommendations for specific users:

```bash
python scripts/generate_recommendations.py --config configs/simple_config.yaml \
    --users user_123 user_456 \
    --output recommendations.json
```

Advanced recommendation generation:
```bash
python scripts/generate_recommendations.py --config configs/simple_config.yaml \
    --sample_users 10 \
    --use_diversity \
    --output diverse_recommendations.json \
    --device cuda
```

Alternative user specification methods:
```bash
# From file (one user ID per line)
python scripts/generate_recommendations.py --config configs/simple_config.yaml \
    --user_file user_list.txt

# Random sampling from available users
python scripts/generate_recommendations.py --config configs/simple_config.yaml \
    --sample_users 50
```

### 11. Extract Encoders (If Needed)

Extract user and item encoders from training data:

```bash
python scripts/extract_encoders.py --config configs/simple_config.yaml
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
│   └── checkpoints/     # Organized by model combination
│       ├── encoders/    # Shared user/item encoders
│       ├── resnet_sentence-bert/    # Model-specific checkpoints
│       └── clip_mpnet/              # Model-specific checkpoints
├── cache/               # Feature caches (organized by model)
│   ├── resnet_sentence-bert/
│   └── clip_mpnet/
└── results/             # Evaluation results and figures
```

## Checkpoint Organization

The system organizes checkpoints by model combination to support experimentation with different architectures:

- **Model-specific directories**: `checkpoints/{vision_model}_{language_model}/` contain .pth files
- **Shared encoders**: `checkpoints/encoders/` contains user_encoder.pkl and item_encoder.pkl
- **Backward compatibility**: System automatically finds checkpoints in old organization

## Configuration Options

The system supports extensive configuration through YAML files:

- **Model parameters**: Architecture choices, embedding dimensions, fusion strategies
- **Training settings**: Batch size, learning rate, optimizer configuration
- **Data processing**: Feature normalization, text augmentation, image compression
- **Evaluation options**: Metrics, negative sampling strategies, baseline comparisons

### Example Configurations

Basic configuration (`configs/simple_config.yaml`):
```yaml
model:
  vision_model: resnet
  language_model: sentence-bert
  embedding_dim: 64
  use_contrastive: true

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 30

data:
  numerical_features_cols:
    - view_number
    - comment_number
    - thumbup_number
```

Advanced configuration (`configs/advanced_config.yaml`) includes additional options for:
- Advanced model architecture parameters
- Detailed training configurations
- Comprehensive data processing settings
- Extensive evaluation options

## Performance Considerations

- Use `scripts/precompute_cache.py` to pre-compute features before training
- Enable feature caching to speed up data loading
- Adjust batch size based on GPU memory
- Use negative sampling for efficient evaluation on large datasets
- Configure numerical features to match your dataset

## Requirements

- Python 3.7+
- PyTorch 2.2.1+
- Transformers 4.47.1+
- CUDA-capable GPU (recommended)

See `requirements.txt` for complete dependencies.

## License

This project is provided as-is for research and educational purposes.