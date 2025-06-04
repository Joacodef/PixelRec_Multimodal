# Multimodal Recommender System

This repository provides a comprehensive framework for building and experimenting with multimodal recommender systems. It leverages both visual and textual information from datasets (e.g., those similar in structure to PixelRec/Pixel200k) to provide personalized recommendations. The system is highly configurable, allowing for easy modification of model architectures, data processing pipelines, and evaluation strategies.

---

## Core Features

* **Multimodal Data Integration**: Processes items with associated images, textual descriptions, and numerical features for comprehensive recommendation modeling.

* **Flexible Model Architectures**:
    * Two main model classes: `PretrainedMultimodalRecommender` (default) and `EnhancedMultimodalRecommender` (with cross-modal attention layers)
    * Supports various pre-trained vision models (CLIP, DINO, ResNet, ConvNeXT) and language models (Sentence-BERT, MPNet, BERT, RoBERTa)
    * Configurable fusion mechanisms with multi-head self-attention for combining diverse feature sets
    * Optional contrastive learning for better vision-text alignment (primarily with CLIP-based vision models)
    * Fully configurable network architectures including fusion layers, activation functions, batch normalization, and initialization methods

* **Advanced Data Preprocessing & Handling**:
    * **Data Preprocessing Pipeline**: `scripts/preprocess_data.py` handles text cleaning (HTML removal, Unicode normalization, lowercasing), image validation (corruption checks, dimension verification), and optional image compression/resizing
    * **Flexible Data Splitting**: `scripts/create_splits.py` generates train/validation/test splits using various strategies (stratified, temporal, user-based, item-based) with activity filtering and optional dataset sampling
    * **Efficient Image Caching**: `SharedImageCache` system with configurable strategies (hybrid, disk-only, memory-only, disabled) for accelerated training and inference
    * **Comprehensive Data Loading**: `MultimodalDataset` class handles negative sampling, numerical feature normalization, text augmentation, and robust data validation

* **Robust Training Framework**:
    * Configurable optimizers (AdamW, Adam, SGD) with full parameter control
    * Advanced learning rate scheduling (ReduceLROnPlateau, Cosine, Step)
    * Gradient clipping, early stopping, and checkpoint management
    * Optional Weights & Biases integration for experiment tracking
    * Support for training resumption and model state persistence

* **Comprehensive Evaluation Framework**:
    * **Task-Based Evaluation**: Multiple evaluation scenarios including retrieval, ranking, next-item prediction, cold-start (user/item), and beyond-accuracy metrics
    * **Standard Metrics**: Precision@k, Recall@k, NDCG@k, MAP, MRR, Hit Rate
    * **Advanced Metrics**: Novelty, diversity, catalog coverage, Gini coefficient, serendipity
    * **Baseline Comparisons**: Random, Popularity, ItemKNN, and UserKNN recommenders for benchmarking

* **Inference and Recommendation Generation**:
    * Efficient batch processing for large-scale recommendation generation
    * Diversity-aware recommendation with MMR-like re-ranking
    * Novelty scoring and filtering capabilities
    * Pre-computed embeddings caching for fast inference
    * Support for candidate filtering and seen-item exclusion

* **Modular Architecture**:
    * Clear separation of concerns across configuration, data handling, models, training, evaluation, and inference
    * Extensive YAML-based configuration system with nested dataclasses
    * Consistent interfaces for easy component swapping and experimentation

---

## Directory Structure

```
PixelRec_Multimodal/
├── configs/                    # Configuration files
│   └── default_config.yaml     # Main configuration template
├── data/                       # Data directories (paths configurable)
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed data and scalers
│   ├── splits/                 # Train/val/test splits
│   └── cache/                  # Image tensor cache
├── models/                     # Model checkpoints and encoders
│   └── checkpoints/            # Training checkpoints
├── results/                    # Evaluation results and figures
├── scripts/                    # Execution scripts
│   ├── preprocess_data.py      # Data preprocessing
│   ├── create_splits.py        # Data splitting
│   ├── extract_encoders.py     # Encoder extraction utility
│   ├── train.py               # Model training
│   ├── evaluate.py            # Model evaluation
│   └── generate_recommendations.py  # Recommendation generation
├── src/                        # Source code
│   ├── config.py              # Configuration dataclasses
│   ├── data/                  # Data processing modules
│   │   ├── dataset.py         # MultimodalDataset class
│   │   ├── image_cache.py     # Shared image caching system
│   │   ├── splitting.py       # Data splitting strategies
│   │   └── preprocessing.py   # Preprocessing utilities
│   ├── evaluation/            # Evaluation framework
│   │   ├── tasks.py          # Task-based evaluators
│   │   ├── metrics.py        # Standard metrics
│   │   ├── novelty.py        # Novelty and diversity metrics
│   │   └── advanced_metrics.py  # Additional metrics
│   ├── inference/             # Inference and recommendation
│   │   ├── recommender.py     # Main recommender class
│   │   └── baseline_recommenders.py  # Baseline implementations
│   ├── models/                # Model architectures
│   │   ├── multimodal.py     # Main model classes
│   │   ├── layers.py         # Custom layers
│   │   └── losses.py         # Loss functions
│   └── training/              # Training framework
│       └── trainer.py         # Trainer class
├── requirements.txt           # Python dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

---

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create and activate a Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

The system is operated through command-line scripts that can be configured via YAML files. All scripts support the `--config` parameter to specify configuration files.

### 1. Configuration

Start by configuring `configs/default_config.yaml` or creating a custom configuration file. Key sections include:

- **`model`**: Architecture selection, embedding dimensions, activation functions, attention mechanisms
- **`training`**: Batch size, learning rate, optimizer settings, scheduler configuration
- **`data`**: File paths, preprocessing options, caching settings, splitting parameters
- **`recommendation`**: Top-k values, diversity weights, filtering options

### 2. Data Preprocessing

Process raw data including text cleaning, image validation, and optional compression:

```bash
python scripts/preprocess_data.py --config configs/default_config.yaml
```

Features:
- Validates and optionally compresses/resizes images
- Cleans text data (HTML removal, Unicode normalization)
- Filters items based on availability and quality
- Saves processed data for consistent training

### 3. Data Splitting

Create standardized train/validation/test splits:

```bash
python scripts/create_splits.py --config configs/default_config.yaml
```

Advanced options:
```bash
# Sample dataset before splitting
python scripts/create_splits.py --config configs/default_config.yaml --sample_n 100000
```

Supports multiple splitting strategies:
- **Stratified** (default): Ensures user representation across splits
- **Temporal**: Time-based splits for realistic evaluation
- **User/Item-based**: Complete separation for cold-start evaluation

### 4. Model Training

Train the multimodal recommender:

```bash
python scripts/train.py --config configs/default_config.yaml --device cuda
```

Training features:
```bash
# With experiment tracking
python scripts/train.py --config configs/default_config.yaml \
    --use_wandb --wandb_project MyProject --wandb_entity username

# Resume from checkpoint
python scripts/train.py --config configs/default_config.yaml \
    --resume models/checkpoints/checkpoint.pth
```

The training script automatically:
- Fits user/item encoders on the full dataset
- Creates efficient data loaders with configurable caching
- Saves model checkpoints and training curves
- Supports distributed training and mixed precision

### 5. Model Evaluation

Evaluate trained models using task-specific metrics:

```bash
python scripts/evaluate.py --config configs/default_config.yaml \
    --test_data data/splits/split_1/test.csv \
    --train_data data/splits/split_1/train.csv \
    --eval_task retrieval \
    --recommender_type multimodal \
    --output results/evaluation_results.json
```

Evaluation tasks:
- **`retrieval`**: Novel item discovery (most common)
- **`ranking`**: Full catalog ranking performance
- **`next_item`**: Sequential prediction accuracy
- **`cold_user`/`cold_item`**: Cold-start performance
- **`beyond_accuracy`**: Diversity, novelty, and fairness metrics

Baseline comparison:
```bash
python scripts/evaluate.py --config configs/default_config.yaml \
    --test_data data/splits/split_1/test.csv \
    --train_data data/splits/split_1/train.csv \
    --eval_task retrieval \
    --recommender_type popularity \
    --output results/baseline_results.json
```

### 6. Recommendation Generation

Generate recommendations for users:

```bash
python scripts/generate_recommendations.py --config configs/default_config.yaml \
    --users user_1 user_2 user_3 \
    --output results/recommendations.json
```

Advanced recommendation options:
```bash
# Diverse recommendations with novelty metrics
python scripts/generate_recommendations.py --config configs/default_config.yaml \
    --sample_users 100 \
    --use_diversity \
    --embeddings_cache results/embeddings_cache.pkl
```

Generation modes:
- **Specific users**: `--users user_1 user_2`
- **User file**: `--user_file path/to/users.txt`
- **Random sample**: `--sample_users N`
- **Diversity-aware**: `--use_diversity` (includes novelty metrics)

---

## Configuration Reference

### Model Configuration

```yaml
model:
  model_class: pretrained          # 'pretrained' or 'enhanced'
  vision_model: clip               # clip, dino, resnet, convnext
  language_model: sentence-bert    # sentence-bert, mpnet, bert, roberta
  embedding_dim: 128
  
  # Architecture details
  fusion_hidden_dims: [512, 256, 128]
  fusion_activation: relu          # relu, gelu, tanh, leaky_relu, silu
  num_attention_heads: 4
  use_batch_norm: true
  
  # Cross-modal attention (enhanced model only)
  use_cross_modal_attention: true
  cross_modal_attention_weight: 0.5
  
  # Contrastive learning
  use_contrastive: true
  contrastive_temperature: 0.07
```

### Data Configuration

```yaml
data:
  # Image caching for performance
  cache_processed_images: true
  image_cache_config:
    strategy: hybrid               # hybrid, memory, disk, disabled
    max_memory_items: 1500
    cache_directory: data/cache/image_tensors
    precompute_at_startup: false
  
  # Image preprocessing
  offline_image_compression:
    enabled: true
    compress_if_kb_larger_than: 500
    target_quality: 85
    resize_target_longest_edge: 1024
  
  # Data splitting
  splitting:
    train_final_ratio: 0.6
    val_final_ratio: 0.2
    test_final_ratio: 0.2
    min_interactions_per_user: 5
    min_interactions_per_item: 5
```

### Training Configuration

```yaml
training:
  batch_size: 64
  epochs: 30
  learning_rate: 0.001
  
  # Optimizer settings
  optimizer_type: adamw            # adamw, adam, sgd
  weight_decay: 0.01
  
  # Learning rate scheduling
  use_lr_scheduler: true
  lr_scheduler_type: reduce_on_plateau
  lr_scheduler_patience: 2
  lr_scheduler_factor: 0.5
```

---

## Key Features in Detail

### Image Caching System

The `SharedImageCache` provides efficient image processing with multiple strategies:

- **Hybrid**: Combines memory and disk caching for optimal performance
- **Disk**: Saves all processed images to disk, loads on demand
- **Memory**: Keeps images in RAM with LRU eviction
- **Disabled**: No caching (processes images each time)

### Task-Based Evaluation

The evaluation framework supports multiple realistic scenarios:

- **Retrieval**: Finding new relevant items (filters seen items)
- **Ranking**: Ranking all items including previously seen ones
- **Cold-start**: Performance on new users or items
- **Beyond-accuracy**: Diversity, novelty, and fairness metrics

### Baseline Recommenders

Included baseline implementations for comparison:

- **Random**: Random item selection
- **Popularity**: Most popular items globally
- **ItemKNN**: Item-based collaborative filtering
- **UserKNN**: User-based collaborative filtering

### Advanced Architecture Options

- **Cross-modal Attention**: Enhanced model with vision-text interaction
- **Contrastive Learning**: CLIP-style alignment of vision and text
- **Flexible Fusion**: Configurable multi-layer fusion networks
- **Multiple Initializations**: Xavier, Kaiming initialization schemes

---

## Performance Optimization

### Training Performance
- **Image Caching**: Pre-processes and caches images for faster loading
- **Efficient Data Loading**: Multi-worker data loaders with prefetching
- **Gradient Clipping**: Prevents training instability
- **Mixed Precision**: Automatic mixed precision support (configure via training parameters)

### Inference Performance
- **Batch Processing**: Efficient batch scoring for large-scale recommendations
- **Feature Caching**: Pre-computed item features for faster recommendation generation
- **Optimized Baselines**: Efficient implementations of collaborative filtering methods

### Memory Management
- **LRU Caching**: Automatic memory management for image cache
- **Lazy Loading**: On-demand loading of cached images
- **Configurable Limits**: Control memory usage through configuration

---

## Extensibility

The modular design supports easy extension:

### Adding New Models
1. Create new model class in `src/models/`
2. Add configuration options to `src/config.py`
3. Update model selection logic in training script

### Adding New Evaluation Tasks
1. Implement new evaluator in `src/evaluation/tasks.py`
2. Add task to `EvaluationTask` enum
3. Update evaluation script task mapping

### Adding New Baselines
1. Implement baseline in `src/inference/baseline_recommenders.py`
2. Inherit from `BaselineRecommender` class
3. Add to recommender factory function

### Custom Loss Functions
1. Add new loss to `src/models/losses.py`
2. Update `MultimodalRecommenderLoss` combination logic
3. Configure loss weights in training configuration

---

## Examples

### Quick Start
```bash
# 1. Preprocess data
python scripts/preprocess_data.py --config configs/default_config.yaml

# 2. Create splits
python scripts/create_splits.py --config configs/default_config.yaml

# 3. Train model
python scripts/train.py --config configs/default_config.yaml

# 4. Evaluate
python scripts/evaluate.py --config configs/default_config.yaml \
    --test_data data/splits/split_1/test.csv \
    --train_data data/splits/split_1/train.csv \
    --eval_task retrieval \
    --recommender_type multimodal
```

### Experiment with Different Architectures
```bash
# Enhanced model with cross-modal attention
python scripts/train.py --config configs/enhanced_config.yaml

# Different vision backbone
python scripts/train.py --config configs/dino_config.yaml

# Compare with baselines
python scripts/evaluate.py --config configs/default_config.yaml \
    --eval_task retrieval --recommender_type item_knn
```

### Large-Scale Deployment
```bash
# Sample large dataset
python scripts/create_splits.py --config configs/default_config.yaml --sample_n 1000000

# Train with image caching
python scripts/train.py --config configs/large_scale_config.yaml \
    --use_wandb --wandb_project LargeScale

# Generate recommendations with caching
python scripts/generate_recommendations.py --config configs/default_config.yaml \
    --sample_users 10000 \
    --embeddings_cache results/embeddings_cache.pkl
```

This framework provides a complete solution for multimodal recommendation research and deployment, with extensive configurability and built-in best practices for scalable machine learning systems.
