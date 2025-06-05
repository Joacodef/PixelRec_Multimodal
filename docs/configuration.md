# Configuration Usage Guide

## Overview
We now have two configuration approaches:

1. **`configs/simple_config.yaml`** - Essential settings only (~15 parameters)
2. **`configs/advanced_config.yaml`** - All configurable options (~60+ parameters)

## Quick Start (Recommended)

### Use Simple Config for Most Cases:
```bash
python scripts/train.py --config configs/simple_config.yaml
python scripts/evaluate.py --config configs/simple_config.yaml --test_data data/splits/split_tiny/test.csv --train_data data/splits/split_tiny/train.csv --eval_task retrieval --recommender_type multimodal --output results/simple_results.json
```

### Customize Simple Config:
Just edit the essential parameters in `configs/simple_config.yaml`:

```yaml
model:
  vision_model: clip        # Change to: clip, resnet, dino, convnext
  language_model: mpnet     # Change to: sentence-bert, mpnet, bert, roberta
  embedding_dim: 128        # Change to: 64, 128, 256
  use_contrastive: false    # Disable contrastive learning

training:
  batch_size: 32           # Reduce for smaller GPU
  learning_rate: 0.0005    # Adjust learning rate
  epochs: 50               # Train longer
```

## Advanced Experimentation

### Use Advanced Config for Research:
```bash
python scripts/train.py --config configs/advanced_config.yaml
```

### Advanced Config Allows:
- **Architectural experiments**: fusion layers, attention heads, activation functions
- **Training experiments**: optimizers, schedulers, regularization
- **Data experiments**: augmentation, compression, preprocessing
- **Model experiments**: cross-modal attention, different initializations

## Configuration Inheritance

The system automatically provides sensible defaults for missing parameters:

### Example: Minimal Config
```yaml
# configs/minimal.yaml - Just the essentials
model:
  vision_model: resnet
  embedding_dim: 64

training:
  batch_size: 32
  epochs: 20

data:
  train_data_path: data/splits/my_split/train.csv
  val_data_path: data/splits/my_split/val.csv
  test_data_path: data/splits/my_split/test.csv
```

**This will automatically get:**
- All other model parameters (dropout_rate: 0.3, fusion_activation: relu, etc.)
- All other training parameters (learning_rate: 0.001, optimizer: adamw, etc.)
- All data processing defaults
- Cache configuration defaults

## Migration Guide

### From Old Complex Config:
If you have an existing config with 60+ parameters, you can:

1. **Extract essentials** → Move to `simple_config.yaml`
2. **Keep advanced settings** → Move to `advanced_config.yaml`
3. **Use either one** - both work with the same system

### Backward Compatibility:
Old configs still work! The system automatically:
- Converts old cache parameters to new format
- Provides defaults for missing parameters
- Handles both nested and flat parameter structures

## Recommendations

### For Development/Testing:
- ✅ Use `simple_config.yaml`
- ✅ Focus on: model type, embedding size, batch size, data paths
- ✅ Let everything else use defaults

### For Research/Publication:
- ✅ Use `advanced_config.yaml`
- ✅ Document which parameters you changed from defaults
- ✅ Save final config with results for reproducibility

### For Production:
- ✅ Start with `simple_config.yaml`
- ✅ Gradually tune specific parameters as needed
- ✅ Keep config files in version control

## Configuration Validation

The system validates:
- ✅ Required paths exist
- ✅ Model names are valid
- ✅ Parameter ranges are sensible
- ✅ Backward compatibility

## Example Workflows

### Quick Experiment:
```bash
# 1. Copy simple config
cp configs/simple_config.yaml configs/my_experiment.yaml

# 2. Edit just what you need
vim configs/my_experiment.yaml  # Change vision_model: clip

# 3. Run experiment
python scripts/train.py --config configs/my_experiment.yaml
```

### Detailed Research:
```bash
# 1. Use advanced config as base
cp configs/advanced_config.yaml configs/research_v1.yaml

# 2. Modify architecture parameters
# Edit: fusion_hidden_dims, attention_heads, etc.

# 3. Run with full control
python scripts/train.py --config configs/research_v1.yaml
```