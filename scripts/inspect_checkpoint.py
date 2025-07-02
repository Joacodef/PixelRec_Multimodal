import torch
from pathlib import Path

def inspect_checkpoint_weights(checkpoint_path: str):
    """
    Loads a model checkpoint and inspects the weights of key layers.
    """
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        # Load the checkpoint onto the CPU for inspection
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Check if the expected key exists
        if 'model_state_dict' not in checkpoint:
            print(f"Error: Checkpoint does not contain the key 'model_state_dict'. Found keys: {list(checkpoint.keys())}")
            return
            
        state_dict = checkpoint['model_state_dict']
        print("\n--- Inspecting Key Layer Weights ---\n")

        # Define some key layers to check
        layers_to_check = {
            "User Embedding": "user_embedding.weight",
            "Item Embedding": "item_embedding.weight",
            "Vision Projection (Linear)": "vision_projection.0.weight",
            "Language Projection (Linear)": "language_projection.0.weight",
            "Final Prediction Layer": "prediction_network.0.weight"
        }

        all_weights_are_zero = True
        for name, key in layers_to_check.items():
            if key in state_dict:
                weights = state_dict[key]
                weight_sum = torch.sum(torch.abs(weights)).item()
                print(f"- Layer: '{name}' ({key})")
                print(f"  - Shape: {weights.shape}")
                print(f"  - Sum of Absolute Weights: {weight_sum:.6f}")
                if weight_sum > 1e-6: # Use a small epsilon for floating point
                    all_weights_are_zero = False
            else:
                print(f"- Layer: '{name}' ({key}) -> Key not found in state_dict.")

        print("\n--- Conclusion ---")
        if all_weights_are_zero:
            print("❌ All inspected weights are zero. The checkpoint file is corrupted or invalid.")
        else:
            print("✅ At least some weights are non-zero. The issue may lie elsewhere.")

    except Exception as e:
        print(f"An error occurred while loading the checkpoint: {e}")

if __name__ == '__main__':
    # IMPORTANT: Replace this path with the exact path to your checkpoint file
    path_to_checkpoint = "results/optuna/trial_3/checkpoints/dino_sentence-bert/best_model.pth"
    inspect_checkpoint_weights(path_to_checkpoint)