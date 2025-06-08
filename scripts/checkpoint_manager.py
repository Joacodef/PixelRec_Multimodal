#!/usr/bin/env python
"""
A command-line utility for managing model checkpoints.

This script provides functionalities to list, organize, and summarize model
checkpoints. It helps maintain a structured checkpoint directory by sorting
models into subdirectories based on their specific configurations, which are
extracted from the checkpoint files themselves.
"""
import argparse
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import pandas as pd


def scan_checkpoints(checkpoint_dir: str) -> Dict[str, List[Path]]:
    """
    Scans a checkpoint directory and categorizes its contents.

    This function walks through the specified directory and sorts files into
    four categories: model files (.pth) in the base directory, encoder files (.pkl),
    files already organized into subdirectories, and any other unrecognized files.

    Args:
        checkpoint_dir: The path to the checkpoint directory to scan.

    Returns:
        A dictionary where keys are category names ('models', 'encoders',
        'organized', 'unknown') and values are lists of Path objects.
    """
    base_dir = Path(checkpoint_dir)
    # Returns a default empty structure if the directory does not exist.
    if not base_dir.exists():
        return {'models': [], 'encoders': [], 'organized': [], 'unknown': []}
    
    results = {
        'models': [],      # Model checkpoints (.pth) in the base directory.
        'encoders': [],    # Encoder files (.pkl) in the base directory.
        'organized': [],   # Files already located in subdirectories.
        'unknown': []      # Any other file types not categorized above.
    }
    
    # Recursively iterate through all items in the directory.
    for item in base_dir.rglob('*'):
        if item.is_file():
            relative_path = item.relative_to(base_dir)
            
            # A file is considered organized if it is not in the top-level directory.
            if len(relative_path.parts) > 1:
                results['organized'].append(item)
            # Categorizes files located in the top-level directory.
            elif item.suffix == '.pth':
                results['models'].append(item)
            elif 'encoder' in item.name and item.suffix == '.pkl':
                results['encoders'].append(item)
            else:
                results['unknown'].append(item)
    
    return results


def extract_model_info_from_checkpoint(checkpoint_path: Path) -> Optional[Tuple[str, str]]:
    """
    Loads a checkpoint file and extracts the model configuration.

    This function attempts to load a PyTorch checkpoint and find a dictionary
    containing the model's configuration, specifically the 'vision_model' and
    'language_model' keys, to identify the model architecture.

    Args:
        checkpoint_path: The Path object pointing to the checkpoint file.

    Returns:
        A tuple containing the (vision_model, language_model) names if found;
        otherwise, returns None.
    """
    try:
        # Loads the checkpoint onto the CPU to avoid GPU memory usage.
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Searches for model configuration under common key names.
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            if 'vision_model' in config and 'language_model' in config:
                return config['vision_model'], config['language_model']
        
        # Fallback search in other potential dictionary keys.
        for key in ['config', 'args', 'model_args']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                config = checkpoint[key]
                if 'vision_model' in config and 'language_model' in config:
                    return config['vision_model'], config['language_model']
        
        return None
        
    except Exception as e:
        # Handles cases where the file cannot be read or is not a valid checkpoint.
        print(f"Warning: Could not read checkpoint {checkpoint_path}: {e}")
        return None


def list_checkpoints(checkpoint_dir: str):
    """
    Prints a formatted summary of all checkpoints and their organization status.

    Args:
        checkpoint_dir: The path to the checkpoint directory to be listed.
    """
    print(f"📁 CHECKPOINT DIRECTORY SCAN: {checkpoint_dir}")
    print("=" * 60)
    
    base_dir = Path(checkpoint_dir)
    if not base_dir.exists():
        print(f"❌ Directory does not exist: {checkpoint_dir}")
        return
    
    scan_results = scan_checkpoints(checkpoint_dir)
    
    # Groups organized checkpoints by their parent model directory.
    organized_by_model = {}
    for item in scan_results['organized']:
        relative_path = item.relative_to(base_dir)
        model_combo = relative_path.parts[0]
        if model_combo not in organized_by_model:
            organized_by_model[model_combo] = []
        organized_by_model[model_combo].append(item)
    
    # Prints the list of already organized checkpoints.
    if organized_by_model:
        print("✅ ORGANIZED CHECKPOINTS (Model-Specific Directories):")
        for model_combo, files in organized_by_model.items():
            print(f"\n📂 {model_combo}/")
            for file in files:
                size_mb = file.stat().st_size / (1024*1024)
                print(f"   └── {file.name} ({size_mb:.1f} MB)")
    
    # Prints the list of unorganized model files found in the base directory.
    if scan_results['models']:
        print(f"\n⚠️  UNORGANIZED MODEL CHECKPOINTS (Base Directory):")
        for model_file in scan_results['models']:
            size_mb = model_file.stat().st_size / (1024*1024)
            
            # Attempts to extract model info to provide a helpful suggestion.
            model_info = extract_model_info_from_checkpoint(model_file)
            if model_info:
                vision_model, language_model = model_info
                print(f"   📄 {model_file.name} ({size_mb:.1f} MB) → {vision_model}_{language_model}")
            else:
                print(f"   📄 {model_file.name} ({size_mb:.1f} MB) → Unknown model combination")
    
    # Prints shared files like encoders.
    if scan_results['encoders']:
        print(f"\n🔄 SHARED ENCODERS:")
        for encoder_file in scan_results['encoders']:
            size_kb = encoder_file.stat().st_size / 1024
            print(f"   📄 {encoder_file.name} ({size_kb:.1f} KB)")
    
    # Prints any other unrecognized files.
    if scan_results['unknown']:
        print(f"\n❓ OTHER FILES:")
        for other_file in scan_results['unknown']:
            size_kb = other_file.stat().st_size / 1024
            print(f"   📄 {other_file.name} ({size_kb:.1f} KB)")
    
    # Prints a final summary of the scan results.
    total_files = sum(len(files) for files in scan_results.values())
    organized_count = len(scan_results['organized'])
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total files: {total_files}")
    print(f"   Organized: {organized_count}")
    print(f"   Unorganized models: {len(scan_results['models'])}")
    print(f"   Shared encoders: {len(scan_results['encoders'])}")
    print(f"   Other files: {len(scan_results['unknown'])}")


def migrate_checkpoint(
    checkpoint_path: Path, 
    base_dir: Path, 
    vision_model: str, 
    language_model: str,
    dry_run: bool = False
) -> bool:
    """
    Moves a single checkpoint file to its corresponding model-specific directory.

    Args:
        checkpoint_path: The Path object for the checkpoint to be moved.
        base_dir: The base checkpoint directory.
        vision_model: The name of the vision model.
        language_model: The name of the language model.
        dry_run: If True, prints the action without moving the file.

    Returns:
        True if the file was moved successfully or if in dry-run mode,
        otherwise False.
    """
    model_combo = f"{vision_model}_{language_model}"
    target_dir = base_dir / model_combo
    target_path = target_dir / checkpoint_path.name
    
    # In dry-run mode, simulate the move and report success.
    if dry_run:
        print(f"   [DRY RUN] Would move: {checkpoint_path.name} → {model_combo}/{checkpoint_path.name}")
        return True
    
    # Executes the file move operation.
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(checkpoint_path), str(target_path))
        print(f"   ✅ Moved: {checkpoint_path.name} → {model_combo}/{checkpoint_path.name}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to move {checkpoint_path.name}: {e}")
        return False


def organize_checkpoints(checkpoint_dir: str, dry_run: bool = False):
    """
    Automatically organizes all unorganized checkpoints into subdirectories.

    This function scans the checkpoint directory, identifies unorganized models
    and encoders, and moves them into a structured format:
    - Models -> <checkpoint_dir>/<vision_model>_<language_model>/
    - Encoders -> <checkpoint_dir>/encoders/

    Args:
        checkpoint_dir: The path to the checkpoint directory.
        dry_run: If True, shows what would be moved without performing any actions.
    """
    print(f"🔄 ORGANIZING CHECKPOINTS: {checkpoint_dir}")
    print("=" * 60)
    
    if dry_run:
        print("🧪 DRY RUN MODE - No files will actually be moved")
        print("-" * 60)
    
    base_dir = Path(checkpoint_dir)
    if not base_dir.exists():
        print(f"❌ Directory does not exist: {checkpoint_dir}")
        return
    
    scan_results = scan_checkpoints(checkpoint_dir)
    
    # If no unorganized models are found, the process is complete.
    if not scan_results['models']:
        print("✅ No unorganized model checkpoints found")
        return
    
    # Creates a dedicated directory for shared encoder files.
    encoders_dir = base_dir / 'encoders'
    if scan_results['encoders'] and not encoders_dir.exists():
        if not dry_run:
            encoders_dir.mkdir(exist_ok=True)
        print(f"📁 Created shared encoders directory: {encoders_dir}")
    
    # Moves all found encoder files to the shared directory.
    for encoder_file in scan_results['encoders']:
        if encoder_file.parent != encoders_dir:
            target_path = encoders_dir / encoder_file.name
            
            if dry_run:
                print(f"   [DRY RUN] Would move: {encoder_file.name} → encoders/{encoder_file.name}")
            else:
                try:
                    shutil.move(str(encoder_file), str(target_path))
                    print(f"   ✅ Moved encoder: {encoder_file.name} → encoders/{encoder_file.name}")
                except Exception as e:
                    print(f"   ❌ Failed to move encoder {encoder_file.name}: {e}")
    
    # Processes each unorganized model checkpoint.
    successful_moves = 0
    failed_moves = 0
    unknown_models = 0
    
    for model_file in scan_results['models']:
        print(f"\n📄 Processing: {model_file.name}")
        
        # Extracts model info to determine the correct subdirectory.
        model_info = extract_model_info_from_checkpoint(model_file)
        
        if model_info:
            vision_model, language_model = model_info
            print(f"   📊 Detected: {vision_model} + {language_model}")
            
            # Migrates the file to its new location.
            if migrate_checkpoint(model_file, base_dir, vision_model, language_model, dry_run):
                successful_moves += 1
            else:
                failed_moves += 1
        else:
            print(f"   ❓ Could not determine model combination")
            unknown_models += 1
    
    # Prints a summary of the organization process.
    print(f"\n📊 ORGANIZATION SUMMARY:")
    print(f"   ✅ Successfully processed: {successful_moves}")
    print(f"   ❌ Failed to process: {failed_moves}")
    print(f"   ❓ Unknown model combinations: {unknown_models}")
    
    if unknown_models > 0:
        print(f"\n💡 For unknown models, you can organize manually:")
        print(f"   python scripts/checkpoint_manager.py organize-manual --checkpoint-dir {checkpoint_dir}")


def manual_organization(checkpoint_dir: str):
    """
    Provides an interactive prompt to manually organize unknown checkpoints.

    This function identifies any model checkpoints that could not be automatically
    categorized and prompts the user to assign them to a model combination.

    Args:
        checkpoint_dir: The path to the checkpoint directory.
    """
    print(f"🔧 MANUAL CHECKPOINT ORGANIZATION: {checkpoint_dir}")
    print("=" * 60)
    
    base_dir = Path(checkpoint_dir)
    scan_results = scan_checkpoints(checkpoint_dir)
    
    # Filters for models where metadata extraction failed.
    unorganized_models = []
    for model_file in scan_results['models']:
        if not extract_model_info_from_checkpoint(model_file):
            unorganized_models.append(model_file)
    
    if not unorganized_models:
        print("✅ No unorganized model checkpoints requiring manual intervention")
        return
    
    print(f"Found {len(unorganized_models)} checkpoints requiring manual organization:")
    
    # A predefined list of common model combinations for user selection.
    available_combinations = [
        ("resnet", "sentence-bert"), ("clip", "sentence-bert"), ("dino", "sentence-bert"),
        ("convnext", "sentence-bert"), ("resnet", "mpnet"), ("clip", "mpnet"),
        ("resnet", "bert"), ("clip", "bert"),
    ]
    
    # Iterates through each unorganized file and prompts the user.
    for i, model_file in enumerate(unorganized_models, 1):
        print(f"\n📄 Checkpoint {i}/{len(unorganized_models)}: {model_file.name}")
        size_mb = model_file.stat().st_size / (1024*1024)
        print(f"   Size: {size_mb:.1f} MB")
        
        print("\nAvailable model combinations:")
        for j, (vision, language) in enumerate(available_combinations, 1):
            print(f"   {j}. {vision}_{language}")
        print("   0. Skip this file")
        print("   c. Custom combination")
        
        while True:
            try:
                choice = input(f"\nSelect combination for {model_file.name} (1-{len(available_combinations)}, 0, c): ").strip().lower()
                
                if choice == '0':
                    print("   ⏭️ Skipped")
                    break
                # Handles custom model combinations.
                elif choice == 'c':
                    vision_model = input("Enter vision model (resnet/clip/dino/convnext): ").strip()
                    language_model = input("Enter language model (sentence-bert/mpnet/bert/roberta): ").strip()
                    
                    if vision_model and language_model:
                        if migrate_checkpoint(model_file, base_dir, vision_model, language_model):
                            print(f"   ✅ Organized as: {vision_model}_{language_model}")
                        break
                    else:
                        print("   ❌ Invalid input, please try again")
                # Handles predefined choices.
                else:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(available_combinations):
                        vision_model, language_model = available_combinations[choice_idx]
                        if migrate_checkpoint(model_file, base_dir, vision_model, language_model):
                            print(f"   ✅ Organized as: {vision_model}_{language_model}")
                        break
                    else:
                        print("   ❌ Invalid choice, please try again")
            except (ValueError, KeyboardInterrupt):
                print("   ⏭️ Skipped")
                break


def create_checkpoint_info(checkpoint_dir: str):
    """
    Creates a JSON file summarizing the contents of the checkpoint directory.

    This function scans an organized checkpoint directory and generates a
    `checkpoint_info.json` file with details about each model combination,
    including checkpoint files, sizes, and any available metadata like epoch
    and validation loss.

    Args:
        checkpoint_dir: The path to the (preferably organized) checkpoint directory.
    """
    print(f"📋 CREATING CHECKPOINT INFO: {checkpoint_dir}")
    print("=" * 60)
    
    base_dir = Path(checkpoint_dir)
    if not base_dir.exists():
        print(f"❌ Directory does not exist: {checkpoint_dir}")
        return
    
    # Initializes the structure for the JSON output.
    info = {
        "checkpoint_directory": str(base_dir.absolute()),
        "scan_date": str(pd.Timestamp.now()),
        "organization_type": "model_specific_directories",
        "models": {},
        "shared_files": [],
        "summary": {}
    }
    
    # Scans for model-specific subdirectories.
    for model_dir in base_dir.iterdir():
        if model_dir.is_dir() and '_' in model_dir.name and model_dir.name != 'encoders':
            model_combo = model_dir.name
            model_files = []
            
            # Gathers information about each checkpoint file in the subdirectory.
            for checkpoint_file in model_dir.glob('*.pth'):
                file_info = {
                    "filename": checkpoint_file.name,
                    "size_mb": round(checkpoint_file.stat().st_size / (1024*1024), 2),
                    "path": str(checkpoint_file.relative_to(base_dir))
                }
                
                # Attempts to extract additional metadata from the checkpoint.
                try:
                    checkpoint = torch.load(checkpoint_file, map_location='cpu')
                    if 'epoch' in checkpoint: file_info['epoch'] = checkpoint['epoch']
                    if 'best_val_loss' in checkpoint: file_info['best_val_loss'] = float(checkpoint['best_val_loss'])
                except Exception:
                    pass
                
                model_files.append(file_info)
            
            if model_files:
                info['models'][model_combo] = {
                    "vision_model": model_combo.split('_')[0],
                    "language_model": '_'.join(model_combo.split('_')[1:]),
                    "checkpoints": model_files
                }
    
    # Scans for shared files like encoders.
    encoders_dir = base_dir / 'encoders'
    if encoders_dir.exists():
        for encoder_file in encoders_dir.glob('*.pkl'):
            info['shared_files'].append({
                "filename": encoder_file.name,
                "size_kb": round(encoder_file.stat().st_size / 1024, 2),
                "path": str(encoder_file.relative_to(base_dir)),
                "type": "encoder"
            })
    
    # Calculates summary statistics.
    info['summary'] = {
        "total_model_combinations": len(info['models']),
        "total_checkpoints": sum(len(m['checkpoints']) for m in info['models'].values()),
        "total_shared_files": len(info['shared_files']),
        "total_size_mb": sum(sum(c['size_mb'] for c in m['checkpoints']) for m in info['models'].values())
    }
    
    # Writes the information to a JSON file.
    info_file = base_dir / 'checkpoint_info.json'
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Checkpoint info saved to: {info_file}")
    print(f"📊 Summary:")
    print(f"   Model combinations: {info['summary']['total_model_combinations']}")
    print(f"   Total checkpoints: {info['summary']['total_checkpoints']}")
    print(f"   Shared files: {info['summary']['total_shared_files']}")
    print(f"   Total size: {info['summary']['total_size_mb']:.1f} MB")


def main():
    """
    Main function to parse command-line arguments and run the selected command.
    """
    parser = argparse.ArgumentParser(description="Manage model-specific checkpoint organization")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Defines the 'list' command.
    list_parser = subparsers.add_parser('list', help='List all checkpoints and their organization status')
    list_parser.add_argument('--checkpoint-dir', default='models/checkpoints', help='Checkpoint directory')
    
    # Defines the 'organize' command.
    organize_parser = subparsers.add_parser('organize', help='Automatically organize checkpoints by model combination')
    organize_parser.add_argument('--checkpoint-dir', default='models/checkpoints', help='Checkpoint directory')
    organize_parser.add_argument('--dry-run', action='store_true', help='Show what would be done without moving files')
    
    # Defines the 'organize-manual' command.
    manual_parser = subparsers.add_parser('organize-manual', help='Manually organize checkpoints with unknown model combinations')
    manual_parser.add_argument('--checkpoint-dir', default='models/checkpoints', help='Checkpoint directory')
    
    # Defines the 'info' command.
    info_parser = subparsers.add_parser('info', help='Create a JSON file with checkpoint information')
    info_parser.add_argument('--checkpoint-dir', default='models/checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Executes the appropriate function based on the provided command.
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'list':
        list_checkpoints(args.checkpoint_dir)
    elif args.command == 'organize':
        organize_checkpoints(args.checkpoint_dir, args.dry_run)
    elif args.command == 'organize-manual':
        manual_organization(args.checkpoint_dir)
    elif args.command == 'info':
        create_checkpoint_info(args.checkpoint_dir)


# Ensures the main function runs only when the script is executed directly.
if __name__ == '__main__':
    main()