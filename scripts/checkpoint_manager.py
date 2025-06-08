#!/usr/bin/env python
"""
Checkpoint Manager Utility

Helps manage model-specific checkpoint organization:
- List all checkpoints
- Migrate from old structure to new structure  
- Organize existing checkpoints by model combination
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
    Scan checkpoint directory and categorize files
    
    Returns:
        Dictionary with 'models', 'encoders', 'organized', 'unknown' keys
    """
    base_dir = Path(checkpoint_dir)
    if not base_dir.exists():
        return {'models': [], 'encoders': [], 'organized': [], 'unknown': []}
    
    results = {
        'models': [],      # .pth files in base directory
        'encoders': [],    # encoder files
        'organized': [],   # files already in model subdirectories
        'unknown': []      # other files
    }
    
    for item in base_dir.rglob('*'):
        if item.is_file():
            relative_path = item.relative_to(base_dir)
            
            # Check if it's already organized (in a subdirectory)
            if len(relative_path.parts) > 1:
                results['organized'].append(item)
            
            # Categorize files in base directory
            elif item.suffix == '.pth':
                results['models'].append(item)
            elif 'encoder' in item.name and item.suffix == '.pkl':
                results['encoders'].append(item)
            else:
                results['unknown'].append(item)
    
    return results


def extract_model_info_from_checkpoint(checkpoint_path: Path) -> Optional[Tuple[str, str]]:
    """
    Try to extract model information from checkpoint metadata
    
    Returns:
        Tuple of (vision_model, language_model) or None if not found
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Look for model configuration in checkpoint
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            if 'vision_model' in config and 'language_model' in config:
                return config['vision_model'], config['language_model']
        
        # Look for config in other possible locations
        for key in ['config', 'args', 'model_args']:
            if key in checkpoint:
                config = checkpoint[key]
                if isinstance(config, dict):
                    if 'vision_model' in config and 'language_model' in config:
                        return config['vision_model'], config['language_model']
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not read checkpoint {checkpoint_path}: {e}")
        return None


def list_checkpoints(checkpoint_dir: str):
    """List all checkpoints with their organization status"""
    print(f"📁 CHECKPOINT DIRECTORY SCAN: {checkpoint_dir}")
    print("=" * 60)
    
    base_dir = Path(checkpoint_dir)
    if not base_dir.exists():
        print(f"❌ Directory does not exist: {checkpoint_dir}")
        return
    
    scan_results = scan_checkpoints(checkpoint_dir)
    
    # Show organized checkpoints (in model subdirectories)
    organized_by_model = {}
    for item in scan_results['organized']:
        relative_path = item.relative_to(base_dir)
        model_combo = relative_path.parts[0]
        if model_combo not in organized_by_model:
            organized_by_model[model_combo] = []
        organized_by_model[model_combo].append(item)
    
    if organized_by_model:
        print("✅ ORGANIZED CHECKPOINTS (Model-Specific Directories):")
        for model_combo, files in organized_by_model.items():
            print(f"\n📂 {model_combo}/")
            for file in files:
                size_mb = file.stat().st_size / (1024*1024)
                print(f"   └── {file.name} ({size_mb:.1f} MB)")
    
    # Show unorganized model files
    if scan_results['models']:
        print(f"\n⚠️  UNORGANIZED MODEL CHECKPOINTS (Base Directory):")
        for model_file in scan_results['models']:
            size_mb = model_file.stat().st_size / (1024*1024)
            
            # Try to extract model info
            model_info = extract_model_info_from_checkpoint(model_file)
            if model_info:
                vision_model, language_model = model_info
                print(f"   📄 {model_file.name} ({size_mb:.1f} MB) → {vision_model}_{language_model}")
            else:
                print(f"   📄 {model_file.name} ({size_mb:.1f} MB) → Unknown model combination")
    
    # Show shared files
    if scan_results['encoders']:
        print(f"\n🔄 SHARED ENCODERS:")
        for encoder_file in scan_results['encoders']:
            size_kb = encoder_file.stat().st_size / 1024
            print(f"   📄 {encoder_file.name} ({size_kb:.1f} KB)")
    
    # Show other files
    if scan_results['unknown']:
        print(f"\n❓ OTHER FILES:")
        for other_file in scan_results['unknown']:
            size_kb = other_file.stat().st_size / 1024
            print(f"   📄 {other_file.name} ({size_kb:.1f} KB)")
    
    # Summary
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
    Migrate a single checkpoint to model-specific directory
    
    Returns:
        True if successful, False otherwise
    """
    model_combo = f"{vision_model}_{language_model}"
    target_dir = base_dir / model_combo
    target_path = target_dir / checkpoint_path.name
    
    if dry_run:
        print(f"   [DRY RUN] Would move: {checkpoint_path.name} → {model_combo}/{checkpoint_path.name}")
        return True
    
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
    Organize unorganized checkpoints by model combination
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
    
    if not scan_results['models']:
        print("✅ No unorganized model checkpoints found")
        return
    
    # Create encoders directory for shared files
    encoders_dir = base_dir / 'encoders'
    if scan_results['encoders'] and not encoders_dir.exists():
        if not dry_run:
            encoders_dir.mkdir(exist_ok=True)
        print(f"📁 Created shared encoders directory: {encoders_dir}")
    
    # Move encoder files to shared directory
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
    
    # Process model checkpoints
    successful_moves = 0
    failed_moves = 0
    unknown_models = 0
    
    for model_file in scan_results['models']:
        print(f"\n📄 Processing: {model_file.name}")
        
        # Try to extract model info from checkpoint
        model_info = extract_model_info_from_checkpoint(model_file)
        
        if model_info:
            vision_model, language_model = model_info
            print(f"   📊 Detected: {vision_model} + {language_model}")
            
            if migrate_checkpoint(model_file, base_dir, vision_model, language_model, dry_run):
                successful_moves += 1
            else:
                failed_moves += 1
        else:
            print(f"   ❓ Could not determine model combination")
            unknown_models += 1
    
    # Summary
    print(f"\n📊 ORGANIZATION SUMMARY:")
    print(f"   ✅ Successfully processed: {successful_moves}")
    print(f"   ❌ Failed to process: {failed_moves}")
    print(f"   ❓ Unknown model combinations: {unknown_models}")
    
    if unknown_models > 0:
        print(f"\n💡 For unknown models, you can organize manually:")
        print(f"   python scripts/checkpoint_manager.py organize-manual --checkpoint-dir {checkpoint_dir}")


def manual_organization(checkpoint_dir: str):
    """
    Interactive manual organization for unknown checkpoints
    """
    print(f"🔧 MANUAL CHECKPOINT ORGANIZATION: {checkpoint_dir}")
    print("=" * 60)
    
    base_dir = Path(checkpoint_dir)
    scan_results = scan_checkpoints(checkpoint_dir)
    
    unorganized_models = []
    for model_file in scan_results['models']:
        model_info = extract_model_info_from_checkpoint(model_file)
        if not model_info:
            unorganized_models.append(model_file)
    
    if not unorganized_models:
        print("✅ No unorganized model checkpoints requiring manual intervention")
        return
    
    print(f"Found {len(unorganized_models)} checkpoints requiring manual organization:")
    
    available_combinations = [
        ("resnet", "sentence-bert"),
        ("clip", "sentence-bert"), 
        ("dino", "sentence-bert"),
        ("convnext", "sentence-bert"),
        ("resnet", "mpnet"),
        ("clip", "mpnet"),
        ("resnet", "bert"),
        ("clip", "bert"),
    ]
    
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
                elif choice == 'c':
                    vision_model = input("Enter vision model (resnet/clip/dino/convnext): ").strip()
                    language_model = input("Enter language model (sentence-bert/mpnet/bert/roberta): ").strip()
                    
                    if vision_model and language_model:
                        if migrate_checkpoint(model_file, base_dir, vision_model, language_model):
                            print(f"   ✅ Organized as: {vision_model}_{language_model}")
                        break
                    else:
                        print("   ❌ Invalid input, please try again")
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
    """Create a JSON file with checkpoint information"""
    print(f"📋 CREATING CHECKPOINT INFO: {checkpoint_dir}")
    print("=" * 60)
    
    base_dir = Path(checkpoint_dir)
    if not base_dir.exists():
        print(f"❌ Directory does not exist: {checkpoint_dir}")
        return
    
    info = {
        "checkpoint_directory": str(base_dir.absolute()),
        "scan_date": str(pd.Timestamp.now()),
        "organization_type": "model_specific_directories",
        "models": {},
        "shared_files": [],
        "summary": {}
    }
    
    # Scan organized checkpoints
    for model_dir in base_dir.iterdir():
        if model_dir.is_dir() and '_' in model_dir.name and model_dir.name != 'encoders':
            model_combo = model_dir.name
            model_files = []
            
            for checkpoint_file in model_dir.glob('*.pth'):
                file_info = {
                    "filename": checkpoint_file.name,
                    "size_mb": round(checkpoint_file.stat().st_size / (1024*1024), 2),
                    "path": str(checkpoint_file.relative_to(base_dir))
                }
                
                # Try to extract epoch and loss info
                try:
                    checkpoint = torch.load(checkpoint_file, map_location='cpu')
                    if 'epoch' in checkpoint:
                        file_info['epoch'] = checkpoint['epoch']
                    if 'best_val_loss' in checkpoint:
                        file_info['best_val_loss'] = float(checkpoint['best_val_loss'])
                except:
                    pass
                
                model_files.append(file_info)
            
            if model_files:
                info['models'][model_combo] = {
                    "vision_model": model_combo.split('_')[0],
                    "language_model": '_'.join(model_combo.split('_')[1:]),
                    "checkpoints": model_files
                }
    
    # Scan shared files
    encoders_dir = base_dir / 'encoders'
    if encoders_dir.exists():
        for encoder_file in encoders_dir.glob('*.pkl'):
            info['shared_files'].append({
                "filename": encoder_file.name,
                "size_kb": round(encoder_file.stat().st_size / 1024, 2),
                "path": str(encoder_file.relative_to(base_dir)),
                "type": "encoder"
            })
    
    # Add summary
    info['summary'] = {
        "total_model_combinations": len(info['models']),
        "total_checkpoints": sum(len(model_info['checkpoints']) for model_info in info['models'].values()),
        "total_shared_files": len(info['shared_files']),
        "total_size_mb": sum(
            sum(checkpoint['size_mb'] for checkpoint in model_info['checkpoints']) 
            for model_info in info['models'].values()
        )
    }
    
    # Save info file
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
    parser = argparse.ArgumentParser(description="Manage model-specific checkpoint organization")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all checkpoints and their organization status')
    list_parser.add_argument('--checkpoint-dir', default='models/checkpoints', help='Checkpoint directory')
    
    # Organize command
    organize_parser = subparsers.add_parser('organize', help='Automatically organize checkpoints by model combination')
    organize_parser.add_argument('--checkpoint-dir', default='models/checkpoints', help='Checkpoint directory')
    organize_parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually moving files')
    
    # Manual organize command
    manual_parser = subparsers.add_parser('organize-manual', help='Manually organize checkpoints with unknown model combinations')
    manual_parser.add_argument('--checkpoint-dir', default='models/checkpoints', help='Checkpoint directory')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Create a JSON file with checkpoint information')
    info_parser.add_argument('--checkpoint-dir', default='models/checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
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


if __name__ == '__main__':
    main()