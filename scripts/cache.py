#!/usr/bin/env python
"""
Command-line interface for managing multimodal feature caches.

This script provides a set of tools to interact with the feature caches
generated during the data preprocessing and training pipelines. It allows
for listing available caches, viewing their statistics, and clearing them
either individually or all at once.
"""
import sys
from pathlib import Path

# Add the project's root directory to the system path.
# This allows the script to import modules from the 'src' directory (e.g., cache utilities)
# when run from the command line.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.cache_utils import (
    print_cache_summary, 
    clear_cache, 
    clear_all_caches, 
    get_cache_stats,
    cache_exists
)


def main():
    """
    Parses command-line arguments and executes the corresponding cache management command.

    This function serves as the main entry point for the script. It reads arguments
    directly from the command line to determine which action to perform on the
    feature caches. The available commands are 'list', 'clear', and 'stats'.
    """
    # If no command is provided, display the help message with usage instructions.
    if len(sys.argv) < 2:
        print("📁 CACHE MANAGEMENT")
        print("Usage:")
        print("  python scripts/cache.py list")
        print("  python scripts/cache.py clear <vision_model>_<language_model>")
        print("  python scripts/cache.py clear --all")
        print("  python scripts/cache.py stats <vision_model>_<language_model>")
        print("\nExamples:")
        print("  python scripts/cache.py list")
        print("  python scripts/cache.py clear resnet_sentence-bert")
        print("  python scripts/cache.py stats clip_mpnet")
        return
    
    # Standardize the command to lowercase for case-insensitive matching.
    command = sys.argv[1].lower()
    
    # Handles the 'list' command to display a summary of all available caches.
    if command == "list":
        print_cache_summary()
    
    # Handles the 'clear' command to remove specified caches.
    elif command == "clear":
        # Ensures that a target for clearing is specified.
        if len(sys.argv) > 2:
            # Handles the '--all' flag to remove every cache directory.
            if sys.argv[2] == "--all":
                # Prompts the user for confirmation before performing a destructive action.
                response = input("⚠️  Clear ALL caches? This cannot be undone. (y/N): ")
                if response.lower() == 'y':
                    cleared_count = clear_all_caches()
                    print(f"✅ Cleared {cleared_count} caches")
                else:
                    print("❌ Cancelled")
            # Handles clearing a specific cache based on the model combination.
            else:
                model_combo = sys.argv[2]
                # Validates the model combination format.
                if '_' in model_combo:
                    vision, language = model_combo.split('_', 1)
                    # Checks if the specified cache exists before attempting to clear.
                    if cache_exists(vision, language):
                        # Prompts the user for confirmation.
                        response = input(f"⚠️  Clear cache for {vision} + {language}? (y/N): ")
                        if response.lower() == 'y':
                            clear_cache(vision, language)
                        else:
                            print("❌ Cancelled")
                    else:
                        print(f"📭 Cache not found for {vision} + {language}")
                else:
                    print("❌ Format should be: vision_language (e.g., resnet_sentence-bert)")
        else:
            # Provides usage instructions if the 'clear' command is used incorrectly.
            print("❌ Usage: python scripts/cache.py clear <vision_model>_<language_model> or --all")
    
    # Handles the 'stats' command to display statistics for a specific cache.
    elif command == "stats":
        # Ensures that a target cache is specified.
        if len(sys.argv) > 2:
            model_combo = sys.argv[2]
            # Validates the model combination format.
            if '_' in model_combo:
                vision, language = model_combo.split('_', 1)
                stats = get_cache_stats(vision, language)
                
                print(f"📊 CACHE STATS: {vision} + {language}")
                print("-" * 40)
                
                # Displays detailed statistics if the cache exists.
                if stats["exists"]:
                    print(f"✅ Cache exists")
                    print(f"   Files: {stats['files']:,}")
                    print(f"   Size: {stats['size_mb']:.1f} MB")
                    print(f"   Path: {stats['path']}")
                # Informs the user if the cache does not exist.
                else:
                    print(f"📭 Cache does not exist")
                    expected_path = f"cache/{vision}_{language}"
                    print(f"   Expected path: {expected_path}")
            else:
                print("❌ Format should be: vision_language (e.g., resnet_sentence-bert)")
        else:
            # Provides usage instructions if the 'stats' command is used incorrectly.
            print("❌ Usage: python scripts/cache.py stats <vision_model>_<language_model>")
    
    # Handles any unknown commands.
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: list, clear, stats")


# Ensures that the main() function is called only when the script is executed directly.
if __name__ == "__main__":
    main()