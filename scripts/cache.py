#!/usr/bin/env python
"""
Simple CLI for cache management
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.cache_utils import (
    print_cache_summary, 
    clear_cache, 
    clear_all_caches, 
    get_cache_stats,
    cache_exists
)


def main():
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
    
    command = sys.argv[1].lower()
    
    if command == "list":
        print_cache_summary()
    
    elif command == "clear":
        if len(sys.argv) > 2:
            if sys.argv[2] == "--all":
                response = input("⚠️  Clear ALL caches? This cannot be undone. (y/N): ")
                if response.lower() == 'y':
                    cleared = clear_all_caches()
                    print(f"✅ Cleared {cleared} caches")
                else:
                    print("❌ Cancelled")
            else:
                model_combo = sys.argv[2]
                if '_' in model_combo:
                    vision, language = model_combo.split('_', 1)
                    if cache_exists(vision, language):
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
            print("❌ Usage: python scripts/cache.py clear <vision_model>_<language_model>")
    
    elif command == "stats":
        if len(sys.argv) > 2:
            model_combo = sys.argv[2]
            if '_' in model_combo:
                vision, language = model_combo.split('_', 1)
                stats = get_cache_stats(vision, language)
                
                print(f"📊 CACHE STATS: {vision} + {language}")
                print("-" * 40)
                
                if stats["exists"]:
                    print(f"✅ Cache exists")
                    print(f"   Files: {stats['files']:,}")
                    print(f"   Size: {stats['size_mb']:.1f} MB")
                    print(f"   Path: {stats['path']}")
                else:
                    print(f"📭 Cache does not exist")
                    expected_path = f"cache/{vision}_{language}"
                    print(f"   Expected path: {expected_path}")
            else:
                print("❌ Format should be: vision_language (e.g., resnet_sentence-bert)")
        else:
            print("❌ Usage: python scripts/cache.py stats <vision_model>_<language_model>")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: list, clear, stats")


if __name__ == "__main__":
    main()