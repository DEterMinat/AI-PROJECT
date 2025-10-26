#!/usr/bin/env python3
"""
Dataset Comparison Utility
=========================

Utility script to compare different medical datasets and provide recommendations.

Usage:
    python -m src.utils.check_data
    python -c "from src.utils.check_data import compare_datasets; compare_datasets()"
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple


def load_dataset_info(dataset_prefix: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load dataset and summary information for a given prefix.

    Args:
        dataset_prefix: Dataset timestamp prefix (e.g., '20251007_235239')

    Returns:
        Tuple of (data_dict, summary_dict)
    """
    data_dir = Path("data/model_ready")

    # Load training data
    train_file = data_dir / f"train_{dataset_prefix}.json"
    summary_file = data_dir / f"split_summary_{dataset_prefix}.json"

    if not train_file.exists() or not summary_file.exists():
        raise FileNotFoundError(f"Dataset files not found for prefix: {dataset_prefix}")

    with open(train_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(summary_file, 'r') as f:
        summary = json.load(f)

    return data, summary


def compare_datasets(dataset1_prefix: str = "20251007_235239",
                    dataset2_prefix: str = "20251008_100314") -> None:
    """
    Compare two datasets and provide recommendations.

    Args:
        dataset1_prefix: First dataset prefix
        dataset2_prefix: Second dataset prefix
    """
    try:
        # Load datasets
        data1, summary1 = load_dataset_info(dataset1_prefix)
        data2, summary2 = load_dataset_info(dataset2_prefix)

        print("\n" + "="*60)
        print("📊 DATASET COMPARISON UTILITY")
        print("="*60 + "\n")

        print(f"📊 Dataset {dataset1_prefix} (LARGE):")
        print(f"   Total: {summary1['counts']['total']:,} samples")
        print(f"   Train: {summary1['counts']['train']:,}")
        print(f"   Val: {summary1['counts']['val']:,}")
        print(f"   Test: {summary1['counts']['test']:,}")

        print(f"\n📊 Dataset {dataset2_prefix} (CURRENT):")
        print(f"   Total: {summary2['counts']['total']:,} samples")
        print(f"   Train: {summary2['counts']['train']:,}")
        print(f"   Val: {summary2['counts']['val']:,}")
        print(f"   Test: {summary2['counts']['test']:,}")

        # Calculate improvement
        total1 = summary1['counts']['total']
        total2 = summary2['counts']['total']
        gain = total1 - total2
        gain_percent = (gain / total2 * 100) if total2 > 0 else 0

        print("\n✅ RECOMMENDATION:")
        if total1 > total2:
            print(f"   Use {dataset1_prefix} dataset: {total1:,} samples")
            print(f"   Increase from: {total2:,} → {total1:,}")
            print(f"   Gain: +{gain:,} samples (+{gain_percent:.1f}%)")
        else:
            print(f"   Use {dataset2_prefix} dataset: {total2:,} samples")
            print(f"   Current dataset is larger by {-gain:,} samples")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Available datasets:")
        data_dir = Path("data/model_ready")
        if data_dir.exists():
            files = list(data_dir.glob("split_summary_*.json"))
            for file in files:
                prefix = file.stem.replace("split_summary_", "")
                print(f"  - {prefix}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    # Default comparison
    compare_datasets()
