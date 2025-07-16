#!/usr/bin/env python3
"""
Dataset CSV Generator for Structured Datasets

This script generates CSV files for datasets organized in the following structure:
dataset_root/
├── train/
│   ├── rgb/
│   └── dsm/
├── valid/
│   ├── rgb/
│   └── dsm/
└── test/
    ├── rgb/
    └── dsm/

The script creates CSV files compatible with the existing loaddata.py script.
"""

import os
import glob
import argparse
from pathlib import Path


def get_matching_files(rgb_dir, dsm_dir, rgb_patterns=None, dsm_patterns=None):
    """
    Find matching RGB and DSM files based on identical base filenames.
    
    Args:
        rgb_dir: Path to RGB images directory
        dsm_dir: Path to DSM images directory
        rgb_patterns: List of patterns for RGB files (default: ["*.tif", "*.png", "*.jpg", "*.jpeg"])
        dsm_patterns: List of patterns for DSM files (default: ["*.tif", "*.png", "*.jpg", "*.jpeg"])
    
    Returns:
        List of tuples (rgb_path, dsm_path)
    """
    if rgb_patterns is None:
        rgb_patterns = ["*.tif", "*.png", "*.jpg", "*.jpeg"]
    if dsm_patterns is None:
        dsm_patterns = ["*.tif", "*.png", "*.jpg", "*.jpeg"]
    
    # Get all RGB files with supported extensions
    rgb_files = []
    for pattern in rgb_patterns:
        rgb_files.extend(glob.glob(os.path.join(rgb_dir, pattern)))
    rgb_files.sort()
    
    # Get all DSM files with supported extensions
    dsm_files = []
    for pattern in dsm_patterns:
        dsm_files.extend(glob.glob(os.path.join(dsm_dir, pattern)))
    dsm_files.sort()
    
    # Create dictionaries mapping base names to file paths
    rgb_base_to_path = {}
    for rgb_path in rgb_files:
        rgb_filename = os.path.basename(rgb_path)
        base_name = os.path.splitext(rgb_filename)[0]  # Simple: just remove extension
        rgb_base_to_path[base_name] = rgb_path
    
    dsm_base_to_path = {}
    for dsm_path in dsm_files:
        dsm_filename = os.path.basename(dsm_path)
        base_name = os.path.splitext(dsm_filename)[0]  # Simple: just remove extension
        dsm_base_to_path[base_name] = dsm_path
    
    # Find matching pairs based on identical base names
    matched_pairs = []
    unmatched_rgb = []
    
    for base_name, rgb_path in rgb_base_to_path.items():
        if base_name in dsm_base_to_path:
            dsm_path = dsm_base_to_path[base_name]
            matched_pairs.append((rgb_path, dsm_path))
        else:
            unmatched_rgb.append(os.path.basename(rgb_path))
    
    # Report unmatched files
    if unmatched_rgb:
        print(f"Warning: Found {len(unmatched_rgb)} RGB files without matching DSM files")
        if len(unmatched_rgb) <= 5:  # Show a few examples
            print(f"Examples: {unmatched_rgb}")
    
    unmatched_dsm = set(dsm_base_to_path.keys()) - set(rgb_base_to_path.keys())
    if unmatched_dsm:
        print(f"Warning: Found {len(unmatched_dsm)} DSM files without matching RGB files")
        if len(unmatched_dsm) <= 5:  # Show a few examples
            dsm_examples = [f"{name}{os.path.splitext(dsm_base_to_path[name])[1]}" for name in list(unmatched_dsm)[:5]]
            print(f"Examples: {dsm_examples}")
    
    return matched_pairs


def generate_csv_for_split(dataset_root, split_name, output_dir, dataset_name, rgb_patterns=None, dsm_patterns=None):
    """
    Generate CSV file for a specific dataset split.
    
    Args:
        dataset_root: Root directory of the dataset
        split_name: Name of the split (train, valid, test)
        output_dir: Directory to save the CSV file
        dataset_name: Name of the dataset for CSV filename
        rgb_patterns: List of patterns for RGB files
        dsm_patterns: List of patterns for DSM files
    """
    if rgb_patterns is None:
        rgb_patterns = ["*.tif", "*.png", "*.jpg", "*.jpeg"]
    if dsm_patterns is None:
        dsm_patterns = ["*.tif", "*.png", "*.jpg", "*.jpeg"]
    split_dir = os.path.join(dataset_root, split_name)
    rgb_dir = os.path.join(split_dir, 'rgb')
    dsm_dir = os.path.join(split_dir, 'dsm')
    
    # Check if directories exist
    if not os.path.exists(rgb_dir):
        print(f"Warning: RGB directory not found: {rgb_dir}")
        return
    
    if not os.path.exists(dsm_dir):
        print(f"Warning: DSM directory not found: {dsm_dir}")
        return
    
    # Print directory info for debugging
    rgb_extensions = [pattern.replace('*.', '') for pattern in rgb_patterns]
    dsm_extensions = [pattern.replace('*.', '') for pattern in dsm_patterns]
    rgb_files_count = len([f for f in os.listdir(rgb_dir) if f.lower().split('.')[-1] in [ext.lower() for ext in rgb_extensions]])
    dsm_files_count = len([f for f in os.listdir(dsm_dir) if f.lower().split('.')[-1] in [ext.lower() for ext in dsm_extensions]])
    print(f"Processing {split_name} split: {rgb_files_count} RGB files, {dsm_files_count} DSM files")
    
    # Get matching file pairs
    matched_pairs = get_matching_files(rgb_dir, dsm_dir, rgb_patterns, dsm_patterns)
    
    if not matched_pairs:
        print(f"Warning: No matching file pairs found for {split_name} split")
        return
    
    # Generate CSV filename
    csv_filename = f"{split_name}_{dataset_name}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Write CSV file
    with open(csv_path, 'w') as f:
        for rgb_path, dsm_path in matched_pairs:
            f.write(f"{rgb_path},{dsm_path}\n")
    
    print(f"Generated {csv_filename} with {len(matched_pairs)} file pairs")


def combine_train_valid_splits(dataset_root, output_dir, dataset_name, rgb_patterns=None, dsm_patterns=None):
    """
    Combine train and valid splits into a single train CSV file.
    
    Args:
        dataset_root: Root directory of the dataset
        output_dir: Directory to save the CSV file
        dataset_name: Name of the dataset for CSV filename
        rgb_patterns: List of patterns for RGB files
        dsm_patterns: List of patterns for DSM files
    """
    if rgb_patterns is None:
        rgb_patterns = ["*.tif", "*.png", "*.jpg", "*.jpeg"]
    if dsm_patterns is None:
        dsm_patterns = ["*.tif", "*.png", "*.jpg", "*.jpeg"]
    train_dir = os.path.join(dataset_root, 'train')
    valid_dir = os.path.join(dataset_root, 'valid')
    
    all_pairs = []
    
    # Get train pairs
    if os.path.exists(train_dir):
        train_rgb_dir = os.path.join(train_dir, 'rgb')
        train_dsm_dir = os.path.join(train_dir, 'dsm')
        
        if os.path.exists(train_rgb_dir) and os.path.exists(train_dsm_dir):
            rgb_extensions = [pattern.replace('*.', '') for pattern in rgb_patterns]
            dsm_extensions = [pattern.replace('*.', '') for pattern in dsm_patterns]
            rgb_files_count = len([f for f in os.listdir(train_rgb_dir) if f.lower().split('.')[-1] in [ext.lower() for ext in rgb_extensions]])
            dsm_files_count = len([f for f in os.listdir(train_dsm_dir) if f.lower().split('.')[-1] in [ext.lower() for ext in dsm_extensions]])
            print(f"Processing train split: {rgb_files_count} RGB files, {dsm_files_count} DSM files")
            
            train_pairs = get_matching_files(train_rgb_dir, train_dsm_dir, rgb_patterns, dsm_patterns)
            all_pairs.extend(train_pairs)
            print(f"Found {len(train_pairs)} pairs in train split")
        else:
            print("Warning: Train RGB or DSM directory not found")
    else:
        print("Warning: Train directory not found")
    
    # Get valid pairs if they exist
    if os.path.exists(valid_dir):
        valid_rgb_dir = os.path.join(valid_dir, 'rgb')
        valid_dsm_dir = os.path.join(valid_dir, 'dsm')
        
        if os.path.exists(valid_rgb_dir) and os.path.exists(valid_dsm_dir):
            rgb_extensions = [pattern.replace('*.', '') for pattern in rgb_patterns]
            dsm_extensions = [pattern.replace('*.', '') for pattern in dsm_patterns]
            rgb_files_count = len([f for f in os.listdir(valid_rgb_dir) if f.lower().split('.')[-1] in [ext.lower() for ext in rgb_extensions]])
            dsm_files_count = len([f for f in os.listdir(valid_dsm_dir) if f.lower().split('.')[-1] in [ext.lower() for ext in dsm_extensions]])
            print(f"Processing valid split: {rgb_files_count} RGB files, {dsm_files_count} DSM files")
            
            valid_pairs = get_matching_files(valid_rgb_dir, valid_dsm_dir, rgb_patterns, dsm_patterns)
            all_pairs.extend(valid_pairs)
            print(f"Found {len(valid_pairs)} pairs in valid split")
            print("Combining train and valid splits into single train CSV")
        else:
            print("Valid RGB or DSM directory not found - using only train split")
    else:
        print("Valid directory not found - using only train split")
    
    if not all_pairs:
        print("Error: No matching file pairs found in train or valid splits")
        return
    
    # Generate combined train CSV
    csv_filename = f"train_{dataset_name}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    with open(csv_path, 'w') as f:
        for rgb_path, dsm_path in all_pairs:
            f.write(f"{rgb_path},{dsm_path}\n")
    
    print(f"Generated combined {csv_filename} with {len(all_pairs)} total file pairs")


def main():
    parser = argparse.ArgumentParser(description='Generate CSV files for structured datasets')
    parser.add_argument('dataset_path', help='Path to the dataset root directory')
    parser.add_argument('--dataset-name', '-n', help='Name of the dataset (for CSV filenames)')
    parser.add_argument('--output-dir', '-o', default='./dataset', 
                       help='Output directory for CSV files (default: ./dataset)')
    parser.add_argument('--splits', nargs='+', default=['test'],
                       help='Dataset splits to process individually (default: test)')
    parser.add_argument('--combine-train-valid', action='store_true', default=True,
                       help='Combine train and valid splits into single train CSV (default: True)')
    parser.add_argument('--rgb-extensions', nargs='+', default=['tif', 'png', 'jpg', 'jpeg'],
                       help='Supported RGB file extensions (default: tif png jpg jpeg)')
    parser.add_argument('--dsm-extensions', nargs='+', default=['tif', 'png', 'jpg', 'jpeg'],
                       help='Supported DSM file extensions (default: tif png jpg jpeg)')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return
    
    # Auto-generate dataset name if not provided
    if not args.dataset_name:
        args.dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert extensions to glob patterns
    rgb_patterns = [f"*.{ext}" for ext in args.rgb_extensions]
    dsm_patterns = [f"*.{ext}" for ext in args.dsm_extensions]
    
    print(f"Processing dataset: {args.dataset_path}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"RGB extensions: {args.rgb_extensions}")
    print(f"DSM extensions: {args.dsm_extensions}")
    print("-" * 50)
    
    # Combine train and valid splits (default behavior)
    if args.combine_train_valid:
        combine_train_valid_splits(args.dataset_path, args.output_dir, args.dataset_name, rgb_patterns, dsm_patterns)
    else:
        # Process train and valid separately if requested
        if 'train' in args.splits:
            generate_csv_for_split(args.dataset_path, 'train', args.output_dir, args.dataset_name, rgb_patterns, dsm_patterns)
        if 'valid' in args.splits:
            generate_csv_for_split(args.dataset_path, 'valid', args.output_dir, args.dataset_name, rgb_patterns, dsm_patterns)
    
    # Process other splits individually
    for split in args.splits:
        if split not in ['train', 'valid']:  # Skip train/valid as they're handled above
            generate_csv_for_split(args.dataset_path, split, args.output_dir, args.dataset_name, rgb_patterns, dsm_patterns)
    
    print("-" * 50)
    print("CSV generation complete!")


if __name__ == "__main__":
    main()
