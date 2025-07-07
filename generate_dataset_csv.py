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
    Find matching RGB and DSM files based on filename.
    
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
    
    matched_pairs = []
    
    for rgb_path in rgb_files:
        rgb_filename = os.path.basename(rgb_path)
        
        # Extract base name (remove extension and any prefixes like RGB_)
        base_name = rgb_filename
        if rgb_filename.startswith('RGB_'):
            base_name = rgb_filename[4:]  # Remove 'RGB_' prefix
        
        # Remove extension
        base_name = os.path.splitext(base_name)[0]
        
        # Look for corresponding DSM file with multiple possible extensions
        dsm_extensions = ['.tif', '.png', '.jpg', '.jpeg']
        possible_dsm_names = []
        
        for ext in dsm_extensions:
            possible_dsm_names.extend([
                f"DSM_{base_name}{ext}",
                f"{base_name}{ext}",
                f"dsm_{base_name}{ext}",
                os.path.splitext(rgb_filename)[0].replace('RGB', 'DSM') + ext,
                os.path.splitext(rgb_filename)[0].replace('rgb', 'dsm') + ext
            ])
        
        dsm_path = None
        for dsm_name in possible_dsm_names:
            potential_path = os.path.join(dsm_dir, dsm_name)
            if os.path.exists(potential_path):
                dsm_path = potential_path
                break
        
        if dsm_path:
            matched_pairs.append((rgb_path, dsm_path))
        else:
            print(f"Warning: No matching DSM file found for {rgb_filename}")
    
    # Check for DSM files without RGB matches - support multiple extensions
    dsm_files = []
    for pattern in dsm_patterns:
        dsm_files.extend(glob.glob(os.path.join(dsm_dir, pattern)))
    
    dsm_basenames = set()
    for dsm_file in dsm_files:
        dsm_filename = os.path.basename(dsm_file)
        dsm_basenames.add(os.path.splitext(dsm_filename)[0])
    
    rgb_basenames = set()
    for rgb_path, _ in matched_pairs:
        rgb_filename = os.path.basename(rgb_path)
        base_name = rgb_filename
        if rgb_filename.startswith('RGB_'):
            base_name = rgb_filename[4:]
        rgb_basenames.add(os.path.splitext(base_name)[0])
    
    unmatched_dsm = dsm_basenames - {os.path.splitext(os.path.basename(pair[1]))[0] for pair in matched_pairs}
    if unmatched_dsm:
        print(f"Warning: Found {len(unmatched_dsm)} DSM files without RGB matches")
    
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
