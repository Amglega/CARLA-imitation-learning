#!/usr/bin/python
# -*- coding: utf-8 -*-

# Script to remove duplicate points from centroid CSV file
# The script reads a CSV file containing (x, y) points, removes duplicates while preserving the order of first occurrence, and writes the unique points back to a CSV file. 
# It also prints statistics about the number of original points, unique points, and duplicates removed.

import csv
import argparse
import os


def remove_duplicate_points(input_csv, output_csv=None):
    """
    Read a CSV file with (x, y) points and remove duplicates.
    Preserves the order of first occurrence.
    
    Args:
        input_csv (str): Path to input CSV file with x,y columns
        output_csv (str): Path to output CSV file. If None, overwrites input file.
    """
    
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found")
        return False
    
    # Read the input CSV
    points = []
    try:
        with open(input_csv, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Read header
            
            for row in reader:
                if len(row) >= 2:
                    try:
                        x = int(row[0])
                        y = int(row[1])
                        points.append((x, y))
                    except ValueError:
                        print(f"Warning: Could not parse row {row}, skipping")
                        continue
    except Exception as e:
        print(f"Error reading input file: {e}")
        return False
    
    # Remove duplicates while preserving order
    unique_points = []
    seen = set()
    
    for point in points:
        if point not in seen:
            unique_points.append(point)
            seen.add(point)
    
    # Determine output file
    if output_csv is None:
        output_csv = input_csv
    
    # Write unique points to output CSV
    try:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])  # Write header
            
            for x, y in unique_points:
                writer.writerow([x, y])
    
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False
    
    # Print statistics
    original_count = len(points)
    unique_count = len(unique_points)
    duplicates_removed = original_count - unique_count
    
    print(f"\nProcessing complete!")
    print(f"Original points: {original_count}")
    print(f"Unique points: {unique_count}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Output file: {output_csv}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate points from centroid CSV file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="centroid_data.csv",
        help="Input CSV file with x,y columns (default: centroid_data.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file. If not specified, overwrites input file"
    )
    
    args = parser.parse_args()
    
    success = remove_duplicate_points(args.input, args.output)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
