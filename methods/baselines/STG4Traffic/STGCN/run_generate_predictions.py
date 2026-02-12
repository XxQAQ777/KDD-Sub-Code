#!/usr/bin/env python3
"""
Wrapper script to run prediction generation with proper environment
"""
import subprocess
import sys
import os
import argparse

def run_command(cmd):
    """Run a shell command and print output using bash"""
    print(f"Running: {cmd}")
    # Use bash explicitly instead of default sh
    result = subprocess.run(cmd, shell=True, executable='/bin/bash', capture_output=False, text=True)
    return result.returncode

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate STGCN predictions for both datasets')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use: cuda:0, cuda:1, or cpu (default: cuda:0)')
    parser.add_argument('--output_dir', type=str, default='./predictions_npy',
                        help='Directory to save NPY files (default: ./predictions_npy)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for testing (default: 64)')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Change to STGCN directory
    stgcn_dir = "/home/xiaoxiao/FlowGNN/STG4Traffic-main/TrafficSpeed/STGCN"
    os.chdir(stgcn_dir)
    print(f"Working directory: {os.getcwd()}\n")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("="*80)
    print("STGCN Prediction NPY Generator")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {output_dir}\n")

    # Generate predictions for PEMSBAY
    print("="*80)
    print("Generating predictions for PEMSBAY...")
    print("="*80)
    cmd_pemsbay = f"python generate_predictions_npy.py --dataset PEMSBAY --device {args.device} --batch_size {args.batch_size} --output_dir {output_dir}"
    ret1 = run_command(cmd_pemsbay)

    if ret1 == 0:
        print("\n✓ PEMSBAY predictions generated successfully\n")
    else:
        print("\n✗ Failed to generate PEMSBAY predictions\n")

    # Generate predictions for METRLA
    print("="*80)
    print("Generating predictions for METRLA...")
    print("="*80)
    cmd_metrla = f"python generate_predictions_npy.py --dataset METRLA --device {args.device} --batch_size {args.batch_size} --output_dir {output_dir}"
    ret2 = run_command(cmd_metrla)

    if ret2 == 0:
        print("\n✓ METRLA predictions generated successfully\n")
    else:
        print("\n✗ Failed to generate METRLA predictions\n")

    # Summary
    print("="*80)
    print("Summary")
    print("="*80)
    if ret1 == 0 and ret2 == 0:
        print("✓ All predictions generated successfully!")
        print(f"\nOutput files in {output_dir}:")
        run_command(f"ls -lh {output_dir}/*.npy")
    else:
        print("Some predictions failed. Please check the errors above.")

    return 0 if (ret1 == 0 and ret2 == 0) else 1

if __name__ == "__main__":
    sys.exit(main())
