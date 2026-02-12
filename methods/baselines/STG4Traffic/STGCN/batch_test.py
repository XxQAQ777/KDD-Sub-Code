#!/usr/bin/env python
"""
批量测试脚本 - 支持多数据集并行或顺序测试

使用方法:
    python batch_test.py --mode sequential  # 顺序测试
    python batch_test.py --mode parallel    # 并行测试（需要多GPU）
    python batch_test.py --datasets METRLA  # 只测试METR-LA
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Batch testing script for STGCN')

    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['METRLA', 'PEMSBAY'],
                        choices=['METRLA', 'PEMSBAY'],
                        help='Datasets to test')

    parser.add_argument('--mode', type=str, default='sequential',
                        choices=['sequential', 'parallel'],
                        help='Testing mode: sequential or parallel')

    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for sequential mode')

    parser.add_argument('--devices', type=str, nargs='+',
                        default=['cuda:0', 'cuda:1'],
                        help='Devices for parallel mode (one per dataset)')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for testing')

    parser.add_argument('--model_paths', type=str, nargs='+', default=[],
                        help='Model paths (optional, one per dataset)')

    return parser.parse_args()


def run_test(dataset, device, batch_size, model_path=None):
    """运行单个测试"""
    cmd = [
        'python', 'test_and_plot.py',
        '--dataset', dataset,
        '--device', device,
        '--batch_size', str(batch_size)
    ]

    if model_path:
        cmd.extend(['--model_path', model_path])

    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd)
    return result.returncode


def sequential_test(args):
    """顺序测试多个数据集"""
    print(f"\n{'='*80}")
    print("Sequential Testing Mode")
    print(f"{'='*80}\n")

    results = {}

    for idx, dataset in enumerate(args.datasets):
        model_path = args.model_paths[idx] if idx < len(args.model_paths) else None

        start_time = datetime.now()
        return_code = run_test(dataset, args.device, args.batch_size, model_path)
        end_time = datetime.now()

        elapsed = (end_time - start_time).total_seconds()
        results[dataset] = {
            'success': return_code == 0,
            'elapsed': elapsed
        }

        if return_code == 0:
            print(f"\n✓ {dataset} completed in {elapsed:.1f}s\n")
        else:
            print(f"\n✗ {dataset} failed!\n")
            return False

    # 打印汇总
    print(f"\n{'='*80}")
    print("Testing Summary")
    print(f"{'='*80}\n")

    for dataset, result in results.items():
        status = '✓ PASS' if result['success'] else '✗ FAIL'
        print(f"{dataset:12s}: {status} ({result['elapsed']:.1f}s)")

    print(f"\n{'='*80}\n")

    return all(r['success'] for r in results.values())


def parallel_test(args):
    """并行测试多个数据集"""
    print(f"\n{'='*80}")
    print("Parallel Testing Mode")
    print(f"{'='*80}\n")

    if len(args.datasets) > len(args.devices):
        print(f"Warning: {len(args.datasets)} datasets but only {len(args.devices)} devices")
        print("Some tests will run sequentially")

    processes = []

    for idx, dataset in enumerate(args.datasets):
        device = args.devices[idx % len(args.devices)]
        model_path = args.model_paths[idx] if idx < len(args.model_paths) else None

        cmd = [
            'python', 'test_and_plot.py',
            '--dataset', dataset,
            '--device', device,
            '--batch_size', str(args.batch_size)
        ]

        if model_path:
            cmd.extend(['--model_path', model_path])

        print(f"Launching {dataset} on {device}...")
        proc = subprocess.Popen(cmd)
        processes.append((dataset, proc))

    print(f"\nAll processes launched. Waiting for completion...\n")

    # 等待所有进程完成
    results = {}
    for dataset, proc in processes:
        proc.wait()
        results[dataset] = proc.returncode == 0

    # 打印汇总
    print(f"\n{'='*80}")
    print("Testing Summary")
    print(f"{'='*80}\n")

    for dataset, success in results.items():
        status = '✓ PASS' if success else '✗ FAIL'
        print(f"{dataset:12s}: {status}")

    print(f"\n{'='*80}\n")

    return all(results.values())


def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print("STGCN Batch Testing Script")
    print(f"{'='*80}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Mode: {args.mode}")
    print(f"Batch Size: {args.batch_size}")

    if args.mode == 'sequential':
        print(f"Device: {args.device}")
    else:
        print(f"Devices: {', '.join(args.devices)}")

    print(f"{'='*80}\n")

    # 检查脚本是否存在
    if not os.path.exists('test_and_plot.py'):
        print("Error: test_and_plot.py not found!")
        print("Please run this script from the STGCN directory.")
        sys.exit(1)

    # 执行测试
    if args.mode == 'sequential':
        success = sequential_test(args)
    else:
        success = parallel_test(args)

    # 显示结果目录
    if success:
        print("All tests completed successfully!")
        print("\nResults saved in:")
        subprocess.run(['ls', '-lhtr', '.'], capture_output=False)
        subprocess.run(['bash', '-c', 'ls -dt test_results_* | head -10'], capture_output=False)
    else:
        print("Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
