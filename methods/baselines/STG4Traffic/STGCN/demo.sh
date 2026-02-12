#!/bin/bash

# ============================================================================
# STGCN 测试脚本演示
# 展示所有可用的测试方式
# ============================================================================

echo "============================================================================"
echo "STGCN Testing Demo Script"
echo "============================================================================"
echo ""
echo "This script demonstrates all available testing methods."
echo "Choose one of the options below or press Ctrl+C to exit."
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数：打印选项
print_option() {
    echo -e "${GREEN}$1${NC}: $2"
}

# 显示菜单
echo "Available options:"
echo ""
print_option "1" "Test METR-LA dataset (default settings)"
print_option "2" "Test PEMS-BAY dataset (default settings)"
print_option "3" "Test both datasets sequentially (using shell script)"
print_option "4" "Test both datasets sequentially (using Python script)"
print_option "5" "Test both datasets in parallel (requires 2 GPUs)"
print_option "6" "Test with custom model path"
print_option "7" "Test with custom device (GPU/CPU)"
print_option "8" "Show all available models"
print_option "9" "View help documentation"
print_option "0" "Exit"
echo ""

# 读取用户选择
read -p "Enter your choice [0-9]: " choice
echo ""

case $choice in
    1)
        echo -e "${BLUE}Testing METR-LA dataset...${NC}"
        echo ""
        echo "Command: python test_and_plot.py --dataset METRLA"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        python test_and_plot.py --dataset METRLA
        ;;

    2)
        echo -e "${BLUE}Testing PEMS-BAY dataset...${NC}"
        echo ""
        echo "Command: python test_and_plot.py --dataset PEMSBAY"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        python test_and_plot.py --dataset PEMSBAY
        ;;

    3)
        echo -e "${BLUE}Testing both datasets sequentially (shell script)...${NC}"
        echo ""
        echo "Command: bash test_both_datasets.sh"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        bash test_both_datasets.sh
        ;;

    4)
        echo -e "${BLUE}Testing both datasets sequentially (Python script)...${NC}"
        echo ""
        echo "Command: python batch_test.py --mode sequential"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        python batch_test.py --mode sequential
        ;;

    5)
        echo -e "${BLUE}Testing both datasets in parallel...${NC}"
        echo ""
        echo -e "${YELLOW}Note: This requires at least 2 GPUs${NC}"
        echo "Command: python batch_test.py --mode parallel --devices cuda:0 cuda:1"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        python batch_test.py --mode parallel --devices cuda:0 cuda:1
        ;;

    6)
        echo -e "${BLUE}Test with custom model path${NC}"
        echo ""
        echo "Available datasets: METRLA, PEMSBAY"
        read -p "Enter dataset name: " dataset
        read -p "Enter model path: " model_path
        echo ""
        echo "Command: python test_and_plot.py --dataset $dataset --model_path $model_path"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        python test_and_plot.py --dataset "$dataset" --model_path "$model_path"
        ;;

    7)
        echo -e "${BLUE}Test with custom device${NC}"
        echo ""
        echo "Available datasets: METRLA, PEMSBAY"
        echo "Available devices: cuda:0, cuda:1, cpu"
        read -p "Enter dataset name: " dataset
        read -p "Enter device: " device
        echo ""
        echo "Command: python test_and_plot.py --dataset $dataset --device $device"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to cancel..."
        python test_and_plot.py --dataset "$dataset" --device "$device"
        ;;

    8)
        echo -e "${BLUE}Searching for available models...${NC}"
        echo ""
        echo "METR-LA models:"
        find ../log/STGCN/METRLA -name "*best_model.pth" 2>/dev/null || echo "  No models found"
        echo ""
        echo "PEMS-BAY models:"
        find ../log/STGCN/PEMSBAY -name "*best_model.pth" 2>/dev/null || echo "  No models found"
        echo ""
        ;;

    9)
        echo -e "${BLUE}Available documentation:${NC}"
        echo ""
        echo "1. Quick Reference:"
        echo "   cat QUICK_REFERENCE.md"
        echo ""
        echo "2. Full Usage Guide:"
        echo "   cat TEST_USAGE.md"
        echo ""
        echo "3. Improvements Summary:"
        echo "   cat README_IMPROVEMENTS.md"
        echo ""
        read -p "Which document do you want to view? [1-3]: " doc_choice
        case $doc_choice in
            1) less QUICK_REFERENCE.md ;;
            2) less TEST_USAGE.md ;;
            3) less README_IMPROVEMENTS.md ;;
            *) echo "Invalid choice" ;;
        esac
        ;;

    0)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo -e "${YELLOW}Invalid choice. Please run the script again.${NC}"
        exit 1
        ;;
esac

echo ""
echo "============================================================================"
echo "Testing completed!"
echo "============================================================================"
echo ""
echo "Results are saved in directories starting with 'test_results_'"
echo ""
echo "To view results:"
echo "  ls -lhtr test_results_*/"
echo ""
echo "To view metrics:"
echo "  cat test_results_*/metrics_summary.txt"
echo ""
echo "To view visualizations:"
echo "  open test_results_*/*.png"
echo ""
