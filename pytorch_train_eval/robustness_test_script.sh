#!/bin/bash

# Robustness Test Script
# Usage: ./robustness_test_script.sh <spawn_points_csv> <town_name> <model> <model_path> <test_type> <num_runs>

# Check if correct number of arguments provided
if [ $# -lt 6 ]; then
    echo "Usage: $0 <spawn_points_csv> <town_name> <model> <model_path> <test_type> <num_runs>"
    echo ""
    echo "Example: $0 ../common_utils/Town01_spawn_points_robustness_test.csv Town01 pilotnet ./pilotnet_model_best_141.pth velocity_test 7"
    echo ""
    echo "test_type options: velocity_test, position_test, random_control_test"
    exit 1
fi

# Parse arguments
SPAWN_POINTS_CSV=$1
TOWN_NAME=$2
MODEL=$3
MODEL_PATH=$4
TEST_TYPE=$5
NUM_RUNS=$6

# Validate test_type
case "$TEST_TYPE" in
    velocity_test|position_test|random_control_test)
        ;;
    *)
        echo "Error: Invalid test_type '$TEST_TYPE'"
        echo "Valid options: velocity_test, position_test, random_control_test"
        exit 1
        ;;
esac

# Check if spawn points CSV exists
if [ ! -f "$SPAWN_POINTS_CSV" ]; then
    echo "Error: Spawn points CSV file not found: $SPAWN_POINTS_CSV"
    exit 1
fi

# Check if model path exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

# Create output directory
OUTPUT_DIR="robustness_test_results_${TEST_TYPE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Robustness Test Batch Runner"
echo "=========================================="
echo "Spawn Points CSV: $SPAWN_POINTS_CSV"
echo "Town Name: $TOWN_NAME"
echo "Model: $MODEL"
echo "Model Path: $MODEL_PATH"
echo "Test Type: $TEST_TYPE"
echo "Number of Runs: $NUM_RUNS"
echo "Output Directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run the test NUM_RUNS times
for i in $(seq 1 $NUM_RUNS); do
    echo "[Run $i/$NUM_RUNS] Starting robustness test..."

    OUTPUT_FILE="$OUTPUT_DIR/run_${i}.log"

    # Build the python command based on test type
    PYTHON_CMD="python3 robustness_test.py"
    PYTHON_CMD="$PYTHON_CMD --spawn_points_csv $SPAWN_POINTS_CSV"
    PYTHON_CMD="$PYTHON_CMD --town_name $TOWN_NAME"
    PYTHON_CMD="$PYTHON_CMD --model $MODEL"
    PYTHON_CMD="$PYTHON_CMD --model_path $MODEL_PATH"

    # Add test-specific flag
    case "$TEST_TYPE" in
        velocity_test)
            PYTHON_CMD="$PYTHON_CMD --velocity_test True"
            ;;
        position_test)
            PYTHON_CMD="$PYTHON_CMD --position_test True"
            ;;
        random_control_test)
            PYTHON_CMD="$PYTHON_CMD --random_control_test True"
            ;;
    esac

    # Run the command and save output
    echo "Output saved to: $OUTPUT_FILE"
    $PYTHON_CMD > "$OUTPUT_FILE" 2>&1

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[Run $i/$NUM_RUNS] ✓ Completed successfully"
    else
        echo "[Run $i/$NUM_RUNS] ✗ Failed with exit code $EXIT_CODE"
    fi
    echo ""
done

echo "=========================================="
echo "All runs completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "=========================================="

# Print summary file locations
echo ""
echo "Log files:"
ls -lh "$OUTPUT_DIR"/run_*.log

# Create a summary script that extracts key metrics
echo ""
echo "Creating summary analysis..."
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
{
    echo "ROBUSTNESS TEST BATCH SUMMARY"
    echo "=============================="
    echo "Test Type: $TEST_TYPE"
    echo "Number of Runs: $NUM_RUNS"
    echo "Timestamp: $(date)"
    echo ""
    echo "Individual Run Results:"
    echo "----------------------"

    for i in $(seq 1 $NUM_RUNS); do
        OUTPUT_FILE="$OUTPUT_DIR/run_${i}.log"
        echo ""
        echo "Run $i:"

        case "$TEST_TYPE" in
            velocity_test)
                grep -A 10 "VELOCITY TEST SUMMARY" "$OUTPUT_FILE" | head -n 6
                ;;
            position_test)
                grep -A 10 "POSITION TEST SUMMARY" "$OUTPUT_FILE" | head -n 6
                ;;
            random_control_test)
                grep -A 5 "RANDOM CONTROL TEST SUMMARY" "$OUTPUT_FILE" | head -n 5
                ;;
        esac
    done
} > "$SUMMARY_FILE"

echo "Summary saved to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"
