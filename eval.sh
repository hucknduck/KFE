#!/bin/bash

mkdir -p logs results/records batches
chmod +x slurm/array.sh slurm/worker.sbatch

show_menu() {
    echo "=========================================="
    echo "           Evaluation Menu"
    echo "=========================================="
    echo "1. Eval 1 - Run evaluation orchestrator"
    echo "2. Option 2 (Not implemen ted)"
    echo "3. Option 3 (Not implemented)"
    echo "=========================================="
    echo -n "Please select an option (1-3): "
}

eval_1() {
    echo "Running evaluation orchestrator..."
    python3 tools/eval_orchestrator.py
    echo "Generating plots and summary..."
    python3 tools/plot_results.py
}


not_implemented() {
    echo "This option is not yet implemented."
    echo "Press Enter to return to menu..."
    read
}

# Display menu and handle choice
clear
show_menu
read choice

case $choice in
    1)
        eval_1
        ;;
    2)
        not_implemented
        ;;
    3)
        not_implemented
        ;;
    *)
        echo "Invalid option. Please select 1, 2, or 3."
        exit 1
        ;;
esac
