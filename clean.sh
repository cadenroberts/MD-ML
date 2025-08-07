#!/bin/bash

for dir in $PSCRATCH/benchark_test_v5/*; do
    sim_dir="$dir/simulation"
    pdb_file="$sim_dir/final_state.pdb"

    # Case 1: simulation/ is empty
    if [ -d "$sim_dir" ] && [ -z "$(ls -A "$sim_dir")" ]; then
        echo "Removing $dir (empty simulation/)"
        rm -r "$dir"
        continue
    fi

    # Case 2: final_state.pdb exists but doesn't contain DNA
    if [ -f "$pdb_file" ]; then
        if ! grep -E '\s(D[ATGC]|[ATGC])\s' "$pdb_file" > /dev/null; then
            echo "Removing $dir (no DNA in final_state.pdb)"
            rm -r "$dir"
        fi
    fi

    echo "Fully Clean"
done

