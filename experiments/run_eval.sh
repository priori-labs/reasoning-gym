#!/bin/bash
# Load environment variables from .env
SCRIPT_DIR="$(dirname "$0")"
set -a
source "$SCRIPT_DIR/.env"
set +a

# Check for --quick flag
QUICK=false
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--quick" ]]; then
        QUICK=true
    else
        ARGS+=("$arg")
    fi
done

if $QUICK; then
    # Find the config file from args
    CONFIG=""
    for i in "${!ARGS[@]}"; do
        if [[ "${ARGS[$i]}" == "--config" ]]; then
            CONFIG="${ARGS[$((i+1))]}"
            break
        fi
    done

    if [[ -n "$CONFIG" && -f "$CONFIG" ]]; then
        # Create temp config with size=1
        TEMP_CONFIG=$(mktemp).yaml
        sed 's/default_size: [0-9]*/default_size: 1/' "$CONFIG" > "$TEMP_CONFIG"

        # Replace config path in args
        for i in "${!ARGS[@]}"; do
            if [[ "${ARGS[$i]}" == "--config" ]]; then
                ARGS[$((i+1))]="$TEMP_CONFIG"
                break
            fi
        done

        echo "Quick test mode: running with 1 sample per dataset"
        python "$SCRIPT_DIR/../eval/eval.py" "${ARGS[@]}"
        rm -f "$TEMP_CONFIG" "${TEMP_CONFIG%.yaml}"
    else
        echo "Error: --quick requires a valid --config file"
        exit 1
    fi
else
    exec python "$SCRIPT_DIR/../eval/eval.py" "${ARGS[@]}"
fi
