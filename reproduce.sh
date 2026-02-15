#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIELD_FILE="$PROJECT_DIR/outputs/pathology_fields/dyed_lifted_polyps_to_polyps.npz"
OUTPUT_DIR="$PROJECT_DIR/outputs/counterfactual_pair"
MODE="${1:-all}"

export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_DIR"

derive_field() {
    python scripts/derive_pathology_field.py \
        --positive_prompt "An endoscopic image of dyed lifted polyps" \
        --negative_prompt "An endoscopic image of polyps" \
        --num_seeds 50 \
        --num_steps 20 \
        --layer_start 8 \
        --layer_end 16 \
        --output "$FIELD_FILE"
}

render_pair() {
    python scripts/render_counterfactual_pair.py \
        --pathology_field "$FIELD_FILE" \
        --positive_prompt "An endoscopic image of dyed lifted polyps" \
        --negative_prompt "An endoscopic image of polyps" \
        --seed 42 \
        --num_steps 20 \
        --layer_start 8 \
        --layer_end 16 \
        --strength 2.5 \
        --output_dir "$OUTPUT_DIR"
}

case "$MODE" in
    derive)
        derive_field
        ;;
    render)
        render_pair
        ;;
    all)
        derive_field
        render_pair
        ;;
    *)
        echo "Usage: bash reproduce.sh [derive|render|all]" >&2
        exit 2
        ;;
esac
