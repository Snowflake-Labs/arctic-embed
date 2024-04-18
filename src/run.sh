#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH=""
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi


OUTPUT_DIR="$MODEL_NAME_OR_PATH-outputs"
mkdir -p "${OUTPUT_DIR}"

python -u mteb_retrieval.py \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --output-dir "${OUTPUT_DIR}" "$@" \

echo "done retrieval"

python -u mteb_general.py \
    --model-name-or-path "${MODEL_NAME_OR_PATH}" \
    --task-types "STS" "Summarization" "PairClassification" "Classification" "Reranking" "Clustering" "BitextMining" \
    --output-dir "${OUTPUT_DIR}" "$@" \
    --l2-normalize True

echo "done non-retrieval"

