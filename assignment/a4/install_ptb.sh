#!/bin/bash

# Helper script to unpack and install the Penn Treebank data.

set -e

pushd "$(dirname $0)"

# Get NLTK data path
NLTK_DATA_DIR=$(python -c 'import nltk; print(nltk.data.path[0])')
echo "NLTK data directory: ${NLTK_DATA_DIR}"
unzip -q ptb.zip -d "${NLTK_DATA_DIR}/corpora"

# Check that NLTK can read the correct file ids.
python -c "from helpers import verify_ptb_install; verify_ptb_install()"
