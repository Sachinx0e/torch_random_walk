#!/bin/bash
set -e
export FORCE_CUDA=0
export MAX_JOBS=8
python setup.py develop
pytest -s 