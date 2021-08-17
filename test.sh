#!/bin/bash
set -e
export FORCE_CUDA=0
export MAX_JOBS=16
python setup.py develop
pytest -s 