#!/bin/bash
export FORCE_CUDA=1
export MAX_JOBS=8
rm -r build
python setup.py clean
python setup.py build -j 16
python setup.py bdist_wheel 