#!/bin/bash
export MAX_JOBS=8
rm -r build
rm -r dist
python setup.py clean
python setup.py build -j 16
python setup.py bdist_wheel
python setup.py install 