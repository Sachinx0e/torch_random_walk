#!/bin/bash
conda env export > environment.yml
pip freeze > requirements.txt