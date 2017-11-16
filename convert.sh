#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

find . -type f -name \*.ipynb -exec python ~/loadrc/pythonrc/autopep8-ipynb.py {} \;
find . -type f -name \*.ipynb -exec jupyter nbconvert --to=python --template=python.tpl {} \;
find ./deeplearning*/  -type f -name \*.py -exec autopep8 --in-place --aggressive {} \;
