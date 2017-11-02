#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

find . -type f -name \*.ipynb -exec jupyter nbconvert --to=python {} \;
find . -type f -name \*.py -exec autopep8 --in-place --aggressive {} \;
