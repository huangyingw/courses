#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

/media/volgrp/anaconda2/bin/python ./deeplearning1/nbs/lesson1.py
