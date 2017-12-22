#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

source /etc/profile
source ~/.profile
cd deeplearning1/nbs
/media/volgrp/anaconda2/bin/python lesson1.py
