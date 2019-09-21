#!/bin/bash

python3 -m venv .venv
source .venv/bin/activate 
pip install --upgrade pip setuptools wheel --no-cache-dir
pip install numpy==1.16.4 --no-cache-dir
pip install -r requirements_gpu.txt

echo "Done!"
