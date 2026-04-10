#!/usr/bin/env bash
# Launch both PCA Analysis programs

echo "Starting PCA Analysis CLI..."
python pca-analysis.py &

echo "Starting PCA Analysis GUI..."
python pca_analysis_gui.py
