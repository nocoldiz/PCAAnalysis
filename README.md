# PCA Analysis Tool

A desktop application for performing Principal Component Analysis (PCA) with interactive controls, preset datasets, and rich visualizations.

## Features

- Load preset datasets (Iris, Wine, Breast Cancer, Blobs, Circles) or your own CSV
- Configure number of components, train/test split, and standardization
- Visualize PCA scatter plots, variance explained, decision boundaries, and component loadings
- Optional Logistic Regression classifier with accuracy and confusion matrix

## Requirements

- Python 3.8+
- pip

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/PCAAnalysis.git
cd PCAAnalysis
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

> `tkinter` is included with standard Python installations on Windows and macOS.
> On Linux, install it separately if needed:
> ```bash
> sudo apt install python3-tk   # Debian/Ubuntu
> sudo dnf install python3-tkinter  # Fedora
> ```

## Running

### Windows

```bat
launch.bat
```

### macOS / Linux

```bash
chmod +x launch.sh
./launch.sh
```

Or run directly:

```bash
python pca_analysis_gui.py
```

## Using a CSV file

The app expects the **last column** to be the target/label column. All other columns are treated as numeric features. String labels are encoded automatically.

Example format:

```
feature_1,feature_2,feature_3,label
5.1,3.5,1.4,setosa
4.9,3.0,1.4,setosa
...
```

A sample dataset (`Wine.csv`) is included in the repository.

## Jupyter Notebook

A companion notebook `Principal_Component_Analysis_with_Python(1).ipynb` is included for step-by-step exploration of PCA concepts.

```bash
pip install notebook
jupyter notebook
```
