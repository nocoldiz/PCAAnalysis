# PCA Analysis Tool

A desktop application for performing Principal Component Analysis (PCA) with interactive controls, preset datasets, and rich visualizations.

## Features

- Load preset datasets (Iris, Wine, Breast Cancer, Blobs, Circles) or your own CSV
- **Raman spectroscopy viewer** — built-in synthetic spectra (Semiconductors, Minerals, Polymers) or load your own Raman CSV
- Interactive Raman spectrum plot with live crosshair, normalize, and stack-offset toggles
- Configure PCA components, train/test split, and standardization
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

## Using a general CSV file

The app expects the **last column** to be the target/label column. All other columns are treated as numeric features. String labels are encoded automatically.

```
feature_1,feature_2,feature_3,label
5.1,3.5,1.4,setosa
4.9,3.0,1.4,setosa
```

A sample dataset (`Wine.csv`) is included in the repository.

## Raman spectroscopy

### Built-in presets

Select from the **RAMAN DATA** panel on the left:

| Preset | Materials |
|--------|-----------|
| Semiconductors | Silicon (521 cm⁻¹), Diamond (1332 cm⁻¹), Graphene (D/G/2D bands) |
| Minerals | Calcite, Quartz, TiO2 (Anatase) |
| Polymers | Polystyrene, PMMA, Polyethylene |

Each preset generates 20 synthetic spectra per material with realistic Lorentzian peaks, amplitude jitter, and noise. The spectra are also fed directly into the PCA pipeline.

### Loading a Raman CSV

Two formats are supported:

**Format A** — first column is the wavenumber axis, one spectrum per remaining column (most common; matches `raman_sample.csv`):

```
Wavenumber,Silicon_1,Silicon_2,Diamond_1,Diamond_2
100.0,0.001,0.002,0.000,0.001
...
521.0,0.998,1.012,0.003,0.002
...
1332.0,0.002,0.001,0.995,1.008
```

**Format B** — transposed; first column is sample label, column headers are wavenumber values:

```
label,100.0,200.0,...,3300.0
Silicon_1,0.001,0.003,...,0.000
Diamond_1,0.000,0.001,...,0.002
```

A ready-to-use sample file (`raman_sample.csv`) with 15 spectra (Silicon × 5, Diamond × 5, Graphene × 5) is included in the repository.

### Interactive viewer controls

| Control | Effect |
|---------|--------|
| Hover mouse | Live crosshair + wavenumber/intensity readout |
| Normalize | Scale each spectrum to max = 1 |
| Stack with offset | Separate overlapping spectra vertically |
| Toolbar (zoom/pan) | Matplotlib navigation toolbar |

## Jupyter Notebook

A companion notebook `Principal_Component_Analysis_with_Python(1).ipynb` is included for step-by-step exploration of PCA concepts.

```bash
pip install notebook
jupyter notebook
```
