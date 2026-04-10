# ============================================================
# PCA ANALYSIS — Texas Instruments Python Calculator Script
# Compatible: TI-84 Plus CE Python, TI-Nspire CX II
#
# Pure Python — no numpy, pandas, or sklearn required.
# ti_plotlib used for scatter / scree plots; falls back to
# ASCII output on a standard PC.
#
# Bundled sample data:
#   Dataset A — Raman spectral features (3 materials × 5 samples)
#   Dataset B — Iris-like flower measurements (3 classes × 15 samples)
#
# Usage on calc: Open in editor and press Run.
# Usage on PC  : python ti_pca.py
# ============================================================

import math

try:
    import ti_plotlib as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ============================================================
# EMBEDDED SAMPLE DATASETS
# ============================================================

# -- Dataset A: Raman spectral features ----------------------
# Features: peak height at 521, 1332, 1580, 2700 cm-1
# Generated from Lorentzian model + small simulated noise
# Classes: 0=Silicon  1=Diamond  2=Graphene
RAMAN_DATA = [
    # Silicon (class 0) — dominant peak at 521 cm-1
    [0.97, 0.01, 0.01, 0.00],
    [0.92, 0.03, 0.00, 0.01],
    [1.00, 0.02, 0.02, 0.00],
    [0.95, 0.00, 0.01, 0.02],
    [0.89, 0.02, 0.03, 0.00],
    # Diamond (class 1) — dominant peak at 1332 cm-1
    [0.01, 0.98, 0.00, 0.00],
    [0.02, 1.00, 0.01, 0.00],
    [0.00, 0.93, 0.02, 0.01],
    [0.01, 0.96, 0.00, 0.02],
    [0.03, 0.91, 0.01, 0.00],
    # Graphene (class 2) — D(1350~1332), G(1580), 2D(2700) bands
    [0.00, 0.23, 1.00, 0.64],
    [0.01, 0.25, 0.97, 0.68],
    [0.00, 0.21, 1.03, 0.61],
    [0.02, 0.26, 0.95, 0.70],
    [0.00, 0.20, 1.05, 0.63],
]
RAMAN_LABELS = [0]*5 + [1]*5 + [2]*5
RAMAN_NAMES  = ["Silicon", "Diamond", "Graphene"]
RAMAN_FEATS  = ["I(521)", "I(1332)", "I(1580)", "I(2700)"]

# -- Dataset B: Iris-like flower measurements ----------------
# Features: sepal_len, sepal_wid, petal_len, petal_wid (cm)
# Simplified 15-sample subset of the classic Iris dataset
# Classes: 0=Setosa  1=Versicolor  2=Virginica
IRIS_DATA = [
    # Setosa (class 0) — small petals
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    # Versicolor (class 1) — medium petals
    [7.0, 3.2, 4.7, 1.4],
    [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5],
    [5.5, 2.3, 4.0, 1.3],
    [6.5, 2.8, 4.6, 1.5],
    # Virginica (class 2) — large petals
    [6.3, 3.3, 6.0, 2.5],
    [5.8, 2.7, 5.1, 1.9],
    [7.1, 3.0, 5.9, 2.1],
    [6.3, 2.9, 5.6, 1.8],
    [6.5, 3.0, 5.8, 2.2],
]
IRIS_LABELS = [0]*5 + [1]*5 + [2]*5
IRIS_NAMES  = ["Setosa", "Versicol", "Virginic"]
IRIS_FEATS  = ["SepLen", "SepWid", "PetLen", "PetWid"]

# Colors for scatter plot (R, G, B)
CLASS_COLORS = [
    (0,   120, 255),   # blue
    (255, 80,  30),    # orange-red
    (30,  180, 60),    # green
]

# ============================================================
# PURE-PYTHON MATRIX / STATS UTILITIES
# ============================================================

def mean_col(data):
    """Return list of column means for a 2-D list."""
    n = len(data)
    p = len(data[0])
    return [sum(data[i][j] for i in range(n)) / n for j in range(p)]

def std_col(data, means):
    """Return list of column standard deviations (population)."""
    n = len(data)
    p = len(data[0])
    return [
        math.sqrt(sum((data[i][j] - means[j]) ** 2 for i in range(n)) / n)
        for j in range(p)
    ]

def standardize(data):
    """Zero-mean, unit-variance scaling. Returns (scaled, means, stds)."""
    means = mean_col(data)
    stds  = std_col(data, means)
    n, p  = len(data), len(data[0])
    scaled = [
        [(data[i][j] - means[j]) / (stds[j] if stds[j] > 1e-10 else 1.0)
         for j in range(p)]
        for i in range(n)
    ]
    return scaled, means, stds

def mat_T(A):
    """Transpose a 2-D list."""
    rows, cols = len(A), len(A[0])
    return [[A[r][c] for r in range(rows)] for c in range(cols)]

def mat_mul(A, B):
    """Matrix multiply A (m×k) × B (k×n) → (m×n)."""
    m, k, n = len(A), len(A[0]), len(B[0])
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            C[i][j] = sum(A[i][t] * B[t][j] for t in range(k))
    return C

def cov_matrix(X_scaled):
    """Compute p×p covariance matrix from n×p scaled data."""
    n = len(X_scaled)
    Xt = mat_T(X_scaled)          # p × n
    C  = mat_mul(Xt, X_scaled)    # p × p
    p  = len(C)
    return [[C[i][j] / (n - 1) for j in range(p)] for i in range(p)]

def dot(u, v):
    """Dot product of two equal-length lists."""
    return sum(u[i] * v[i] for i in range(len(u)))

def norm_vec(v):
    """L2 norm of a list."""
    return math.sqrt(sum(x * x for x in v))

def mat_vec(A, v):
    """Multiply matrix A (n×n) by column vector v (n,)."""
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

def normalize_vec(v):
    """Return unit-length version of v."""
    n = norm_vec(v)
    if n < 1e-12:
        return v
    return [x / n for x in v]

def power_iteration(A, n_iter=200):
    """Find the dominant eigenpair of symmetric matrix A via power iteration.

    Returns (eigenvalue, eigenvector) with the largest |eigenvalue|.
    n_iter=200 is sufficient for 4×4 matrices on TI hardware.
    """
    p = len(A)
    # Start with a non-trivial vector (ones normalised)
    v = normalize_vec([1.0] * p)
    for _ in range(n_iter):
        v = normalize_vec(mat_vec(A, v))
    eigenval = dot(v, mat_vec(A, v))
    return eigenval, v

def deflate(A, eigenval, eigenvec):
    """Hotelling deflation: A  ←  A − λ·v·vᵀ  (removes top component)."""
    p = len(A)
    return [
        [A[i][j] - eigenval * eigenvec[i] * eigenvec[j] for j in range(p)]
        for i in range(p)
    ]

def top_k_eigenpairs(C, k=2):
    """Return top-k (eigenvalue, eigenvector) pairs of covariance matrix C."""
    pairs = []
    A = [row[:] for row in C]          # copy
    for _ in range(k):
        lam, v = power_iteration(A)
        pairs.append((lam, v))
        A = deflate(A, lam, v)
    return pairs

def project(X_scaled, eigenvecs):
    """Project n×p scaled data onto eigenvecs (k×p) → n×k scores."""
    n = len(X_scaled)
    k = len(eigenvecs)
    return [
        [dot(X_scaled[i], eigenvecs[j]) for j in range(k)]
        for i in range(n)
    ]

def explained_variance_ratio(C, eigenvalues):
    """Fraction of total variance explained by each component."""
    total = sum(C[i][i] for i in range(len(C)))   # trace of C
    if total < 1e-12:
        return [0.0] * len(eigenvalues)
    return [lam / total for lam in eigenvalues]

# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_sep():
    print("-" * 26)

def print_header(title):
    print("=" * 26)
    print(title.center(26))
    print("=" * 26)

def print_loadings(eigenvecs, feat_names):
    """Print component loading table."""
    p = len(feat_names)
    k = len(eigenvecs)
    print("\nLoadings:")
    header = "Feature   " + "".join(f"  PC{i+1}" for i in range(k))
    print(header)
    print_sep()
    for j in range(p):
        row = f"{feat_names[j]:<9s}"
        for i in range(k):
            row += f" {eigenvecs[i][j]:+.3f}"
        print(row)

def print_variance(evr):
    """Print explained variance bar chart."""
    print("\nExplained variance:")
    cumul = 0.0
    for i, r in enumerate(evr):
        cumul += r
        bar = int(r * 30)
        print(f"PC{i+1}: {'#'*bar} {r*100:.1f}%  (cum {cumul*100:.1f}%)")

def ascii_scatter(scores, labels, class_names):
    """Simple ASCII scatter plot of PC1 vs PC2."""
    W, H = 38, 18
    canvas = [[" "] * W for _ in range(H)]

    pc1 = [s[0] for s in scores]
    pc2 = [s[1] for s in scores]
    x_min, x_max = min(pc1), max(pc1)
    y_min, y_max = min(pc2), max(pc2)
    x_rng = (x_max - x_min) or 1.0
    y_rng = (y_max - y_min) or 1.0

    markers = ["o", "+", "*"]
    for i, (x, y) in enumerate(zip(pc1, pc2)):
        col = int((x - x_min) / x_rng * (W - 1))
        row = H - 1 - int((y - y_min) / y_rng * (H - 1))
        col = max(0, min(W - 1, col))
        row = max(0, min(H - 1, row))
        canvas[row][col] = markers[labels[i] % 3]

    print("\n PC2")
    print("  ^")
    for r in canvas:
        print("  |" + "".join(r))
    print("  +" + "-" * W + "> PC1")
    print("\nLegend:")
    for i, name in enumerate(class_names):
        print(f"  {markers[i]} = {name}")

def ti_scatter(scores, labels, class_names, title="PC1 vs PC2"):
    """Scatter plot with ti_plotlib, one color per class."""
    pc1 = [s[0] for s in scores]
    pc2 = [s[1] for s in scores]

    margin = 0.3
    x_min = min(pc1) - margin
    x_max = max(pc1) + margin
    y_min = min(pc2) - margin
    y_max = max(pc2) + margin

    plt.cls()
    plt.window(x_min, x_max, y_min, y_max)
    plt.axes("on")

    n_classes = len(class_names)
    for cls in range(n_classes):
        xs = [pc1[i] for i in range(len(labels)) if labels[i] == cls]
        ys = [pc2[i] for i in range(len(labels)) if labels[i] == cls]
        if not xs:
            continue
        r, g, b = CLASS_COLORS[cls % len(CLASS_COLORS)]
        plt.color(r, g, b)
        plt.scatter(xs, ys, ".", "")    # empty string = use plt.color()

    plt.title(title, x_min, y_max + 0.05 * (y_max - y_min), "black")
    plt.show()

def ti_scree(evr):
    """Bar/line scree plot of explained variance per component."""
    k = len(evr)
    cumul = []
    c = 0.0
    for r in evr:
        c += r
        cumul.append(c)

    x = list(range(1, k + 1))

    plt.cls()
    plt.window(0, k + 1, 0, 1.1)
    plt.axes("on")
    # Individual bars (approximated with scatter + thick marks)
    plt.color(0, 120, 255)
    plt.scatter(x, evr, ".", "blue")
    plt.plot(x, evr, ".", "blue")
    # Cumulative line
    plt.color(255, 60, 60)
    plt.plot(x, cumul, ".", "red")
    plt.title("Scree (blue=indiv, red=cum)", 0.1, 1.05, "black")
    plt.show()

# ============================================================
# PCA PIPELINE
# ============================================================

def run_pca(data, labels, class_names, feat_names, n_components=2, dataset_name=""):
    """Full PCA pipeline: standardize → covariance → eigenpairs → project → display."""

    print_header("PCA: " + dataset_name)

    n_samples = len(data)
    n_features = len(data[0])
    print(f"Samples  : {n_samples}")
    print(f"Features : {n_features}")
    print(f"Classes  : {len(class_names)}")

    # 1. Standardize
    print("\nStandardizing...")
    X_scaled, means, stds = standardize(data)
    print("Feature means (orig):")
    for j, (f, m) in enumerate(zip(feat_names, means)):
        print(f"  {f}: {m:.3f}  std={stds[j]:.3f}")

    # 2. Covariance matrix
    print("\nCovariance matrix (scaled):")
    C = cov_matrix(X_scaled)
    for row in C:
        print("  " + "  ".join(f"{v:+.3f}" for v in row))

    # 3. Eigenpairs (top n_components)
    k = min(n_components, n_features)
    print(f"\nComputing top {k} components...")
    pairs = top_k_eigenpairs(C, k=k)
    eigenvalues = [lam for (lam, _) in pairs]
    eigenvecs   = [v   for (_,   v) in pairs]

    # 4. Explained variance
    evr = explained_variance_ratio(C, eigenvalues)
    print_variance(evr)

    # 5. Component loadings
    print_loadings(eigenvecs, feat_names)

    # 6. Project data
    scores = project(X_scaled, eigenvecs)

    # 7. Print first few projected samples
    print("\nPC scores (first 6 samples):")
    print("  " + "  ".join(f"PC{i+1}" for i in range(k)))
    for i in range(min(6, n_samples)):
        row = "  ".join(f"{scores[i][j]:+.3f}" for j in range(k))
        label = class_names[labels[i]]
        print(f"  {row}  [{label}]")

    # 8. Class separation: mean PC1 per class
    print("\nMean PC1 per class:")
    for cls, name in enumerate(class_names):
        pc1_vals = [scores[i][0] for i in range(n_samples) if labels[i] == cls]
        if pc1_vals:
            m = sum(pc1_vals) / len(pc1_vals)
            print(f"  {name}: {m:+.3f}")

    # 9. Visualize
    if k >= 2:
        if HAS_PLOT:
            ti_scatter(scores, labels, class_names, dataset_name)
            input("Press Enter for scree plot...")
            ti_scree(evr)
        else:
            ascii_scatter(scores, labels, class_names)
            print()
            print_variance(evr)

    return scores, eigenvecs, evr

# ============================================================
# MAIN MENU
# ============================================================

def main():
    print_header("PCA ANALYSIS")
    print("Select dataset:")
    print(" 1 - Raman spectra")
    print("     (Si/Diamond/Graphene)")
    print(" 2 - Iris flowers")
    print("     (Setosa/Versicol/Virgin)")
    print(" 0 - Quit")

    while True:
        choice = input("\nSelect (0-2): ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            run_pca(
                data        = RAMAN_DATA,
                labels      = RAMAN_LABELS,
                class_names = RAMAN_NAMES,
                feat_names  = RAMAN_FEATS,
                n_components= 2,
                dataset_name= "Raman",
            )

        elif choice == "2":
            run_pca(
                data        = IRIS_DATA,
                labels      = IRIS_LABELS,
                class_names = IRIS_NAMES,
                feat_names  = IRIS_FEATS,
                n_components= 2,
                dataset_name= "Iris",
            )

        else:
            print("Invalid choice")
            continue

        again = input("\nRun again? (y/n): ").strip().lower()
        if again != "y":
            print("Bye!")
            break

main()
