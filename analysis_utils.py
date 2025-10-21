import math
import random
from typing import List, Tuple, Iterable, Dict

# Basic linear algebra utilities using Python lists

def transpose(matrix: List[List[float]]) -> List[List[float]]:
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix)]


def matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    if not A or not B:
        return []
    result = []
    for row in A:
        new_row = []
        for col in zip(*B):
            total = 0.0
            for a, b in zip(row, col):
                total += a * b
            new_row.append(total)
        result.append(new_row)
    return result


def matvec(A: List[List[float]], v: List[float]) -> List[float]:
    return [sum(a * b for a, b in zip(row, v)) for row in A]


def vecdot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def gaussian_elimination(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(A)
    # Augment matrix
    aug = [row[:] + [val] for row, val in zip(A, b)]
    for col in range(n):
        # Partial pivoting
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular or ill-conditioned")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        # Normalize pivot row
        for j in range(col, n + 1):
            aug[col][j] /= pivot
        # Eliminate other rows
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def normal_equation_solve(X: List[List[float]], y: List[float]) -> List[float]:
    Xt = transpose(X)
    XtX = [[vecdot(col_i, col_j) for col_j in Xt] for col_i in Xt]
    Xty = [vecdot(col, y) for col in Xt]
    return gaussian_elimination(XtX, Xty)


def add_intercept_column(features: List[List[float]]) -> List[List[float]]:
    return [[1.0] + row for row in features]


def polynomial_design(x: List[float], degree: int) -> List[List[float]]:
    design = []
    for xi in x:
        row = [xi ** d for d in range(1, degree + 1)]
        design.append(row)
    return add_intercept_column(design)


def predict(design: List[List[float]], beta: List[float]) -> List[float]:
    return [vecdot(row, beta) for row in design]


def mse(y_true: List[float], y_pred: List[float]) -> float:
    n = len(y_true)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n if n else float('nan')


def loocv_error(x: List[float], y: List[float], degree: int) -> float:
    n = len(x)
    errors = []
    for i in range(n):
        x_train = x[:i] + x[i + 1 :]
        y_train = y[:i] + y[i + 1 :]
        design = polynomial_design(x_train, degree)
        beta = normal_equation_solve(design, y_train)
        x_test_row = [1.0] + [x[i] ** d for d in range(1, degree + 1)]
        y_pred = vecdot(x_test_row, beta)
        errors.append((y[i] - y_pred) ** 2)
    return sum(errors) / n


def combinations(iterable: List[int], r: int) -> Iterable[Tuple[int, ...]]:
    pool = list(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


def subset_metrics(X: List[List[float]], y: List[float], feature_indices: Tuple[int, ...], sigma_full_sq: float) -> Dict[str, float]:
    subset_design = [[row[i] for i in feature_indices] for row in X]
    subset_design = add_intercept_column(subset_design)
    beta = normal_equation_solve(subset_design, y)
    fitted = predict(subset_design, beta)
    n = len(y)
    p = len(feature_indices) + 1
    residuals = [yi - fi for yi, fi in zip(y, fitted)]
    sse = sum(r ** 2 for r in residuals)
    mse_val = sse / (n - p)
    y_mean = sum(y) / n
    sst = sum((yi - y_mean) ** 2 for yi in y)
    adj_r2 = 1 - (sse / (n - p)) / (sst / (n - 1))
    cp = sse / sigma_full_sq - (n - 2 * p)
    bic = n * math.log(sse / n) + p * math.log(n)
    return {
        'beta': beta,
        'sse': sse,
        'adj_r2': adj_r2,
        'cp': cp,
        'bic': bic,
        'p': p,
        'fitted': fitted,
    }


def forward_stepwise(X: List[List[float]], y: List[float], sigma_full_sq: float) -> List[Dict[str, object]]:
    remaining = list(range(len(X[0])))
    selected = []
    history = []
    while remaining:
        best = None
        best_metrics = None
        for idx in remaining:
            trial = selected + [idx]
            metrics = subset_metrics(X, y, tuple(sorted(trial)), sigma_full_sq)
            if best_metrics is None or metrics['bic'] < best_metrics['bic']:
                best_metrics = metrics
                best = idx
        selected.append(best)
        remaining.remove(best)
        history.append({'features': tuple(sorted(selected)), **best_metrics})
    return history


def backward_stepwise(X: List[List[float]], y: List[float], sigma_full_sq: float) -> List[Dict[str, object]]:
    selected = list(range(len(X[0])))
    history = []
    while selected:
        metrics_full = subset_metrics(X, y, tuple(sorted(selected)), sigma_full_sq)
        history.append({'features': tuple(sorted(selected)), **metrics_full})
        if len(selected) == 1:
            break
        best_idx = None
        best_metrics = None
        for idx in selected:
            trial = [i for i in selected if i != idx]
            metrics = subset_metrics(X, y, tuple(sorted(trial)), sigma_full_sq)
            if best_metrics is None or metrics['bic'] < best_metrics['bic']:
                best_metrics = metrics
                best_idx = idx
        selected.remove(best_idx)
    return history


def standardize_columns(X: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    means = [sum(col) / len(col) for col in zip(*X)]
    stds = []
    standardized = []
    for row in X:
        new_row = []
        for j, val in enumerate(row):
            mean = means[j]
            variance = sum((col[j] - mean) ** 2 for col in X) / len(X)
            std = math.sqrt(variance) if variance > 0 else 1.0
            stds.append(std)
            new_row.append((val - mean) / std if std > 0 else 0.0)
        standardized.append(new_row)
    # The above loop computed stds repeatedly; fix to compute once
    stds = []
    for j in range(len(X[0])):
        mean = means[j]
        variance = sum((row[j] - mean) ** 2 for row in X) / len(X)
        stds.append(math.sqrt(variance) if variance > 0 else 1.0)
    standardized = []
    for row in X:
        new_row = []
        for j, val in enumerate(row):
            std = stds[j]
            mean = means[j]
            new_row.append((val - mean) / std if std > 0 else 0.0)
        standardized.append(new_row)
    return standardized, means, stds


def lasso_coordinate_descent(X: List[List[float]], y: List[float], lam: float, max_iter: int = 1000, tol: float = 1e-6) -> Tuple[List[float], float]:
    n = len(X)
    p = len(X[0])
    X_std, means, stds = standardize_columns(X)
    beta = [0.0 for _ in range(p)]
    residuals = y[:]
    def soft_threshold(z, gamma):
        if z > gamma:
            return z - gamma
        if z < -gamma:
            return z + gamma
        return 0.0
    for iteration in range(max_iter):
        max_change = 0.0
        for j in range(p):
            # Compute partial residual excluding feature j
            residuals = [y[i] - sum(beta[k] * X_std[i][k] for k in range(p) if k != j) for i in range(n)]
            rho = sum(X_std[i][j] * residuals[i] for i in range(n))
            new_beta = soft_threshold(rho / n, lam / (2 * n))
            change = abs(beta[j] - new_beta)
            beta[j] = new_beta
            if change > max_change:
                max_change = change
        if max_change < tol:
            break
    intercept = sum(y) / n - sum(beta[j] * means[j] / stds[j] for j in range(p))
    beta_rescaled = [beta[j] / stds[j] for j in range(p)]
    return [intercept] + beta_rescaled, intercept


def k_fold_indices(n: int, k: int, seed: int = 0) -> List[List[int]]:
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)
    folds = [[] for _ in range(k)]
    for idx, val in enumerate(indices):
        folds[idx % k].append(val)
    return folds


def k_fold_cv(X: List[List[float]], y: List[float], k: int, build_model) -> float:
    folds = k_fold_indices(len(X), k)
    errors = []
    for fold in folds:
        train_idx = [i for i in range(len(X)) if i not in fold]
        test_idx = fold
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_test = [y[i] for i in test_idx]
        beta = build_model(X_train, y_train)
        preds = [vecdot([1.0] + row, beta) for row in X_test]
        errors.extend((yt - yp) ** 2 for yt, yp in zip(y_test, preds))
    return sum(errors) / len(errors)

# Plotting utilities using SVG output

def svg_scatter(x: List[float], y: List[float], width: int = 600, height: int = 400, filename: str = 'scatter.svg', title: str = 'Scatter', xlabel: str = 'X', ylabel: str = 'Y', curves: List[Tuple[List[float], List[float], str]] = None) -> None:
    if not x:
        return
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    padding = 40
    def scale_x(val):
        if max_x == min_x:
            return padding
        return padding + (val - min_x) / (max_x - min_x) * (width - 2 * padding)
    def scale_y(val):
        if max_y == min_y:
            return height - padding
        return height - padding - (val - min_y) / (max_y - min_y) * (height - 2 * padding)
    points = '\n'.join(f'<circle cx="{scale_x(xi):.2f}" cy="{scale_y(yi):.2f}" r="3" fill="#1f77b4" />' for xi, yi in zip(x, y))
    axis_lines = [
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="black" />',
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="black" />',
    ]
    labels = [
        f'<text x="{width/2}" y="{height - 5}" text-anchor="middle" font-size="14">{xlabel}</text>',
        f'<text x="15" y="{height/2}" transform="rotate(-90 15,{height/2})" text-anchor="middle" font-size="14">{ylabel}</text>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-size="16">{title}</text>',
    ]
    curve_paths = []
    if curves:
        colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        for idx, (cx, cy, label) in enumerate(curves):
            path_points = ' '.join(f'L {scale_x(px):.2f} {scale_y(py):.2f}' for px, py in zip(cx, cy))
            if path_points:
                path_points = 'M ' + path_points[2:]
            color = colors[idx % len(colors)]
            curve_paths.append(
                f'<path d="{path_points}" fill="none" stroke="{color}" stroke-width="2" />'
            )
            if label:
                labels.append(f'<text x="{width - padding}" y="{padding + 15 * (idx + 1)}" text-anchor="end" font-size="12" fill="{color}">{label}</text>')
    svg_content = '\n'.join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        *axis_lines,
        points,
        *(curve_paths or []),
        *labels,
        '</svg>'
    ])
    with open(filename, 'w') as f:
        f.write(svg_content)


def linspace(start: float, end: float, num: int) -> List[float]:
    if num == 1:
        return [start]
    step = (end - start) / (num - 1)
    return [start + i * step for i in range(num)]


def truncated_power_basis(x: List[float], knots: List[float], degree: int = 3) -> List[List[float]]:
    features = []
    for xi in x:
        row = [xi ** d for d in range(1, degree + 1)]
        for knot in knots:
            val = xi - knot
            row.append(val ** degree if val > 0 else 0.0)
        features.append(row)
    return add_intercept_column(features)


def natural_cubic_spline_basis(x: List[float], knots: List[float]) -> List[List[float]]:
    # Implementation based on natural spline truncated power basis
    def d(x_val, knot):
        return ((x_val - knot) ** 3) if x_val > knot else 0.0
    features = []
    K = len(knots)
    for xi in x:
        row = [xi, xi ** 2, xi ** 3]
        last = knots[-1]
        first = knots[0]
        def basis_term(k):
            denom = knots[-1] - knots[k]
            if denom == 0:
                return 0.0
            return (d(xi, knots[k]) - d(xi, last)) / denom
        terms = [basis_term(k) for k in range(K - 1)]
        row.extend(terms)
        features.append([1.0] + row)
    return features


def smoothing_spline(x: List[float], y: List[float], lam: float) -> List[float]:
    n = len(x)
    if n < 3:
        return y[:]
    # Sort data
    sorted_indices = sorted(range(n), key=lambda i: x[i])
    x_sorted = [x[i] for i in sorted_indices]
    y_sorted = [y[i] for i in sorted_indices]
    # Build second-difference penalty matrix
    D = [[0.0 for _ in range(n)] for _ in range(n - 2)]
    for i in range(n - 2):
        D[i][i] = 1.0
        D[i][i + 1] = -2.0
        D[i][i + 2] = 1.0
    DtD = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            DtD[i][j] = sum(D[k][i] * D[k][j] for k in range(n - 2))
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    A = [[I[i][j] + lam * DtD[i][j] for j in range(n)] for i in range(n)]
    coeffs = gaussian_elimination(A, y_sorted)
    fitted = coeffs[:]
    result = [0.0] * n
    for idx, original_idx in enumerate(sorted_indices):
        result[original_idx] = fitted[idx]
    return result


def evaluate_polynomial(coeffs: List[float], x: float) -> float:
    return sum(coeff * (x ** i) for i, coeff in enumerate(coeffs))


def polynomial_fit(x: List[float], y: List[float], degree: int) -> List[float]:
    design = polynomial_design(x, degree)
    return normal_equation_solve(design, y)


def read_auto_dataset(path: str) -> Tuple[List[Dict[str, object]], List[float], List[float], List[float]]:
    import csv
    rows = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # Clean horsepower: remove '?' and convert to float
    cleaned = []
    x_vals = []
    y_vals = []
    horsepower_vals = []
    for row in rows:
        horsepower = row['horsepower']
        if horsepower == '?':
            continue
        try:
            hp = float(horsepower)
        except ValueError:
            continue
        mpg = float(row['mpg'])
        cylinders = int(row['cylinders'])
        displacement = float(row['displacement'])
        weight = float(row['weight'])
        acceleration = float(row['acceleration'])
        year = int(row['year'])
        cleaned.append({
            'mpg': mpg,
            'cylinders': cylinders,
            'horsepower': hp,
            'displacement': displacement,
            'weight': weight,
            'acceleration': acceleration,
            'year': year,
            'origin': int(row['origin']),
            'name': row['name'],
        })
        x_vals.append(hp)
        y_vals.append(mpg)
        horsepower_vals.append(hp)
    return cleaned, x_vals, y_vals, horsepower_vals


def build_polynomial_features(x: List[float], degree: int) -> List[List[float]]:
    return [[x_val ** d for d in range(1, degree + 1)] for x_val in x]


def center_data(values: List[float]) -> Tuple[List[float], float]:
    mean = sum(values) / len(values)
    centered = [val - mean for val in values]
    return centered, mean


def variance(values: List[float]) -> float:
    mean = sum(values) / len(values)
    return sum((val - mean) ** 2 for val in values) / len(values)


def lasso_path_cv(X: List[List[float]], y: List[float], lambdas: List[float], k: int = 5) -> Tuple[float, Dict[float, float], Dict[float, List[float]]]:
    results = {}
    coeffs = {}
    for lam in lambdas:
        def builder(X_train, y_train, lam=lam):
            beta, _ = lasso_coordinate_descent(X_train, y_train, lam)
            return beta
        error = k_fold_cv(X, y, k, lambda Xt, yt, lam=lam: lasso_coordinate_descent(Xt, yt, lam)[0])
        model_beta, _ = lasso_coordinate_descent(X, y, lam)
        results[lam] = error
        coeffs[lam] = model_beta
    best_lambda = min(results, key=results.get)
    return best_lambda, results, coeffs



def moving_average_smoother(x: List[float], y: List[float], window: int = 25) -> List[float]:
    if not x:
        return []
    paired = sorted(zip(x, y))
    xs = [p[0] for p in paired]
    ys = [p[1] for p in paired]
    n = len(xs)
    half = max(1, window // 2)
    smoothed = []
    for idx, val in enumerate(xs):
        start = max(0, idx - half)
        end = min(n, idx + half + 1)
        avg = sum(ys[start:end]) / (end - start)
        smoothed.append(avg)
    # map back to original order using interpolation
    result = []
    for original in x:
        if original <= xs[0]:
            result.append(smoothed[0])
        elif original >= xs[-1]:
            result.append(smoothed[-1])
        else:
            for i in range(1, n):
                if xs[i] >= original:
                    ratio = (original - xs[i-1]) / (xs[i] - xs[i-1]) if xs[i] != xs[i-1] else 0.0
                    value = smoothed[i-1] + ratio * (smoothed[i] - smoothed[i-1])
                    result.append(value)
                    break
    return result

def backfitting_gam(X: Dict[str, List[float]], y: List[float], lam: float = 1.0, max_iter: int = 20, tol: float = 1e-4, window: int = 25) -> Dict[str, object]:
    predictors = list(X.keys())
    n = len(y)
    intercept = sum(y) / n
    functions = {name: [0.0] * n for name in predictors}
    for iteration in range(max_iter):
        max_change = 0.0
        for name in predictors:
            partial = [y[i] - intercept - sum(functions[p][i] for p in predictors if p != name) for i in range(n)]
            fitted = moving_average_smoother(X[name], partial, window)
            mean_f = sum(fitted) / n
            fitted = [val - mean_f for val in fitted]
            diff = max(abs(functions[name][i] - fitted[i]) for i in range(n))
            functions[name] = fitted
            if diff > max_change:
                max_change = diff
        intercept = sum(y[i] - sum(functions[name][i] for name in predictors) for i in range(n)) / n
        if max_change < tol:
            break
    residuals = [y[i] - intercept - sum(functions[name][i] for name in predictors) for i in range(n)]
    sigma = math.sqrt(sum(r * r for r in residuals) / n)
    se = {name: sigma for name in predictors}
    return {
        'intercept': intercept,
        'functions': functions,
        'residuals': residuals,
        'sigma': sigma,
        'se': se,
    }


def svg_line_plot(x: List[float], lines: List[Tuple[List[float], str, str]], filename: str, title: str, xlabel: str, ylabel: str) -> None:
    if not x:
        return
    min_x, max_x = min(x), max(x)
    min_y = min(min(values) for values, _, _ in lines)
    max_y = max(max(values) for values, _, _ in lines)
    padding = 40
    width, height = 600, 400
    def scale_x(val):
        if max_x == min_x:
            return padding
        return padding + (val - min_x) / (max_x - min_x) * (width - 2 * padding)
    def scale_y(val):
        if max_y == min_y:
            return height - padding
        return height - padding - (val - min_y) / (max_y - min_y) * (height - 2 * padding)
    axis_lines = [
        f'<line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="black" />',
        f'<line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="black" />',
    ]
    labels = [
        f'<text x="{width/2}" y="{height - 5}" text-anchor="middle" font-size="14">{xlabel}</text>',
        f'<text x="15" y="{height/2}" transform="rotate(-90 15,{height/2})" text-anchor="middle" font-size="14">{ylabel}</text>',
        f'<text x="{width/2}" y="25" text-anchor="middle" font-size="16">{title}</text>',
    ]
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
    line_paths = []
    for idx, (values, label, style) in enumerate(lines):
        color = colors[idx % len(colors)]
        path_points = ' '.join(f'L {scale_x(px):.2f} {scale_y(py):.2f}' for px, py in zip(x, values))
        if path_points:
            path_points = 'M ' + path_points[2:]
        line_paths.append(f'<path d="{path_points}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{style}" />')
        if label:
            labels.append(f'<text x="{width - padding}" y="{padding + 15 * (idx + 1)}" text-anchor="end" font-size="12" fill="{color}">{label}</text>')
    svg_content = '\n'.join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        *axis_lines,
        *(line_paths or []),
        *labels,
        '</svg>'
    ])
    with open(filename, 'w') as f:
        f.write(svg_content)

def smoothing_spline_predict(x: List[float], y: List[float], lam: float, x_new: List[float]) -> List[float]:
    fitted = smoothing_spline(x, y, lam)
    paired = sorted(zip(x, fitted))
    xs = [p[0] for p in paired]
    ys = [p[1] for p in paired]
    results = []
    for val in x_new:
        if val <= xs[0]:
            results.append(ys[0])
        elif val >= xs[-1]:
            results.append(ys[-1])
        else:
            for i in range(1, len(xs)):
                if xs[i] >= val:
                    x0, y0 = xs[i-1], ys[i-1]
                    x1, y1 = xs[i], ys[i]
                    ratio = (val - x0) / (x1 - x0) if x1 != x0 else 0.0
                    results.append(y0 + ratio * (y1 - y0))
                    break
    return results
