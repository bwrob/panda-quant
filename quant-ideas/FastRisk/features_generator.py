import json
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import pdist, cdist

# -----------------------------------------------------------------------------
# 1) Registry of custom functions (string → callable)
#
# Each callable here must accept a NumPy array `X_sub` of shape (n_samples, n_feats_sub)
# and return a 1D NumPy array of length n_samples.
# -----------------------------------------------------------------------------
def percent_all(X_sub: np.ndarray) -> np.ndarray:
    """
    Example: sum each row across columns, divide by 100.
    X_sub can be (n_samples, n_feats_sub). We sum along axis=1.
    """
    return X_sub.sum(axis=1) / 100.0

def zscore_all(X_sub: np.ndarray) -> np.ndarray:
    """
    Example: take the first column only, compute its z-score.
    If X_sub has multiple columns, we only zscore column 0.
    """
    col = X_sub[:, 0]
    return (col - col.mean()) / col.std()

def sum_squares(X_sub: np.ndarray) -> np.ndarray:
    """
    Example: sum of squares of each row across all columns.
    """
    return np.sum(X_sub ** 2, axis=1)

CUSTOM_FUNCTIONS = {
    'percent_all':  percent_all,
    'zscore_all':   zscore_all,
    'sum_squares':  sum_squares,
    # You can add more registry‐named functions here, all taking a NumPy array.
}


# -----------------------------------------------------------------------------
# 2) Helper: Generate Sobol centers + median‐σ in d dimensions
# -----------------------------------------------------------------------------
def generate_sobol_centers(X: np.ndarray, n_centers: int):
    """
    Input:
      X : np.ndarray, shape (n_samples, n_features)
      n_centers : int
    Returns:
      centers : np.ndarray, shape (n_centers, n_features)
      sigma   : float = median pairwise distance among those centers
    """
    # 1) Find feature‐wise min & max
    mins = X.min(axis=0)
    maxs = X.max(axis=0)

    # 2) Scrambled Sobol sampler
    d = X.shape[1]
    sampler = qmc.Sobol(d, scramble=True)

    m = int(np.ceil(np.log2(n_centers)))
    all_pts = sampler.random_base2(m)       # shape = (2^m, d)
    sobol_pts = all_pts[:n_centers, :]      # shape = (n_centers, d)

    # 3) Scale [0,1]^d → [mins, maxs]^d
    centers = qmc.scale(sobol_pts, mins, maxs)

    # 4) Compute median pairwise distance
    pairwise = pdist(centers)
    sigma = float(np.median(pairwise))

    return centers, sigma


# -----------------------------------------------------------------------------
# 3) Helper: Generate all exponent‐vectors summing to “degree”
# -----------------------------------------------------------------------------
def generate_exponent_vectors(n_vars: int, degree: int):
    """
    Yield all tuples (e1, …, e_n_vars) of nonnegative ints s.t. sum(e) = degree.
    """
    if n_vars == 1:
        yield (degree,)
    else:
        for i in range(degree + 1):
            for tail in generate_exponent_vectors(n_vars - 1, degree - i):
                yield (i,) + tail


# -----------------------------------------------------------------------------
# 4) The NumPy‐only FeatureGenerator (JSON‐serializable)
# -----------------------------------------------------------------------------
class FeatureGenerator:
    """
    A NumPy‐based, JSON‐serializable feature‐generation class.

    Supports:
      • Multivariate polynomials (exact total degree) over a list of columns
      • Univariate log
      • Univariate hat (triangular basis)
      • RBF blocks (explicit centers+sigma or Sobol automatic) over any subset of columns
      • “Global” Sobol‐RBF over all columns via rbf_auto
      • Custom functions (registry‐named or lambda‐string) on multiple columns

    Methods:
      create_features()  → (new_matrix, new_names)
      to_dict()/to_json()    → serialize recipe (no raw Python objects)
      from_dict()/from_json() → reconstruct from JSON (needs feature_values again)
    """

    def __init__(
        self,
        feature_defs: dict,
        feature_names: list,
        feature_values: np.ndarray,
        rbf_auto: dict = None
    ):
        """
        Parameters
        ----------
        feature_defs : dict
          Keys are arbitrary identifiers. Each value is a dict with:
            'type': one of ['polynomial','log','hat','rbf','custom']
            depending on 'type', other keys:

          1) 'type':'polynomial'
             • 'degree': int ≥ 1
             • optional 'features': list[str]  (subset of feature_names)
               if omitted, defaults to [feat] (univariate).

          2) 'type':'log'
             (no extra keys)

          3) 'type':'hat'
             • 'centers': list[float]
             • 'width': float

          4) 'type':'rbf'
             • optional 'features': list[str] (subset of feature_names, default = all)
             AND EITHER:
               a) 'centers': list[list_of_floats]  (each sublist length = len(features))
                  'sigma': float
               OR
               b) 'n_centers': int  (auto Sobol)

          5) 'type':'custom'
             • 'features': list[str]  (subset of feature_names; defaults to [feat])
             • 'functions': list of dicts, each either:
               { 'fn': '<registry_key>',   'suffix': '<suffix>' }
               { 'expr': '<python_lambda_string>', 'suffix': '<suffix>' }
             If 'fn' is present, lookup CUSTOM_FUNCTIONS[fn], call with a NumPy array
               of shape (n_samples, n_features_sub).
             If 'expr' is present, eval the string to get a Python lambda, which should
               take len(features) 1D arrays as arguments.

        feature_names : list[str]
          Names of the original columns, used to index into feature_values.

        feature_values : np.ndarray, shape (n_samples, n_original_features)

        rbf_auto : dict or None, default=None
          If provided, must be { 'n_centers': int }, which triggers a global Sobol‐RBF
          over all original columns, creating columns: 'rbf_auto_0', …, 'rbf_auto_{n_centers−1}'.
        """
        # Convert feature_values to NumPy array if needed
        if not isinstance(feature_values, np.ndarray):
            raise ValueError("feature_values must be a NumPy array (shape n_samples × n_features).")
        self.df = feature_values  # purely NumPy
        self.feature_names = feature_names
        self.feature_defs = feature_defs

        # Validate rbf_auto
        if rbf_auto is not None:
            if not (
                isinstance(rbf_auto, dict)
                and 'n_centers' in rbf_auto
                and isinstance(rbf_auto['n_centers'], int)
                and rbf_auto['n_centers'] >= 1
            ):
                raise ValueError("rbf_auto must be None or { 'n_centers': positive_int }.")
            self.rbf_auto = rbf_auto
        else:
            self.rbf_auto = None

    def create_features(self):
        """
        Run through feature_defs (and rbf_auto) and build all new features.

        Returns
        -------
        new_matrix : np.ndarray, shape (n_samples, total_new_cols)
        new_names  : list[str], names of each new column
        """
        new_cols = {}
        n_samples, n_features = self.df.shape

        # ------------------------------------------------------------
        # STEP 1: Global auto‐RBF over all original columns (if requested)
        # ------------------------------------------------------------
        if self.rbf_auto is not None:
            n_centers_global = self.rbf_auto['n_centers']
            centers_global, sigma_global = generate_sobol_centers(self.df, n_centers_global)
            distances_global = cdist(self.df, centers_global)  # shape = (n_samples, n_centers_global)
            for j in range(n_centers_global):
                new_cols[f"rbf_auto_{j}"] = np.exp(- (distances_global[:, j] ** 2) / (2 * sigma_global ** 2))

        # ------------------------------------------------------------
        # STEP 2: Loop over each feature definition
        # ------------------------------------------------------------
        for feat, definition in self.feature_defs.items():
            ftype = definition.get('type')

            # If ftype=='polynomial' with its own 'features' list, skip presence‐check of feat in feature_names.
            if not (ftype == 'polynomial' and 'features' in definition):
                if feat not in self.feature_names:
                    raise KeyError(f"Feature '{feat}' not found in feature_names.")

            # ===== POLYNOMIAL (possibly multivariate) =====
            if ftype == 'polynomial':
                degree = definition.get('degree')
                feats_list = definition.get('features', [feat])
                if not (isinstance(degree, int) and degree >= 1):
                    raise ValueError(f"'degree' must be a positive int for polynomial on '{feat}'.")
                if not isinstance(feats_list, list) or any(f not in self.feature_names for f in feats_list):
                    raise KeyError(f"'features' must be a list of valid columns for polynomial on '{feat}'.")

                # Build a NumPy submatrix: X_sub shape = (n_samples, len(feats_list))
                idxs = [self.feature_names.index(f) for f in feats_list]
                X_sub = self.df[:, idxs]  # shape = (n_samples, len(feats_list))

                # (A) Univariate: len(feats_list) == 1
                if len(feats_list) == 1:
                    col_data = X_sub[:, 0]  # 1D array
                    for d in range(1, degree + 1):
                        new_cols[f"{feats_list[0]}_pow{d}"] = col_data ** d

                # (B) Multivariate: len(feats_list) ≥ 2
                else:
                    for exponents in generate_exponent_vectors(len(feats_list), degree):
                        term_name_parts = []
                        term_array = np.ones(n_samples, dtype=float)
                        for var_idx, exp in enumerate(exponents):
                            if exp > 0:
                                var_name = feats_list[var_idx]
                                term_array *= (self.df[:, idxs[var_idx]] ** exp)
                                term_name_parts.append(f"{var_name}{exp}")
                        col_name = "_".join(term_name_parts)
                        new_cols[col_name] = term_array

            # ===== LOG =====
            elif ftype == 'log':
                idx = self.feature_names.index(feat)
                col_data = self.df[:, idx]
                if np.any(col_data <= 0):
                    raise ValueError(f"Non-positive values in '{feat}' for log transform.")
                new_cols[f"{feat}_log"] = np.log(col_data)

            # ===== HAT (triangular) =====
            elif ftype == 'hat':
                idx = self.feature_names.index(feat)
                col_data = self.df[:, idx]
                centers_list = definition.get('centers')
                width = definition.get('width')
                if centers_list is None or width is None:
                    raise ValueError(f"'centers' and 'width' required for hat on '{feat}'.")
                if not isinstance(centers_list, list) or not isinstance(width, (int, float)):
                    raise ValueError(f"Invalid types for 'hat' parameters on '{feat}'.")
                for c in centers_list:
                    new_cols[f"{feat}_hat_{c}"] = np.maximum(0.0, 1.0 - (np.abs(col_data - c) / width))

            # ===== RBF =====
            elif ftype == 'rbf':
                feats_to_use = definition.get('features', self.feature_names.copy())
                if not isinstance(feats_to_use, list) or any(f not in self.feature_names for f in feats_to_use):
                    raise KeyError(f"'features' must be a list of valid columns for rbf on '{feat}'.")
                idxs = [self.feature_names.index(f) for f in feats_to_use]
                X_sub = self.df[:, idxs]  # shape = (n_samples, len(feats_to_use))

                explicit_centers = definition.get('centers')
                sigma_val       = definition.get('sigma')
                n_centers_sub   = definition.get('n_centers')

                # (A) explicit centers + sigma
                if explicit_centers is not None and sigma_val is not None:
                    centers_arr = np.asarray(explicit_centers)
                    if centers_arr.ndim != 2 or centers_arr.shape[1] != X_sub.shape[1]:
                        raise ValueError(f"Invalid 'centers' shape for rbf on '{feat}'.")
                    sigma_use = float(sigma_val)

                # (B) automatic Sobol on that subspace
                elif explicit_centers is None and isinstance(n_centers_sub, int):
                    centers_arr, sigma_use = generate_sobol_centers(X_sub, n_centers_sub)

                else:
                    raise ValueError(
                        f"For rbf on '{feat}', supply either:\n"
                        f"  • 'centers' (list of length = n_centers_sub each) AND 'sigma',\n"
                        f"    OR\n"
                        f"  • 'n_centers' (int) alone."
                    )

                distances_sub = cdist(X_sub, centers_arr)  # shape = (n_samples, n_centers_sub)
                joined_feats  = "_".join(feats_to_use)
                for j in range(centers_arr.shape[0]):
                    new_cols[f"rbf_{joined_feats}_c{j}"] = np.exp(- (distances_sub[:, j] ** 2) / (2 * sigma_use ** 2))

            # ===== CUSTOM (now supports multiple features) =====
            elif ftype == 'custom':
                feats_list = definition.get('features', [feat])
                if not isinstance(feats_list, list) or any(f not in self.feature_names for f in feats_list):
                    raise KeyError(f"'features' must be a list of valid columns for custom on '{feat}'.")
                idxs = [self.feature_names.index(f) for f in feats_list]
                X_sub = self.df[:, idxs]  # shape = (n_samples, len(feats_list))

                funcs_list = definition.get('functions')
                if funcs_list is None or not isinstance(funcs_list, list):
                    raise ValueError(f"'functions' must be a list of dicts for custom on '{feat}'.")

                for fn_entry in funcs_list:
                    suffix = fn_entry.get('suffix')
                    # (A) Registry‐named function:
                    if 'fn' in fn_entry:
                        fn_name = fn_entry['fn']
                        if fn_name not in CUSTOM_FUNCTIONS:
                            raise KeyError(f"Custom function '{fn_name}' not in registry.")
                        func = CUSTOM_FUNCTIONS[fn_name]
                        # func expects a NumPy array (n_samples × n_feats_sub)
                        new_cols[f"{feat}_{suffix}"] = func(X_sub)

                    # (B) Lambda‐string: e.g. "lambda x, y: x + 2*y"
                    elif 'expr' in fn_entry:
                        expr_str = fn_entry['expr']
                        try:
                            func = eval(expr_str)   # e.g. a Python lambda
                        except Exception as e:
                            raise ValueError(f"Invalid lambda expression '{expr_str}' for custom on '{feat}': {e}")
                        # We call func on separate 1D arrays for each feature
                        arrays = [X_sub[:, i] for i in range(X_sub.shape[1])]
                        new_cols[f"{feat}_{suffix}"] = func(*arrays)

                    else:
                        raise ValueError(f"Each custom entry must have 'fn' or 'expr' for '{feat}'.")

            else:
                raise ValueError(f"Unsupported feature type '{ftype}' for '{feat}'.")

        # STEP 3: Stack all new columns into a NumPy matrix
        new_names = list(new_cols.keys())
        # Build a 2D array of shape (n_samples, len(new_cols))
        new_matrix = np.column_stack([new_cols[name] for name in new_names])
        return new_matrix, new_names

    # ------------------------------------------------------------------------
    # JSON‐Serialization Helpers
    # ------------------------------------------------------------------------
    def to_dict(self):
        """
        Return a JSON‐safe dict representing this generator’s configuration.
        Custom functions store 'fn' or 'expr' (strings), not raw callables.
        """
        serializable_defs = {}

        for feat, definition in self.feature_defs.items():
            ftype = definition['type']
            entry = {'type': ftype}

            if ftype == 'polynomial':
                entry['degree'] = definition['degree']
                if 'features' in definition:
                    entry['features'] = definition['features']

            elif ftype == 'log':
                pass

            elif ftype == 'hat':
                entry['centers'] = definition['centers']
                entry['width']   = definition['width']

            elif ftype == 'rbf':
                if 'features' in definition:
                    entry['features'] = definition['features']
                if 'centers' in definition and 'sigma' in definition:
                    entry['centers'] = definition['centers']
                    entry['sigma']   = definition['sigma']
                elif 'n_centers' in definition:
                    entry['n_centers'] = definition['n_centers']
                else:
                    raise ValueError(f"Cannot serialize rbf '{feat}'—missing parameters.")

            elif ftype == 'custom':
                if 'features' in definition:
                    entry['features'] = definition['features']
                funcs_list = definition.get('functions', [])
                serialized_funcs = []
                for fn_entry in funcs_list:
                    if 'fn' in fn_entry:
                        fn_name = fn_entry['fn']
                        if fn_name not in CUSTOM_FUNCTIONS:
                            raise KeyError(f"Function '{fn_name}' not in registry; cannot serialize.")
                        serialized_funcs.append({'fn': fn_name, 'suffix': fn_entry['suffix']})
                    elif 'expr' in fn_entry:
                        serialized_funcs.append({'expr': fn_entry['expr'], 'suffix': fn_entry['suffix']})
                    else:
                        raise ValueError(f"Invalid custom entry for '{feat}' during to_dict().")
                entry['functions'] = serialized_funcs

            else:
                raise ValueError(f"Cannot serialize feature type '{ftype}' for '{feat}'.")

            serializable_defs[feat] = entry

        return {
            'feature_defs': serializable_defs,
            'feature_names': self.feature_names,
            'rbf_auto': self.rbf_auto
        }

    def to_json(self, **json_kwargs):
        """
        Return a JSON string of this generator’s configuration.
        By default uses indent=2; override via json_kwargs.
        """
        return json.dumps(self.to_dict(), **json_kwargs)

    @classmethod
    def from_dict(cls, config_dict, feature_values: np.ndarray):
        """
        Reconstruct a FeatureGenerator from a JSON‐safe dict (as returned by to_dict()).
        Must pass `feature_values` (the data matrix) again.
        """
        reconstructed_defs = {}
        for feat, definition in config_dict['feature_defs'].items():
            ftype = definition['type']
            entry = {'type': ftype}

            if ftype == 'polynomial':
                entry['degree'] = definition['degree']
                if 'features' in definition:
                    entry['features'] = definition['features']

            elif ftype == 'log':
                pass

            elif ftype == 'hat':
                entry['centers'] = definition['centers']
                entry['width']   = definition['width']

            elif ftype == 'rbf':
                if 'features' in definition:
                    entry['features'] = definition['features']
                if 'centers' in definition and 'sigma' in definition:
                    entry['centers'] = definition['centers']
                    entry['sigma']   = definition['sigma']
                elif 'n_centers' in definition:
                    entry['n_centers'] = definition['n_centers']
                else:
                    raise ValueError(f"Invalid rbf definition for '{feat}' in from_dict().")

            elif ftype == 'custom':
                if 'features' in definition:
                    entry['features'] = definition['features']
                funcs_list = definition.get('functions', [])
                reconstructed_funcs = []
                for fn_entry in funcs_list:
                    if 'fn' in fn_entry:
                        fn_name = fn_entry['fn']
                        if fn_name not in CUSTOM_FUNCTIONS:
                            raise KeyError(f"Function '{fn_name}' not in registry.")
                        reconstructed_funcs.append({'fn': fn_name, 'suffix': fn_entry['suffix']})
                    elif 'expr' in fn_entry:
                        reconstructed_funcs.append({'expr': fn_entry['expr'], 'suffix': fn_entry['suffix']})
                    else:
                        raise ValueError(f"Invalid custom entry for '{feat}' in from_dict().")
                entry['functions'] = reconstructed_funcs

            else:
                raise ValueError(f"Unsupported feature type '{ftype}' for '{feat}' in from_dict().")

            reconstructed_defs[feat] = entry

        feature_names = config_dict['feature_names']
        rbf_auto = config_dict.get('rbf_auto')
        return cls(
            feature_defs=        reconstructed_defs,
            feature_names=       feature_names,
            feature_values=      feature_values,
            rbf_auto=            rbf_auto
        )

    @classmethod
    def from_json(cls, json_str, feature_values: np.ndarray):
        """
        Reconstruct a FeatureGenerator from a JSON string (as returned by to_json()).
        Must pass `feature_values` (the data matrix) again.
        """
        obj = json.loads(json_str)
        return cls.from_dict(obj, feature_values)


# --- Feature Engineering and Normalization for Options (and potentially other products) ---
def engineer_option_features(
    s0_values: np.ndarray, vol_values: np.ndarray, order: int = 2
) -> tuple[np.ndarray, list[str]]:
    """
    Engineers polynomial features for option pricing based on S0 and Vol.
    """
    s0, vol = np.asarray(s0_values), np.asarray(vol_values)
    if s0.shape != vol.shape or s0.ndim != 1:
        raise ValueError("Invalid s0/vol shapes. They must be 1D and of the same length.")
    
    features = [s0, vol]
    feature_names = ['S0_eng', 'Vol_eng']
    
    if order >= 2:
        features.extend([s0**2, vol**2, s0 * vol])
        feature_names.extend(['S0^2_eng', 'Vol^2_eng', 'S0*Vol_eng'])
    if order >= 3:
        features.extend([s0**3, vol**3, (s0**2)*vol, s0*(vol**2)])
        feature_names.extend(['S0^3_eng', 'Vol^3_eng', 'S0^2*Vol_eng', 'S0*Vol^2_eng'])
    if order >= 4:
        features.extend([s0**4, vol**4, (s0**3)*vol, s0*(vol**3), (s0**2)*(vol**2)])
        feature_names.extend(['S0^4_eng', 'Vol^4_eng', 'S0^3*Vol_eng', 'S0*Vol^3_eng', 'S0^2*Vol^2_eng'])
    
    return np.vstack(features).T, feature_names

def normalize_features(
    features: np.ndarray, means: np.ndarray = None, stds: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalizes features using mean and standard deviation.
    Calculates means and stds if not provided.
    """
    if features.ndim != 2:
        raise ValueError("Features must be a 2D array.")
    
    if means is None or stds is None:
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0)
        stds[stds < 1e-8] = 1.0  # Avoid division by zero or very small numbers
        
    return (features - means) / stds, means, stds