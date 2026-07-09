
__author__ = "Qihao Cheng"



"""Data processing utilities for atomic DFT calculations."""

import numpy as np
from typing import Optional, Dict, Any


# Valid smooth methods
VALID_SMOOTH_METHODS = ['lowpass', 'savgol', 'moving_avg', 'spline', 'gaussian', 'exp_weighted', 'cascade']

# Error messages
RHO_NOT_NDARRAY_ERROR = \
    "parameter 'rho' must be a numpy.ndarray, get type {} instead."
RHO_NOT_2D_ARRAY_ERROR = \
    "parameter 'rho' must be a 2D numpy.ndarray, get dimension {} instead."
VXC_TARGET_PHYSICAL_NOT_NDARRAY_ERROR = \
    "parameter 'vxc_target_physical' must be a numpy.ndarray, get type {} instead."
VXC_TARGET_PHYSICAL_NOT_2D_ARRAY_ERROR = \
    "parameter 'vxc_target_physical' must be a 2D numpy.ndarray, get dimension {} instead."
VXC_NOT_NDARRAY_ERROR = \
    "parameter 'vxc' must be a numpy.ndarray, get type {} instead."
R_NOT_NDARRAY_ERROR = \
    "parameter 'r' must be a numpy.ndarray, get type {} instead."
VXC_AND_R_NOT_SAME_LENGTH_ERROR = \
    "vxc and r must have the same length: {} vs {}."
R_THRESHOLD_NOT_FLOAT_ERROR = \
    "parameter 'r_threshold' must be a float, get type {} instead."
R_THRESHOLD_NOT_POSITIVE_ERROR = \
    "parameter 'r_threshold' must be positive, get {} instead."
METHOD_NOT_STRING_ERROR = \
    "parameter 'method' must be a string, get type {} instead."
METHOD_NOT_IN_VALID_LIST_ERROR = \
    "parameter 'method' must be in {}, get {} instead."
KWARGS_NOT_DICT_ERROR = \
    "parameter 'kwargs' must be a dict, get type {} instead."

UNKNOWN_SMOOTH_METHOD_ERROR = \
    "Unknown smoothing method: {}. Choose from: {}."


class DataProcessor:
    

    @staticmethod
    def smooth_vxc_data(
        vxc         : np.ndarray,
        r           : np.ndarray,
        r_threshold : float = 5.0, 
        method      : str   = 'savgol', 
        **kwargs
    ):
        """
        Smooth vxc values for r > r_threshold to reduce numerical instability and high-frequency oscillations.
        
        Common smoothing methods for filtering high-frequency noise:
        1. 'lowpass'      - Low-pass Butterworth filter (RECOMMENDED for high-frequency noise)
        2. 'savgol'       - Savitzky-Golay filter : a local polynomial regression-based smoothing method that reduces 
                            noise while preserving the shape, height, and position of signal features such as peaks, 
                            making it particularly suitable for spectroscopic and physical data analysis.

        3. 'moving_avg'   - Moving average (simple, good for high-frequency filtering)
        4. 'spline'       - Spline smoothing (controllable smoothness)
        5. 'gaussian'     - Gaussian filter (good smoothing, adjustable strength)
        6. 'exp_weighted' - Exponentially weighted moving average (smooth for large r)
        7. 'cascade'      - Apply multiple smoothing methods in sequence (strongest filtering)
        

        Parameters
        ----------
        vxc : np.ndarray
            vxc values to smooth
        r : np.ndarray
            Radius values (same length as vxc)
        r_threshold : float
            Threshold radius. Values with r > r_threshold will be smoothed.
        method : str
            Smoothing method: 'lowpass' , 'savgol'(default), 'moving_avg', 'spline', 'gaussian', 'exp_weighted', 'cascade'
        **kwargs : Optional[Dict]
            Additional parameters for smoothing methods:
            - lowpass: cutoff (default: 0.05), order (default: 6) - lower cutoff = stronger filtering
            - savgol: window_length (default: min(30% of data, len(data)//2*2+1)), polyorder (default: 2)
            - moving_avg: window_size (default: 25% of data, min 25) - larger = stronger filtering
            - spline: s (smoothing factor, default: len(data) * variance * 0.8) - larger = stronger
            - gaussian: sigma (default: max(2.0, 1% of data length)) - larger = stronger filtering
            - exp_weighted: alpha (default: 0.15) - smaller = stronger filtering
            - cascade: methods (list), kwargs_list (list of kwargs for each method)
        
        Returns
        -------
        vxc_smoothed : np.ndarray
            Smoothed vxc values (only r > r_threshold is smoothed)
        """

        # Convert to numpy arrays
        vxc = np.asarray(vxc).copy()
        r = np.asarray(r)

        # Check arguments
        assert isinstance(vxc, np.ndarray), \
            VXC_NOT_NDARRAY_ERROR.format(type(vxc))
        assert isinstance(r, np.ndarray), \
            R_NOT_NDARRAY_ERROR.format(type(r))
        assert len(vxc) == len(r), \
            VXC_AND_R_NOT_SAME_LENGTH_ERROR.format(len(vxc), len(r))
        assert isinstance(r_threshold, float), \
            R_THRESHOLD_NOT_FLOAT_ERROR.format(type(r_threshold))
        assert r_threshold > 0, \
            R_THRESHOLD_NOT_POSITIVE_ERROR.format(r_threshold)
        assert isinstance(method, str), \
            METHOD_NOT_STRING_ERROR.format(type(method))
        assert method in VALID_SMOOTH_METHODS, \
            METHOD_NOT_IN_VALID_LIST_ERROR.format(VALID_SMOOTH_METHODS, method)
        assert isinstance(kwargs, dict), \
            KWARGS_NOT_DICT_ERROR.format(type(kwargs))
        
        # Find indices where r > r_threshold
        large_r_mask = r > r_threshold
        
        if not np.any(large_r_mask):
            # No points to smooth
            return vxc
        
        # Extract data for smoothing
        vxc_large_r = vxc[large_r_mask]
        r_large_r   = r[large_r_mask]
        
        if len(vxc_large_r) < 3:
            # Not enough points to smooth
            return vxc
        
        # Apply smoothing based on method
        if method == 'lowpass':
            # Low-pass Butterworth filter - BEST for filtering high-frequency oscillations
            from scipy.signal import butter, filtfilt
            # Normalized cutoff frequency (0 < cutoff < 1, where 1 is Nyquist frequency)
            # Lower cutoff = stronger filtering (removes more high frequencies)
            cutoff = kwargs.get('cutoff', 0.05)  # Default: filter out frequencies > 5% of Nyquist (stronger)
            order = kwargs.get('order', 6)  # Filter order (higher = sharper cutoff, increased default)
            
            # Design Butterworth low-pass filter
            b, a = butter(order, cutoff, btype='low', analog=False)
            
            # Apply filter (filtfilt applies forward and backward for zero phase distortion)
            padlen = min(3 * max(len(a), len(b)), len(vxc_large_r) - 1)
            vxc_smoothed_large_r = filtfilt(
                b, a, vxc_large_r, padlen=padlen
            )
        
        elif method == 'savgol':
            from scipy.signal import savgol_filter
            # Use larger window and lower polyorder for better high-frequency filtering
            # Default: use up to 30% of data points for window (much larger for stronger filtering)
            default_window = min(int(len(vxc_large_r) * 0.3) // 2 * 2 + 1, len(vxc_large_r))
            default_window = max(default_window, 21)  # At least 21
            window_length = kwargs.get('window_length', default_window)
            polyorder = kwargs.get('polyorder', 2)  # Lower order for stronger smoothing
            # Ensure window_length is odd and <= len(data).
            window_length = min(window_length, len(vxc_large_r))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length < polyorder + 2:
                polyorder = max(1, window_length - 2)
            
            vxc_smoothed_large_r = savgol_filter(vxc_large_r, window_length, polyorder)
        
        elif method == 'moving_avg':
            # Much larger window for better high-frequency filtering
            # Default: use up to 25% of data points (much stronger filtering)
            default_window = max(int(len(vxc_large_r) * 0.25), 25)  # At least 25, up to 25% of data
            window_size = kwargs.get('window_size', default_window)
            window_size = min(window_size, len(vxc_large_r))
            # Use convolution for moving average
            kernel = np.ones(window_size) / window_size
            vxc_smoothed_large_r = np.convolve(vxc_large_r, kernel, mode='same')
            # Handle boundaries
            for i in range(window_size // 2):
                vxc_smoothed_large_r[i] = np.mean(vxc_large_r[:i+window_size//2+1])
                vxc_smoothed_large_r[-(i+1)] = np.mean(vxc_large_r[-(i+window_size//2+1):])
        
        elif method == 'spline':
            from scipy.interpolate import UnivariateSpline
            # Calculate smoothing factor based on data variance
            # Larger s = stronger smoothing (filters more high-frequency noise)
            data_variance = np.var(vxc_large_r)
            # Default: use 80% of variance (much stronger smoothing)
            s = kwargs.get('s', len(vxc_large_r) * data_variance * 0.8)
            spline = UnivariateSpline(
                r_large_r,
                vxc_large_r,
                s=s,
                k=min(3, len(vxc_large_r) - 1),
            )
            vxc_smoothed_large_r = spline(r_large_r)
        
        elif method == 'gaussian':
            from scipy.ndimage import gaussian_filter1d
            # Larger sigma = stronger smoothing (better for high-frequency filtering)
            # Default: adaptive sigma based on data length (stronger for longer sequences)
            default_sigma = max(2.0, len(vxc_large_r) * 0.01)  # At least 2.0, or 1% of data length
            sigma = kwargs.get('sigma', default_sigma)
            vxc_smoothed_large_r = gaussian_filter1d(vxc_large_r, sigma=sigma)
        
        elif method == 'exp_weighted':
            # Lower alpha = stronger smoothing (more weight on previous values)
            alpha = kwargs.get('alpha', 0.15)  # Further decreased for stronger filtering
            vxc_smoothed_large_r = np.zeros_like(vxc_large_r)
            vxc_smoothed_large_r[0] = vxc_large_r[0]
            for i in range(1, len(vxc_large_r)):
                vxc_smoothed_large_r[i] = alpha * vxc_large_r[i] + (1 - alpha) * vxc_smoothed_large_r[i-1]
        
        elif method == 'cascade':
            # Apply multiple smoothing methods in sequence for strongest filtering
            methods_list = kwargs.get('methods', ['lowpass', 'moving_avg'])
            # Default: stronger parameters for cascade
            default_window = max(int(len(vxc_large_r) * 0.2), 20)
            kwargs_list = kwargs.get('kwargs_list', [
                {'cutoff': 0.05, 'order': 6},  # Strong lowpass
                {'window_size': default_window}  # Large moving average
            ])
            if len(methods_list) != len(kwargs_list):
                raise ValueError("cascade methods and kwargs_list must have equal length")
            
            vxc_smoothed_large_r = vxc_large_r.copy()
            for m, kws in zip(methods_list, kwargs_list):
                # Apply each smoothing method directly (avoid recursion)
                if m == 'lowpass':
                    from scipy.signal import butter, filtfilt
                    cutoff = kws.get('cutoff', 0.05)
                    order = kws.get('order', 6)
                    b, a = butter(order, cutoff, btype='low', analog=False)
                    padlen = min(
                        3 * max(len(a), len(b)),
                        len(vxc_smoothed_large_r) - 1,
                    )
                    vxc_smoothed_large_r = filtfilt(
                        b, a, vxc_smoothed_large_r, padlen=padlen
                    )
                elif m == 'moving_avg':
                    window_size = kws.get('window_size', default_window)
                    window_size = min(window_size, len(vxc_smoothed_large_r))
                    kernel = np.ones(window_size) / window_size
                    vxc_smoothed_large_r = np.convolve(vxc_smoothed_large_r, kernel, mode='same')
                elif m == 'gaussian':
                    from scipy.ndimage import gaussian_filter1d
                    sigma = kws.get('sigma', max(2.0, len(vxc_smoothed_large_r) * 0.01))
                    vxc_smoothed_large_r = gaussian_filter1d(vxc_smoothed_large_r, sigma=sigma)
                elif m == 'savgol':
                    from scipy.signal import savgol_filter
                    window_length = kws.get(
                        'window_length',
                        min(21, len(vxc_smoothed_large_r)),
                    )
                    window_length = min(window_length, len(vxc_smoothed_large_r))
                    if window_length % 2 == 0:
                        window_length -= 1
                    polyorder = min(
                        kws.get('polyorder', 2),
                        window_length - 1,
                    )
                    vxc_smoothed_large_r = savgol_filter(
                        vxc_smoothed_large_r,
                        window_length,
                        polyorder,
                    )
                elif m == 'spline':
                    from scipy.interpolate import UnivariateSpline
                    smoothing = kws.get(
                        's',
                        len(vxc_smoothed_large_r)
                        * np.var(vxc_smoothed_large_r)
                        * 0.8,
                    )
                    spline = UnivariateSpline(
                        r_large_r,
                        vxc_smoothed_large_r,
                        s=smoothing,
                        k=min(3, len(vxc_smoothed_large_r) - 1),
                    )
                    vxc_smoothed_large_r = spline(r_large_r)
                elif m == 'exp_weighted':
                    alpha = kws.get('alpha', 0.15)
                    smoothed = np.empty_like(vxc_smoothed_large_r)
                    smoothed[0] = vxc_smoothed_large_r[0]
                    for i in range(1, len(smoothed)):
                        smoothed[i] = (
                            alpha * vxc_smoothed_large_r[i]
                            + (1 - alpha) * smoothed[i - 1]
                        )
                    vxc_smoothed_large_r = smoothed
                else:
                    raise ValueError(
                        UNKNOWN_SMOOTH_METHOD_ERROR.format(m, VALID_SMOOTH_METHODS)
                    )
        
        else:
            raise ValueError(UNKNOWN_SMOOTH_METHOD_ERROR.format(method, VALID_SMOOTH_METHODS))
        
        # Replace smoothed values
        vxc[large_r_mask] = vxc_smoothed_large_r
        
        return vxc



    @staticmethod
    def symlog(x, linthresh=0.002):
        """
        Symmetric logarithm transformation.
        
        Parameters
        ----------
        x : array-like
            Input values
        linthresh : float
            Linear threshold for symlog
        
        Returns
        -------
        Transformed values
        """
        x = np.asarray(x)
        return np.sign(x) * np.log1p(np.abs(x) / linthresh) * linthresh



    @staticmethod
    def symexp(y, linthresh=0.002):
        """
        Inverse of symlog (symmetric exponential).
        
        Parameters
        ----------
        y : array-like
            Symlog-transformed values
        linthresh : float
            Linear threshold (must match symlog)
        
        Returns
        -------
        Original values
        """
        y = np.asarray(y)
        return np.sign(y) * (np.expm1(np.abs(y) / linthresh)) * linthresh


    # Function to calculate correct weights for symlog-transformed targets
    @staticmethod
    def calculate_symlog_weights(rho, vxc_target_physical, linthresh, min_weight=1e-6):
        """
        Calculate weights for loss function when targets are symlog-transformed.
        
        The weight is designed to approximate the original rho-weighted loss in physical space:
        L_original = rho * |vxc_pred - vxc_true|
        
        IMPORTANT: This function uses the target Vxc (not delta_vxc) for weight calculation,
        as the weight should be based on the target potential magnitude.
        
        Parameters
        ----------
        rho : array-like
            Density values (physical space)
        vxc_target_physical : array-like
            Target vxc values in physical space (v_x + v_c)
            This is used instead of delta_vxc because weights should reflect
            the magnitude of the target potential
        linthresh : float
            Linear threshold for symlog (default: 0.001)
        
        Returns
        -------
        weights : array
            Weights for loss function
        """
        rho = np.asarray(rho)
        vxc_target_physical = np.asarray(vxc_target_physical)

        # Type check
        assert isinstance(rho, np.ndarray), \
            RHO_NOT_NDARRAY_ERROR.format(type(rho))
        assert isinstance(vxc_target_physical, np.ndarray), \
            VXC_TARGET_PHYSICAL_NOT_NDARRAY_ERROR.format(type(vxc_target_physical))

        # Shape check
        if rho.ndim == 1:
            rho = rho.reshape(-1, 1)
        if vxc_target_physical.ndim == 1:
            vxc_target_physical = vxc_target_physical.reshape(-1, 1)
        
        # Dimension check
        assert rho.ndim == 2, \
            RHO_NOT_2D_ARRAY_ERROR.format(type(rho))
        assert vxc_target_physical.ndim == 2, \
            VXC_TARGET_PHYSICAL_NOT_2D_ARRAY_ERROR.format(type(vxc_target_physical))


        # Calculate absolute value of symlog derivative using target vxc
        abs_vxc = np.abs(vxc_target_physical)
        
        # symlog'(vxc) = 1/linthresh if |vxc| <= linthresh, else sign(vxc)/|vxc|
        # |symlog'(vxc)| = 1/linthresh if |vxc| <= linthresh, else 1/|vxc|
        # Avoid divide-by-zero by only dividing on the valid branch.
        abs_symlog_derivative = np.empty_like(abs_vxc, dtype=float)
        small_mask = abs_vxc <= linthresh
        abs_symlog_derivative[small_mask] = 1.0 / linthresh
        abs_symlog_derivative[~small_mask] = 1.0 / abs_vxc[~small_mask]
        
        # Step 3: Calculate weight = rho / |symlog'(vxc)|
        # For |vxc| <= linthresh: weight = rho * linthresh
        # For |vxc| > linthresh: weight = rho * |vxc|
        weights = rho / abs_symlog_derivative
        
        # Step 4: Ensure weights are non-negative
        # Take absolute value of rho to ensure non-negative weights
        # (rho should already be non-negative, but this is a safety check)
        weights = np.abs(weights)
        
        # Additional safety: clip any negative or zero weights to a small positive value
        # This should not happen if rho is positive, but just in case
        weights = np.maximum(weights, min_weight)
        
        return weights



    @staticmethod
    def normalize_weights_by_atom(weights, atomic_numbers, min_weight_ratio=1e-2):
        """
        Normalize weights so that each atom contributes equally to the total loss.
        
        This is important because different atoms have different numbers of electrons
        and different numbers of data points. Without normalization, atoms with more
        data points or higher electron density would dominate the loss.
        
        The normalization ensures:
        - Each atom's total weighted contribution is equal
        - Atoms with more data points will have lower per-sample weights
        - Atoms with fewer data points will have higher per-sample weights
        - All weights are bounded below by max(weights) * min_weight_ratio to avoid numerical issues
        
        Parameters
        ----------
        weights : array-like
            Original weights for each sample
        atomic_numbers : array-like
            Atomic numbers for each sample (same length as weights)
        min_weight_ratio : float
            Minimum weight ratio relative to maximum weight (default: 1e-2).
            The minimum weight will be max(normalized_weights) * min_weight_ratio.
            All normalized weights will be at least this value to avoid numerical issues.
        
        Returns
        -------
        normalized_weights : array
            Normalized weights such that each atom contributes equally,
            with all weights >= max(normalized_weights) * min_weight_ratio
        """
        weights = np.asarray(weights)
        atomic_numbers = np.asarray(atomic_numbers)
        
        if len(weights) != len(atomic_numbers):
            raise ValueError(f"weights and atomic_numbers must have the same length: "
                            f"{len(weights)} vs {len(atomic_numbers)}")
        
        # Get unique atoms
        unique_atoms = np.unique(atomic_numbers)
        
        # Calculate total weight per atom
        atom_total_weights = {}
        atom_counts = {}
        for atom_z in unique_atoms:
            atom_mask = atomic_numbers == atom_z
            atom_total_weights[atom_z] = np.sum(weights[atom_mask])
            atom_counts[atom_z] = np.sum(atom_mask)
        
        # Calculate the target total weight per atom (use mean of all atoms)
        target_total_weight = np.mean(list(atom_total_weights.values()))
        
        # Normalize weights for each atom
        normalized_weights = weights.copy()
        for atom_z in unique_atoms:
            atom_mask = atomic_numbers == atom_z
            atom_total = atom_total_weights[atom_z]
            
            if atom_total > 0:
                # Scale weights so that this atom's total equals the target
                # normalized_weight = original_weight * (target_total / atom_total)
                normalized_weights[atom_mask] = weights[atom_mask] * (target_total_weight / atom_total)
            else:
                # If atom has zero total weight, set to uniform (shouldn't happen in practice)
                # Use a small positive value to avoid division issues
                normalized_weights[atom_mask] = target_total_weight / max(atom_counts[atom_z], 1)
        
        # Apply minimum weight bound after normalization
        # Minimum weight is max(normalized_weights) * min_weight_ratio
        max_weight = np.max(normalized_weights)
        min_weight = max_weight * min_weight_ratio
        normalized_weights = np.maximum(normalized_weights, min_weight)
        
        return normalized_weights



    @classmethod
    def inverse_transform_features(cls, X_transformed, scaler_X=None, transform_params=None, feature_idx=None):
        """
        Inverse transform features back to physical values.
        
        Parameters
        ----------
        X_transformed : array-like
            Transformed features (after scaling and/or symlog)
            Can be a single feature column or full feature matrix
        scaler_X : RobustScaler or None
            Scaler used for features (if any)
        transform_params : dict or None
            Transform parameters from prepare_data
        feature_idx : int or None
            If provided and X_transformed is a single column, specifies which feature index
            this column corresponds to (0-based). If None, assumes it's the first feature.
            
        Returns
        -------
        X_physical : array
            Features in physical space (same shape as input)
        """
        X = np.asarray(X_transformed).copy()
        original_shape = X.shape
        
        # Handle case where X is a single column but scaler expects full feature matrix
        if scaler_X is not None:
            # Check if X has fewer columns than scaler expects
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            n_features_input = X.shape[1]
            n_features_scaler = scaler_X.n_features_in_ if hasattr(scaler_X, 'n_features_in_') else None
            
            if n_features_scaler is not None and n_features_input < n_features_scaler:
                # Need to create a full feature matrix for inverse transform
                # Create a matrix with zeros for other features
                X_full = np.zeros((X.shape[0], n_features_scaler))
                if feature_idx is None:
                    feature_idx = 0
                X_full[:, feature_idx] = X[:, 0]
                X = X_full
            elif n_features_input == 1 and feature_idx is not None:
                # Single column with known index - create full matrix
                if n_features_scaler is not None:
                    X_full = np.zeros((X.shape[0], n_features_scaler))
                    X_full[:, feature_idx] = X[:, 0]
                    X = X_full
        
        # Step 1: Inverse scaling (if scaler was used)
        if scaler_X is not None:
            X = scaler_X.inverse_transform(X)
        
        # Step 2: Inverse symlog (if symlog was used)
        if transform_params is not None and transform_params.get('use_symlog_features', False):
            linthresh = transform_params.get('linthresh', 0.002)
            X = cls.symexp(X, linthresh=linthresh)
        
        # If we expanded to full matrix, extract only the requested feature
        if feature_idx is not None and X.shape[1] > 1:
            X = X[:, feature_idx:feature_idx+1]
        
        # Restore original shape if input was 1D
        if len(original_shape) == 1 and X.ndim == 2 and X.shape[1] == 1:
            X = X.flatten()
        
        return X



    @classmethod
    def inverse_transform_predictions(
        cls,
        y_pred           : np.ndarray,
        scaler_y         : Optional[Any] = None,
        transform_params : Optional[Dict] = None
    ) -> np.ndarray:
        """
        Inverse transform predictions back to physical space.
        
        Parameters
        ----------
        y_pred : np.ndarray
            Predictions (transformed space)
        scaler_y : Scaler or None
            Target scaler (if used)
        transform_params : dict or None
            Transform parameters
        
        Returns
        -------
        np.ndarray
            Predictions in physical space
        """
        y = np.asarray(y_pred).copy()
        
        # Step 1: Inverse scaling
        if scaler_y is not None:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y = scaler_y.inverse_transform(y)
        
        # Step 2: Inverse symlog
        if transform_params is not None and transform_params.get('use_symlog_targets', False):
            linthresh = transform_params.get('linthresh', 0.002)
            y = cls.symexp(y, linthresh=linthresh)
        
        return y.flatten() if y.ndim > 1 and y.shape[1] == 1 else y




