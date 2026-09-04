
from __future__ import annotations

__author__ = "Qihao Cheng"

import os
import traceback
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union, Literal, get_args, get_origin
from dataclasses import dataclass, field

# Import from sklearn
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
    ScalerType = Optional[Union[StandardScaler, RobustScaler]]

except ImportError:
    SKLEARN_AVAILABLE = False
    ScalerType = Any

# Error messages for AtomicDataset
DATA_ROOT_NOT_STRING_ERROR = \
    "parameter 'data_root' must be a string, get type {} instead."
DATA_ROOT_NOT_EXIST_ERROR = \
    "data root '{}' does not exist. Please create an empty directory if you want to generate data, or check if the directory is correct."
SCF_XC_FUNCTIONAL_NOT_STRING_ERROR = \
    "parameter 'scf_xc_functional' must be a string, get type {} instead."
SCF_XC_FUNCTIONAL_NOT_IN_VALID_LIST_ERROR = \
    "parameter 'scf_xc_functional' must be in {}, get {} instead."
FORWARD_PASS_XC_FUNCTIONAL_NOT_NONE_STR_OR_LIST_ERROR = \
    "parameter 'forward_pass_xc_functionals' must be None, string, or list of strings, get type {} instead."
FORWARD_PASS_XC_FUNCTIONAL_LIST_NOT_LIST_ERROR = \
    "parameter 'forward_pass_xc_functional_list' must be a list, get {} instead."
FORWARD_PASS_XC_FUNCTIONAL_LIST_NOT_LIST_OF_STRINGS_ERROR = \
    "parameter 'forward_pass_xc_functional_list' must be a list of strings, get {} instead."
FORWARD_PASS_XC_FUNCTIONAL_LIST_NOT_IN_VALID_LIST_ERROR = \
    "parameter 'forward_pass_xc_functional_list' must be in {}, get {} instead."
FEATURES_LIST_NOT_LIST_ERROR = \
    "parameter 'features_list' must be a list, get {} instead."
FEATURES_LIST_NOT_LIST_OF_STRINGS_ERROR = \
    "parameter 'features_list' must be a list of strings, get {} instead."
FEATURES_LIST_NOT_IN_VALID_LIST_ERROR = \
    "parameter 'features_list' must be in {}, get {} instead."


RADIUS_CUTOFF_RHO_THRESHOLD_NOT_FLOAT_ERROR = \
    "parameter 'radius_cutoff_rho_threshold' must be a float, get type {} instead."
RADIUS_CUTOFF_V_X_THRESHOLD_NOT_FLOAT_ERROR = \
    "parameter 'radius_cutoff_v_x_threshold' must be a float, get type {} instead."
RADIUS_CUTOFF_V_C_THRESHOLD_NOT_FLOAT_ERROR = \
    "parameter 'radius_cutoff_v_c_threshold' must be a float, get type {} instead."
RADIUS_CUTOFF_RHO_THRESHOLD_NOT_POSITIVE_ERROR = \
    "parameter 'radius_cutoff_rho_threshold' must be positive, get {} instead."
RADIUS_CUTOFF_V_X_THRESHOLD_NOT_POSITIVE_ERROR = \
    "parameter 'radius_cutoff_v_x_threshold' must be positive, get {} instead."
RADIUS_CUTOFF_V_C_THRESHOLD_NOT_POSITIVE_ERROR = \
    "parameter 'radius_cutoff_v_c_threshold' must be positive, get {} instead."

SMOOTH_RADIUS_THRESHOLD_NOT_FLOAT_ERROR = \
    "parameter 'smooth_radius_threshold' must be a float, get type {} instead."
SMOOTH_RADIUS_THRESHOLD_NOT_POSITIVE_ERROR = \
    "parameter 'smooth_radius_threshold' must be positive, get {} instead."
SMOOTH_METHOD_NOT_STRING_ERROR = \
    "parameter 'smooth_method' must be a string, get type {} instead."
SMOOTH_METHOD_NOT_IN_VALID_LIST_ERROR = \
    "parameter 'smooth_method' must be in {}, get {} instead."
SMOOTH_KWARGS_NOT_DICT_ERROR = \
    "parameter 'smooth_kwargs' must be a dict, get type {} instead."


CONFIGURATION_DATA_LIST_NOT_LIST_ERROR = \
    "parameter 'configuration_data_list' must be a list, get {} instead."
CONFIGURATION_DATA_LIST_NOT_LIST_OF_SINGLE_CONFIGURATION_DATA_ERROR = \
    "parameter 'configuration_data_list' must be a list of SingleConfigurationData, invalid element types: {}."

ATOMIC_NUMBER_NOT_INTEGER_ERROR = \
    "atomic number must be an integer, get {} instead."
ATOMIC_NUMBER_NOT_IN_DATASET_ERROR = \
    "atomic number {} is not in the dataset."
FUNCTIONAL_NOT_IN_DATASET_ERROR = \
    "functional '{}' is not in dataset: scf={}, forward_pass={}."

FEATURE_NAME_NOT_STRING_ERROR = \
    "parameter 'feature_name' must be a string, get {} instead."
FEATURE_NAME_NOT_IN_FEATURES_LIST_ERROR = \
    "parameter 'feature_name' must be in {}, get {} instead."

WEIGHTS_MODE_NOT_STRING_ERROR = \
    "parameter 'weights_mode' must be a string, get type {} instead."
WEIGHTS_MODE_NOT_IN_VALID_LIST_ERROR = \
    "parameter 'weights_mode' must be in {}, get {} instead."

TARGET_COMPONENT_NOT_PROVIDED_ERROR = \
    "parameter 'target_component' must be provided for symlog weights mode, get None instead."
TARGET_FUNCTIONAL_NOT_PROVIDED_ERROR = \
    "parameter 'target_functional' must be provided for symlog weights mode, get None instead."
LINTHRESH_TARGETS_NOT_PROVIDED_ERROR = \
    "parameter 'linthresh_targets' must be provided for symlog weights mode, get None instead."

# Error messages for DataManager
ATOMIC_NUMBER_LIST_NOT_LIST_ERROR = \
    "parameter 'atomic_number_list' must be a list, get {} instead."
ATOMIC_NUMBER_LIST_NOT_LIST_OF_INTEGERS_ERROR = \
    "parameter 'atomic_number_list' must be a list of integers, get {} instead."
ATOMIC_NUMBER_LIST_NOT_IN_VALID_RANGE_ERROR = \
    "parameter 'atomic_number_list' must be a list of integers in the range 1-92, get {} instead."

# Deprecated parameter warnings

ATOMIC_NUMBER_LIST_AND_CONFIGURATION_INDEX_LIST_BOTH_SPECIFIED_ERROR = \
    "Cannot specify both 'atomic_number_list' (deprecated) and 'configuration_index_list'. " \
    "Use 'configuration_index_list' only."
USE_RADIUS_CUTOFF_NOT_BOOL_ERROR = \
    "parameter 'use_radius_cutoff' must be a boolean, get {} instead."
USE_FEATURE_ROUND_OFF_NOT_BOOL_ERROR = \
    "parameter 'use_feature_round_off' must be a boolean, get {} instead."
SMOOTH_VXC_NOT_BOOL_ERROR = \
    "parameter 'smooth_vxc' must be a boolean, get {} instead."
CLOSE_SHELL_ONLY_NOT_BOOL_ERROR = \
    "parameter 'close_shell_only' must be a boolean, get {} instead."
INCLUDE_ENERGY_DENSITY_NOT_BOOL_ERROR = \
    "parameter 'include_energy_density' must be a boolean, get {} instead."
INCLUDE_INTERMEDIATE_NOT_BOOL_ERROR = \
    "parameter 'include_intermediate' must be a boolean, get {} instead."
PRINT_DEBUG_INFO_NOT_BOOL_ERROR = \
    "parameter 'print_debug_info' must be a boolean, get {} instead."
PRINT_SUMMARY_NOT_BOOL_ERROR = \
    "parameter 'print_summary' must be a boolean, get {} instead."

REFERENCE_FUNCTIONAL_NOT_STRING_ERROR = \
    "parameter 'reference_functional' must be a string, get {} instead."
REFERENCE_FUNCTIONAL_NOT_IN_DATASET_ERROR = \
    "parameter 'reference_functional' '{}' is not in dataset: scf={}, forward_pass={}."
TARGET_FUNCTIONAL_NOT_STRING_ERROR = \
    "parameter 'target_functional' must be a string, get {} instead."
TARGET_FUNCTIONAL_NOT_IN_DATASET_ERROR = \
    "parameter 'target_functional' '{}' is not in dataset: scf={}, forward_pass={}."
TARGET_COMPONENT_NOT_IN_VALID_LIST_ERROR = \
    "parameter 'target_component' must be in {}, get {} instead."

SKLEARN_NOT_AVAILABLE_FOR_DATA_PREPROCESSING_ERROR = \
    "sklearn is required for data preprocessing, but it is not available. To install sklearn, run 'pip install scikit-learn'."


# Warning messages
V_XC_IS_ALREADY_SMOOTHED_WARNING = \
    "V_xc is already smoothed for this dataset, skipping smoothing"
WEIGHTS_DATA_ALREADY_UPDATED_WARNING = \
    "Weights data is already updated for this dataset, skipping update"

# Deprecated arguments warning messages
USE_CUTOFF_DEPRECATED_WARNING = \
    "WARNING: parameter 'use_cutoff' is now deprecated, use parameter 'use_radius_cutoff' instead"
CUTOFF_RHO_THRESHOLD_DEPRECATED_WARNING = \
    "WARNING: parameter 'cutoff_rho_threshold' is now deprecated, use parameter 'radius_cutoff_rho_threshold' instead"
CUTOFF_V_X_THRESHOLD_DEPRECATED_WARNING = \
    "WARNING: parameter 'cutoff_v_x_threshold' is now deprecated, use parameter 'radius_cutoff_v_x_threshold' instead"
CUTOFF_V_C_THRESHOLD_DEPRECATED_WARNING = \
    "WARNING: parameter 'cutoff_v_c_threshold' is now deprecated, use parameter 'radius_cutoff_v_c_threshold' instead"
FINITE_ELEMENTS_DEPRECATED_WARNING = \
    "WARNING: parameter 'finite_elements' is now deprecated, use 'finite_elements_number' instead."
FINITE_ELEMENTS_NUMBER_AND_FINITE_ELEMENTS_BOTH_SPECIFIED_ERROR = \
    "Cannot specify both 'finite_elements_number' and deprecated 'finite_elements'. Use 'finite_elements_number' only."
ATOMIC_NUMBER_LIST_DEPRECATED_WARNING = \
    "WARNING: parameter 'atomic_number_list' is deprecated. Use 'configuration_index_list' instead. \n" \
    "\t Atomic numbers are now read from meta.json files, not from folder names."


VALID_TARGET_COMPONENTS = {"v_xc", "v_x", "v_c", "v_x_v_c"}
VALID_WEIGHTS_MODES     = {"symlog", "density"}


def format_error_message(e: Exception, context: str = "") -> Tuple[str, str]:
    """Format error message with type and enhanced context.

    Args:
        e: Exception instance
        context: Additional context string
        
    Returns:
        Tuple of (error_summary, full_traceback)
    """
    error_type = type(e).__name__
    error_msg = str(e)
    
    # Enhance KeyError messages
    if isinstance(e, KeyError):
        error_msg = f"Missing key '{error_msg}' (possibly a string formatting issue)"
    
    summary = f"[{error_type}] {error_msg}"
    if context:
        summary = f"{context}: {summary}"
    
    return summary, traceback.format_exc()


# Import other modules after format_error_message is defined
# (data_loading.py needs format_error_message, so it must be defined first)
from .data_generation import DataGenerator, XC_FUNCTIONAL_OEP_DEFAULT
from .data_loading import (
    DataLoader,
    SingleConfigurationData,
    FEATURE_ALIASES,
    NORMALIZED_VALID_FEATURES_LIST_FOR_POTENTIAL,
    format_invalid_feature_error,
)
from .data_processing import DataProcessor, VALID_SMOOTH_METHODS
from ..utils.occupation_states import OccupationInfo

# Type aliases
# (v_x, v_c, e_x, e_c), where e_x and e_c are only available if include_energy_density is True
XCDataType = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]


class TypeCheckMixin:
    
    @staticmethod
    def check_type(value, expected_type, name):
        def _raise_type_error():
            raise TypeError(
                f"Attribute {name} must be of type {expected_type}, get {type(value)} instead"
            )

        if expected_type is Any:
            return

        origin = get_origin(expected_type)
        if origin is None:
            if not isinstance(value, expected_type):
                _raise_type_error()
            return

        if origin is Union:
            valid_types = tuple(
                t for t in get_args(expected_type) if t is not type(None)
            )
            if value is None:
                return
            if valid_types and isinstance(value, valid_types):
                return
            _raise_type_error()

        if not isinstance(value, origin):
            _raise_type_error()

    @staticmethod
    def check_dim(value, expected_dim, name):
        if value.ndim != expected_dim:
            raise ValueError(f"Attribute {name} must have {expected_dim} dimensions, get {value.ndim} instead")
    
    @staticmethod
    def check_shape(value, expected_shape, name):
        if value.shape != expected_shape:
            raise ValueError(f"Attribute {name} must have shape {expected_shape}, get {value.shape} instead")
    

    @staticmethod
    def check_is_not_none(value, name, condition_name, condition_value):
        if value is None:
            raise ValueError(f"Attribute {name} must NOT be None when {condition_name} is {condition_value}, get {value} instead")


@dataclass
class VxcDataLoader(TypeCheckMixin):
    """
    Data class for V_xc data loader. Here, V_xc is the exchange-correlation potential.
    """

    # data attributes (transformed)
    X : np.ndarray  # (n_samples, n_features)
    y : np.ndarray  # (n_samples, n_targets)
    weights_for_training      : np.ndarray  # (n_samples, n_targets)
    atomic_numbers_per_sample : np.ndarray  # (n_samples,)
    
    # parameters for documentation
    target_functional      : str
    target_component       : str
    target_mode            : str
    reference_functional   : Optional[str]
    features_list          : List[str]

    # optional parameters for scaling
    scale_features         : bool = True
    scale_targets          : bool = True
    scaler_type_features   : str = 'robust'
    scaler_type_targets    : str = 'robust'
    scaler_kwargs_features : Dict[str, Any] = field(default_factory=dict)
    scaler_kwargs_targets  : Dict[str, Any] = field(default_factory=dict)
    scaler_X               : ScalerType = None
    scaler_y               : ScalerType = None

    # optional parameters for symlog transformation
    use_symlog_features    : bool = True
    use_symlog_targets     : bool = True
    linthresh_features     : Optional[float] = 0.002
    linthresh_targets      : Optional[float] = 0.002


    def __post_init__(self):
        # type checks
        self.check_type(self.X                         , np.ndarray      , "X")
        self.check_type(self.y                         , np.ndarray      , "y")
        self.check_type(self.weights_for_training      , np.ndarray      , "weights_for_training")
        self.check_type(self.atomic_numbers_per_sample , np.ndarray      , "atomic_numbers_per_sample")
        self.check_type(self.target_functional         , str             , "target_functional")
        self.check_type(self.target_component          , str             , "target_component")
        self.check_type(self.target_mode               , str             , "target_mode")
        self.check_type(self.reference_functional      , Optional[str]   , "reference_functional")
        self.check_type(self.features_list             , List[str]       , "features_list")
        self.check_type(self.scale_features            , bool            , "scale_features")
        self.check_type(self.scale_targets             , bool            , "scale_targets")
        self.check_type(self.scaler_type_features      , str             , "scaler_type_features")
        self.check_type(self.scaler_type_targets       , str             , "scaler_type_targets")
        self.check_type(self.scaler_kwargs_features    , Dict[str, Any]  , "scaler_kwargs_features")
        self.check_type(self.scaler_kwargs_targets     , Dict[str, Any]  , "scaler_kwargs_targets")
        self.check_type(self.scaler_X                  , ScalerType      , "scaler_X")
        self.check_type(self.scaler_y                  , ScalerType      , "scaler_y")
        self.check_type(self.use_symlog_features       , bool            , "use_symlog_features")
        self.check_type(self.use_symlog_targets        , bool            , "use_symlog_targets")
        self.check_type(self.linthresh_features        , Optional[float] , "linthresh_features")
        self.check_type(self.linthresh_targets         , Optional[float] , "linthresh_targets")

        # dimension checks
        self.check_dim(self.X, 2, "X")
        self.check_dim(self.y, 2, "y")
        self.check_dim(self.weights_for_training, 2, "weights_for_training")
        self.check_dim(self.atomic_numbers_per_sample, 1, "atomic_numbers_per_sample")

        # shape checks
        self.check_shape(self.X, (self.n_samples, self.n_features), "X")
        self.check_shape(self.y, (self.n_samples, self.n_targets) , "y")
        self.check_shape(self.weights_for_training, (self.n_samples, self.n_targets), "weights_for_training")
        self.check_shape(self.atomic_numbers_per_sample, (self.n_samples,), "atomic_numbers_per_sample")

        # value checks
        if self.scale_features:
            self.check_is_not_none(self.scaler_X, "scaler_X", "scale_features", True)
        if self.scale_targets:
            self.check_is_not_none(self.scaler_y, "scaler_y", "scale_targets", True)
        if self.use_symlog_features:
            self.check_is_not_none(self.linthresh_features, "linthresh_features", "use_symlog_features", True)
        if self.use_symlog_targets:
            self.check_is_not_none(self.linthresh_targets, "linthresh_targets", "use_symlog_targets", True)

        # other checks
        assert self.target_mode in ["absolute", "delta"]
        

    def print_info(self, label: str = "Vxc Data Loader"):
        """
        Print information about the V_xc data loader.
        """
        print(f"\n{'='*75}")
        print(f"{label} Summary".center(75))
        print(f"{'='*75}")
        print(f"Number of atoms            : {self.n_atoms}")
        print(f"Number of samples          : {self.n_samples}")
        print(f"Number of features         : {self.n_features}")
        print(f"Number of targets          : {self.n_targets}")
        print(f"Target functional          : {self.target_functional}")
        print(f"Target component           : {self.target_component}")
        print(f"Target mode                : {self.target_mode}")
        print(f"Reference functional       : {self.reference_functional}")
        print(f"Features in features list  : {len(self.features_list)} channels")
        for idx, feature in enumerate(self.features_list):
            suffix = " (repeated)" if feature in self.features_list[:idx] else ""
            print(f"    - Channel {idx + 1}: {feature}{suffix}")
        print()
        print(f"shape of X                 : Array of shape {self.X.shape}")
        print(f"shape of y                 : Array of shape {self.y.shape}")
        print(f"shape of weights           : Array of shape {self.weights_for_training.shape}")
        print(f"shape of atomic_numbers    : Array of shape {self.atomic_numbers_per_sample.shape}")
        print(f"{'='*75}")


    def get_features_data(self, feature_name_list: List[str]) -> np.ndarray:
        """
        Get features data for a given feature list.
        """
        for feature_name in feature_name_list:
            assert feature_name in self.features_list, \
                FEATURE_NAME_NOT_IN_FEATURES_LIST_ERROR.format(feature_name, self.features_list)
        
        feature_index_list = [self.features_list.index(feature_name) for feature_name in feature_name_list]
        return self.X[:, feature_index_list]


    @property
    def n_atoms(self) -> int:
        return len(np.unique(self.atomic_numbers_per_sample))

    @property
    def n_features(self) -> int:
        return len(self.features_list)

    @property
    def n_samples(self) -> int:
        return len(self.atomic_numbers_per_sample)

    @property
    def n_targets(self) -> int:
        return self.y.shape[1]

    @property
    def atomic_number_list(self) -> List[int]:
        return np.unique(self.atomic_numbers_per_sample).tolist()


    def split_data_by_atom(
        self,
        test_size           : float = 0.2,
        val_size            : float = 0.1,
        random_state        : int   = 42, 
        ensure_train_atoms  : Optional[List[int]] = None
    ) -> Tuple["VxcDataLoader", "VxcDataLoader", "VxcDataLoader"]:
        """
        Split data ensuring atoms don't leak between train/val/test sets.
        
        Parameters
        ----------
        test_size : float
            Proportion of atoms for test set
        val_size : float
            Proportion of atoms for validation set
        random_state : int
            Random seed
        ensure_train_atoms : list or None
            List of atomic numbers that must be in training set (e.g., [0, 1, 2, ..., 20])
        
        Returns
        -------
        train_loader, val_loader, test_loader : VxcDataLoader
            Data loaders split by atom without leakage.
        """
        atomic_numbers = self.atomic_numbers_per_sample
        unique_atoms = np.unique(atomic_numbers)
        
        # Ensure specified atoms are in training set
        if ensure_train_atoms is not None:
            ensure_train_atoms = np.array(ensure_train_atoms)
            # Find atoms that exist in the data and should be in training set
            atoms_guaranteed_train = np.intersect1d(unique_atoms, ensure_train_atoms)
            # Remaining atoms to split
            atoms_to_split = np.setdiff1d(unique_atoms, atoms_guaranteed_train)
            
            if len(atoms_guaranteed_train) > 0:
                print(f"Ensuring atoms {atoms_guaranteed_train.tolist()} are in training set")
        else:
            atoms_guaranteed_train = np.array([], dtype=int)
            atoms_to_split = unique_atoms
        
        # Split remaining atoms into train/val/test
        if len(atoms_to_split) > 0:
            atoms_train_temp, atoms_temp = train_test_split(
                atoms_to_split, test_size=(test_size + val_size), random_state=random_state
            )
            val_ratio = val_size / (test_size + val_size)
            atoms_val, atoms_test = train_test_split(
                atoms_temp, test_size=(1 - val_ratio), random_state=random_state
            )
            
            # Combine guaranteed training atoms with randomly split training atoms
            atoms_train = np.concatenate([atoms_guaranteed_train, atoms_train_temp])
        else:
            # All atoms are guaranteed to be in training set
            atoms_train = atoms_guaranteed_train
            atoms_val = np.array([], dtype=int)
            atoms_test = np.array([], dtype=int)
        
        # Create masks
        train_mask = np.isin(atomic_numbers, atoms_train)
        val_mask   = np.isin(atomic_numbers, atoms_val)
        test_mask  = np.isin(atomic_numbers, atoms_test)

        def _subset(mask: np.ndarray) -> "VxcDataLoader":
            return VxcDataLoader(
                X                         = self.X[mask],
                y                         = self.y[mask],
                weights_for_training      = self.weights_for_training[mask],
                atomic_numbers_per_sample = atomic_numbers[mask],
                target_functional         = self.target_functional,
                target_component          = self.target_component,
                target_mode               = self.target_mode,
                reference_functional      = self.reference_functional,
                features_list             = self.features_list,
                scale_features            = self.scale_features,
                scale_targets             = self.scale_targets,
                scaler_type_features      = self.scaler_type_features,
                scaler_type_targets       = self.scaler_type_targets,
                scaler_kwargs_features    = self.scaler_kwargs_features,
                scaler_kwargs_targets     = self.scaler_kwargs_targets,
                scaler_X                  = self.scaler_X,
                scaler_y                  = self.scaler_y,
                use_symlog_features       = self.use_symlog_features,
                use_symlog_targets        = self.use_symlog_targets,
                linthresh_features        = self.linthresh_features,
                linthresh_targets         = self.linthresh_targets,
            )

        return _subset(train_mask), _subset(val_mask), _subset(test_mask)



@dataclass
class ExcDataLoader(TypeCheckMixin):
    """
    Data class for Energy data loader. Here, Exc is the exchange-correlation energy density.
    """

    # data attributes (transformed)
    X : np.ndarray  # (n_samples, n_features)
    y : np.ndarray  # (n_samples, n_targets)
    weights_for_training      : np.ndarray  # (n_samples, n_targets)
    atomic_numbers_per_sample : np.ndarray  # (n_samples,)
    
    # parameters for documentation
    target_functional    : str
    target_component     : str
    target_mode          : str
    reference_functional : Optional[str]
    features_list        : List[str]

    # optional parameters for scaling
    scale_features       : bool = True
    scale_targets        : bool = True
    scaler_type_features : str = 'robust'
    scaler_type_targets  : str = 'robust'
    scaler_kwargs_features : Dict[str, Any] = field(default_factory=dict)
    scaler_kwargs_targets  : Dict[str, Any] = field(default_factory=dict)
    scaler_X             : ScalerType = None
    scaler_y             : ScalerType = None

    # optional parameters for symlog transformation
    use_symlog_features  : bool = True
    use_symlog_targets   : bool = True
    linthresh_features   : Optional[float] = 0.002
    linthresh_targets    : Optional[float] = 0.002

    # optional configuration-level data (for configuration-based training)
    configuration_ids_per_sample : Optional[np.ndarray] = None
    quadrature_nodes_per_sample  : Optional[np.ndarray] = None
    y_vxc                        : Optional[np.ndarray] = None
    weights_for_training_vxc     : Optional[np.ndarray] = None
    derivative_matrix_list       : Optional[List[np.ndarray]] = None
    laplacian_matrix_list        : Optional[List[np.ndarray]] = None


    def __post_init__(self):
        # type checks
        self.check_type(self.X                         , np.ndarray      , "X")
        self.check_type(self.y                         , np.ndarray      , "y")
        self.check_type(self.weights_for_training      , np.ndarray      , "weights_for_training")
        self.check_type(self.atomic_numbers_per_sample , np.ndarray      , "atomic_numbers_per_sample")
        self.check_type(self.target_functional         , str             , "target_functional")
        self.check_type(self.target_component          , str             , "target_component")
        self.check_type(self.target_mode               , str             , "target_mode")
        self.check_type(self.reference_functional      , Optional[str]   , "reference_functional")
        self.check_type(self.features_list             , List[str]       , "features_list")
        self.check_type(self.scale_features            , bool            , "scale_features")
        self.check_type(self.scale_targets             , bool            , "scale_targets")
        self.check_type(self.scaler_type_features      , str             , "scaler_type_features")
        self.check_type(self.scaler_type_targets       , str             , "scaler_type_targets")
        self.check_type(self.scaler_kwargs_features    , Dict[str, Any]  , "scaler_kwargs_features")
        self.check_type(self.scaler_kwargs_targets     , Dict[str, Any]  , "scaler_kwargs_targets")
        self.check_type(self.scaler_X                  , ScalerType      , "scaler_X")
        self.check_type(self.scaler_y                  , ScalerType      , "scaler_y")
        self.check_type(self.use_symlog_features       , bool            , "use_symlog_features")
        self.check_type(self.use_symlog_targets        , bool            , "use_symlog_targets")
        self.check_type(self.linthresh_features        , Optional[float] , "linthresh_features")
        self.check_type(self.linthresh_targets         , Optional[float] , "linthresh_targets")
        self.check_type(self.configuration_ids_per_sample, Optional[np.ndarray], "configuration_ids_per_sample")
        self.check_type(self.quadrature_nodes_per_sample , Optional[np.ndarray], "quadrature_nodes_per_sample")
        self.check_type(self.y_vxc                       , Optional[np.ndarray], "y_vxc")
        self.check_type(self.weights_for_training_vxc    , Optional[np.ndarray], "weights_for_training_vxc")
        self.check_type(self.derivative_matrix_list      , Optional[List[np.ndarray]], "derivative_matrix_list")
        self.check_type(self.laplacian_matrix_list       , Optional[List[np.ndarray]], "laplacian_matrix_list")

        # dimension checks
        self.check_dim(self.X, 2, "X")
        self.check_dim(self.y, 2, "y")
        self.check_dim(self.weights_for_training, 2, "weights_for_training")
        self.check_dim(self.atomic_numbers_per_sample, 1, "atomic_numbers_per_sample")
        if self.configuration_ids_per_sample is not None:
            self.check_dim(self.configuration_ids_per_sample, 1, "configuration_ids_per_sample")
        if self.quadrature_nodes_per_sample is not None:
            self.check_dim(self.quadrature_nodes_per_sample, 1, "quadrature_nodes_per_sample")
        if self.y_vxc is not None:
            self.check_dim(self.y_vxc, 2, "y_vxc")
        if self.weights_for_training_vxc is not None:
            self.check_dim(self.weights_for_training_vxc, 2, "weights_for_training_vxc")

        # shape checks
        self.check_shape(self.X, (self.n_samples, self.n_features), "X")
        self.check_shape(self.y, (self.n_samples, self.n_targets) , "y")
        self.check_shape(self.weights_for_training, (self.n_samples, self.n_targets), "weights_for_training")
        self.check_shape(self.atomic_numbers_per_sample, (self.n_samples,), "atomic_numbers_per_sample")
        if self.configuration_ids_per_sample is not None:
            self.check_shape(self.configuration_ids_per_sample, (self.n_samples,), "configuration_ids_per_sample")
        if self.quadrature_nodes_per_sample is not None:
            self.check_shape(self.quadrature_nodes_per_sample, (self.n_samples,), "quadrature_nodes_per_sample")
        if self.y_vxc is not None:
            self.check_shape(self.y_vxc, (self.n_samples, self.n_targets), "y_vxc")
        if self.weights_for_training_vxc is not None:
            self.check_shape(self.weights_for_training_vxc, (self.n_samples, self.n_targets), "weights_for_training_vxc")

        # value checks
        if self.scale_features:
            self.check_is_not_none(self.scaler_X, "scaler_X", "scale_features", True)
        if self.scale_targets:
            self.check_is_not_none(self.scaler_y, "scaler_y", "scale_targets", True)
        if self.use_symlog_features:
            self.check_is_not_none(self.linthresh_features, "linthresh_features", "use_symlog_features", True)
        if self.use_symlog_targets:
            self.check_is_not_none(self.linthresh_targets, "linthresh_targets", "use_symlog_targets", True)

        # other checks
        assert self.target_mode in ["absolute", "delta"]
        

    def print_info(self, label: str = "Vxc Data Loader"):
        """
        Print information about the V_xc data loader.
        """
        print(f"\n{'='*75}")
        print(f"{label} Summary".center(75))
        print(f"{'='*75}")
        print(f"Number of atoms            : {self.n_atoms}")
        print(f"Number of samples          : {self.n_samples}")
        print(f"Number of features         : {self.n_features}")
        print(f"Number of targets          : {self.n_targets}")
        print(f"Target functional          : {self.target_functional}")
        print(f"Target component           : {self.target_component}")
        print(f"Target mode                : {self.target_mode}")
        print(f"Reference functional       : {self.reference_functional}")
        print(f"Features in features list  : {len(self.features_list)} channels")
        for idx, feature in enumerate(self.features_list):
            suffix = " (repeated)" if feature in self.features_list[:idx] else ""
            print(f"    - Channel {idx + 1}: {feature}{suffix}")
        print()
        print(f"shape of X                 : Array of shape {self.X.shape}")
        print(f"shape of y                 : Array of shape {self.y.shape}")
        print(f"shape of weights           : Array of shape {self.weights_for_training.shape}")
        print(f"shape of atomic_numbers    : Array of shape {self.atomic_numbers_per_sample.shape}")
        print(f"{'='*75}")


    def get_features_data(self, feature_name_list: List[str]) -> np.ndarray:
        """
        Get features data for a given feature list.
        """
        for feature_name in feature_name_list:
            assert feature_name in self.features_list, \
                FEATURE_NAME_NOT_IN_FEATURES_LIST_ERROR.format(feature_name, self.features_list)
        
        feature_index_list = [self.features_list.index(feature_name) for feature_name in feature_name_list]
        return self.X[:, feature_index_list]


    @property
    def n_atoms(self) -> int:
        return len(np.unique(self.atomic_numbers_per_sample))

    @property
    def n_features(self) -> int:
        return len(self.features_list)

    @property
    def n_samples(self) -> int:
        return len(self.atomic_numbers_per_sample)

    @property
    def n_targets(self) -> int:
        return self.y.shape[1]

    @property
    def atomic_number_list(self) -> List[int]:
        return np.unique(self.atomic_numbers_per_sample).tolist()


    def split_data_by_atom(
        self,
        test_size           : float = 0.2,
        val_size            : float = 0.1,
        random_state        : int   = 42, 
        ensure_train_atoms  : Optional[List[int]] = None
    ) -> Tuple["VxcDataLoader", "VxcDataLoader", "VxcDataLoader"]:
        """
        Split data ensuring atoms don't leak between train/val/test sets.
        
        Parameters
        ----------
        test_size : float
            Proportion of atoms for test set
        val_size : float
            Proportion of atoms for validation set
        random_state : int
            Random seed
        ensure_train_atoms : list or None
            List of atomic numbers that must be in training set (e.g., [0, 1, 2, ..., 20])
        
        Returns
        -------
        train_loader, val_loader, test_loader : VxcDataLoader
            Data loaders split by atom without leakage.
        """
        atomic_numbers = self.atomic_numbers_per_sample
        unique_atoms = np.unique(atomic_numbers)
        
        # Ensure specified atoms are in training set
        if ensure_train_atoms is not None:
            ensure_train_atoms = np.array(ensure_train_atoms)
            # Find atoms that exist in the data and should be in training set
            atoms_guaranteed_train = np.intersect1d(unique_atoms, ensure_train_atoms)
            # Remaining atoms to split
            atoms_to_split = np.setdiff1d(unique_atoms, atoms_guaranteed_train)
            
            if len(atoms_guaranteed_train) > 0:
                print(f"Ensuring atoms {atoms_guaranteed_train.tolist()} are in training set")
        else:
            atoms_guaranteed_train = np.array([], dtype=int)
            atoms_to_split = unique_atoms
        
        # Split remaining atoms into train/val/test
        if len(atoms_to_split) > 0:
            atoms_train_temp, atoms_temp = train_test_split(
                atoms_to_split, test_size=(test_size + val_size), random_state=random_state
            )
            val_ratio = val_size / (test_size + val_size)
            atoms_val, atoms_test = train_test_split(
                atoms_temp, test_size=(1 - val_ratio), random_state=random_state
            )
            
            # Combine guaranteed training atoms with randomly split training atoms
            atoms_train = np.concatenate([atoms_guaranteed_train, atoms_train_temp])
        else:
            # All atoms are guaranteed to be in training set
            atoms_train = atoms_guaranteed_train
            atoms_val = np.array([], dtype=int)
            atoms_test = np.array([], dtype=int)
        
        # Create masks
        train_mask = np.isin(atomic_numbers, atoms_train)
        val_mask   = np.isin(atomic_numbers, atoms_val)
        test_mask  = np.isin(atomic_numbers, atoms_test)

        def _subset(mask: np.ndarray) -> "VxcDataLoader":
            return VxcDataLoader(
                X                         = self.X[mask],
                y                         = self.y[mask],
                weights_for_training      = self.weights_for_training[mask],
                atomic_numbers_per_sample = atomic_numbers[mask],
                target_functional         = self.target_functional,
                target_component          = self.target_component,
                target_mode               = self.target_mode,
                reference_functional      = self.reference_functional,
                features_list             = self.features_list,
                scale_features            = self.scale_features,
                scale_targets             = self.scale_targets,
                scaler_type_features      = self.scaler_type_features,
                scaler_type_targets       = self.scaler_type_targets,
                scaler_kwargs_features    = self.scaler_kwargs_features,
                scaler_kwargs_targets     = self.scaler_kwargs_targets,
                scaler_X                  = self.scaler_X,
                scaler_y                  = self.scaler_y,
                use_symlog_features       = self.use_symlog_features,
                use_symlog_targets        = self.use_symlog_targets,
                linthresh_features        = self.linthresh_features,
                linthresh_targets         = self.linthresh_targets,
            )

        return _subset(train_mask), _subset(val_mask), _subset(test_mask)


    def iter_configuration_batches(
        self,
        batch_size : int,
        shuffle    : bool = True,
        seed       : Optional[int] = None
    ):
        """
        Yield batches of configurations (not grid samples).
        Each batch is a list of dictionaries, one per configuration.
        """
        if self.configuration_ids_per_sample is None:
            raise ValueError("'configuration_ids_per_sample' must be provided for configuration batching.")
        if self.quadrature_nodes_per_sample is None:
            raise ValueError("'quadrature_nodes_per_sample' must be provided for configuration batching.")

        configuration_ids_unique = np.unique(self.configuration_ids_per_sample)
        configuration_ids = configuration_ids_unique.copy()
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(configuration_ids)

        config_id_to_index = None
        if self.derivative_matrix_list is not None or self.laplacian_matrix_list is not None:
            if self.derivative_matrix_list is not None and len(self.derivative_matrix_list) != len(configuration_ids_unique):
                raise ValueError("'derivative_matrix_list' length must match number of configurations.")
            if self.laplacian_matrix_list is not None and len(self.laplacian_matrix_list) != len(configuration_ids_unique):
                raise ValueError("'laplacian_matrix_list' length must match number of configurations.")
            config_id_to_index = {cid: idx for idx, cid in enumerate(configuration_ids_unique)}

        for start in range(0, len(configuration_ids), batch_size):
            batch_config_ids = configuration_ids[start:start + batch_size]
            batch = []
            for config_id in batch_config_ids:
                mask = self.configuration_ids_per_sample == config_id
                # keep grid order within configuration
                sort_idx = np.argsort(self.quadrature_nodes_per_sample[mask])
                indices = np.where(mask)[0][sort_idx]

                config_data = {
                    "configuration_id": int(config_id),
                    "X": self.X[indices],
                    "y_exc": self.y[indices],
                    "weights_exc": self.weights_for_training[indices],
                    "atomic_numbers": self.atomic_numbers_per_sample[indices],
                    "quadrature_nodes": self.quadrature_nodes_per_sample[indices],
                }
                if self.y_vxc is not None:
                    config_data["y_vxc"] = self.y_vxc[indices]
                if self.weights_for_training_vxc is not None:
                    config_data["weights_vxc"] = self.weights_for_training_vxc[indices]
                if self.derivative_matrix_list is not None and config_id_to_index is not None:
                    config_data["derivative_matrix"] = self.derivative_matrix_list[config_id_to_index[config_id]]
                if self.laplacian_matrix_list is not None and config_id_to_index is not None:
                    config_data["laplacian_matrix"] = self.laplacian_matrix_list[config_id_to_index[config_id]]

                batch.append(config_data)

            yield batch



@dataclass
class AtomicDataset:
    """
    Data class for atomic dataset.

    Parameters
    ----------
    data_root                       : str, Dataset root directory
    scf_xc_functional               : str, SCF XC functional
    forward_pass_xc_functional_list : List[str], Forward pass XC functional list
    features_list                   : List[str], Features list
    radius_cutoff_rho_threshold            : float, Radius cutoff rho threshold
    radius_cutoff_v_x_threshold            : float, Radius cutoff v_x threshold
    radius_cutoff_v_c_threshold            : float, Radius cutoff v_c threshold
    smooth_radius_threshold         : Optional[float], Smooth radius threshold
    smooth_method                   : Optional[str], Smooth method
    smooth_kwargs                   : Dict[str, Any], Smooth kwargs
    configuration_data_list         : List[SingleConfigurationData], Configuration data list
    shared_derivative_matrix        : Optional[np.ndarray], Shared derivative matrix stored at dataset root
    """

    # Basic attributes
    data_root                       : str
    scf_xc_functional               : str
    forward_pass_xc_functional_list : List[str]
    features_list                   : List[str]

    # Other attributes
    radius_cutoff_rho_threshold     : float           = 1e-6
    radius_cutoff_v_x_threshold     : float           = 1e-8
    radius_cutoff_v_c_threshold     : float           = 1e-8
    smooth_radius_threshold         : Optional[float] = None
    smooth_method                   : Optional[str]   = None
    smooth_kwargs                   : Dict[str, Any]  = field(default_factory=dict)

    # Data attributes
    configuration_data_list         : List[SingleConfigurationData] = field(default_factory=list)
    shared_derivative_matrix       : Optional[np.ndarray] = None  # Shared derivative matrix stored at dataset root


    def __post_init__(self):

        # Type and value checks
        AtomicDataManager.check_data_root(self.data_root)
        AtomicDataManager.check_features_list(self.features_list)
        AtomicDataManager.check_scf_xc_functional(self.scf_xc_functional)
        AtomicDataManager.check_forward_pass_xc_functional_list(self.forward_pass_xc_functional_list)
        
        AtomicDataManager.check_radius_cutoff_threshold(
            self.radius_cutoff_rho_threshold,
            self.radius_cutoff_v_x_threshold,
            self.radius_cutoff_v_c_threshold,
        )
        AtomicDataManager.check_configuration_data_list(self.configuration_data_list)


        # Initialize data attributes
        self.v_xc_is_smoothed        = False
        self.potential_weights_data_is_updated = False
        self.energy_weights_data_is_updated    = False
        self.include_energy_density  = self._get_include_energy_density()

        # Cached data attributes
        self.cached_atomic_numbers_unique            : Optional[np.ndarray] = None
        self.cached_atomic_numbers_per_configuration : Optional[np.ndarray] = None
        self.cached_atomic_numbers_per_sample        : Optional[np.ndarray] = None
    
        self.cached_cutoff_radii                     : Optional[np.ndarray] = None
        self.cached_cutoff_indices                   : Optional[np.ndarray] = None
        self.cached_quadrature_nodes                 : Optional[np.ndarray] = None
        self.cached_configuration_ids                : Optional[List[int]]  = None

        self.cached_features_data                    : Optional[np.ndarray] = None
        self.cached_scf_xc_data                      : Optional[XCDataType] = None
        self.cached_forward_pass_xc_data_list        : Optional[List[XCDataType]] = None


    def _get_include_energy_density(self) -> bool:
        """
        Check if energy density is included.
        """
        include_energy_density = True

        # Check if energy density is included in SCF XC data
        for configuration_data in self.configuration_data_list:
            if configuration_data.scf_xc_data[2] is None or configuration_data.scf_xc_data[3] is None:
                include_energy_density = False
                break
        
        # Check if energy density is included in forward pass XC data
        for configuration_data in self.configuration_data_list:
            for forward_pass_xc_data in configuration_data.forward_pass_xc_data_list:
                if forward_pass_xc_data[2] is None or forward_pass_xc_data[3] is None:
                    include_energy_density = False
                    break

        return include_energy_density



    def smooth_v_xc(
        self,
        smooth_radius_threshold : float = 5.0,
        smooth_method           : str   = 'savgol',
        smooth_kwargs           : Dict[str, Any] = field(default_factory=dict),
    ) -> None:
        """
        Smooth V_xc data.
        """
        # Type and value check
        AtomicDataManager.check_smooth_parameters(smooth_radius_threshold, smooth_method, smooth_kwargs)

        # Smooth V_xc data if not already smoothed
        if not self.v_xc_is_smoothed:

            # Update smoothing parameters
            self.smooth_radius_threshold = smooth_radius_threshold
            self.smooth_method           = smooth_method
            self.smooth_kwargs           = smooth_kwargs

            # Helper function to smooth XC data if needed
            def smooth_if_needed(v_x, v_c, quadrature_nodes):
                v_x = DataProcessor.smooth_vxc_data(v_x, quadrature_nodes, self.smooth_radius_threshold, self.smooth_method, *self.smooth_kwargs)
                v_c = DataProcessor.smooth_vxc_data(v_c, quadrature_nodes, self.smooth_radius_threshold, self.smooth_method, *self.smooth_kwargs)
                return v_x, v_c
            
            # Smooth each configuration data
            for configuration_data in self.configuration_data_list:
                # Smooth SCF XC data
                scf_v_x, scf_v_c = smooth_if_needed(configuration_data.scf_xc_data[0], configuration_data.scf_xc_data[1], configuration_data.quadrature_nodes)
                configuration_data.scf_xc_data = (scf_v_x, scf_v_c, configuration_data.scf_xc_data[2], configuration_data.scf_xc_data[3])
                
                # Smooth forward pass XC data
                for idx, forward_pass_xc_data in enumerate(configuration_data.forward_pass_xc_data_list):
                    forward_pass_v_x, forward_pass_v_c = smooth_if_needed(forward_pass_xc_data[0], forward_pass_xc_data[1], configuration_data.quadrature_nodes)
                    configuration_data.forward_pass_xc_data_list[idx] = (forward_pass_v_x, forward_pass_v_c, forward_pass_xc_data[2], forward_pass_xc_data[3])
            
            # Update the smoothed flag
            self.v_xc_is_smoothed = True

            # Clean the cached data since they are no longer valid
            self.cached_scf_xc_data = None
            self.cached_forward_pass_xc_data_list = None

            # Clear the weights data if they are updated before, since the weights data can depends on the smoothed XC data
            if self.potential_weights_data_is_updated or self.energy_weights_data_is_updated:
                for configuration_data in self.configuration_data_list:
                    configuration_data.clear_weights_data()
                self.potential_weights_data_is_updated = False
                self.energy_weights_data_is_updated = False

        else:
            print(V_XC_IS_ALREADY_SMOOTHED_WARNING)



    def print_info(self):
        """
        Print information about the atomic dataset.
        """

        print(f"{'='*75}")
        print(f"Loaded Data Summary".center(75))
        print(f"{'='*75}")
        
        print(f"Number of atoms                    : {self.n_atoms}")
        print(f"Number of configurations           : {self.n_configurations}")
        print(f"Total samples loaded               : {self.n_samples}")
        print(f"V_xc is smoothed                   : {self.v_xc_is_smoothed}")
        print(f"Include energy density             : {self.include_energy_density}")
        print(f"Cutoff radii range                 : [{np.min(self.cutoff_radii):.6f}, {np.max(self.cutoff_radii):.6f}]")
        print(f"SCF XC functional                  : {self.scf_xc_functional}")
        print(f"Forward pass XC functional list    : {self.forward_pass_xc_functional_list}")
        print(f"Features in features list          : {len(self.features_list)} channels")
        for idx, feature in enumerate(self.features_list):
            suffix = " (repeated)" if feature in self.features_list[:idx] else ""
            print(f"    - Channel {idx + 1}: {feature}{suffix}")

        print()
        print(f"radius_cutoff_rho_threshold        : {self.radius_cutoff_rho_threshold}")
        print(f"radius_cutoff_v_x_threshold        : {self.radius_cutoff_v_x_threshold}")
        print(f"radius_cutoff_v_c_threshold        : {self.radius_cutoff_v_c_threshold}")
        print(f"smooth_radius_threshold            : {self.smooth_radius_threshold}")
        print(f"smooth_method                      : {self.smooth_method}")
        print(f"smooth_kwargs                      : {self.smooth_kwargs}")
        print(f"v_xc_is_smoothed                   : {self.v_xc_is_smoothed}")
        print(f"potential_weights_data_is_updated  : {self.potential_weights_data_is_updated}")
        print(f"energy_weights_data_is_updated     : {self.energy_weights_data_is_updated}")
        print(f"include_energy_density             : {self.include_energy_density}")
        # Print derivative matrix information
        print(f"Shared derivative matrix exists    : {self.shared_derivative_matrix is not None}")
        if self.shared_derivative_matrix is not None:
            print(f"Shared derivative matrix shape     : {self.shared_derivative_matrix.shape}")


        print()
        print(f"shape of cutoff_radii              : Array of shape {self.cutoff_radii.shape}")
        print(f"shape of cutoff_indices            : Array of shape {self.cutoff_indices.shape}")
        print(f"shape of atomic_numbers_per_sample : Array of shape {self.atomic_numbers_per_sample.shape}")
        print(f"shape of quadrature_nodes          : Array of shape {self.quadrature_nodes.shape}")
        print(f"shape of configuration_ids         : List of {len(self.configuration_ids)} integers")
        print(f"shape of features_data             : {self.features_data.shape}")
        print(f"shape of scf_xc_data               : Tuple of 4 elements")
        print(f"    - v_x: Array of shape {self.scf_xc_data[0].shape}")
        print(f"    - v_c: Array of shape {self.scf_xc_data[1].shape}")
        print( "    - e_x: {}".format(f"Array of shape {self.scf_xc_data[2].shape}" if self.scf_xc_data[2] is not None else "None"))
        print( "    - e_c: {}".format(f"Array of shape {self.scf_xc_data[3].shape}" if self.scf_xc_data[3] is not None else "None"))
        print(f"shape of forward_pass_xc_data_list : List of {len(self.forward_pass_xc_data_list)} tuples, each with 4 elements")
        if len(self.forward_pass_xc_data_list) > 0:
            print(f"    - v_x: Array of shape {self.forward_pass_xc_data_list[0][0].shape}")
            print(f"    - v_c: Array of shape {self.forward_pass_xc_data_list[0][1].shape}")
            print( "    - e_x: {}".format(f"Array of shape {self.forward_pass_xc_data_list[0][2].shape}" if self.forward_pass_xc_data_list[0][2] is not None else "None"))
            print( "    - e_c: {}".format(f"Array of shape {self.forward_pass_xc_data_list[0][3].shape}" if self.forward_pass_xc_data_list[0][3] is not None else "None"))
        else:
            print("    - No forward pass XC data")
        

        
        print(f"{'='*75}")


    # Data preparation methods, for potential data
    def prepare_potential_dataloader(
        self,
        # required parameters
        target_functional      : str,
        target_component       : str,
        reference_functional   : Optional[str],

        # optional parameters for scaling
        scale_features         : bool = True,
        scale_targets          : bool = True,
        scaler_type_features   : str = 'robust',
        scaler_type_targets    : str = 'robust',
        scaler_kwargs_features : Optional[Dict[str, Any]] = None,
        scaler_kwargs_targets  : Optional[Dict[str, Any]] = None,

        # optional parameters for symlog transformation
        use_symlog_features    : bool = True,
        use_symlog_targets     : bool = True,
        linthresh_features     : Optional[float] = 0.002,
        linthresh_targets      : Optional[float] = 0.002,

        # optional parameters for weights calculation
        min_weight_ratio       : float = 1e-2,
    ) -> VxcDataLoader:
        """
        Prepare data with symlog transformation and scaling, for potential data.
        
        Parameters
        ----------
        target_functional : str
            Target XC functional used to construct training targets
        target_component : str
            Target component to learn: "v_xc", "v_x", "v_c", or "v_x_v_c"
        reference_functional : str or None
            Reference XC functional for delta learning. If None, use absolute targets
        scale_features : bool
            Whether to scale features
        scale_targets : bool
            Whether to scale targets
        use_symlog_features : bool
            Whether to apply symlog to features
        use_symlog_targets : bool
            Whether to apply symlog to targets
        linthresh : float
            Linear threshold for symlog
        scaler_type_features : str
            Type of feature scaler: 'robust' or 'standard'
        scaler_type_targets : str
            Type of target scaler: 'robust' or 'standard'
        use_radial_jacobian : bool
            Whether to use radial Jacobian for weights calculation, namely if the extra 
            factor of 4 * pi * r^2 should be included in the weights calculation.
        
        Returns
        -------
        VxcDataLoader
            Prepared data loader
        """

        # Type and value checks
        if not SKLEARN_AVAILABLE:
            raise ImportError(SKLEARN_NOT_AVAILABLE_FOR_DATA_PREPROCESSING_ERROR)

        # Check if target functional exists in the dataset
        if not isinstance(target_functional, str):
            raise ValueError(TARGET_FUNCTIONAL_NOT_STRING_ERROR.format(target_functional))
        if not self.exists_functional(target_functional):
            raise ValueError(TARGET_FUNCTIONAL_NOT_IN_DATASET_ERROR.format(target_functional, self.scf_xc_functional, self.forward_pass_xc_functional_list))
        
        # Check if reference functional exists in the dataset
        if reference_functional is not None:
            if not isinstance(reference_functional, str):
                raise ValueError(REFERENCE_FUNCTIONAL_NOT_STRING_ERROR.format(reference_functional))
            if not self.exists_functional(reference_functional):
                raise ValueError(REFERENCE_FUNCTIONAL_NOT_IN_DATASET_ERROR.format(reference_functional, self.scf_xc_functional, self.forward_pass_xc_functional_list))

        if scaler_kwargs_features is None:
            scaler_kwargs_features = {}
        if scaler_kwargs_targets is None:
            scaler_kwargs_targets = {}

        # Determine target mode
        if reference_functional is None:
            target_mode = "absolute"
        else:
            target_mode = "delta"


        # Extract features and targets
        X = self.features_data.copy()
        atomic_numbers = self.atomic_numbers_per_sample

        # Extract target component data
        target_component_data = self.extract_component(self.get_xc_data(target_functional), target_component)
        if target_mode == "delta":
            reference_component_data = self.extract_component(self.get_xc_data(reference_functional), target_component)
            y = target_component_data - reference_component_data
        else:
            y = target_component_data

        # Apply symlog to features if requested
        if use_symlog_features:
            X = DataProcessor.symlog(X, linthresh=linthresh_features)

        # Scale features if requested
        scaler_X = None
        if scale_features:
            if scaler_type_features == 'robust':
                scaler_X = RobustScaler(**scaler_kwargs_features)
            else:
                scaler_X = StandardScaler(**scaler_kwargs_features)
            X = scaler_X.fit_transform(X)

        # Apply symlog to targets if requested
        if use_symlog_targets:
            y = DataProcessor.symlog(y, linthresh=linthresh_targets)

        # Ensure 2D targets for downstream scalers
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Scale targets if requested
        scaler_y = None
        if scale_targets:
            if scaler_type_targets == 'robust':
                scaler_y = RobustScaler(**scaler_kwargs_targets)
            else:
                scaler_y = StandardScaler(**scaler_kwargs_targets)
            y = scaler_y.fit_transform(y)
        
        # calculate weights for training
        # Note: weights_mode is now use_normalization_factor (bool)
        # If use_symlog_targets is True, we use the new formula with normalization
        # If use_symlog_targets is False, we still use the new formula but with N_v^(c) = 1
        # Use quadrature_weights from each configuration (quadrature_weights_filtered)
        # update_potential_weights_data will automatically read them from configuration_data
        self.update_potential_weights_data(
            use_normalization_factor = use_symlog_targets,  # Use normalization when symlog is enabled
            min_weight_ratio         = min_weight_ratio,
            target_component         = target_component,
            target_functional        = target_functional,
            reference_functional     = reference_functional,
            linthresh_targets        = linthresh_targets,
            scaler_y                 = scaler_y,
        )
        weights_for_training = self.potential_weights_data

        # create dataloader
        dataloader = VxcDataLoader(
            # data attributes
            X = X,
            y = y,
            weights_for_training      = weights_for_training,
            atomic_numbers_per_sample = atomic_numbers,

            # parameters for documentation
            target_functional         = target_functional,
            target_component          = target_component,
            target_mode               = target_mode,
            reference_functional      = reference_functional,
            features_list             = self.features_list,

            # optional parameters for scaling
            scale_features            = scale_features,
            scale_targets             = scale_targets,
            scaler_type_features      = scaler_type_features,
            scaler_type_targets       = scaler_type_targets,
            scaler_kwargs_features    = scaler_kwargs_features,
            scaler_kwargs_targets     = scaler_kwargs_targets,
            scaler_X                  = scaler_X,
            scaler_y                  = scaler_y,

            # optional parameters for symlog transformation
            use_symlog_features       = use_symlog_features,
            use_symlog_targets        = use_symlog_targets,
            linthresh_features        = linthresh_features,
            linthresh_targets         = linthresh_targets,
        )

        return dataloader


    # Data preparation methods, for energy density data
    def prepare_energy_dataloader(
        self,
        # required parameters
        target_functional       : str,
        target_component        : str,
        reference_functional    : Optional[str],

        # optional parameters for scaling
        scale_features          : bool = True,
        scale_targets           : bool = True,
        scaler_type_features    : str = 'robust',
        scaler_type_potential   : str = 'robust',
        scaler_type_energy      : str = 'robust',
        scaler_kwargs_features  : Optional[Dict[str, Any]] = None,
        scaler_kwargs_potential : Optional[Dict[str, Any]] = None,
        scaler_kwargs_energy    : Optional[Dict[str, Any]] = None,

        # optional parameters for symlog transformation
        use_symlog_features     : bool = True,
        use_symlog_potential    : bool = True,
        use_symlog_energy       : bool = True,
        linthresh_features      : Optional[float] = 0.002,
        linthresh_potential     : Optional[float] = 0.002,
        linthresh_energy        : Optional[float] = 0.002,

        # optional parameters for weights calculation
        min_weight_ratio_potential    : float = 1e-2,
        min_weight_ratio_energy       : float = 1e-2,
        use_radial_jacobian_potential : bool = True,
        use_radial_jacobian_energy    : bool = True,
    ) -> ExcDataLoader:
        """
        Prepare data with symlog transformation and scaling, for energy density data.
        
        Parameters
        ----------
        target_functional : str
            Target XC functional used to construct training targets
        target_component : str
            Target component to learn: "v_xc", "v_x", "v_c", or "v_x_v_c"
        reference_functional : str or None
            Reference XC functional for delta learning. If None, use absolute targets
        scale_features : bool
            Whether to scale features
        scale_targets : bool
            Whether to scale targets
        use_symlog_features : bool
            Whether to apply symlog to features
        use_symlog_targets : bool
            Whether to apply symlog to targets
        linthresh : float
            Linear threshold for symlog
        scaler_type_features : str
            Type of feature scaler: 'robust' or 'standard'
        scaler_type_targets : str
            Type of target scaler: 'robust' or 'standard'
        use_radial_jacobian : bool
            Whether to use radial Jacobian for weights calculation, namely if the extra 
            factor of 4 * pi * r^2 should be included in the weights calculation.
        
        Returns
        -------
        VxcDataLoader
            Prepared data loader
        """

        # Type and value checks
        if not SKLEARN_AVAILABLE:
            raise ImportError(SKLEARN_NOT_AVAILABLE_FOR_DATA_PREPROCESSING_ERROR)

        # Check if target functional exists in the dataset
        if not isinstance(target_functional, str):
            raise ValueError(TARGET_FUNCTIONAL_NOT_STRING_ERROR.format(target_functional))
        if not self.exists_functional(target_functional):
            raise ValueError(TARGET_FUNCTIONAL_NOT_IN_DATASET_ERROR.format(target_functional, self.scf_xc_functional, self.forward_pass_xc_functional_list))
        
        # Check if reference functional exists in the dataset
        if reference_functional is not None:
            if not isinstance(reference_functional, str):
                raise ValueError(REFERENCE_FUNCTIONAL_NOT_STRING_ERROR.format(reference_functional))
            if not self.exists_functional(reference_functional):
                raise ValueError(REFERENCE_FUNCTIONAL_NOT_IN_DATASET_ERROR.format(reference_functional, self.scf_xc_functional, self.forward_pass_xc_functional_list))

        if scaler_kwargs_features is None:
            scaler_kwargs_features = {}
        if scaler_kwargs_potential is None:
            scaler_kwargs_potential = {}
        if scaler_kwargs_energy is None:
            scaler_kwargs_energy = {}

        # Determine target mode
        if reference_functional is None:
            target_mode = "absolute"
        else:
            target_mode = "delta"


        # Extract features and targets
        X = self.features_data.copy()
        atomic_numbers = self.atomic_numbers_per_sample

        # Extract target component data
        target_component_data = self.extract_component(self.get_xc_data(target_functional), target_component)
        if target_mode == "delta":
            reference_component_data = self.extract_component(self.get_xc_data(reference_functional), target_component)
            y = target_component_data - reference_component_data
        else:
            y = target_component_data

        # Apply symlog to features if requested
        if use_symlog_features:
            X = DataProcessor.symlog(X, linthresh=linthresh_features)

        # Scale features if requested
        scaler_X = None
        if scale_features:
            if scaler_type_features == 'robust':
                scaler_X = RobustScaler(**scaler_kwargs_features)
            else:
                scaler_X = StandardScaler(**scaler_kwargs_features)
            X = scaler_X.fit_transform(X)

        # Apply symlog to targets if requested
        if use_symlog_energy:
            y = DataProcessor.symlog(y, linthresh=linthresh_energy)

        # Ensure 2D targets for downstream scalers
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Scale targets if requested
        scaler_y = None
        if scale_targets:
            if scaler_type_energy == 'robust':
                scaler_y = RobustScaler(**scaler_kwargs_energy)
            else:
                scaler_y = StandardScaler(**scaler_kwargs_energy)
            y = scaler_y.fit_transform(y)
        
        # For potential weights, we need to prepare potential data first to get the potential scaler
        # This is needed to get σ_v for the weights calculation
        # Prepare potential data temporarily to get potential scaler
        potential_target_component_data = self.extract_component(self.get_xc_data(target_functional), target_component)
        if target_mode == "delta":
            potential_reference_component_data = self.extract_component(self.get_xc_data(reference_functional), target_component)
            y_potential = potential_target_component_data - potential_reference_component_data
        else:
            y_potential = potential_target_component_data
        
        # Apply symlog to potential targets if requested
        if use_symlog_potential:
            y_potential = DataProcessor.symlog(y_potential, linthresh=linthresh_potential)
        
        # Ensure 2D for scaler
        if y_potential.ndim == 1:
            y_potential = y_potential.reshape(-1, 1)
        
        # Create potential scaler
        scaler_y_potential = None
        if scaler_type_potential == 'robust':
            scaler_y_potential = RobustScaler(**scaler_kwargs_potential)
        else:
            scaler_y_potential = StandardScaler(**scaler_kwargs_potential)
        scaler_y_potential.fit(y_potential)  # Fit but don't transform, we just need the scaler
        
        # calculate weights for training
        # Note: weights_mode is now use_normalization_factor (bool)
        # If use_symlog_potential is True, we use the new formula with normalization
        # If use_symlog_potential is False, we still use the new formula but with N_v^(c) = 1
        self.update_potential_weights_data(
            use_normalization_factor = use_symlog_potential,  # Use normalization when symlog is enabled
            min_weight_ratio         = min_weight_ratio_potential,
            target_component         = target_component,
            target_functional        = target_functional,
            reference_functional     = reference_functional,
            linthresh_targets        = linthresh_potential,
            scaler_y                 = scaler_y_potential,  # Use potential scaler to get σ_v
        )
        weights_for_training = self.potential_weights_data

        # create dataloader
        dataloader = ExcDataLoader(
            # data attributes
            X = X,
            y = y,
            weights_for_training = weights_for_training,
            atomic_numbers_per_sample = atomic_numbers,

            # parameters for documentation
            target_functional = target_functional,
            target_component = target_component,
            target_mode = target_mode,
            reference_functional = reference_functional,
            features_list = self.features_list,

            # optional parameters for scaling
            scale_features = scale_features,
            scale_targets  = scale_targets,
            scaler_type_features = scaler_type_features,
            scaler_type_targets  = scaler_type_energy,
            scaler_kwargs_features = scaler_kwargs_features,
            scaler_kwargs_targets  = scaler_kwargs_energy,
            scaler_X = scaler_X,
            scaler_y = scaler_y,

            # optional parameters for symlog transformation
            use_symlog_features = use_symlog_features,
            use_symlog_targets  = use_symlog_energy,
            linthresh_features  = linthresh_features,
            linthresh_targets   = linthresh_energy,
        )

        return dataloader


    def update_potential_weights_data(
        self, 
        use_normalization_factor : bool = True,  # If True, use N_v^(c) normalization; if False, N_v^(c) = 1
        min_weight_ratio         : float = 1e-3,
        target_component         : Optional[str] = None,
        target_functional        : Optional[str] = None,
        reference_functional     : Optional[str] = None,
        linthresh_targets        : Optional[float] = None,
        scaler_y                 : ScalerType = None,
    ):
        """
        Update the potential weights data according to the new formula:
        
        weights = (4πr²σ_v w(r)ρ(r)N_v^(c)) / |Symlog'[Δv_xc^(c)(r)]|
        
        where:
        - If use_normalization_factor=True: N_v^(c) = [∑ᵣᵢ (4πrᵢ²σᵥw(rᵢ)ρ(rᵢ))/|Symlog'[Δv_xc^(c)(rᵢ)]|]⁻¹
        - If use_normalization_factor=False: N_v^(c) = 1
        
        Parameters
        ----------
        use_normalization_factor : bool
            Whether to use normalization factor N_v^(c).
            If True, all configurations have the same importance.
            If False, all grid points have the same importance.

        min_weight_ratio : float
            Minimum weight ratio for normalization
        target_component : str
            Target component: "v_xc", "v_x", "v_c", or "v_x_v_c"
        target_functional : str
            Target XC functional
        reference_functional : str, optional
            Reference XC functional for delta learning
        linthresh_targets : float
            Linear threshold for symlog transformation
        scaler_y : StandardScaler or RobustScaler, optional
            Scaler used for targets. Required to get σ_v (standard deviation).
            If None, σ_v will be set to 1.0.
        """
        # Check if necessary parameters are provided
        assert target_component is not None, \
            TARGET_COMPONENT_NOT_PROVIDED_ERROR.format("symlog")
        assert target_functional is not None, \
            TARGET_FUNCTIONAL_NOT_PROVIDED_ERROR.format("symlog")
        assert linthresh_targets is not None, \
            LINTHRESH_TARGETS_NOT_PROVIDED_ERROR.format("symlog")

        # check if weights data is already updated
        if self.potential_weights_data_is_updated:
            print(WEIGHTS_DATA_ALREADY_UPDATED_WARNING)
            return

        # Get σ_v from scaler_y
        sigma_v = 1.0

        if scaler_y is not None:
            if hasattr(scaler_y, 'scale_'):
                # StandardScaler or RobustScaler
                sigma_v = scaler_y.scale_
                # Convert to scalar if it's an array
                if isinstance(sigma_v, np.ndarray):
                    if sigma_v.ndim > 0:
                        sigma_v = float(sigma_v[0])  # Take first component if multi-dimensional
                    else:
                        sigma_v = float(sigma_v)
                else:
                    sigma_v = float(sigma_v)
        

        # Compute weights data for each configuration
        raw_weights_data_list = []
        normalization_sum_list = []  # For computing N_v^(c) if needed

        for idx, configuration_data in enumerate(self.configuration_data_list):
            rho = configuration_data.rho
            quadrature_nodes = configuration_data.quadrature_nodes_filtered.reshape(-1, 1)
            
            # Get quadrature weights for this configuration
            # Default behavior: use quadrature_weights_filtered from configuration_data
            if hasattr(configuration_data, "quadrature_weights_filtered"):
                w = configuration_data.quadrature_weights_filtered.reshape(-1, 1)
            else:
                # Fallback: uniform weights if quadrature weights are not available
                w = np.ones_like(quadrature_nodes)
            
            # Compute Δv_xc^(c)(r)
            v_xc_target_physical = self.extract_component(
                self._get_xc_data_from_single_configuration_data(configuration_data, target_functional), 
                target_component
            )
            if reference_functional is not None:
                v_xc_reference_physical = self.extract_component(
                    self._get_xc_data_from_single_configuration_data(configuration_data, reference_functional), 
                    target_component
                )
                delta_v_xc = v_xc_target_physical - v_xc_reference_physical
            else:
                delta_v_xc = v_xc_target_physical
            
            # Ensure 2D shape
            if delta_v_xc.ndim == 1:
                delta_v_xc = delta_v_xc.reshape(-1, 1)
            if rho.ndim == 1:
                rho = rho.reshape(-1, 1)
            
            # Calculate |Symlog'[Δv_xc^(c)(r)]|
            abs_delta_v_xc = np.abs(delta_v_xc)
            abs_symlog_derivative = np.empty_like(abs_delta_v_xc, dtype=float)
            small_mask = abs_delta_v_xc <= linthresh_targets
            abs_symlog_derivative[small_mask] = 1.0 / linthresh_targets
            abs_symlog_derivative[~small_mask] = 1.0 / abs_delta_v_xc[~small_mask]
            
            # Calculate numerator: 4πr²σ_v w(r)ρ(r)
            numerator = 4 * np.pi * quadrature_nodes**2 * sigma_v * w * rho
            
            # Calculate weights: numerator / |Symlog'[Δv_xc^(c)(r)]|
            weights_data = numerator / abs_symlog_derivative
            
            # Ensure non-negative (allow very small values; actual lower bound set globally later)
            weights_data = np.maximum(weights_data, 0.0)
            
            raw_weights_data_list.append(weights_data)
            
            # If using normalization, compute the sum for N_v^(c)
            if use_normalization_factor:
                normalization_sum = np.sum(numerator / abs_symlog_derivative)
                normalization_sum_list.append(normalization_sum)
            else:
                normalization_sum_list.append(None)

        # Apply normalization factor N_v^(c) if needed
        if use_normalization_factor:
            normalized_weights_data_list = []
            for weights_data, norm_sum in zip(raw_weights_data_list, normalization_sum_list):
                if norm_sum is not None and norm_sum > 0:
                    N_v_c = 1.0 / norm_sum
                    normalized_weights = weights_data * N_v_c
                else:
                    normalized_weights = weights_data
                normalized_weights_data_list.append(normalized_weights)
        else:
            normalized_weights_data_list = raw_weights_data_list

        # Apply minimum weight ratio constraint (global lower bound across all configurations)
        final_weights_data_list = []
        if len(normalized_weights_data_list) > 0:
            # Compute global maximum weight over all configurations
            global_max_weight = max(
                float(np.max(normalized_weights_data))
                for normalized_weights_data in normalized_weights_data_list
            )
            # Global lower bound controlled by min_weight_ratio
            tiny = 1e-12  # purely for numerical safety
            global_min_weight = max(min_weight_ratio * global_max_weight, tiny)
            
            # Debug: Print global bounds info
            if len(normalized_weights_data_list) > 1:
                print(f"Global weight bounds: min={global_min_weight:.6e}, max={global_max_weight:.6e}, ratio={min_weight_ratio:.6e}")
        else:
            global_min_weight = 0.0

        for normalized_weights_data in normalized_weights_data_list:
            # Apply global minimum weight constraint
            final_weights = np.maximum(
                normalized_weights_data,
                global_min_weight
            )
            final_weights_data_list.append(final_weights)

        # set the weights data
        for configuration_data, final_weights_data in \
            zip(self.configuration_data_list, final_weights_data_list):
            configuration_data.set_potential_weights_data(final_weights_data)

        # update the weights data is updated flag
        self.potential_weights_data_is_updated = True


    @staticmethod
    def extract_component(xc_data: XCDataType, target_component: str) -> np.ndarray:
        """
        Extract the component of the XC data.
        """
        v_x, v_c, _, _ = xc_data
        if target_component == "v_x":
            return v_x
        elif target_component == "v_c":
            return v_c
        elif target_component == "v_xc":
            return v_x + v_c
        elif target_component == "v_x_v_c":
            return np.column_stack([v_x, v_c])
        else:
            raise ValueError(TARGET_COMPONENT_NOT_IN_VALID_LIST_ERROR.format(target_component, VALID_TARGET_COMPONENTS))


    @staticmethod
    def calculate_symlog_weights(
        rho                 : np.ndarray,
        vxc_target_physical : np.ndarray,
        linthresh           : float,
        min_weight          : float = 1e-6
    ) -> np.ndarray:
        """
        Calculate symlog-based weights from density and target Vxc values.
        """
        return DataProcessor.calculate_symlog_weights(rho, vxc_target_physical, linthresh, min_weight)


    @staticmethod
    def normalize_weights_by_atom(
        weights          : np.ndarray,
        atomic_numbers   : np.ndarray,
        min_weight_ratio : float = 1e-2
    ) -> np.ndarray:
        """
        Normalize weights per atom to keep relative scales comparable.
        """
        raise NotImplementedError("This method is deprecated. Use method self.update_potential_weights_data() instead.")
        
        return DataProcessor.normalize_weights_by_atom(weights, atomic_numbers, min_weight_ratio)


    @staticmethod
    def normalize_weights_by_configuration(
        raw_weights_data_list         : List[np.ndarray],
        weights_sum_per_configuration : float = 100.0,
        min_weight_ratio              : float = 1e-2,
    ) -> List[np.ndarray]:
        """
        Normalize weights per configuration to keep relative scales comparable.
        Here, we let the weights for each configuration to have the same sum.
        """
        normalized_weights_data_list = []

        for raw_weights_data in raw_weights_data_list:
            # Let the weights for each configuration to have the same sum.
            normalized_weights_data = raw_weights_data / np.sum(raw_weights_data) * weights_sum_per_configuration

            # apply minimum weight constraint
            normalized_weights_data = np.maximum(normalized_weights_data, min_weight_ratio * np.max(normalized_weights_data))
            normalized_weights_data_list.append(normalized_weights_data)

        return normalized_weights_data_list


    def get_scf_folder_path(self, atomic_number: int) -> str:
        """
        Get SCF folder path for a given atomic number.
        Note: This method finds the configuration folder by matching atomic_number in meta.json files.
        It supports both new format (configuration_XXX) and old format (atom_XXX) for backward compatibility.
        """
        assert isinstance(atomic_number, int), \
            ATOMIC_NUMBER_NOT_INTEGER_ERROR.format(atomic_number)
        assert atomic_number in self.atomic_numbers_unique, \
            ATOMIC_NUMBER_NOT_IN_DATASET_ERROR.format(atomic_number)
        
        # Find configuration folder by searching for matching atomic_number in meta.json
        # This handles both new and old folder naming formats
        for config_data in self.configuration_data_list:
            if config_data.atomic_number == atomic_number:
                # Use the folder path from the loaded configuration data
                return os.path.join(config_data.scf_folder_path)
        
        # Fallback: try to find by searching all folders (for backward compatibility)
        # This should rarely be needed if configuration_data_list is properly loaded
        import json
        for item in os.listdir(self.data_root):
            item_path = os.path.join(self.data_root, item)
            if not os.path.isdir(item_path):
                continue
            if item.startswith('configuration_') or item.startswith('atom_'):
                meta_path = os.path.join(item_path, "meta.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as meta_file:
                            meta_data = json.load(meta_file)
                        if int(meta_data.get("atomic_number", 0)) == atomic_number:
                            return os.path.join(item_path, self.scf_xc_functional.lower())
                    except Exception:
                        continue
        
        # If still not found, raise error
        raise ValueError(ATOMIC_NUMBER_NOT_IN_DATASET_ERROR.format(atomic_number))


    def exists_functional(self, functional: str) -> bool:
        """
        Check if a functional exists in the dataset.
        """
        return functional == self.scf_xc_functional or functional in self.forward_pass_xc_functional_list


    def get_xc_data(self, functional: str) -> XCDataType:
        """
        Get XC data for a given functional.
        """
        if functional == self.scf_xc_functional:
            return self.scf_xc_data
        if functional in self.forward_pass_xc_functional_list:
            return self.forward_pass_xc_data_list[self.forward_pass_xc_functional_list.index(functional)]
        raise ValueError(FUNCTIONAL_NOT_IN_DATASET_ERROR.format(functional, self.scf_xc_functional, self.forward_pass_xc_functional_list))


    def _get_xc_data_from_single_configuration_data(self, single_configuration_data: SingleConfigurationData, functional: str) -> XCDataType:
        """
        Get XC data from a single configuration data, only for internal use.
        """
        assert isinstance(single_configuration_data, SingleConfigurationData)

        if functional == self.scf_xc_functional:
            return single_configuration_data.scf_xc_data
        if functional in self.forward_pass_xc_functional_list:
            return single_configuration_data.forward_pass_xc_data_list[self.forward_pass_xc_functional_list.index(functional)]
        raise ValueError(FUNCTIONAL_NOT_IN_DATASET_ERROR.format(functional, self.scf_xc_functional, self.forward_pass_xc_functional_list))


    def get_features_data(self, feature_name: str) -> np.ndarray:
        """
        Get features data for a given feature name.
        """
        assert isinstance(feature_name, str),\
            FEATURE_NAME_NOT_STRING_ERROR.format(feature_name)
        assert feature_name in self.features_list,\
            FEATURE_NAME_NOT_IN_FEATURES_LIST_ERROR.format(feature_name, self.features_list)
        return self.features_data[:, self.features_list.index(feature_name)]


    @property
    def rho(self) -> np.ndarray:
        """
        Get density data for all configurations.
        """
        assert "rho" in self.features_list,\
            FEATURE_NAME_NOT_IN_FEATURES_LIST_ERROR.format("rho", self.features_list)
        return self.features_data[:, self.features_list.index("rho")]


    @property
    def potential_weights_data(self) -> np.ndarray:
        """
        Get weights data for all configurations.
        """
        assert self.potential_weights_data_is_updated
        return np.concatenate([configuration_data.potential_weights_data for configuration_data in self.configuration_data_list], axis=0)


    @property
    def atomic_numbers_unique(self) -> np.ndarray:
        """
        Get unique atomic numbers.
        """
        if self.cached_atomic_numbers_unique is None:
            self.cached_atomic_numbers_unique = np.sort(np.unique(self.atomic_numbers_per_configuration))
        return self.cached_atomic_numbers_unique
    

    @property
    def atomic_numbers_per_configuration(self) -> np.ndarray:
        """
        Get atomic numbers per configuration.
        """
        if self.cached_atomic_numbers_per_configuration is None:
            self.cached_atomic_numbers_per_configuration = np.array([configuration_data.atomic_number for configuration_data in self.configuration_data_list])
        return self.cached_atomic_numbers_per_configuration
    

    @property
    def atomic_numbers_per_sample(self) -> np.ndarray:
        """
        Get atomic numbers per sample.
        """
        if self.cached_atomic_numbers_per_sample is None:
            self.cached_atomic_numbers_per_sample = np.concatenate([configuration_data.atomic_numbers for configuration_data in self.configuration_data_list])
        return self.cached_atomic_numbers_per_sample


    @property
    def n_atoms(self) -> int:
        """
        Get number of atomic numbers.
        """
        return len(self.atomic_numbers_unique)
    

    @property
    def n_configurations(self) -> int:
        """
        Get number of configurations.
        """
        return len(self.configuration_data_list)
    

    @property
    def n_samples(self) -> int:
        """
        Get number of samples.
        """
        return len(self.atomic_numbers_per_sample)
    

    @property
    def cutoff_radii(self) -> np.ndarray:
        """
        Get cutoff radii.
        """
        if self.cached_cutoff_radii is None:
            self.cached_cutoff_radii = np.array([configuration_data.cutoff_radius for configuration_data in self.configuration_data_list])
        return self.cached_cutoff_radii
    

    @property
    def cutoff_indices(self) -> np.ndarray:
        """
        Get cutoff indices.
        """
        if self.cached_cutoff_indices is None:
            self.cached_cutoff_indices = np.array([configuration_data.cutoff_idx for configuration_data in self.configuration_data_list])
        return self.cached_cutoff_indices


    @property
    def quadrature_nodes(self) -> np.ndarray:
        """
        Get quadrature nodes for all configurations.
        """
        if self.cached_quadrature_nodes is None:
            self.cached_quadrature_nodes = np.concatenate([configuration_data.quadrature_nodes_filtered for configuration_data in self.configuration_data_list])
        return self.cached_quadrature_nodes


    @property
    def configuration_ids(self) -> List[int]:
        """
        Get configuration ids for all configurations.
        """
        if self.cached_configuration_ids is None:
            self.cached_configuration_ids = [configuration_data.configuration_id for configuration_data in self.configuration_data_list]
        return self.cached_configuration_ids


    @property
    def features_data(self) -> np.ndarray:
        """
        Get features data for all configurations.
        """
        if self.cached_features_data is None:
            self.cached_features_data = np.concatenate([configuration_data.features_data for configuration_data in self.configuration_data_list])
        return self.cached_features_data


    @property
    def scf_xc_data(self) -> XCDataType:
        """
        Get SCF XC data for all configurations.
        """
        if self.cached_scf_xc_data is None:
            scf_xc_data_list = [configuration_data.scf_xc_data for configuration_data in self.configuration_data_list]
            self.cached_scf_xc_data = (
                np.concatenate([v_x for v_x, _, _, _ in scf_xc_data_list]),
                np.concatenate([v_c for _, v_c, _, _ in scf_xc_data_list]),
                np.concatenate([e_x for _, _, e_x, _ in scf_xc_data_list]) if self.include_energy_density else None,
                np.concatenate([e_c for _, _, _, e_c in scf_xc_data_list]) if self.include_energy_density else None,
            )
        return self.cached_scf_xc_data
    

    @property
    def forward_pass_xc_data_list(self) -> List[XCDataType]:
        """
        Get forward pass XC data for all configurations.
        """
        if self.cached_forward_pass_xc_data_list is None:

            # nest forward pass XC data into lists
            forward_pass_xc_data_list_list = [[] for _ in range(len(self.forward_pass_xc_functional_list))]
            for configuration_data in self.configuration_data_list:
                for idx, forward_pass_xc_data in enumerate(configuration_data.forward_pass_xc_data_list):
                    forward_pass_xc_data_list_list[idx].append(forward_pass_xc_data)

            # convert lists to numpy arrays
            self.cached_forward_pass_xc_data_list = [
                (
                    np.concatenate([v_x for v_x, _, _, _ in forward_pass_xc_data_list]),
                    np.concatenate([v_c for _, v_c, _, _ in forward_pass_xc_data_list]),
                    np.concatenate([e_x for _, _, e_x, _ in forward_pass_xc_data_list]) if self.include_energy_density else None,
                    np.concatenate([e_c for _, _, _, e_c in forward_pass_xc_data_list]) if self.include_energy_density else None,
                )
                for forward_pass_xc_data_list in forward_pass_xc_data_list_list
            ]
        return self.cached_forward_pass_xc_data_list





class AtomicDataManager:
    
    """
    Main class for managing atomic data, consisting of:
    - DataGenerator : generating atomic data
    - DataLoader    : loading atomic data
    - DataProcessor : processing atomic data
    """

    def __init__(self,
        data_root                   : str,
        scf_xc_functional           : str,
        forward_pass_xc_functionals : Optional[str | list[str]],
        auto_confirm                : bool = False,  # If True, automatically confirm prompts without user input
    ):

        """
        Args:
            data_root                   : Root directory of the dataset
            scf_xc_functional           : SCF XC functional (used for full SCF calculation to convergence)
            forward_pass_xc_functionals : Forward pass XC functional(s), if not None, will perform forward pass for each functional based on SCF results
            auto_confirm                : If True, automatically confirm prompts without user input (useful for batch/background jobs). Defaults to False.
        """

        # Check if data root exists
        if not os.path.exists(data_root):
            # Safety confirmation prompt for creating empty directory
            print("\n" + "=" * 60)
            print("WARNING: data root '{}' does not found.".format(data_root))
            print("=" * 60)
            if auto_confirm:
                print("\nAuto-confirming: Creating directory...")
                os.makedirs(data_root)
            else:
                if input("\nDo you want to create an empty directory? (y/n): ").strip().lower() == 'y':
                    os.makedirs(data_root)
                else:
                    print("Operation cancelled by user.")
                    exit(0)
            
            print("Directory created successfully.")


        # Process forward_pass_xc_functionals: None -> [], str -> [str], list -> list
        if forward_pass_xc_functionals is None:
            forward_pass_xc_functional_list = []
        elif isinstance(forward_pass_xc_functionals, str):
            forward_pass_xc_functional_list = [forward_pass_xc_functionals]
        elif isinstance(forward_pass_xc_functionals, list):
            forward_pass_xc_functional_list = forward_pass_xc_functionals
        else:
            raise TypeError(FORWARD_PASS_XC_FUNCTIONAL_NOT_NONE_STR_OR_LIST_ERROR.format(type(forward_pass_xc_functionals)))


        # Type and value checks
        self.check_data_root(data_root)
        self.check_scf_xc_functional(scf_xc_functional)
        self.check_forward_pass_xc_functional_list(forward_pass_xc_functional_list)


        # Initialize attributes
        self.data_root                       : str       = data_root
        self.scf_xc_functional               : str       = scf_xc_functional
        self.forward_pass_xc_functional_list : List[str] = forward_pass_xc_functional_list
        self.auto_confirm                    : bool      = auto_confirm

        # Initialize tools
        self.generator : DataGenerator = DataGenerator()
        self.loader    : DataLoader    = DataLoader()
        self.processor : DataProcessor = DataProcessor()

        # Initialize default parameters
        ## For data generation
        self.default_radius_cutoff_rho_threshold : float = 1e-6
        self.default_radius_cutoff_v_x_threshold : float = 1e-8
        self.default_radius_cutoff_v_c_threshold : float = 1e-8

        ## For data processing
        self.default_smooth_radius_threshold : float = 5.0
        self.default_smooth_method           : str = 'savgol'
        self.default_smooth_kwargs           : Dict = {}


    # Data generation methods
    def generate_data(self, 
        # Required arguments
        atomic_number_list        : list[int | float], 
        n_electrons_list          : Optional[list[int | float]] = None,
        use_oep                   : Optional[bool] = None,  # If None, uses default from XC_FUNCTIONAL_OEP_DEFAULT based on scf_xc_functional

        # Optional arguments controlling the contents of the dataset
        save_energy_density       : bool  = False,
        save_intermediate         : bool  = False,
        save_full_spectrum        : bool  = False,
        save_derivative_matrix    : bool  = False,
        start_configuration_index : int   = 1,

        # Optional arguments controlling the generation process
        # Grid, basis, and mesh parameters
        domain_size               : float = 20.0,
        finite_elements_number    : int   = 35,
        polynomial_order          : int   = 20,
        quadrature_point_number   : int   = 43,
        oep_basis_number          : int   = 5,
        mesh_type                 : str   = "polynomial",
        mesh_concentration        : float = 2.0,
        mesh_spacing              : float = 0.1,

        # SCF convergence parameters
        scf_tolerance             : float = 1e-8,
        max_scf_iterations        : int   = 500,
        max_scf_iterations_outer  : Optional[int] = None,
        use_pulay_mixing          : bool  = True,
        use_preconditioner        : bool  = True,
        pulay_mixing_parameter    : float = 1.0,
        pulay_mixing_history      : int   = 7,
        pulay_mixing_frequency    : int   = 3,
        linear_mixing_alpha1      : float = 0.75,
        linear_mixing_alpha2      : float = 0.95,

        # Advanced functional parameters
        hybrid_mixing_parameter           : float = None,
        frequency_quadrature_point_number : int   = None,
        angular_momentum_cutoff           : int   = None,
        double_hybrid_flag                : bool  = None,
        oep_mixing_parameter              : float = None,
        enable_parallelization            : bool  = None,

        # Debugging and verbose parameters
        verbose                   : bool  = True,
        overwrite                 : Optional[bool] = None,  # If True, automatically confirm prompts without user input. If None, uses instance default.

        # deprecated arguments
        finite_elements           : Optional[int] = None,
    ):
        """
        Generate atomic dataset, based on SCF and forward pass XC functionals.
        

        Required arguments
        ------------------
        `atomic_number_list` : list[int | float]
            List of atomic numbers to generate data for (can be fractional).
        `n_electrons_list` : list[int | float] | None
            List of number of electrons to generate data for (can be fractional). If None, defaults to atomic_number_list.

        Dataset content control
        -----------------------
        `save_energy_density` : bool
            Whether to save energy density in the dataset. Defaults to False.
        `save_intermediate` : bool
            Whether to save intermediate information during SCF. Defaults to False.
        `save_full_spectrum` : bool
            Whether to save optional full-spectrum outputs. Defaults to False.
            Source spectra required by OEP forward targets are always computed
            and saved, including intermediate states, regardless of this option.
        `save_derivative_matrix` : bool
            Whether to save derivative matrix. Most systems have the same derivative matrix when using
            the same grid/basis/mesh parameters, so a shared derivative matrix is saved at the dataset root.
            If an atom's derivative matrix differs from the shared one, it is saved locally and recorded in meta.json.
            Defaults to False.
        `start_configuration_index` : int
            Starting configuration index for generated folders. Configuration folders will be named
            configuration_XXX starting from this index. Default is 1.
            For example, if start_configuration_index=5, the first atom will be saved as configuration_005,
            the second as configuration_006, etc.

        Grid, basis, and mesh parameters
        --------------------------------
        `domain_size` : float
            Radial computational domain size in atomic units (typically 10-30 Bohr). Defaults to 20.0.
        `finite_elements_number` : int
            Number of finite elements in the computational domain. Defaults to 35.
        `polynomial_order` : int
            Polynomial order of basis functions within each finite element (typically 20-40). Defaults to 20.
        `quadrature_point_number` : int
            Number of quadrature points for numerical integration (recommended: 3-4x polynomial_order). Defaults to 43.
        `oep_basis_number` : int
            Basis size used in OEP calculations when enabled. Defaults to 5.
        `mesh_type` : str
            Mesh distribution type ('exponential', 'polynomial', 'uniform'). Defaults to 'polynomial'.
        `mesh_concentration` : float
            Mesh concentration parameter (controls point density distribution). Defaults to 2.0.
        `mesh_spacing` : float
            Used to set the output uniform mesh spacing, irrelevant during SCF calculation. Defaults to 0.1.

        Self-consistent field (SCF) convergence parameters
        --------------------------------------------------
        `scf_tolerance` : float
            SCF convergence tolerance (typically 1e-8). Defaults to 1e-8 (1e-6 for SCAN/RSCAN/R2SCAN functionals).
        `max_scf_iterations` : int
            Maximum number of inner SCF iterations. If None, uses default (500). Defaults to None.
        `max_scf_iterations_outer` : int | None
            Maximum number of outer SCF iterations (for functionals requiring outer loop like HF, EXX, RPA, PBE0).
            If None, uses default (50 when needed, otherwise not used). Defaults to None.
        `use_pulay_mixing` : bool
            True for Pulay mixing for SCF convergence, False for linear mixing. Defaults to True.
        `use_preconditioner` : bool
            Flag for using preconditioner for SCF convergence. Defaults to True.
        `pulay_mixing_parameter` : float
            Pulay mixing parameter. Defaults to 1.0.
        `pulay_mixing_history` : int
            Pulay mixing history. Defaults to 7.
        `pulay_mixing_frequency` : int
            Pulay mixing frequency. Defaults to 3.
        `linear_mixing_alpha1` : float
            Linear mixing parameter (alpha_1). Defaults to 0.75.
        `linear_mixing_alpha2` : float
            Linear mixing parameter (alpha_2). Defaults to 0.95.

        Advanced functional parameters
        ------------------------------
        `hybrid_mixing_parameter` : float
            Mixing parameter for hybrid/double-hybrid functionals. Defaults to 1.0.
        `frequency_quadrature_point_number` : int
            Number of frequency quadrature points for RPA calculations. Defaults to 25.
        `angular_momentum_cutoff` : int
            Maximum angular momentum quantum number to include. Defaults to 4.
        `double_hybrid_flag` : bool
            Flag for double-hybrid functional methods. Defaults to False.
        `oep_mixing_parameter` : float
            Scaling parameter (λ) for OEP exchange/correlation potentials. Defaults to 1.0.
        `enable_parallelization` : bool
            Flag for parallelization of RPA calculations. Defaults to False.

        Debugging and verbose parameters
        --------------------------------
        `verbose` : bool
            Whether to print information during execution. Defaults to True.
        `overwrite` : bool | None
            If True, automatically confirm prompts without user input (useful for batch/background jobs).
            If None, uses the value set in AtomicDataManager.__init__(). Defaults to None.
        """
        
        # Handle deprecated finite_elements parameter
        if finite_elements is not None:
            if finite_elements_number != 35 and finite_elements_number != finite_elements:
                # Check if finite_elements_number was explicitly set (not using default) and conflicts with finite_elements
                raise ValueError(FINITE_ELEMENTS_NUMBER_AND_FINITE_ELEMENTS_BOTH_SPECIFIED_ERROR)
            finite_elements_number = finite_elements
            print(FINITE_ELEMENTS_DEPRECATED_WARNING)
        
        # Use instance default if overwrite not explicitly provided
        if overwrite is None:
            overwrite = self.auto_confirm
        
        # Get default use_oep value if not provided
        if use_oep is None:
            use_oep = XC_FUNCTIONAL_OEP_DEFAULT.get(self.scf_xc_functional, False)
        
        # Safety confirmation prompt
        print("\n" + "="*75)
        print("WARNING: This script will generate/overwrite dataset files.")
        print("This operation may take a long time and will modify existing data in {}".format(self.data_root))
        print("="*75)
        if overwrite:
            print("\nAuto-confirming: Proceeding with data generation...")
        else:
            user_input = input("\nDo you want to proceed? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Operation cancelled by user.")
                exit(0)
        print("\nStarting data generation...\n")

        self.generator.generate_data(
            # Required arguments
            data_root                   = self.data_root,
            atomic_number_list          = atomic_number_list,
            n_electrons_list            = n_electrons_list,
            use_oep                     = use_oep,
            scf_xc_functional           = self.scf_xc_functional,
            forward_pass_xc_functionals = self.forward_pass_xc_functional_list,

            # Arguments controlling the contents of the dataset
            save_energy_density         = save_energy_density,
            save_intermediate           = save_intermediate,
            save_full_spectrum          = save_full_spectrum,
            save_derivative_matrix      = save_derivative_matrix,
            start_configuration_index   = start_configuration_index,

            # Arguments controlling the generation process
            # Grid, basis, and mesh parameters
            domain_size                 = domain_size,
            finite_elements_number      = finite_elements_number,
            polynomial_order            = polynomial_order,
            quadrature_point_number     = quadrature_point_number,
            oep_basis_number            = oep_basis_number,
            mesh_type                   = mesh_type,
            mesh_concentration          = mesh_concentration,
            mesh_spacing                = mesh_spacing,

            # SCF convergence parameters
            scf_tolerance               = scf_tolerance,
            max_scf_iterations          = max_scf_iterations,
            max_scf_iterations_outer    = max_scf_iterations_outer,
            use_pulay_mixing            = use_pulay_mixing,
            use_preconditioner          = use_preconditioner,
            pulay_mixing_parameter      = pulay_mixing_parameter,
            pulay_mixing_history        = pulay_mixing_history,
            pulay_mixing_frequency      = pulay_mixing_frequency,
            linear_mixing_alpha1        = linear_mixing_alpha1,
            linear_mixing_alpha2        = linear_mixing_alpha2,

            # Advanced functional parameters
            hybrid_mixing_parameter           = hybrid_mixing_parameter,
            frequency_quadrature_point_number = frequency_quadrature_point_number,
            angular_momentum_cutoff           = angular_momentum_cutoff,
            double_hybrid_flag                = double_hybrid_flag,
            oep_mixing_parameter              = oep_mixing_parameter,
            enable_parallelization            = enable_parallelization,

            # Debugging and verbose parameters
            verbose                     = verbose,
        )


    # Data loading methods
    def load_data(self,
        # Required arguments
        configuration_index_list    : Optional[List[int]] = None, # If None, load data for all configurations
        features_list               : List[str] = ["rho", "grad_rho", "lap_rho", "hartree", "lda_xc"],

        # Control arguments
        use_radius_cutoff           : bool = False,
        use_feature_round_off       : bool = False,
        smooth_vxc                  : bool = False,
        close_shell_only            : bool = False,
        include_energy_density      : bool = False,
        include_intermediate        : bool = False,
        print_debug_info            : bool = False,
        print_summary               : bool = False,

        # Additional arguments, for parameter 'tuning'
        radius_cutoff_rho_threshold : Optional[float] = None,
        radius_cutoff_v_x_threshold : Optional[float] = None,
        radius_cutoff_v_c_threshold : Optional[float] = None,
        smooth_radius_threshold     : Optional[float] = None,
        smooth_method               : Optional[str]   = None,
        smooth_kwargs               : Optional[Dict]  = None,

        # Deprecated arguments
        use_cutoff                  : Literal[None] = None,
        cutoff_rho_threshold        : Literal[None] = None,
        cutoff_v_x_threshold        : Literal[None] = None,
        cutoff_v_c_threshold        : Literal[None] = None,
        atomic_number_list          : Optional[List[int]] = None,  # Deprecated: use configuration_index_list instead
    ) -> AtomicDataset:
        """
        Load training data for all configurations or specified ones.
        
        Parameters
        ----------
        configuration_index_list : Optional[List[int]]
            List of configuration indices (1-based) to load. If None, loads all available configurations.
            Atomic numbers are read from meta.json files in each atom folder.
        features_list : List[str]
            List of features to load
        
        use_radius_cutoff : bool
            Whether to apply cutoff filtering to truncate the radial grid
        use_feature_round_off : bool
            Whether to apply feature round-off (e.g., lower/upper bounds for small rho or potentials)
        smooth_vxc : bool
            Whether to apply smoothing to vxc data
        close_shell_only : bool
            Whether to only load data for closed shell atoms
        include_energy_density : bool
            Whether to also load energy density data
        include_intermediate : bool
            Whether to also load data from intermediate iteration folders (outer_iter_XX)
        print_debug_info : bool
            Whether to print debug information
        
        radius_cutoff_rho_threshold : Optional[float]
            Threshold for rho data when use_radius_cutoff is True
        radius_cutoff_v_x_threshold : Optional[float]
            Threshold for v_x data when use_radius_cutoff is True
        radius_cutoff_v_c_threshold : Optional[float]
            Threshold for v_c data when use_radius_cutoff is True
        smooth_radius_threshold : Optional[float]
            Radius threshold for smoothing. Values with r > r_smooth_threshold will be smoothed.
            Default is 5.0.
        smooth_method : Optional[str]
            Smoothing method: 'lowpass' , 'savgol'(default), 'moving_avg', 'spline', 'gaussian', 'exp_weighted', 'cascade'
            - 'lowpass': Low-pass Butterworth filter (RECOMMENDED for high-frequency oscillations)
            - 'savgol': Savitzky-Golay filter (preserves data shape, but may not filter high-freq well)
            - 'moving_avg': Moving average (simple, good for high-frequency filtering)
            - 'spline': Spline smoothing (controllable smoothness)
            - 'gaussian': Gaussian filter (good smoothing, adjustable strength)
            - 'exp_weighted': Exponentially weighted moving average (smooth for large r)
            - 'cascade': Apply multiple smoothing methods in sequence (strongest filtering)
        smooth_kwargs : Optional[Dict]
            Additional parameters for smoothing methods:
            - lowpass: cutoff (default: 0.05), order (default: 6) - lower cutoff = stronger filtering
            - savgol: window_length (default: min(30% of data, len(data)//2*2+1)), polyorder (default: 2)
            - moving_avg: window_size (default: 25% of data, min 25) - larger = stronger filtering
            - spline: s (smoothing factor, default: len(data) * variance * 0.8) - larger = stronger
            - gaussian: sigma (default: max(2.0, 1% of data length)) - larger = stronger filtering
            - exp_weighted: alpha (default: 0.15) - smaller = stronger filtering
            - cascade: methods (list), kwargs_list (list of kwargs for each method)
    
        Deprecated Parameters
        --------------------
        atomic_number_list : Optional[List[int]]
            Deprecated: Use configuration_index_list instead. Atomic numbers are now read from meta.json files.

        Returns
        -------
        AtomicDataset: Atomic dataset
        """

        # handle deprecated arguments
        if use_cutoff is not None:
            use_radius_cutoff = use_cutoff
            print(USE_CUTOFF_DEPRECATED_WARNING)
        if cutoff_rho_threshold is not None:
            radius_cutoff_rho_threshold = cutoff_rho_threshold
            print(CUTOFF_RHO_THRESHOLD_DEPRECATED_WARNING)
        if cutoff_v_x_threshold is not None:
            radius_cutoff_v_x_threshold = cutoff_v_x_threshold
            print(CUTOFF_V_X_THRESHOLD_DEPRECATED_WARNING)
        if cutoff_v_c_threshold is not None:
            radius_cutoff_v_c_threshold = cutoff_v_c_threshold
            print(CUTOFF_V_C_THRESHOLD_DEPRECATED_WARNING)
        
        # Handle deprecated atomic_number_list parameter
        if atomic_number_list is not None:
            if configuration_index_list is not None:
                raise ValueError(ATOMIC_NUMBER_LIST_AND_CONFIGURATION_INDEX_LIST_BOTH_SPECIFIED_ERROR)
            print(ATOMIC_NUMBER_LIST_DEPRECATED_WARNING)
            configuration_index_list = atomic_number_list  # Convert to configuration_index_list

        # Find all configuration indices if not specified
        if configuration_index_list is None:
            configuration_index_list = []
            seen_indices = set()
            for item in os.listdir(self.data_root):
                item_path = os.path.join(self.data_root, item)
                if not os.path.isdir(item_path):
                    continue
                
                # Try new format first (configuration_XXX)
                if item.startswith('configuration_'):
                    try:
                        configuration_index = int(item.split('_')[1])
                        if configuration_index not in seen_indices:
                            configuration_index_list.append(configuration_index)
                            seen_indices.add(configuration_index)
                    except ValueError:
                        continue
                # Fallback to old format for backward compatibility (atom_XXX)
                elif item.startswith('atom_'):
                    try:
                        configuration_index = int(item.split('_')[1])
                        if configuration_index not in seen_indices:
                            configuration_index_list.append(configuration_index)
                            seen_indices.add(configuration_index)
                    except ValueError:
                        continue
            configuration_index_list = sorted(configuration_index_list)

        # set default parameters
        if radius_cutoff_rho_threshold is None:
            radius_cutoff_rho_threshold = self.default_radius_cutoff_rho_threshold
        if radius_cutoff_v_x_threshold is None:
            radius_cutoff_v_x_threshold = self.default_radius_cutoff_v_x_threshold
        if radius_cutoff_v_c_threshold is None:
            radius_cutoff_v_c_threshold = self.default_radius_cutoff_v_c_threshold
        if smooth_radius_threshold is None:
            smooth_radius_threshold = self.default_smooth_radius_threshold
        if smooth_method is None:
            smooth_method = self.default_smooth_method
        if smooth_kwargs is None:
            smooth_kwargs = self.default_smooth_kwargs

        # Type and value check
        # Check configuration_index_list (list of integers >= 1)
        if not isinstance(configuration_index_list, list):
            raise TypeError("parameter 'configuration_index_list' must be a list, get {} instead.".format(type(configuration_index_list)))
        if not all(isinstance(idx, int) and idx >= 1 for idx in configuration_index_list):
            raise TypeError("parameter 'configuration_index_list' must be a list of integers >= 1, get {} instead.".format(configuration_index_list))
        
        AtomicDataManager.check_features_list(features_list)
        features_list = DataLoader.check_and_normalize_features_list(features_list)

        assert isinstance(use_radius_cutoff, bool), \
            USE_RADIUS_CUTOFF_NOT_BOOL_ERROR.format(type(use_radius_cutoff))
        assert isinstance(use_feature_round_off, bool), \
            USE_FEATURE_ROUND_OFF_NOT_BOOL_ERROR.format(type(use_feature_round_off))
        assert isinstance(smooth_vxc, bool), \
            SMOOTH_VXC_NOT_BOOL_ERROR.format(type(smooth_vxc))
        assert isinstance(close_shell_only, bool), \
            CLOSE_SHELL_ONLY_NOT_BOOL_ERROR.format(type(close_shell_only))
        assert isinstance(include_energy_density, bool), \
            INCLUDE_ENERGY_DENSITY_NOT_BOOL_ERROR.format(type(include_energy_density))
        assert isinstance(include_intermediate, bool), \
            INCLUDE_INTERMEDIATE_NOT_BOOL_ERROR.format(type(include_intermediate))
        assert isinstance(print_debug_info, bool), \
            PRINT_DEBUG_INFO_NOT_BOOL_ERROR.format(type(print_debug_info))
        assert isinstance(print_summary, bool), \
            PRINT_SUMMARY_NOT_BOOL_ERROR.format(type(print_summary))
        
        AtomicDataManager.check_radius_cutoff_threshold(
            radius_cutoff_rho_threshold,
            radius_cutoff_v_x_threshold,
            radius_cutoff_v_c_threshold,
        )
        AtomicDataManager.check_smooth_parameters(smooth_radius_threshold, smooth_method, smooth_kwargs)

        
        # Filter to closed shell atoms only if requested
        if close_shell_only:
            filtered_config_indices = []
            for configuration_index in configuration_index_list:
                # Find configuration folder (with backward compatibility)
                config_folder = DataLoader._find_configuration_folder(self.data_root, configuration_index)
                if config_folder is None:
                    continue
                meta_path = os.path.join(config_folder, "meta.json")
                if os.path.exists(meta_path):
                    try:
                        import json
                        with open(meta_path, "r") as meta_file:
                            meta_data = json.load(meta_file)
                        z = int(meta_data.get("atomic_number", 0))
                        if z > 0:
                            occupation_info = OccupationInfo(z_nuclear=z, z_valence=z, all_electron_flag=True)
                            if occupation_info.closed_shell_flag:
                                filtered_config_indices.append(configuration_index)
                    except Exception as e:
                        print(f"Warning: Could not determine closed shell status for configuration {configuration_index}: {e}. Skipping...")
                        continue
            configuration_index_list = filtered_config_indices
            print(f"Filtered to {len(configuration_index_list)} closed shell configurations: {configuration_index_list}")
        
        # Load data for each configuration
        configuration_data_list, skipped_atoms = DataLoader.load_data(
            data_root                   = self.data_root,
            scf_xc_functional           = self.scf_xc_functional,
            forward_pass_xc_functionals = self.forward_pass_xc_functional_list,
            features_list               = features_list,
            configuration_index_list    = configuration_index_list,
            use_radius_cutoff           = use_radius_cutoff,
            use_feature_round_off       = use_feature_round_off,
            include_energy_density      = include_energy_density,
            include_intermediate        = include_intermediate,
            print_debug_info            = print_debug_info,
            radius_cutoff_rho_threshold = radius_cutoff_rho_threshold,
            radius_cutoff_v_x_threshold = radius_cutoff_v_x_threshold,
            radius_cutoff_v_c_threshold = radius_cutoff_v_c_threshold,
        )

        # Load shared derivative matrix if any configuration uses it
        shared_derivative_matrix = None
        shared_derivative_matrix_path = os.path.join(self.data_root, "derivative_matrix.npy")
        if os.path.exists(shared_derivative_matrix_path):
            # Check if any configuration uses shared derivative matrix
            for config_data in configuration_data_list:
                if hasattr(config_data, 'derivative_matrix_use_shared') and config_data.derivative_matrix_use_shared:
                    # Load shared derivative matrix once
                    shared_derivative_matrix = np.load(shared_derivative_matrix_path)
                    break

        dataset = AtomicDataset(
            # Basic attributes
            data_root                       = self.data_root,
            scf_xc_functional               = self.scf_xc_functional,
            forward_pass_xc_functional_list = self.forward_pass_xc_functional_list,
            features_list                   = features_list,
            
            # Other attributes
            radius_cutoff_rho_threshold     = radius_cutoff_rho_threshold,
            radius_cutoff_v_x_threshold     = radius_cutoff_v_x_threshold,
            radius_cutoff_v_c_threshold     = radius_cutoff_v_c_threshold,

            # Data attributes
            configuration_data_list         = configuration_data_list,
            shared_derivative_matrix        = shared_derivative_matrix,
        )

        if smooth_vxc:
            dataset.smooth_v_xc(smooth_radius_threshold, smooth_method, smooth_kwargs)

        if print_debug_info or print_summary:
            dataset.print_info()
        
        return dataset



    def inverse_transform_features(
        self,
        X_transformed    : np.ndarray,
        scaler_X         : Optional[Any] = None,
        transform_params : Optional[Dict] = None,
        feature_idx      : Optional[int] = None
    ) -> np.ndarray:
        """
        Inverse transform features back to physical space.
        """
        return DataProcessor.inverse_transform_features(X_transformed, scaler_X, transform_params, feature_idx)


    def inverse_transform_predictions(
        self,
        y_pred           : np.ndarray,
        scaler_y         : Optional[Any] = None,
        transform_params : Optional[Dict] = None
    ) -> np.ndarray:
        """
        Inverse transform predictions back to physical space.
        """
        return DataProcessor.inverse_transform_predictions(y_pred, scaler_y, transform_params)


    # Type and value checks
    @staticmethod
    def check_data_root(data_root) -> None:
        """
        Check if the data root is a valid directory.
        """
        if not isinstance(data_root, str):
            raise TypeError(DATA_ROOT_NOT_STRING_ERROR.format(type(data_root)))
        if not os.path.exists(data_root):
            raise FileNotFoundError(DATA_ROOT_NOT_EXIST_ERROR.format(data_root))

    @staticmethod
    def check_scf_xc_functional(scf_xc_functional) -> None:
        """
        Check if the SCF XC functional is a valid string.
        """
        from ..solver import VALID_XC_FUNCTIONAL_LIST
        if not isinstance(scf_xc_functional, str):
            raise TypeError(SCF_XC_FUNCTIONAL_NOT_STRING_ERROR.format(type(scf_xc_functional)))
        if scf_xc_functional not in VALID_XC_FUNCTIONAL_LIST:
            raise ValueError(SCF_XC_FUNCTIONAL_NOT_IN_VALID_LIST_ERROR.format(VALID_XC_FUNCTIONAL_LIST, scf_xc_functional))

    @staticmethod
    def check_forward_pass_xc_functional_list(forward_pass_xc_functional_list) -> None:
        """
        Check if the forward pass XC functional list is a valid list of strings.
        """
        from ..solver import VALID_XC_FUNCTIONAL_LIST
        if not isinstance(forward_pass_xc_functional_list, list):
            raise TypeError(FORWARD_PASS_XC_FUNCTIONAL_LIST_NOT_LIST_ERROR.format(type(forward_pass_xc_functional_list)))
        if not all(isinstance(functional, str) for functional in forward_pass_xc_functional_list):
            raise TypeError(FORWARD_PASS_XC_FUNCTIONAL_LIST_NOT_LIST_OF_STRINGS_ERROR.format(type(forward_pass_xc_functional_list)))
        if not all(functional in VALID_XC_FUNCTIONAL_LIST for functional in forward_pass_xc_functional_list):
            raise ValueError(FORWARD_PASS_XC_FUNCTIONAL_LIST_NOT_IN_VALID_LIST_ERROR.format(VALID_XC_FUNCTIONAL_LIST, forward_pass_xc_functional_list))


    @staticmethod
    def check_features_list(features_list) -> None:
        """
        Check if the features list is a valid list of strings.
        """
        if not isinstance(features_list, list):
            raise TypeError(FEATURES_LIST_NOT_LIST_ERROR.format(type(features_list)))
        if not all(isinstance(feature, str) for feature in features_list):
            raise TypeError(FEATURES_LIST_NOT_LIST_OF_STRINGS_ERROR.format(type(features_list)))
        for feature in features_list:
            normalized_feature = FEATURE_ALIASES.get(feature, feature)
            if normalized_feature not in NORMALIZED_VALID_FEATURES_LIST_FOR_POTENTIAL:
                raise ValueError(
                    format_invalid_feature_error(
                        feature,
                        NORMALIZED_VALID_FEATURES_LIST_FOR_POTENTIAL,
                    )
                )



    @staticmethod
    def check_radius_cutoff_threshold(
        radius_cutoff_rho_threshold,
        radius_cutoff_v_x_threshold,
        radius_cutoff_v_c_threshold,
    ) -> None:
        """
        Check if the cutoff rho threshold is a valid float.
        """

        # Cutoff rho threshold
        if not isinstance(radius_cutoff_rho_threshold, float):
            raise TypeError(RADIUS_CUTOFF_RHO_THRESHOLD_NOT_FLOAT_ERROR.format(type(radius_cutoff_rho_threshold)))
        if radius_cutoff_rho_threshold <= 0:
            raise ValueError(RADIUS_CUTOFF_RHO_THRESHOLD_NOT_POSITIVE_ERROR.format(radius_cutoff_rho_threshold))
        
        # Cutoff v_x threshold
        if radius_cutoff_v_x_threshold is not None:
            if not isinstance(radius_cutoff_v_x_threshold, float):
                raise TypeError(RADIUS_CUTOFF_V_X_THRESHOLD_NOT_FLOAT_ERROR.format(type(radius_cutoff_v_x_threshold)))
            if radius_cutoff_v_x_threshold <= 0:
                raise ValueError(RADIUS_CUTOFF_V_X_THRESHOLD_NOT_POSITIVE_ERROR.format(radius_cutoff_v_x_threshold))
        
        # Cutoff v_c threshold
        if radius_cutoff_v_c_threshold is not None:
            if not isinstance(radius_cutoff_v_c_threshold, float):
                raise TypeError(RADIUS_CUTOFF_V_C_THRESHOLD_NOT_FLOAT_ERROR.format(type(radius_cutoff_v_c_threshold)))
            if radius_cutoff_v_c_threshold <= 0:
                raise ValueError(RADIUS_CUTOFF_V_C_THRESHOLD_NOT_POSITIVE_ERROR.format(radius_cutoff_v_c_threshold))

    @staticmethod
    def check_smooth_parameters(smooth_radius_threshold, smooth_method, smooth_kwargs) -> None:
        """
        Check if the smooth radius threshold is a valid float.
        """
        # Smooth radius threshold
        if not isinstance(smooth_radius_threshold, float):
            raise TypeError(SMOOTH_RADIUS_THRESHOLD_NOT_FLOAT_ERROR.format(type(smooth_radius_threshold)))
        if smooth_radius_threshold <= 0:
            raise ValueError(SMOOTH_RADIUS_THRESHOLD_NOT_POSITIVE_ERROR.format(smooth_radius_threshold))

        # Smooth method
        if not isinstance(smooth_method, str):
            raise TypeError(SMOOTH_METHOD_NOT_STRING_ERROR.format(type(smooth_method)))
        if smooth_method not in VALID_SMOOTH_METHODS:
            raise ValueError(SMOOTH_METHOD_NOT_IN_VALID_LIST_ERROR.format(VALID_SMOOTH_METHODS, smooth_method))

        # Smooth kwargs
        if not isinstance(smooth_kwargs, dict):
            raise TypeError(SMOOTH_KWARGS_NOT_DICT_ERROR.format(type(smooth_kwargs)))


    @staticmethod
    def check_configuration_data_list(configuration_data_list) -> None:
        """
        Check if the configuration data list is a valid list of SingleConfigurationData.
        """
        if not isinstance(configuration_data_list, list):
            raise TypeError(CONFIGURATION_DATA_LIST_NOT_LIST_ERROR.format(type(configuration_data_list)))
        if not all(isinstance(configuration_data, SingleConfigurationData) for configuration_data in configuration_data_list):
            invalid_types = {
                type(configuration_data)
                for configuration_data in configuration_data_list
                if not isinstance(configuration_data, SingleConfigurationData)
            }
            raise TypeError(
                CONFIGURATION_DATA_LIST_NOT_LIST_OF_SINGLE_CONFIGURATION_DATA_ERROR.format(invalid_types)
            )


    @staticmethod
    def check_atomic_number_list(atomic_number_list: List[int]) -> None:
        """
        Check if the atomic number list is a valid list of integers.
        """
        if not isinstance(atomic_number_list, list):
            raise TypeError(ATOMIC_NUMBER_LIST_NOT_LIST_ERROR.format(type(atomic_number_list)))
        if not all(isinstance(atomic_number, int) for atomic_number in atomic_number_list):
            raise TypeError(ATOMIC_NUMBER_LIST_NOT_LIST_OF_INTEGERS_ERROR.format(type(atomic_number_list)))
        if not all(atomic_number >= 1 and atomic_number <= 92 for atomic_number in atomic_number_list):
            raise ValueError(ATOMIC_NUMBER_LIST_NOT_IN_VALID_RANGE_ERROR.format(atomic_number_list))


    # Default parameter 'setting' methods
    def set_default_radius_cutoff_rho_threshold(self, radius_cutoff_rho_threshold: float):
        assert isinstance(radius_cutoff_rho_threshold, float), \
            RADIUS_CUTOFF_RHO_THRESHOLD_NOT_FLOAT_ERROR.format(type(radius_cutoff_rho_threshold))
        assert radius_cutoff_rho_threshold > 0, \
            RADIUS_CUTOFF_RHO_THRESHOLD_NOT_POSITIVE_ERROR.format(radius_cutoff_rho_threshold)
        self.default_radius_cutoff_rho_threshold = radius_cutoff_rho_threshold

    def set_default_radius_cutoff_v_x_threshold(self, radius_cutoff_v_x_threshold: float):
        assert isinstance(radius_cutoff_v_x_threshold, float), \
            RADIUS_CUTOFF_V_X_THRESHOLD_NOT_FLOAT_ERROR.format(type(radius_cutoff_v_x_threshold))
        assert radius_cutoff_v_x_threshold > 0, \
            RADIUS_CUTOFF_V_X_THRESHOLD_NOT_POSITIVE_ERROR.format(radius_cutoff_v_x_threshold)
        self.default_radius_cutoff_v_x_threshold = radius_cutoff_v_x_threshold

    def set_default_radius_cutoff_v_c_threshold(self, radius_cutoff_v_c_threshold: float):
        assert isinstance(radius_cutoff_v_c_threshold, float), \
            RADIUS_CUTOFF_V_C_THRESHOLD_NOT_FLOAT_ERROR.format(type(radius_cutoff_v_c_threshold))
        assert radius_cutoff_v_c_threshold > 0, \
            RADIUS_CUTOFF_V_C_THRESHOLD_NOT_POSITIVE_ERROR.format(radius_cutoff_v_c_threshold)
        self.default_radius_cutoff_v_c_threshold = radius_cutoff_v_c_threshold

    def set_default_smooth_radius_threshold(self, smooth_radius_threshold: float):
        assert isinstance(smooth_radius_threshold, float), \
            SMOOTH_RADIUS_THRESHOLD_NOT_FLOAT_ERROR.format(type(smooth_radius_threshold))
        assert smooth_radius_threshold > 0, \
            SMOOTH_RADIUS_THRESHOLD_NOT_POSITIVE_ERROR.format(smooth_radius_threshold)
        self.default_smooth_radius_threshold = smooth_radius_threshold
    
    def set_default_smooth_method(self, smooth_method: str):
        assert isinstance(smooth_method, str), \
            SMOOTH_METHOD_NOT_STRING_ERROR.format(type(smooth_method))
        assert smooth_method in VALID_SMOOTH_METHODS, \
            SMOOTH_METHOD_NOT_IN_VALID_LIST_ERROR.format(VALID_SMOOTH_METHODS, smooth_method)
        self.default_smooth_method = smooth_method
    
    def set_default_smooth_kwargs(self, smooth_kwargs: Dict):
        assert isinstance(smooth_kwargs, dict), \
            SMOOTH_KWARGS_NOT_DICT_ERROR.format(type(smooth_kwargs))
        self.default_smooth_kwargs = smooth_kwargs
