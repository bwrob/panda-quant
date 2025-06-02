# portfolio.py
"""
Contains classes for defining and analyzing portfolios of financial instruments.
The Portfolio class allows pricing using either TFF models (retrieved from a cache)
or full pricers.
"""
import numpy as np
import abc
# Import custom classes for product definitions and pricers
from features_generator import engineer_option_features, normalize_features
from product_definitions import ProductStaticBase, EuropeanOptionStatic
# Import the engineer_option_features function if it exists in another module
from pricers import PricerBase, QuantLibBondPricer, FastBondPricer, BlackScholesPricer
# Ensure that the custom classes are imported correctly
from quantlib_custom_serializer import custom_json_serializer
from datetime import date
# Ensure that ProductStaticBase is imported if we are checking for it in to_dict
from product_definitions import ProductStaticBase
# Ensure that PricerBase and its derivatives are imported for pricer handling
from pricers import PricerBase, QuantLibBondPricer, FastBondPricer, BlackScholesPricer
from tff_approximator import TensorFunctionalForm

class PortfolioBase(abc.ABC):
    """
    Abstract base class for a portfolio of financial instruments.
    """
    def __init__(self):
        # Stores details for each *position* in the portfolio.
        # Multiple positions can refer to the same underlying instrument_id if TFFs are cached.
        self.positions: list[dict] = []

    @abc.abstractmethod
    def add_position(self, *args, **kwargs):
        """Adds a position (instrument holding) to the portfolio."""
        pass

    @abc.abstractmethod
    def price_portfolio(self,
                        raw_market_scenarios: np.ndarray,
                        scenario_factor_names: list[str],
                        portfolio_rate_pillar_times: np.ndarray = None # For bond pricers
                        ) -> np.ndarray:
        """
        Prices all instruments in the portfolio for given market scenarios.

        Args:
            raw_market_scenarios (np.ndarray): 2D array of market scenarios
                                               (N_scenarios, N_total_market_factors).
            scenario_factor_names (list[str]): Names of the columns in raw_market_scenarios.
            portfolio_rate_pillar_times (np.ndarray, optional): 1D array of rate pillar times.
                                                               Required if portfolio contains bonds
                                                               priced with full QuantLibBondPricer.
        Returns:
            np.ndarray: 1D array of aggregated portfolio prices for each scenario (N_scenarios,).
        """
        pass

class Portfolio(PortfolioBase):
    """
    A portfolio where each instrument can be priced using either a pre-fitted
    Tensor Functional Form (TFF) model (retrieved from a cache) or its original full pricer.
    """
    def __init__(self):
        super().__init__()
        self.tff_model_cache: dict = {}
        # Cache structure:
        # { instrument_id: {
        #     'tff_model': TensorFunctionalForm_object,
        #     'raw_tff_input_names': list_of_names, # Raw factors TFF was trained on
        #     'normalization_params': dict_of_norm_params, # Includes engineered_feature_names
        #     'option_feature_order': int
        #   }, ...
        # }

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the portfolio.

        Returns:
            dict: Dictionary representation of the portfolio.
        """
        return {
            'positions': [p.to_dict() if hasattr(p, 'to_dict') else p
                for p in self.positions],
            'tff_model_cache': self.tff_model_cache
        }

    def cache_tff_model(self,
                        instrument_id: str,
                        tff_model: TensorFunctionalForm,
                        raw_tff_input_names: list[str],
                        normalization_params: dict,
                        option_feature_order: int = 0):
        """
        Explicitly caches a fitted TFF model and its associated parameters.

        Args:
            instrument_id (str): Unique ID for the instrument type this TFF represents.
            tff_model (TensorFunctionalForm): The pre-fitted TFF model.
            raw_tff_input_names (list[str]): Names of raw market factors TFF is based on.
            normalization_params (dict): Normalization params from TFF calibration.
                                         Expected keys: 'means', 'stds', 'engineered_feature_names', 'is_engineered'.
            option_feature_order (int, optional): Order of feature engineering if option TFF.
        """
        if not instrument_id:
            raise ValueError("instrument_id must be provided for caching TFF model.")
        if not isinstance(tff_model, TensorFunctionalForm):
            raise TypeError("tff_model must be an instance of TensorFunctionalForm.")
        if raw_tff_input_names is None or not isinstance(raw_tff_input_names, list):
            raise ValueError("raw_tff_input_names (list of strings) must be provided for caching TFF model.")
        if normalization_params is None or not isinstance(normalization_params, dict):
            raise ValueError("normalization_params (dict) must be provided for caching TFF model.")

        self.tff_model_cache[instrument_id] = {
            'tff_model': tff_model,
            'raw_tff_input_names': raw_tff_input_names,
            'normalization_params': normalization_params,
            'option_feature_order': option_feature_order
        }

    def from_dict(self, portfolio_dict: dict):
        """
        Loads a portfolio from a dictionary representation. Similar to cache_tff_model

        Args:
            portfolio_dict (dict): Dictionary representation of the portfolio.
        """



    def add_position(self,
                       instrument_id: str,
                       product_static: ProductStaticBase,
                       num_holdings: int = 1,
                       pricing_engine_type: str = 'tff',
                       direct_tff_config: dict = None,
                       full_pricer_instance: PricerBase = None,
                       full_pricer_kwargs: dict = None):
        """
        Adds a position (an instrument holding) to the portfolio.
        If pricing_engine_type is 'tff', it retrieves the TFF model from the cache
        using the instrument_id. The TFF model must have been cached previously using `cache_tff_model`.

        Args:
            instrument_id (str): Unique ID for this instrument type. Used for TFF caching/lookup.
            product_static (ProductStaticBase): Static definition of the product for this position.
            num_holdings (int, optional): Number of units. Defaults to 1.
            pricing_engine_type (str, optional): 'tff' or 'full'. Defaults to 'tff'.
            full_pricer_instance (PricerBase, optional): A full pricer instance (required if type is 'full').
            full_pricer_kwargs (dict, optional): Keyword arguments for the full pricer's price method.
        """
        if not isinstance(num_holdings, int) or num_holdings <= 0:
                raise ValueError("num_holdings must be a positive integer.")
        if not instrument_id:
            raise ValueError("instrument_id must be provided for the position.")

        position_detail = {
            'instrument_id': instrument_id,
            'product_static': product_static,
            'num_holdings': num_holdings,
            'engine_type': pricing_engine_type.lower()
        }

        if position_detail['engine_type'] == 'tff':
            if direct_tff_config is not None:
                # A TFF model and its configuration are being provided directly as a dictionary
                if not isinstance(direct_tff_config, dict):
                    raise TypeError("direct_tff_config must be a dictionary.")

                model_dict = direct_tff_config.get('model_dict')
                raw_names = direct_tff_config.get('raw_input_names')
                norm_params = direct_tff_config.get('normalization_params')

                # Default option_feature_order to 0 if not explicitly in config
                opt_order = direct_tff_config.get('option_feature_order', 0)

                if model_dict is None or not isinstance(model_dict, dict):
                    raise ValueError("direct_tff_config is missing 'model_dict' or it's not a dictionary.")
                if raw_names is None or not isinstance(raw_names, list):
                    raise ValueError("direct_tff_config is missing 'raw_input_names' or it's not a list.")
                if norm_params is None or not isinstance(norm_params, dict):
                    raise ValueError("direct_tff_config is missing 'normalization_params' or it's not a dictionary.")

                # Deserialize the TFF model itself from model_dict
                try:
                    tff_model_instance = TensorFunctionalForm.from_dict(model_dict)
                except Exception as e:
                    raise ValueError(f"Failed to deserialize TFF model from direct_tff_config['model_dict']: {e}")

                position_detail['pricer_engine'] = tff_model_instance
                position_detail['raw_tff_input_names'] = raw_names
                position_detail['normalization_params'] = norm_params
                position_detail['option_feature_order'] = opt_order

            else:
                # No direct TFF config provided, so use the cache via instrument_id
                if instrument_id not in self.tff_model_cache:
                    raise ValueError(
                        f"TFF model for instrument_id '{instrument_id}' not found in cache. "
                        "Either fit and cache it first, or provide its configuration via 'direct_tff_config'."
                    )

                cached_data = self.tff_model_cache[instrument_id]
                position_detail['pricer_engine'] = cached_data['tff_model']
                position_detail['raw_tff_input_names'] = cached_data['raw_tff_input_names']
                position_detail['normalization_params'] = cached_data['normalization_params']
                position_detail['option_feature_order'] = cached_data['option_feature_order']

        elif position_detail['engine_type'] == 'full':
            if not isinstance(full_pricer_instance, PricerBase):
                raise TypeError("full_pricer_instance must be an instance of PricerBase if engine_type is 'full'.")
            position_detail['pricer_engine'] = full_pricer_instance
            position_detail['full_pricer_kwargs'] = full_pricer_kwargs or {}
        else:
            raise ValueError(f"Unsupported pricing_engine_type: {pricing_engine_type}. Choose 'tff' or 'full'.")

        self.positions.append(position_detail)

    def load_portfolio_from_specs(self, portfolio_specs: list[dict]):
        """
        Loads multiple positions into the portfolio from a list of specifications.

        Each specification in the list is a dictionary that should conform to
        the parameters expected by the `add_position` method.

        Args:
            portfolio_specs (list[dict]): A list of dictionaries, where each dictionary
                                          defines a position to be added.
                                          Expected keys in each dict:
                                            'instrument_id' (str)
                                            'product_static_object' (ProductStaticBase)
                                            'num_holdings' (int)
                                            'pricing_engine_type' (str: 'tff' or 'full')
                                            Optional for 'tff' type:
                                              'direct_tff_config' (dict, containing 'model_dict',
                                                                   'raw_input_names', 'normalization_params',
                                                                   'option_feature_order')
                                            Optional for 'full' type:
                                              'full_pricer_instance' (PricerBase)
                                              'full_pricer_kwargs' (dict)
        """
        if not isinstance(portfolio_specs, list):
            raise TypeError("portfolio_specs must be a list of dictionaries.")

        for i, item_spec in enumerate(portfolio_specs):
            if not isinstance(item_spec, dict):
                raise TypeError(f"Each item in portfolio_specs must be a dictionary. Found type {type(item_spec)} at index {i}.")

            try:
                self.add_position(
                    instrument_id=item_spec['instrument_id'],
                    product_static=item_spec['product_static_object'],
                    num_holdings=item_spec.get('num_holdings', 1), # Default if not specified
                    pricing_engine_type=item_spec.get('pricing_engine_type', 'tff'), # Default

                    direct_tff_config=item_spec.get('direct_tff_config'), # Will be None if key doesn't exist

                    full_pricer_instance=item_spec.get('full_pricer_instance'),
                    full_pricer_kwargs=item_spec.get('full_pricer_kwargs')
                )
            except KeyError as e:
                raise ValueError(f"Missing required key {e} in portfolio_spec at index {i}: {item_spec}")
            except Exception as e:
                raise RuntimeError(f"Error adding position from spec at index {i} ('{item_spec.get('instrument_id', 'Unknown ID')}'): {e}")

        print(f"Successfully loaded {len(self.positions)} positions into the portfolio from specifications.")

    def price_portfolio(self,
                        raw_market_scenarios: np.ndarray,
                        scenario_factor_names: list[str],
                        portfolio_rate_pillar_times: np.ndarray = None
                        ) -> np.ndarray:
        """
        Prices all positions in the portfolio for given market scenarios.

        Args:
            raw_market_scenarios (np.ndarray): 2D array of raw market scenarios
                                               (N_scenarios, N_total_market_factors).
            scenario_factor_names (list[str]): Names of the columns in raw_market_scenarios.
            portfolio_rate_pillar_times (np.ndarray, optional): 1D array of rate pillar times (numeric tenors).
                                                               Required if portfolio contains bonds
                                                               priced with full QuantLibBondPricer or FastBondPricer.

        Returns:
            np.ndarray: 1D array of aggregated portfolio prices for each scenario (N_scenarios,).
        """
        if not self.positions:
            return np.array([])

        num_scenarios = raw_market_scenarios.shape[0]
        portfolio_prices_per_scenario = np.zeros(num_scenarios, dtype=float)

        for position_detail in self.positions:
            pricer_engine = position_detail['pricer_engine']
            engine_type = position_detail['engine_type']
            num_holdings = position_detail['num_holdings']

            instrument_prices_this_instrument = np.zeros(num_scenarios, dtype=float)

            if engine_type == 'tff':
                tff_model: TensorFunctionalForm = pricer_engine
                # These are the names of the RAW factors the TFF was originally trained on.
                raw_tff_input_factor_names_for_this_tff = position_detail['raw_tff_input_names']
                norm_params = position_detail['normalization_params']
                opt_feat_order = position_detail['option_feature_order']

                # Select the relevant columns from the global raw_market_scenarios
                try:
                    indices_of_raw_factors_in_global_scenarios = [scenario_factor_names.index(name) for name in raw_tff_input_factor_names_for_this_tff]
                except ValueError as e:
                    raise ValueError(f"A TFF input name in {raw_tff_input_factor_names_for_this_tff} not found in scenario_factor_names {scenario_factor_names} for instrument_id '{position_detail['instrument_id']}'. Error: {e}")

                current_raw_inputs_for_tff = raw_market_scenarios[:, indices_of_raw_factors_in_global_scenarios]

                inputs_for_tff_evaluation = current_raw_inputs_for_tff # Default

                # opt_feat_order should be available from position_detail if it's an option TFF
                opt_feat_order = position_detail.get('option_feature_order', 0)

                if isinstance(position_detail['product_static'], EuropeanOptionStatic) and \
                   norm_params.get('is_engineered', False): # Check if TFF was trained with engineered features

                    s0_factor_actual_name_port = None
                    vol_factor_actual_name_port = None
                    # These indices are relative to raw_tff_input_factor_names_for_this_tff
                    # and thus to the columns of current_raw_inputs_for_tff
                    s0_idx_in_tff_inputs = -1
                    vol_idx_in_tff_inputs = -1

                    # For an option TFF, raw_tff_input_factor_names_for_this_tff is expected to be [full_s0_name, full_vol_name]
                    if len(raw_tff_input_factor_names_for_this_tff) == 2:
                        for i, name in enumerate(raw_tff_input_factor_names_for_this_tff):
                            if name.upper().endswith("_S0"): # Using suffix matching
                                s0_factor_actual_name_port = name
                                s0_idx_in_tff_inputs = i
                            elif name.upper().endswith("_VOLATILITY") or name.upper().endswith("_VOL"): # Using suffix matching
                                vol_factor_actual_name_port = name
                                vol_idx_in_tff_inputs = i

                        if s0_idx_in_tff_inputs == -1 or vol_idx_in_tff_inputs == -1 or s0_idx_in_tff_inputs == vol_idx_in_tff_inputs:
                            raise ValueError(
                                f"Portfolio pricing: Could not identify distinct S0 and Volatility factors for option "
                                f"'{position_detail['instrument_id']}' from its TFF input names: "
                                f"{raw_tff_input_factor_names_for_this_tff}."
                            )
                    else:
                        raise ValueError(
                            f"Portfolio pricing: Option TFF for '{position_detail['instrument_id']}' with engineered features "
                            f"expects 2 raw input factors (S0, Vol), but "
                            f"TFF was trained on {len(raw_tff_input_factor_names_for_this_tff)} names: "
                            f"{raw_tff_input_factor_names_for_this_tff}."
                        )

                    # current_raw_inputs_for_tff columns are ordered as per raw_tff_input_factor_names_for_this_tff
                    s0_scenarios_raw = current_raw_inputs_for_tff[:, s0_idx_in_tff_inputs]
                    vol_scenarios_raw = current_raw_inputs_for_tff[:, vol_idx_in_tff_inputs]

                    engineered_features_test, _ = engineer_option_features( # Changed name for clarity
                        s0_scenarios_raw, vol_scenarios_raw, order=opt_feat_order
                    )
                    # Use normalization_params['means'] and ['stds'] from the cached TFF model data
                    inputs_for_tff_evaluation, _, _ = normalize_features(
                        engineered_features_test, # Changed name
                        norm_params['means'],
                        norm_params['stds']
                    )

                instrument_prices_this_instrument = tff_model(inputs_for_tff_evaluation)

            elif engine_type == 'full':
                full_pricer: PricerBase = pricer_engine
                pricer_kwargs = position_detail.get('full_pricer_kwargs', {})

                if isinstance(full_pricer, (QuantLibBondPricer, FastBondPricer)):
                    if portfolio_rate_pillar_times is None:
                        raise ValueError("portfolio_rate_pillar_times must be provided for full QL/Fast bond pricing in portfolio.")

                    num_rate_pillars = len(portfolio_rate_pillar_times)
                    # Select rate columns from raw_market_scenarios
                    rate_indices = []
                    try:
                        # Attempt to find by name if portfolio_rate_pillar_times contains names (should be numeric tenors)
                        # For simplicity, assume scenario_factor_names aligns with rate_pillar_names for the rate part
                        rate_indices = [scenario_factor_names.index(name) for name in portfolio_rate_pillar_times if isinstance(name, str) and name in scenario_factor_names]
                        if len(rate_indices) != num_rate_pillars: # Fallback if names don't match
                             rate_indices = list(range(num_rate_pillars))
                    except (ValueError, IndexError, TypeError):
                        # Fallback if portfolio_rate_pillar_times are not strings or not found
                        rate_indices = list(range(num_rate_pillars))

                    market_data_for_bond_pricer = raw_market_scenarios[:, rate_indices]

                    if isinstance(full_pricer, QuantLibBondPricer) and full_pricer.is_convertible and 'S0' in scenario_factor_names:
                        s0_col_idx = scenario_factor_names.index('S0')
                        market_data_for_bond_pricer = np.hstack((
                            market_data_for_bond_pricer,
                            raw_market_scenarios[:, s0_col_idx, np.newaxis]
                        ))

                    instrument_prices_this_instrument = full_pricer.price(
                        pillar_times=portfolio_rate_pillar_times, # These are the numeric tenors
                        market_scenario_data=market_data_for_bond_pricer,
                        **pricer_kwargs
                    )
                elif isinstance(full_pricer, BlackScholesPricer):
                    s0_idx = scenario_factor_names.index('S0')
                    vol_idx = scenario_factor_names.index('Volatility')
                    s0_scens = raw_market_scenarios[:, s0_idx]
                    vol_scens = raw_market_scenarios[:, vol_idx]
                    instrument_prices_this_instrument = full_pricer.price(
                        stock_price=s0_scens,
                        volatility=vol_scens,
                        **pricer_kwargs
                    )
                else:
                    raise TypeError(f"Unsupported full pricer type in portfolio: {type(full_pricer)} for instrument_id '{position_detail['instrument_id']}'")
            else:
                raise ValueError(f"Unknown engine_type: {engine_type} for instrument_id '{position_detail['instrument_id']}'")

            # Ensure consistent shapes for aggregation
            if instrument_prices_this_instrument.ndim == 0:
                instrument_prices_this_instrument = np.full(num_scenarios, float(instrument_prices_this_instrument))
            elif len(instrument_prices_this_instrument) != num_scenarios:
                 instrument_prices_this_instrument = instrument_prices_this_instrument.flatten()
                 if len(instrument_prices_this_instrument) != num_scenarios:
                    raise ValueError(f"Price array shape mismatch for instrument '{position_detail['instrument_id']}'. Expected ({num_scenarios},), got {instrument_prices_this_instrument.shape}")

            portfolio_prices_per_scenario += instrument_prices_this_instrument * num_holdings

        return portfolio_prices_per_scenario


