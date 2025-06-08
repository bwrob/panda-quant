"""
Contains classes for generating market scenarios for different risk factors.
SimpleRandomScenarioGenerator can now generate scenarios for a targeted list of factors,
including credit spread curves.
SimpleScaledRandomScenarioGenerator wraps a SimpleRandomScenarioGenerator to scale its shocks.
"""
import numpy as np
import abc
from functools import lru_cache

class ScenarioGeneratorBase(abc.ABC):
    @abc.abstractmethod
    def generate_scenarios(self, num_scenarios: int, target_factor_names: list[str] = None) -> tuple[np.ndarray, list[str]]:
        pass

class SimpleRandomScenarioGenerator(ScenarioGeneratorBase):
    def __init__(self,
                 base_rates_map: dict[str, float] = None, # e.g., {"USD_IR_0.25Y": 0.02}
                 rate_factor_shock_std_dev_map: dict[str, float] = None,
                 base_s0_map: dict[str, float] = None, # e.g., {"USD_STOCKA_S0": 100}
                 s0_shock_config_map: dict[str, tuple[str, float]] = None, # ('dist_type', param)
                 base_vol_map: dict[str, float] = None, # e.g., {"USD_STOCKA_VOL": 0.20}
                 vol_shock_config_map: dict[str, tuple[str, float]] = None,

                 base_credit_spread_curves_map: dict[str, np.ndarray] = None, # e.g., {"USD_FIN_AA_CS": np.array([0.005, ...])}
                 credit_spread_curve_tenors_map: dict[str, np.ndarray] = None, # e.g., {"USD_FIN_AA_CS": np.array([0.25, ...])}
                 # Shock for individual points on a credit spread curve
                 credit_spread_point_shock_std_dev: float = 0.0005,

                 default_rate_shock_std_dev: float = 0.0020, # 10 bps
                 default_s0_shock_config: tuple[str, float] = ('normal_relative', 0.01), # 1% relative normal shock
                 default_vol_shock_config: tuple[str, float] = ('normal_absolute', 0.01), # 1% absolute normal shock
                 random_seed: int = None):

        self.base_rates_map = base_rates_map if base_rates_map is not None else {}
        self.rate_factor_shock_std_dev_map = rate_factor_shock_std_dev_map if rate_factor_shock_std_dev_map is not None else {}
        
        self.base_s0_map = base_s0_map if base_s0_map is not None else {}
        self.s0_shock_config_map = s0_shock_config_map if s0_shock_config_map is not None else {}
        
        self.base_vol_map = base_vol_map if base_vol_map is not None else {}
        self.vol_shock_config_map = vol_shock_config_map if vol_shock_config_map is not None else {}

        self.base_credit_spread_curves_map = base_credit_spread_curves_map if base_credit_spread_curves_map is not None else {}
        self.credit_spread_curve_tenors_map = credit_spread_curve_tenors_map if credit_spread_curve_tenors_map is not None else {}
        self.credit_spread_point_shock_std_dev = credit_spread_point_shock_std_dev

        self.default_rate_shock_std_dev = default_rate_shock_std_dev
        self.default_s0_shock_config = default_s0_shock_config
        self.default_vol_shock_config = default_vol_shock_config

        self.rng = np.random.default_rng(random_seed)

        # Store base values for individual credit spread factor points for easier lookup
        # Factor name format: CURVE_NAME_TenorY, e.g., "USD_FIN_AA_CS_0.25Y"
        self.base_credit_spread_points_map = {}
        for curve_name, tenors in self.credit_spread_curve_tenors_map.items():
            if curve_name in self.base_credit_spread_curves_map:
                spread_values = self.base_credit_spread_curves_map[curve_name]
                if len(tenors) == len(spread_values):
                    for i, tenor in enumerate(tenors):
                        # Ensure tenor format is consistent, e.g., two decimal places
                        factor_point_name = f"{curve_name}_{tenor:.2f}Y" 
                        self.base_credit_spread_points_map[factor_point_name] = spread_values[i]
                else:
                    # Consider logging this warning
                    print(f"Warning: Mismatch between tenors and spread values for credit curve '{curve_name}'.")
        
        # Consolidate all known factor names for default generation if target_factor_names is None
        self._configured_factor_names_ordered = sorted(list(
            set(self.base_rates_map.keys()) |
            set(self.base_s0_map.keys()) |
            set(self.base_vol_map.keys()) |
            set(self.base_credit_spread_points_map.keys()) 
        ))


    def get_base_value_for_factor(self, factor_name: str) -> float | None:
        """Returns the base value for a given factor name."""
        if factor_name in self.base_rates_map:
            return self.base_rates_map[factor_name]
        if factor_name in self.base_s0_map: # Covers S0, potentially single-value DIVYIELD, CS
            return self.base_s0_map[factor_name]
        if factor_name in self.base_vol_map:
            return self.base_vol_map[factor_name]
        if factor_name in self.base_credit_spread_points_map: # For points on a CS curve
            return self.base_credit_spread_points_map[factor_name]
        # print(f"Warning: Base value for factor '{factor_name}' not found in any specific map.")
        return None

    # Add cache for configured factor names to avoid recomputing
    #@lru_cache(maxsize=None)
    def generate_scenarios(self, num_scenarios: int, target_factor_names: list[str] = None) -> tuple[np.ndarray, list[str]]:
        if target_factor_names is None:
            factors_to_generate = self._configured_factor_names_ordered
        else:
            factors_to_generate = target_factor_names

        if not factors_to_generate:
            return np.array([]).reshape(num_scenarios, 0), []

        all_scenario_columns = []

        for factor_name in factors_to_generate:
            factor_column = np.zeros(num_scenarios)
            base_value = self.get_base_value_for_factor(factor_name)
            
            if base_value is None and target_factor_names is not None:
                 raise ValueError(f"Target factor name '{factor_name}' is not configured or has no base value in this scenario generator.")
            elif base_value is None: # Should not happen if factors_to_generate comes from _configured_factor_names_ordered
                continue


            if factor_name in self.base_rates_map:
                shock_std_dev = self.rate_factor_shock_std_dev_map.get(factor_name, self.default_rate_shock_std_dev)
                shocks = self.rng.normal(loc=0.0, scale=shock_std_dev, size=num_scenarios)
                shocks[0] = 0.0 # Ensure first scenario is always the base value
                factor_column = base_value + shocks
            
            elif factor_name in self.base_s0_map: # S0, or other single-value factors like DIVYIELD, CS_FLAT
                shock_type, shock_param = self.s0_shock_config_map.get(factor_name, self.default_s0_shock_config)
                
                if shock_type.lower() == 'normal_relative': # shock_param is percentage
                    actual_std_dev = shock_param * abs(base_value) if abs(base_value) > 1e-9 else shock_param
                    shocks = self.rng.normal(loc=0.0, scale=actual_std_dev, size=num_scenarios)
                    shocks[0] = 0.0 # Ensure first scenario is always the base value
                    factor_column = base_value + shocks
                elif shock_type.lower() == 'normal_absolute': # shock_param is absolute std_dev
                    shocks = self.rng.normal(loc=0.0, scale=shock_param, size=num_scenarios)
                    shocks[0] = 0.0 # Ensure first scenario is always the base value
                    factor_column = base_value + shocks
                elif shock_type.lower() == 'uniform_relative': # shock_param is percentage half-width
                    half_width = shock_param * abs(base_value) if abs(base_value) > 1e-9 else shock_param
                    factor_column = self.rng.uniform(base_value - half_width, base_value + half_width, size=num_scenarios)
                else: raise ValueError(f"Unsupported shock_type: {shock_type} for S0-like factor {factor_name}")
                
                if "S0" in factor_name.upper() or "VOL" in factor_name.upper(): # Ensure S0 and Vol are non-negative
                    factor_column = np.maximum(factor_column, 1e-6)
                elif "_CS" in factor_name.upper() or "DIVYIELD" in factor_name.upper(): # Ensure CS and DivYield are non-negative
                    factor_column = np.maximum(factor_column, 0.0)

            elif factor_name in self.base_vol_map:
                shock_type, shock_param = self.vol_shock_config_map.get(factor_name, self.default_vol_shock_config)
                if shock_type.lower() == 'normal_relative':
                    actual_std_dev = shock_param * abs(base_value) if abs(base_value) > 1e-9 else shock_param
                    shocks = self.rng.normal(loc=0.0, scale=actual_std_dev, size=num_scenarios)
                    shocks[0] = 0.0 # Ensure first scenario is always the base value
                    factor_column = base_value + shocks
                elif shock_type.lower() == 'normal_absolute':
                    shocks = self.rng.normal(loc=0.0, scale=shock_param, size=num_scenarios)
                    shocks[0] = 0.0 # Ensure first scenario is always the base value
                    factor_column = base_value + shocks
                else: raise ValueError(f"Unsupported shock_type: {shock_type} for Vol factor {factor_name}")
                factor_column = np.maximum(factor_column, 1e-6) # Vol must be positive

            elif factor_name in self.base_credit_spread_points_map: # Point on a credit spread curve
                shocks = self.rng.normal(loc=0.0, scale=self.credit_spread_point_shock_std_dev, size=num_scenarios)
                shocks[0] = 0.0 # Ensure first scenario is always the base value
                factor_column = base_value + shocks
                factor_column = np.maximum(factor_column, 0.0) # Spreads must be non-negative
            
            else: # Should be covered by the initial check for base_value if target_factor_names is used
                if target_factor_names is not None: # Only raise if explicitly targeted and not found
                    raise ValueError(f"Factor '{factor_name}' was targeted but no generation rule was found.")
                # If not targeted, and somehow missed, skip it.
                continue


            all_scenario_columns.append(factor_column[:, np.newaxis])

        if not all_scenario_columns: # If factors_to_generate was non-empty but no columns were made (e.g. all unknown)
             return np.array([]).reshape(num_scenarios, 0), []
             
        return np.hstack(all_scenario_columns), factors_to_generate
