from dateutil.relativedelta import relativedelta # For MBSPricer

# Imports from other modules in this project
from product_definitions import (
    ProductStaticBase, QuantLibBondStaticBase, CallableBondStaticBase,
    ConvertibleBondStaticBase, EuropeanOptionStatic, MBSPoolStatic
)
from prepayment_models import (
    PrepaymentModelBase, ConstantCPRModel, PSAModel, RefiIncentivePrepaymentModel, create_prepayment_model_from_config
)

# --- Import PricerBase and specific pricers needed by the factory ---
from base_pricer import PricerBase
from mbs_pricer import MBSPricer

from custom_importer import import_class_from_module

def create_pricer(product_static: ProductStaticBase, pricer_config: dict) -> PricerBase:
    """
    Factory for creating PricerBase instances from a product static definition
    and a pricer configuration dictionary.

    The pricer_config should specify:
    - 'pricer_module_name' (str): The module where the pricer class is defined (e.g., 'fast_bond_pricer').
    - 'pricer_class_name' (str): The name of the pricer class (e.g., 'FastBondPricer').
    - 'pricer_params' (dict, optional): Parameters to pass to the pricer's __init__.
                                        This can include a 'prepayment_model_config'
                                        for MBSPricer.
    """
    module_name = pricer_config.get("pricer_module_name")
    class_name = pricer_config.get("pricer_class_name")
    init_params = pricer_config.get("pricer_params", {}).copy() # Make a copy to modify

    if not module_name:
        raise KeyError("Pricer configuration must include 'pricer_module_name'.")
    if not class_name:
        raise KeyError("Pricer configuration must include 'pricer_class_name'.")

    try:
        # Ensure PricerBase is available in the scope where import_class_from_module is called,
        # or that import_class_from_module can find it.
        pricer_class = import_class_from_module(module_name, class_name, expected_base_class=PricerBase)
    except (ImportError, AttributeError, TypeError) as e:
        raise ValueError(f"Failed to load pricer class '{class_name}' from module '{module_name}': {e}")

    # Special handling for MBSPricer's prepayment_model
    if pricer_class == MBSPricer and "prepayment_model_config" in init_params:
        pm_config = init_params.pop("prepayment_model_config") # Remove from init_params
        prepayment_model_instance = create_prepayment_model_from_config(pm_config)
        init_params["prepayment_model"] = prepayment_model_instance # Add instance to params
    elif pricer_class == MBSPricer and "prepayment_model" not in init_params:
        raise ValueError("MBSPricer requires 'prepayment_model' or 'prepayment_model_config' in pricer_params.")


    # Instantiate the pricer with product_static and other parameters
    try:
        # The first argument to pricer __init__ is product_static
        return pricer_class(product_static, **init_params)
    except TypeError as e:
        # This can catch issues like missing required arguments or unexpected arguments
        raise TypeError(f"Error instantiating pricer '{class_name}' with product_static and params {init_params}: {e}")