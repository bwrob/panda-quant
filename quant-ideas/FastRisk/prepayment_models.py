"""
Contains classes for modeling mortgage prepayments.
Includes ConstantCPRModel, PSAModel, and RefiIncentivePrepaymentModel.
"""
import abc
import numpy as np
from custom_importer import import_class_from_module

class PrepaymentModelBase(abc.ABC):
    """Abstract base class for prepayment models."""
    @abc.abstractmethod
    def get_smm(self, period_info: dict) -> float:
        """
        Calculates the Single Monthly Mortality (SMM) for the current period.

        Args:
            period_info (dict): A dictionary containing information relevant to
                                the current period for prepayment calculation, e.g.,
                                'age_months': current age of the pool in months.
                                'wac_pool': weighted average coupon of the pool.
                                'current_market_mortgage_rate_cm': current market mortgage rate.
                                (and other factors as models become more complex).

        Returns:
            float: The Single Monthly Mortality rate (as a decimal, e.g., 0.005).
        """
        pass

class ConstantCPRModel(PrepaymentModelBase):
    """Models prepayment based on a constant annual CPR."""
    def __init__(self, cpr_annual: float):
        if not (0.0 <= cpr_annual <= 1.0):
            raise ValueError("Annual CPR must be between 0 and 1.")
        self.cpr_annual = cpr_annual

    def get_smm(self, period_info: dict) -> float:
        """Calculates SMM from the constant annual CPR."""
        # SMM = 1 - (1 - CPR)^(1/12)
        return 1.0 - (1.0 - self.cpr_annual)**(1.0/12.0)

class PSAModel(PrepaymentModelBase):
    """Models prepayment based on the PSA (Public Securities Association) standard."""
    def __init__(self, psa_multiplier: float): # e.g., 100.0 for 100% PSA
        if psa_multiplier < 0:
            raise ValueError("PSA multiplier cannot be negative.")
        self.psa_multiplier = psa_multiplier / 100.0 # Convert to decimal factor

    def get_smm(self, period_info: dict) -> float:
        """
        Calculates SMM based on the PSA model.
        Requires 'age_months' in period_info.
        """
        age_months = period_info.get('age_months')
        if age_months is None:
            raise ValueError("'age_months' must be provided in period_info for PSAModel.")

        if age_months <= 0:
            cpr_psa_benchmark = 0.0
        elif age_months < 30:
            # CPR increases linearly from 0% to 6% over 30 months (0.2% per month)
            cpr_psa_benchmark = 0.06 * (age_months / 30.0)
        else:
            # CPR is 6% for months 30 and beyond
            cpr_psa_benchmark = 0.06

        cpr_actual = cpr_psa_benchmark * self.psa_multiplier
        cpr_actual = min(max(cpr_actual, 0.0), 1.0) # Clamp CPR between 0 and 1

        return 1.0 - (1.0 - cpr_actual)**(1.0/12.0)


class RefiIncentivePrepaymentModel(PrepaymentModelBase):
    """
    Models prepayment based on refinancing incentive.
    SMM = A + B * arctan(C * (WAC_pool - C_M) + D)
    WAC_pool: Weighted Average Coupon of the mortgage pool.
    C_M: Current market mortgage rate.
    A, B, C, D are model parameters.
    """
    def __init__(self, A: float = 0.003, B: float = 0.02, C: float = 10.0, D: float = -0.5):
        self.A = A
        self.B = B
        self.C = C
        self.D = D

    def get_smm(self, period_info: dict) -> float:
        """
        Calculates SMM based on the refinancing incentive model.
        Requires 'wac_pool' and 'current_market_mortgage_rate_cm' in period_info.
        """
        wac_pool = period_info.get('wac_pool')
        current_market_mortgage_rate_cm = period_info.get('current_market_mortgage_rate_cm')

        if wac_pool is None or current_market_mortgage_rate_cm is None:
            raise ValueError("'wac_pool' and 'current_market_mortgage_rate_cm' must be provided "
                             "in period_info for RefiIncentivePrepaymentModel.")

        refi_incentive = wac_pool - current_market_mortgage_rate_cm
        smm = self.A + self.B * np.arctan(self.C * refi_incentive + self.D)

        # SMM should be non-negative and typically not excessively large (e.g., > 0.5 for a month)
        # Clamping SMM to a reasonable range, e.g., [0, 0.9] to avoid (1-SMM) being negative.
        return min(max(smm, 0.0), 0.9)

def create_prepayment_model_from_config(config: dict) -> PrepaymentModelBase:
    """
    Factory function to create an instance of a prepayment model from a configuration dictionary.

    Args:
        config (dict): Configuration dictionary specifying the prepayment model to instantiate.
            Must contain the following keys:
                - 'module_name' (str): The name of the module containing the model class.
                - 'class_name' (str): The name of the model class to instantiate.
                - 'params' (dict, optional): A dictionary of parameters to pass to the model's constructor.

    Returns:
        PrepaymentModelBase: An instance of the specified prepayment model class.
    """
    module_name = config.get("module_name")
    class_name = config.get("class_name")
    model_params = config.get("params", {})

    if not module_name:
        raise KeyError("Prepayment model configuration must include 'module_name'.")
    if not class_name:
        raise KeyError("Prepayment model configuration must include 'class_name'.")

    try:
        # Ensure the class is imported from prepayment_models or a valid custom module
        # and that it inherits from PrepaymentModelBase
        model_class = import_class_from_module(module_name, class_name, expected_base_class=PrepaymentModelBase)
    except (ImportError, AttributeError, TypeError) as e:
        raise ValueError(f"Failed to load prepayment model class '{class_name}' from module '{module_name}': {e}")

    if not hasattr(model_class, '__init__'):
        raise NotImplementedError(f"Class '{class_name}' does not have an '__init__' method.")

    # Instantiate the model with its specific parameters
    try:
        return model_class(**model_params)
    except TypeError as e:
        raise TypeError(f"Error instantiating prepayment model '{class_name}' with params {model_params}: {e}")