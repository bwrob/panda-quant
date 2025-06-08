import abc
import QuantLib as ql
import numpy as np # Added for type hint consistency, though PricerBase.price uses string
from product_definitions import ProductStaticBase

class PricerBase(abc.ABC):
    """
    Abstract base class for all pricers.
    """
    def __init__(self, product_static: ProductStaticBase):
        self.product_static: ProductStaticBase = product_static
        # Common QL settings setup, can be overridden if needed by specific pricers
        if hasattr(product_static, 'ql_valuation_date'):
            ql.Settings.instance().evaluationDate = product_static.ql_valuation_date
        elif hasattr(product_static, 'valuation_date_py'):
             eval_date = product_static.valuation_date_py
             ql.Settings.instance().evaluationDate = ql.Date(eval_date.day, eval_date.month, eval_date.year)


    @abc.abstractmethod
    def price(self, **kwargs) -> np.ndarray:
        """
        Price a single scenario or vector of inputs.
        Must be overridden by each pricer subclass.
        """
        pass

    @abc.abstractmethod
    def price_scenarios(
        self,
        raw_market_scenarios: np.ndarray,
        scenario_factor_names: list[str],
        rate_pillars: np.ndarray | None = None,
        **price_kwargs
    ) -> np.ndarray:
        """
        Slice out the relevant columns from `raw_market_scenarios` and
        call `.price(...)`.  Must be implemented per pricer type.
        """
        pass