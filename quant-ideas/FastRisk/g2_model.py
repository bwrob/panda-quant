"""
Contains the G2Calibrator class for calibrating the G2++ interest rate model.
"""
import QuantLib as ql

class G2Calibrator:
    """
    Calibrates a G2++ (two-factor Hull-White) interest rate model
    to market cap/floor quotes.

    The G2++ model has 5 parameters: a, sigma, b, eta, rho.
    - a: mean reversion speed of the first factor (x)
    - sigma: volatility of the first factor (x)
    - b: mean reversion speed of the second factor (y)
    - eta: volatility of the second factor (y)
    - rho: correlation between the Wiener processes driving x and y

    Args:
        ts_handle (ql.YieldTermStructureHandle): Handle to the initial yield term structure
                                                 to which the model will be fitted. This term
                                                 structure defines the initial discount curve.
        index (ql.IborIndex or ql.OvernightIndex): The interest rate index underlying the
                                                   calibration instruments (e.g., caps/floors).
                                                   This index is used by the CapHelper to
                                                   determine caplet cashflows and fixings.
    """
    def __init__(self, ts_handle: ql.YieldTermStructureHandle, index: ql.InterestRateIndex):
        self.ts_handle: ql.YieldTermStructureHandle = ts_handle
        self.index: ql.InterestRateIndex = index
        # Initialize a G2 model. The parameters (a, sigma, b, eta, rho) will be calibrated.
        # The model is initialized with the provided term structure.
        self.model: ql.G2 = ql.G2(self.ts_handle)

    def calibrate(
        self,
        periods: list[ql.Period],
        quotes: list[ql.QuoteHandle], # Market volatilities (or prices) for caps/floors
        instrument_type: str = 'cap', # 'cap' or 'floor'
        strike_offset: float = 0.0, # For caps/floors, strike is often ATM + offset
                                    # If quotes are vols, strike might be implied or need to be set.
                                    # If quotes are prices, strike is inherent.
                                    # For simplicity, this example assumes vols and derives ATM strike.
        optimization_method: ql.OptimizationMethod = None,
        end_criteria: ql.EndCriteria = None,
        engine_steps: int = 100, # Increased default steps for tree engine
        vol_type: ql.VolatilityType = ql.Normal # ql.Normal or ql.ShiftedLognormal
    ) -> tuple[float, float, float, float, float]:
        """
        Calibrates the G2 model parameters to market cap/floor volatilities or prices.

        Args:
            periods (list[ql.Period]): List of ql.Period objects for cap/floor tenors.
            quotes (list[ql.QuoteHandle]): List of ql.QuoteHandle objects containing market
                                           volatilities or prices.
            instrument_type (str): 'cap' or 'floor'.
            strike_offset (float): Offset from ATM strike if applicable.
                                   If 0.0, ATM strike is used.
            optimization_method (ql.OptimizationMethod, optional): QuantLib optimization method.
            end_criteria (ql.EndCriteria, optional): QuantLib end criteria.
            engine_steps (int, optional): Time steps for TreeCapFloorEngine.
            vol_type (ql.VolatilityType, optional): Type of volatility in quotes.

        Returns:
            tuple[float, float, float, float, float]: Calibrated (a, sigma, b, eta, rho).
        """
        if len(periods) != len(quotes):
            raise ValueError("Length of periods and quotes must match.")
        if len(periods) < 5:
            print(f"Warning: Number of calibration instruments ({len(periods)}) is less than 5. "
                  "G2++ calibration might be unstable or fail.")

        helpers = []
        for period_obj, quote_handle in zip(periods, quotes):
            # Determine strike: For simplicity, assume ATM strike if not directly provided.
            # A more robust setup would take strikes explicitly or derive them carefully.
            # Here, we'll use a placeholder for strike determination.
            # If quote is volatility, CapHelper needs a strike.
            # If quote is price, strike is implicit in the instrument.
            # For vol quotes, often ATM strike is used.
            
            # This part is simplified: assumes quote_handle is for volatility
            # and we need to construct a cap/floor with a strike.
            # A common approach is to use ATM strikes.
            # For this example, we'll create an ATM cap/floor.
            # The strike for an ATM cap/floor is the forward rate of the index.
            # This requires the index to be able to provide forward rates.
            
            # Create a Cap/Floor instrument to get its ATM rate (simplification)
            # This is a bit circular if we are calibrating to vols of these instruments.
            # A proper setup would involve a consistent way to define strikes for calibration.
            # For now, let's assume the quote is a price quote or the strike is handled externally
            # if it's a vol quote. The ql.CapHelper takes a vol quote.

            # If using ql.CapHelper with vol quotes, it internally creates a cap/floor
            # with an ATM strike based on the index and its term structure.
            
            if instrument_type.lower() == 'cap':
                helper = ql.CapHelper(
                    period_obj,
                    quote_handle, # This is a handle to the market volatility
                    self.index,
                    self.index.frequency(), # Coupon frequency from index
                    self.index.dayCounter(),
                    self.index.fixedLegDayCounter(), # Not strictly used for cap helper with vol
                    False, # includeFirstCaplet
                    self.ts_handle, # Discounting curve
                    ql.BlackCalibrationHelper.RelativePriceError, # Error type
                    vol_type, # Volatility type of the quote
                    0.0 if vol_type == ql.Normal else 0.01 # Shift for Lognormal
                )
            elif instrument_type.lower() == 'floor':
                 helper = ql.FloorHelper(
                    period_obj,
                    quote_handle,
                    self.index, # Strike is ATM for FloorHelper by default
                    self.index.frequency(),
                    self.index.dayCounter(),
                    self.index.fixedLegDayCounter(),
                    False,
                    self.ts_handle,
                    ql.BlackCalibrationHelper.RelativePriceError,
                    vol_type,
                    0.0 if vol_type == ql.Normal else 0.01
                )
            else:
                raise ValueError(f"Unsupported instrument_type: {instrument_type}")

            engine = ql.TreeCapFloorEngine(self.model, engine_steps)
            helper.setPricingEngine(engine)
            helpers.append(helper)

        opt_method = optimization_method or ql.LevenbergMarquardt(1e-8, 1e-8, 1e-8)
        crit = end_criteria or ql.EndCriteria(
            maxIterations=10000,
            maxStationaryStateIterations=1000,
            rootEpsilon=1e-7, # Slightly relaxed from default
            functionEpsilon=1e-7,
            gradientNormEpsilon=1e-7
        )

        self.model.calibrate(helpers, opt_method, crit)
        
        # params() returns [a, sigma, b, eta, rho]
        calibrated_params = self.model.params()
        if any(p < 0 for p in calibrated_params[:4]) or not (-1 <= calibrated_params[4] <= 1):
             print(f"Warning: Calibration resulted in potentially invalid G2 params: {calibrated_params}")

        return tuple(calibrated_params)
