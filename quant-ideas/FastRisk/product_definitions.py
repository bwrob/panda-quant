"""
Contains classes for defining the static properties of various financial products.
Added MBSPoolStatic with simple prepayment assumptions.
"""
import QuantLib as ql
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import abc
import numpy as np

def _parse_date_input(date_input):
    """Helper to parse date input which could be date object or ISO string."""
    if isinstance(date_input, datetime):
        return date_input.date()
    if isinstance(date_input, date):
        return date_input
    if isinstance(date_input, str):
        try:
            return date.fromisoformat(date_input)
        except ValueError:
            try:
                dt_obj = datetime.fromisoformat(date_input.replace('Z', '+00:00').replace('z', '+00:00'))
                return dt_obj.date()
            except ValueError:
                raise ValueError(f"Invalid date string format: '{date_input}'. Expected YYYY-MM-DD or ISO datetime.")
    if date_input is None:
        return None
    raise TypeError(f"Unsupported date input type: {type(date_input)}. Value: {date_input}")

def _serialize_date_list(date_list):
    if date_list is None: return None
    return [_parse_date_input(d).isoformat() if _parse_date_input(d) else None for d in date_list]


def _parse_date_list(date_input_list):
    if date_input_list is None: return []
    return [_parse_date_input(d_str) if d_str else None for d_str in date_input_list]


class ProductStaticBase(abc.ABC):
    def __init__(self, valuation_date):
        self.valuation_date_py: date = _parse_date_input(valuation_date)

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, params: dict) -> 'ProductStaticBase':
        pass

    @abc.abstractmethod
    def to_dict(self) -> dict:
        pass


class QuantLibBondStaticBase(ProductStaticBase):
    def __init__(
        self,
        valuation_date, maturity_date, coupon_rate: float,
        face_value: float = 100.0, freq: int = 2,
        calendar = None, day_count = None,
        business_convention: int = None, settlement_days: int = 0,
        currency: str = "USD", index_stub: str = "GENERIC_IR",
        credit_spread_curve_name: str = None
    ):
        super().__init__(valuation_date)
        self.maturity_date_py: date = _parse_date_input(maturity_date)
        self.coupon_rate: float = float(coupon_rate)
        self.face_value: float = float(face_value)
        self.freq: int = int(freq)
        self.settlement_days: int = int(settlement_days)
        self.currency: str = currency
        self.index_stub: str = index_stub if index_stub and index_stub.strip() else "GENERIC_IR"
        self.credit_spread_curve_name: str = credit_spread_curve_name

        if isinstance(calendar, str):
            if calendar.lower() == "target": self.calendar_ql = ql.TARGET()
            elif calendar.lower() == "us_federalreserve": self.calendar_ql = ql.UnitedStates(ql.UnitedStates.FederalReserve)
            else: self.calendar_ql = ql.NullCalendar()
        elif isinstance(calendar, ql.Calendar): self.calendar_ql = calendar
        else:
            if self.currency.upper() == "USD": self.calendar_ql = ql.UnitedStates(ql.UnitedStates.FederalReserve)
            elif self.currency.upper() == "EUR": self.calendar_ql = ql.TARGET()
            else: self.calendar_ql = ql.TARGET()

        if isinstance(day_count, str):
            if day_count.lower() == "actualactualisda": self.day_count_ql = ql.ActualActual(ql.ActualActual.ISDA)
            elif day_count.lower() == "actual360": self.day_count_ql = ql.Actual360()
            elif day_count.lower() == "thirty360": self.day_count_ql = ql.Thirty360(ql.Thirty360.USA)
            else: self.day_count_ql = ql.Actual365Fixed()
        elif isinstance(day_count, ql.DayCounter): self.day_count_ql = day_count
        else: self.day_count_ql = ql.ActualActual(ql.ActualActual.ISDA)

        self.business_convention_ql: int = business_convention if business_convention is not None else ql.Following

        self.ql_valuation_date: ql.Date = ql.Date(self.valuation_date_py.day, self.valuation_date_py.month, self.valuation_date_py.year)
        self.ql_maturity_date: ql.Date = ql.Date(self.maturity_date_py.day, self.maturity_date_py.month, self.maturity_date_py.year)
        self.issue_date_ql: ql.Date = self.ql_valuation_date

        months_in_period = int(12 / self.freq)
        self.schedule: ql.Schedule = ql.Schedule(
            self.issue_date_ql, self.ql_maturity_date,
            ql.Period(months_in_period, ql.Months), self.calendar_ql,
            self.business_convention_ql, self.business_convention_ql,
            ql.DateGeneration.Forward, False)
        self.bond: ql.Bond = ql.FixedRateBond(
            self.settlement_days, self.face_value, self.schedule,
            [self.coupon_rate], self.day_count_ql,
            self.business_convention_ql, self.face_value)

    @classmethod
    def from_dict(cls, params: dict) -> 'QuantLibBondStaticBase':
        return cls(
            valuation_date=params['valuation_date'],
            maturity_date=params['maturity_date'],
            coupon_rate=float(params['coupon_rate']),
            face_value=float(params.get('face_value', 100.0)),
            freq=int(params.get('freq', 2)),
            calendar=params.get('calendar'), day_count=params.get('day_count'),
            business_convention=params.get('business_convention'),
            settlement_days=int(params.get('settlement_days', 0)),
            currency=params.get('currency', "USD"),
            index_stub=params.get('index_stub', "GENERIC_IR"),
            credit_spread_curve_name=params.get('credit_spread_curve_name', None)
        )

    def to_dict(self) -> dict:
        product_type = 'VanillaBond'
        if isinstance(self, CallableBondStaticBase): product_type = 'CallableBond'
        elif isinstance(self, ConvertibleBondStaticBase): product_type = 'ConvertibleBond'

        data = {
            'product_type': product_type,
            'valuation_date': self.valuation_date_py.isoformat(),
            'maturity_date': self.maturity_date_py.isoformat(),
            'coupon_rate': self.coupon_rate, 'face_value': self.face_value,
            'freq': self.freq, 'settlement_days': self.settlement_days,
            'currency': self.currency, 'index_stub': self.index_stub,
            'credit_spread_curve_name': self.credit_spread_curve_name,
            'calendar': self.calendar_ql.name() if self.calendar_ql else None,
            'day_count': self.day_count_ql.name() if self.day_count_ql else None,
        }
        return data


class CallableBondStaticBase(QuantLibBondStaticBase):
    def __init__(
        self, valuation_date, maturity_date, coupon_rate: float,
        call_dates: list, call_prices: list[float], face_value: float = 100.0,
        freq: int = 2, calendar = None, day_count = None,
        business_convention: int = None, settlement_days: int = 0,
        currency: str = "USD", index_stub: str = "GENERIC_IR",
        credit_spread_curve_name: str = None ):
        super().__init__(valuation_date, maturity_date, coupon_rate, face_value, freq,
                         calendar, day_count, business_convention, settlement_days,
                         currency, index_stub, credit_spread_curve_name)
        self.call_dates_py: list[date] = _parse_date_list(call_dates)
        self.call_prices_py: list[float] = [float(p) for p in call_prices]
        self.call_schedule: ql.CallabilitySchedule = ql.CallabilitySchedule()
        if self.call_dates_py:
            for cd_py, cp in zip(self.call_dates_py, self.call_prices_py):
                if cd_py is None: continue
                ql_cd = ql.Date(cd_py.day, cd_py.month, cd_py.year)
                call = ql.Callability(ql.BondPrice(cp, ql.BondPrice.Clean), ql.Callability.Call, ql_cd)
                self.call_schedule.push_back(call)
        self.bond: ql.CallableFixedRateBond = ql.CallableFixedRateBond(
            self.settlement_days, self.face_value, self.schedule, [self.coupon_rate],
            self.day_count_ql, self.business_convention_ql, self.face_value,
            self.issue_date_ql, self.call_schedule)

    @classmethod
    def from_dict(cls, params: dict) -> 'CallableBondStaticBase':
        return cls(
            valuation_date=params['valuation_date'], maturity_date=params['maturity_date'],
            coupon_rate=float(params['coupon_rate']),
            call_dates=params.get('call_dates', []), call_prices=params.get('call_prices',[]),
            face_value=float(params.get('face_value', 100.0)), freq=int(params.get('freq', 2)),
            calendar=params.get('calendar'), day_count=params.get('day_count'),
            business_convention=params.get('business_convention'),
            settlement_days=int(params.get('settlement_days', 0)),
            currency=params.get('currency', "USD"),
            index_stub=params.get('index_stub', "GENERIC_IR"),
            credit_spread_curve_name=params.get('credit_spread_curve_name', None)
        )

    def to_dict(self) -> dict:
      base = super().to_dict(); base.update({
          'product_type': 'CallableBond',
          'call_dates': _serialize_date_list(self.call_dates_py), 'call_prices': self.call_prices_py
      }); return base


class ConvertibleBondStaticBase(QuantLibBondStaticBase):
    def __init__(
        self, valuation_date, issue_date, maturity_date, coupon_rate: float,
        conversion_ratio: float, face_value: float = 100.0, freq: int = 2,
        settlement_days: int = 0, calendar = None, day_count = None,
        business_convention: int = None, exercise_type: str = 'EuropeanAtMaturity',
        currency: str = "USD", index_stub: str = "GENERIC_IR", underlying_symbol: str = None,
        credit_spread_curve_name: str = None ):
        super().__init__(valuation_date, maturity_date, coupon_rate, face_value, freq, calendar,
                         day_count, business_convention, settlement_days, currency, index_stub,
                         credit_spread_curve_name)
        self.issue_date_py: date = _parse_date_input(issue_date)
        self.issue_date_ql = ql.Date(self.issue_date_py.day, self.issue_date_py.month, self.issue_date_py.year)

        months_in_period = int(12 / self.freq)
        self.schedule = ql.Schedule(
            self.issue_date_ql, self.ql_maturity_date, ql.Period(months_in_period, ql.Months),
            self.calendar_ql, self.business_convention_ql, self.business_convention_ql,
            ql.DateGeneration.Forward, False)

        self.conversion_ratio: float = float(conversion_ratio)
        self.exercise_type_str: str = exercise_type
        self.underlying_symbol: str = underlying_symbol

        if self.exercise_type_str == 'EuropeanAtMaturity':
            self.exercise: ql.Exercise = ql.EuropeanExercise(self.ql_maturity_date)
        else: raise ValueError(f"Unsupported exercise type: {self.exercise_type_str}")
        self.convertible_call_schedule: ql.CallabilitySchedule = ql.CallabilitySchedule()
        self.bond: ql.ConvertibleFixedCouponBond = ql.ConvertibleFixedCouponBond(
            self.exercise, self.conversion_ratio, self.convertible_call_schedule,
            self.issue_date_ql, self.settlement_days, [self.coupon_rate],
            self.day_count_ql, self.schedule, self.face_value)

    @classmethod
    def from_dict(cls, params: dict) -> 'ConvertibleBondStaticBase':
        return cls(
            valuation_date=params['valuation_date'], issue_date=params['issue_date'],
            maturity_date=params['maturity_date'], coupon_rate=float(params['coupon_rate']),
            conversion_ratio=float(params['conversion_ratio']),
            face_value=float(params.get('face_value', 100.0)), freq=int(params.get('freq', 2)),
            settlement_days=int(params.get('settlement_days', 0)),
            calendar=params.get('calendar'), day_count=params.get('day_count'),
            business_convention=params.get('business_convention'),
            exercise_type=params.get('exercise_type', 'EuropeanAtMaturity'),
            currency=params.get('currency', "USD"),
            index_stub=params.get('index_stub', "GENERIC_IR"),
            underlying_symbol=params.get('underlying_symbol'),
            credit_spread_curve_name=params.get('credit_spread_curve_name', None)
        )

    def to_dict(self) -> dict:
        base = super().to_dict(); base.update({
            'product_type': 'ConvertibleBond', 'issue_date': self.issue_date_py.isoformat(),
            'conversion_ratio': self.conversion_ratio, 'exercise_type': self.exercise_type_str,
            'underlying_symbol': self.underlying_symbol
        }); return base


class EuropeanOptionStatic(ProductStaticBase):
    def __init__(self, valuation_date, expiry_date, strike_price: float, option_type: str,
                 day_count_convention = None, currency: str = "USD", underlying_symbol: str = None):
        super().__init__(valuation_date)
        self.expiry_date_py: date = _parse_date_input(expiry_date)
        self.strike_price: float = float(strike_price)
        self.currency: str = currency
        self.underlying_symbol: str = underlying_symbol

        if option_type.lower() not in ['call', 'put']: raise ValueError("Option type must be 'call' or 'put'")
        self.option_type: str = option_type.lower()

        ql_valuation_date = ql.Date(self.valuation_date_py.day, self.valuation_date_py.month, self.valuation_date_py.year)
        ql_expiry_date = ql.Date(self.expiry_date_py.day, self.expiry_date_py.month, self.expiry_date_py.year)

        if isinstance(day_count_convention, str):
            if day_count_convention.lower() == "actual365fixed": self.day_count_convention_ql = ql.Actual365Fixed()
            elif day_count_convention.lower() == "actual360": self.day_count_convention_ql = ql.Actual360()
            else: self.day_count_convention_ql = ql.Actual365Fixed()
        elif isinstance(day_count_convention, ql.DayCounter): self.day_count_convention_ql = day_count_convention
        else: self.day_count_convention_ql = ql.Actual365Fixed()

        self.time_to_expiry: float = self.day_count_convention_ql.yearFraction(ql_valuation_date, ql_expiry_date)
        if self.time_to_expiry < 0: self.time_to_expiry = 0.0

    @classmethod
    def from_dict(cls, params: dict) -> 'EuropeanOptionStatic':
        return cls(
            valuation_date=params['valuation_date'], expiry_date=params['expiry_date'],
            strike_price=float(params['strike_price']), option_type=params['option_type'],
            day_count_convention=params.get('day_count_convention'),
            currency=params.get('currency', "USD"),
            underlying_symbol=params.get('underlying_symbol'))

    def to_dict(self) -> dict:
        return {
            'product_type': 'EuropeanOption',
            'valuation_date': self.valuation_date_py.isoformat(),
            'expiry_date': self.expiry_date_py.isoformat(),
            'strike_price': self.strike_price, 'option_type': self.option_type,
            'day_count_convention': self.day_count_convention_ql.name() if self.day_count_convention_ql else None,
            'currency': self.currency, 'underlying_symbol': self.underlying_symbol
        }

# --- NEW: MBS Pool Static Definition ---
class MBSPoolStatic(ProductStaticBase):
    def __init__(self,
                 valuation_date: date,
                 issue_date: date,
                 original_balance: float,
                 current_balance: float,
                 wac: float, # Weighted Average Coupon
                 pass_through_rate: float, # Coupon passed to investors
                 original_term_months: int,
                 age_months: int, # Seasoning
                 prepayment_model_type: str, # e.g., "CPR", "PSA"
                 prepayment_rate_param: float, # CPR value or PSA multiplier
                 delay_days: int = 0,
                 currency: str = "USD",
                 index_stub: str = "GENERIC_IR", # For discount curve
                 credit_spread_curve_name: str = None): # For discount curve
        super().__init__(valuation_date)
        self.issue_date_py: date = _parse_date_input(issue_date)
        self.original_balance: float = float(original_balance)
        self.current_balance: float = float(current_balance)
        self.wac: float = float(wac)
        self.pass_through_rate: float = float(pass_through_rate)
        self.original_term_months: int = int(original_term_months)
        self.age_months: int = int(age_months)
        self.prepayment_model_type: str = prepayment_model_type
        self.prepayment_rate_param: float = float(prepayment_rate_param)
        self.delay_days: int = int(delay_days)
        self.currency: str = currency
        self.index_stub: str = index_stub if index_stub and index_stub.strip() else "GENERIC_IR"
        self.credit_spread_curve_name: str = credit_spread_curve_name

        self.ql_valuation_date: ql.Date = ql.Date(self.valuation_date_py.day, self.valuation_date_py.month, self.valuation_date_py.year)
                
        # Derived: Remaining term
        self.remaining_term_months = self.original_term_months - self.age_months
        if self.remaining_term_months < 0:
            self.remaining_term_months = 0
            print(f"Warning: MBS pool age {self.age_months} exceeds original term {self.original_term_months}. Remaining term set to 0.")

    @classmethod
    def from_dict(cls, params: dict) -> 'MBSPoolStatic':
        return cls(
            valuation_date=params['valuation_date'],
            issue_date=params['issue_date'],
            original_balance=float(params['original_balance']),
            current_balance=float(params['current_balance']),
            wac=float(params['wac']),
            pass_through_rate=float(params['pass_through_rate']),
            original_term_months=int(params['original_term_months']),
            age_months=int(params['age_months']),
            prepayment_model_type=params.get('prepayment_model_type', "CPR"),
            prepayment_rate_param=params.get('prepayment_rate_param', 0.0),
            delay_days=int(params.get('delay_days', 0)),
            currency=params.get('currency', "USD"),
            index_stub=params.get('index_stub', "GENERIC_IR"),
            credit_spread_curve_name=params.get('credit_spread_curve_name', None)
        )

    def to_dict(self) -> dict:
        return {
            'product_type': 'MBSPool',
            'valuation_date': self.valuation_date_py.isoformat(),
            'issue_date': self.issue_date_py.isoformat(),
            'original_balance': self.original_balance,
            'current_balance': self.current_balance,
            'wac': self.wac,
            'pass_through_rate': self.pass_through_rate,
            'original_term_months': self.original_term_months,
            'age_months': self.age_months,
            'prepayment_model_type': self.prepayment_model_type,
            'prepayment_rate_param': self.prepayment_rate_param,
            'delay_days': self.delay_days,
            'currency': self.currency,
            'index_stub': self.index_stub,
            'credit_spread_curve_name': self.credit_spread_curve_name
        }

