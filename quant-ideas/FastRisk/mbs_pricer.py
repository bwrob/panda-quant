import QuantLib as ql
import numpy as np
from datetime import date # Not strictly used but often useful with QL
from dateutil.relativedelta import relativedelta

from base_pricer import PricerBase
from product_definitions import MBSPoolStatic
from prepayment_models import (
    PrepaymentModelBase, ConstantCPRModel, PSAModel, RefiIncentivePrepaymentModel # Added for completeness
)
# Import QuantLibBondPricer to use its static methods for curve building
from quantlib_bond_pricer import QuantLibBondPricer


class MBSPricer(PricerBase):
    def __init__(self, mbs_static: MBSPoolStatic, prepayment_model: PrepaymentModelBase):
        if not isinstance(mbs_static, MBSPoolStatic):
            raise TypeError("MBSPricer requires an MBSPoolStatic instance.")
        if not isinstance(prepayment_model, PrepaymentModelBase): 
            raise TypeError("MBSPricer requires a PrepaymentModelBase instance.")
        super().__init__(mbs_static)
        self.mbs_static: MBSPoolStatic = mbs_static 
        self.prepayment_model = prepayment_model

    def _calculate_scheduled_payment_and_principal(self, balance, wac_monthly, remaining_periods):
        """Calculates monthly scheduled payment and principal portion."""
        if wac_monthly < 1e-9: 
            if remaining_periods == 0: return 0.0, 0.0
            payment = balance / remaining_periods if remaining_periods > 0 else 0.0
            return payment, payment 

        if remaining_periods <=0 : return 0.0, 0.0

        payment = balance * (wac_monthly * (1 + wac_monthly)**remaining_periods) / \
                  ((1 + wac_monthly)**remaining_periods - 1)
        interest_portion = balance * wac_monthly
        scheduled_principal = payment - interest_portion
        return payment, scheduled_principal


    def price(self, pillar_times_rf: np.ndarray, market_scenario_data: np.ndarray,
              credit_spread_pillar_times: np.ndarray = None,
              fixed_market_mortgage_rate_for_prepay: float = None, 
              **kwargs) -> np.ndarray:

        if market_scenario_data.ndim == 1:
            market_scenario_data = market_scenario_data.reshape(1, -1)

        num_scenarios = market_scenario_data.shape[0]
        npvs = np.zeros(num_scenarios)

        eval_date_py = self.mbs_static.valuation_date_py
        eval_date_ql = ql.Date(eval_date_py.day, eval_date_py.month, eval_date_py.year)
        
        dc_discount = ql.ActualActual(ql.ActualActual.ISDA)
        calendar_discount = ql.TARGET() 

        for i in range(num_scenarios):
            scen_data_row = market_scenario_data[i, :]
            ql.Settings.instance().evaluationDate = eval_date_ql 

            num_rf_pillars = len(pillar_times_rf)
            rates_scen_rf = scen_data_row[:num_rf_pillars]
            current_scen_idx = num_rf_pillars

            spreads_scen_cs = None
            cs_pillar_times_to_use = credit_spread_pillar_times

            if hasattr(self.mbs_static, 'credit_spread_curve_name') and \
               self.mbs_static.credit_spread_curve_name and \
               cs_pillar_times_to_use is not None and cs_pillar_times_to_use.size > 0:
                num_cs_pillars = len(cs_pillar_times_to_use)
                if len(scen_data_row) >= current_scen_idx + num_cs_pillars:
                    spreads_scen_cs = scen_data_row[current_scen_idx : current_scen_idx + num_cs_pillars]
                else: cs_pillar_times_to_use = None 
            
            dates_rf_scen = QuantLibBondPricer._build_ql_dates(self.mbs_static, eval_date_ql, pillar_times_rf)
            eff_rates_rf_scen = QuantLibBondPricer._align_rates(self.mbs_static, dates_rf_scen, rates_scen_rf)
            
            final_rates_for_curve = eff_rates_rf_scen
            if cs_pillar_times_to_use is not None and spreads_scen_cs is not None:
                if not np.array_equal(pillar_times_rf, cs_pillar_times_to_use):
                    interp_spreads = np.interp(pillar_times_rf, cs_pillar_times_to_use, spreads_scen_cs)
                else:
                    interp_spreads = spreads_scen_cs
                
                combined_pillar_rates = rates_scen_rf + interp_spreads
                final_rates_for_curve = QuantLibBondPricer._align_rates(self.mbs_static, dates_rf_scen, combined_pillar_rates)

            discount_curve_scen = ql.ZeroCurve(dates_rf_scen, final_rates_for_curve, dc_discount, calendar_discount, ql.Linear(), ql.Continuous, ql.Annual)
            discount_curve_scen.enableExtrapolation()
            discounting_ts_handle_scen = ql.YieldTermStructureHandle(discount_curve_scen)

            current_balance = self.mbs_static.current_balance
            total_npv_for_scenario = 0.0
            wac_monthly = self.mbs_static.wac / 12.0
            pass_through_monthly = self.mbs_static.pass_through_rate / 12.0

            for month_idx in range(1, self.mbs_static.remaining_term_months + 1):
                if current_balance < 1e-2: break 

                age_at_period_start = self.mbs_static.age_months + month_idx -1 
                remaining_periods_at_period_start = self.mbs_static.original_term_months - age_at_period_start
                if remaining_periods_at_period_start <= 0: break

                # interest_accrued_on_wac = current_balance * wac_monthly # Not directly used for investor CF

                _, scheduled_principal = self._calculate_scheduled_payment_and_principal(
                    current_balance, wac_monthly, remaining_periods_at_period_start
                )
                scheduled_principal = min(scheduled_principal, current_balance) 

                period_info = {'age_months': age_at_period_start, 'wac_pool': self.mbs_static.wac}
                if isinstance(self.prepayment_model, RefiIncentivePrepaymentModel):
                    cm_proxy = fixed_market_mortgage_rate_for_prepay
                    if cm_proxy is None: 
                        cm_proxy = discounting_ts_handle_scen.zeroRate(10.0, ql.Continuous, ql.Annual).rate()
                    period_info['current_market_mortgage_rate_cm'] = cm_proxy
                
                smm = self.prepayment_model.get_smm(period_info)
                prepayment_amount = (current_balance - scheduled_principal) * smm 
                prepayment_amount = min(prepayment_amount, current_balance - scheduled_principal) 

                total_principal_paid = scheduled_principal + prepayment_amount
                interest_passed_through = current_balance * pass_through_monthly
                total_investor_cashflow = interest_passed_through + total_principal_paid

                payment_date_actual = eval_date_py + relativedelta(months=month_idx, days=self.mbs_static.delay_days)
                payment_date_ql = ql.Date(payment_date_actual.day, payment_date_actual.month, payment_date_actual.year)
                
                df = discounting_ts_handle_scen.discount(payment_date_ql)

                total_npv_for_scenario += total_investor_cashflow * df
                current_balance -= total_principal_paid

            npvs[i] = total_npv_for_scenario
        return npvs
    
    def price_scenarios(
        self,
        raw_market_scenarios: np.ndarray,
        scenario_factor_names: list[str],
        rate_pillars: np.ndarray | None = None,
        **price_kwargs
    ) -> np.ndarray:
        if rate_pillars is None:
            raise ValueError("rate_pillars must be provided for MBSPricer.")
        n = len(rate_pillars)
        try:
            idx = [scenario_factor_names.index(str(t)) for t in rate_pillars]
            if len(idx) != n:
                idx = list(range(n))
        except Exception:
            idx = list(range(n))
        data = raw_market_scenarios[:, idx]
        return self.price(
            pillar_times_rf=rate_pillars,
            market_scenario_data=data,
            **price_kwargs
        )