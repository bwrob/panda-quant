import numpy as np
import json
from colorama import Fore, Back, Style

def calculate_risk(exposure_dict):  
    
    # Define the order of currencies in the covariance matrix
    cov_order = ['USD', 'EUR', 'GBP', 'CHF', 'CAD', 'JPY']
    
    # Define the covariance matrix for the currency returns
    covariance_matrix = np.array([
        [0.0002, 0.0001, 0.0001, 0.0002, 0.0002, 0.0003],
        [0.0001, 0.0003, 0.0002, 0.0002, 0.0002, 0.0004],
        [0.0001, 0.0002, 0.0004, 0.0002, 0.0002, 0.0003],
        [0.0002, 0.0002, 0.0003, 0.0003, 0.0001, 0.0002],
        [0.0002, 0.0002, 0.0002, 0.0001, 0.0003, 0.0004],
        [0.0003, 0.0004, 0.0003, 0.0002, 0.0004, 0.0008]
    ])

    # Reorder the exposure dictionary to match the order of the covariance matrix
    exposures = np.array([exposure_dict[curr] for curr in cov_order])

    # Calculate the total variance of the portfolio
    variance = np.dot(exposures.T, np.dot(covariance_matrix, exposures))

    # Calculate the standard deviation of the portfolio
    standard_deviation = np.sqrt(variance)
    
    return standard_deviation

def get_fx_rate(ccy):
    fx_rates = {
        'USD': 1.0,
        'EUR': 1.215,  # 1 USD = 1.215 EUR
        'GBP': 1.394,  # 1 USD = 1.394 GBP
        'CHF': 1.096,  # 1 USD = 1.096 CHF
        'CAD': 0.824,  # 1 USD = 0.824 CAD
        'JPY': 0.0092  # 1 USD = 0.0092 JPY
    }
    
    return fx_rates[ccy]

def get_currency_exposure(cpty):
    return {'USD': 10000000, 'EUR': 50000000, 'GBP': 75000000, 'CHF': 25000000, 'CAD': 40000000, 'JPY': 200000000}

def get_limit(cpty):
    return 50000000.0

def add_exposures(exposure_dict1, exposure_dict2):
    # Combine the keys from both dictionaries
    all_currencies = set(exposure_dict1.keys()) | set(exposure_dict2.keys())

    # Create a new dictionary with the combined exposures
    combined_exposure_dict = {}
    for currency in all_currencies:
        exposure1 = exposure_dict1.get(currency, 0)
        exposure2 = exposure_dict2.get(currency, 0)
        combined_exposure = exposure1 + exposure2
        combined_exposure_dict[currency] = combined_exposure

    return combined_exposure_dict

def exposure_from_json(input_json: str):
    input_dict = json.loads(input_json)
    trade_data = input_dict['trade_json']

    notional_fx_rate = get_fx_rate(trade_data['Notional_Currency'])

    notional_usd = trade_data['Notional'] * notional_fx_rate

    exposure_dict = {
        trade_data['Currency1']: notional_usd,
        trade_data['Currency2']: -notional_usd
    }

    return exposure_dict, trade_data['Counterparty']

def get_approval(trade, verbose=True):
    exposure, cpty = exposure_from_json(trade)
    current_exposure = get_currency_exposure(cpty)
    new_exposure = add_exposures(current_exposure, exposure)
    limit = get_limit(cpty)
    risk = calculate_risk(new_exposure)
    approval =  limit > risk
    
    if verbose:
        print()
        print(Style.BRIGHT + "Counterparty = " + cpty)
        print()
        print("Trade exposure:")
        print(exposure)
        print()
        print("Total exposure:")
        print(new_exposure)
        print()
        print(Fore.GREEN + "Risk     = " + str(risk))
        print(Fore.BLUE  + "Limit    = " + str(limit))
        print(Fore.RED   + "Approval = " + str(approval))
        print(Style.RESET_ALL)
        
    return approval, risk, limit, cpty

