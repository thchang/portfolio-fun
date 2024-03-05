from util import *

# Using yahoo finance
PROVIDER = "yfinance"

# Interval is 1 day
INTERVAL = "1d"

# The goal is a 6% annual return
ANNUAL_RETURN = 0.06
DAILY_RETURN = (ANNUAL_RETURN + 1) ** (1 / 365) - 1

# These are symbols for the largest American SW companies
SYMBOLS = {"meta": "Meta Platforms (Facebook)",
           "amzn": "Amazon",
           "aapl": "Apple",
           "nflx": "Netflix",
           "goog": "Alphabet (Google)",
           "msft": "Microsoft",
           "uber": "Uber",
           "lyft": "Lyft",
           "abnb": "Airbnb",
           "nvda": "NVIDIA",
           "amd" : "AMD",
           "intc": "Intel",
           "ibm" : "IBM",
           "dell": "Dell",
           "hpq" : "HP",
           "tsla": "Tesla",
           "orcl": "Oracle",
           "csco": "Cisco",
           "avgo": "Broadcom (VMware)",
           "txn" : "Texas Instruments",
           "adbe": "Adobe",
           "crm" : "Salesforce",
           "intu": "Intuit",
           "adp" : "ADP"
          }

daily_returns = load_returns(SYMBOLS, interval=INTERVAL, provider=PROVIDER)
exp_return, mark_risk, allocation = min_risk(daily_returns, DAILY_RETURN)
print_portfolio(SYMBOLS, allocation, exp_return, mark_risk, tol=1.0e-8)
