
def load_returns(symbols, interval="1d", provider="yfinance", start_date=None):
    """ Load and calculate historical returns over a specified period.

    This is a thinly-veiled wrapper for openbb

    Args:
        symbols (dict): A dictionary of symbol (key), company name (value)
            pairs specifying the companies whose data will be loaded.
        interval (str, optional): The intervals over which returns are
            calculated, one of ["1m", "1h", "1d", "1W", "1M"].
            Defaults to "1d"
        provider (str, optional): The provider for the opening/closing
            costs over the specified intervals. Defaults to "yfinance"
        start_date (str, optional): The start date to calculate historical
            returns over in "yyyy-mm-dd" format. Defaults to one year ago

    Returns:
        ndarray: A 2D array of historical returns. The rows correspond to
        keys from the symbols arg, and the columns correspond to market
        trading days since start_date (earliest first)

    """

    from datetime import date, timedelta
    from openbb import obb
    import numpy as np

    if start_date is None:
        start_date = (date.today() - timedelta(days=365)).isoformat()
    for i, symi in enumerate(symbols):
        dfi = obb.equity.price.historical(symi,
                                          interval=interval,
                                          start_date=start_date,
                                          provider=provider
                                         ).to_df()
        if i == 0:
            opening_prices = dfi["open"].values
            closing_prices = dfi["close"].values
        else:
            opening_prices = np.vstack((opening_prices, dfi["open"].values))
            closing_prices = np.vstack((closing_prices, dfi["close"].values))
    return (closing_prices - opening_prices) / opening_prices


def min_risk(daily_returns, target_daily_return):
    """ Calculate the min-risk portfolio that meets target returns

    Sets up and solves the QP:

    min x' C x
    s.t.
    r' x >= t
    x >= 0
    1' x == 1

    where C is the covariance matrix for the daily returns matrix;
    r is the expected daily return for each row of the daily returns matrix;
    t is the target daily return; and
    x is the optimization variable representing proportions of portfolio
    allocated to each asset type

    Args:
        daily_returns (ndarray): A 2D array of historical returns. The rows
            correspond to asset types and the columns correspond to historical
            daily returns
        target_daily_return (float): The target daily return for this portfolio
            allocation

    Returns:
        float, float, ndarray:
        The expected daily return, which should usually be close to the target,
        The market risk (daily return variance) obtained, and
        The optimal portfolio allocation attaining this risk

    """

    import cvxpy as cp
    import numpy as np

    n = daily_returns.shape[0]
    r = np.mean(daily_returns, axis=1)
    C = daily_returns @ daily_returns.T
    b = target_daily_return
    ones = np.ones(n)
    x = cp.Variable(n)
    qp = cp.Problem(cp.Minimize(cp.quad_form(x, C)),          # Objective
                    [r.T @ x >= b, x >= 0, ones.T @ x == 1])  # Constraints
    qp.solve(solver="ECOS", max_iters=10000, reltol=1.8e-12)
    exp_return = r.T @ x.value
    return exp_return, qp.value, x.value


def print_portfolio(symbols, allocations, expected_return, market_risk, tol=None):
    """ Display a nicely-formatted description of a potential portfolio

    Args:
        symbols (dict): A dictionary of symbol (key), company name (value)
            pairs specifying the potential assets in the portfolio
        allocations (ndarray): A 1D array of proportions of assets allocated
            to each key in symbols
        expected_return (float): The expected daily return for this portfolio
        market_risk (float): The daily risk (variance) for this portfolio
        tol (float, optional): The tolerance for displaying an allocation.
            When None (default) all allocations are shown. When tol>0, only
            allocations whose proportion is greater than tol are shown

    """

    if tol is None:
        tol = 0
    print(f"Expected return: {expected_return}")
    print(f"Risk (standard deviation): {market_risk}")
    print("Portfolio allocation:")
    for i, symi in enumerate(symbols):
        if allocations[i] >= tol:
            print(f"\t{symbols[symi]}: {allocations[i]}")
