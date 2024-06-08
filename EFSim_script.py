# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
import seaborn as sns
from pypfopt import EfficientFrontier
import janitor
from tabulate import tabulate
yf.pdr_override()
import sys
import warnings
warnings.filterwarnings('ignore')

# %%
'''
TODO fix standard deviation in df(??)
TODO add user inputs for stocks and weights
TODO 
'''

# %%
# Import data
def collect_securities():
    print('Set your portolio of two or more stocks; enter stock tickers:')
    print('(press enter when finished)')

    stock_portfolio = []
    stock_num = 1

    run = True

    while run:
        stock = str(input('Stock {}:'.format(stock_num)).upper().strip())
        stock_num += 1
        if stock == "":
            run = False
        else:
            stock_portfolio.append(stock)

    return stock_portfolio

def get_historic_time_horizon():
    time_unit_run = True
    time_unit_count_run = True
    time_delta = 0
    valid_time_units = {'d':'days', 'm':'months', 'y': 'years', 's':'date'}
    while time_unit_run:
    
        time_unit_abbr = input("Enter preferred time unit of stock returns ('d' for days, 'm' for months, 'y' for years, 's' for specified date): ")

        if time_unit_abbr in valid_time_units.keys():
            time_unit_run = False

            while time_unit_count_run:
                if time_unit_abbr == 's':
                    try:
                        date = input('Enter the specified date (MM/DD/YYYY): ')
                        date = date.split('/')
                        time_delta = dt.date(int(date[2]), int(date[0]), int(date[1]))
                        time_unit_count_run = False
                    except ValueError:
                        print('Invalid date. Please try again')

                else:
                    try:
                        time_unit_count = float(input(f'Enter the number of {valid_time_units[time_unit_abbr]}: '))
                        time_unit_count_run = False
                    except ValueError:
                        print('Invalid entry. Please try again')

                    if time_unit_abbr == 'y':
                        time_delta = time_unit_count * 366
                    elif time_unit_abbr == 'm':
                        time_delta = (time_unit_count * 30) + (time_unit_count // 2)
                    else:
                        time_delta = time_unit_count
                    
        else:
            print('Invalid entry. Please try again')
             

    return time_delta

def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Adj Close']
    # returns = (np.log(stockData) - np.log(stockData.shift(1))).dropna()
    returns = (np.log(stockData / stockData.shift(1))).dropna()
    return returns

def get_returns(stock_portfolio, time_delta):
    try:
        returns = get_data(stock_portfolio, dt.datetime.today() - dt.timedelta(time_delta) ,dt.datetime.today())
    except:
        returns = get_data(stock_portfolio, time_delta, dt.datetime.today())
    return returns
    
def calculate_mean_returns(returns, stock_portfolio):
    stocks = returns.columns
    returns_dict = {stock: np.mean(returns[stock]) * 252 for stock in stocks}
    returns_dict = {k: returns_dict[k] for k in stock_portfolio}
    return returns_dict

def calculate_standard_deviation(returns):
    stocks = returns.columns
    std_devs_dict = {stock: np.sqrt(np.var(returns[stock].values) * 252) for stock in stocks}
    return std_devs_dict

def get_covariance_matrix(returns, stock_portfolio):
    cov = returns.cov().reorder_columns(stock_portfolio)
    cov = cov.reindex(stock_portfolio)
    return cov
    # return np.cov(returns.T)

# prompt user for portfolio weights
# give option of either percentages or dollar amount ????
def get_weights(stock_portfolio):
    # weights_dict = {stock: float(input('Enter portfolio weight: ')) for stock in stock_portfolio}
    weights_dict = {stock: 1./len(stock_portfolio) for stock in stock_portfolio}
    return weights_dict

def create_portfolio_df(returns, std_devs, portfolio, weights):
    df_index = ['Annual Return', 'Standard Deviation', 'Weight']
    df = pd.DataFrame(np.nan, index=df_index, columns=stock_portfolio)
    for stock in portfolio:
        df.loc['Annual Return'][stock] = returns[stock]
        df.loc['Standard Deviation'][stock] = std_devs[stock]
        df.loc['Weight'][stock] = weights[stock]
    return df

def calculate_portfolio_stats(df, cov_matrix):
    port_std_dev = np.sqrt((np.dot(np.dot(df.loc['Weight'], cov_matrix), df.loc['Weight'].T) * 252))
    port_return = np.dot(df.loc['Weight'], df.loc['Annual Return'])
    return {'Standard Deviation': port_std_dev, 
            'Return': port_return}

def calc_mvp(mean_returns, std_devs, stock_portfolio, cov_matrix):
    ef = EfficientFrontier(pd.Series(mean_returns), cov_matrix)
    mvp_weights = ef.min_volatility()
    mvp = create_portfolio_df(mean_returns, std_devs, stock_portfolio, mvp_weights)
    mvp_performance = calculate_portfolio_stats(mvp, cov_matrix)
    return mvp, mvp_performance

def calc_orp(mean_returns, std_devs, stock_portfolio, cov_matrix):
    ef = EfficientFrontier(pd.Series(mean_returns), cov_matrix)
    orp_weights = ef.max_sharpe()
    orp = create_portfolio_df(mean_returns, std_devs, stock_portfolio, orp_weights)
    orp_performance = calculate_portfolio_stats(orp, cov_matrix)
    return orp, orp_performance

def plot_ef(mean_returns, cov_matrix, stock_portfolio, mvp_performance, orp_performance, curr_portfolio=None):
    sims = run_mc_sim(mean_returns, cov_matrix, stock_portfolio)
    sns.scatterplot(data=sims, x='Volatility', y='Return', linewidth=0, label='Simulations', color='lightsteelblue')
    plt.plot(mvp_performance['Standard Deviation'], mvp_performance['Return'], 'ro', label='Min. Variance Portfolio')
    plt.plot(orp_performance['Standard Deviation'], orp_performance['Return'], marker='*', color='green', label='Optimal Risky Portfolio', markersize=10)
    plt.legend()
    plt.show()

def get_ef_line(mvp, orp, cov_matrix):
    weight_in_mvp_list = np.arange(-3, 3.1, 0.05)
    test_dict = {'Weight in MVP': weight_in_mvp_list}

    covariance_ef = np.dot(np.dot(mvp.loc['Weight'], cov_matrix), orp.loc['Weight'].T)

    ef_line_sim = pd.DataFrame(test_dict)
    ef_line_sim['return'] = (mvp_performance['Return'] * ef_line_sim['Weight in MVP']) + (orp_performance['Return'] * (1 - ef_line_sim['Weight in MVP']))
    ef_line_sim['standard deviation'] = np.sqrt((ef_line_sim['Weight in MVP']**2 * mvp_performance['Standard Deviation']**2) + ((1 - ef_line_sim['Weight in MVP'])**2 * orp_performance['Standard Deviation']**2) + (2 * ef_line_sim['Weight in MVP'] * (ef_line_sim['Weight in MVP'] - 1) * covariance_ef))
    return ef_line_sim

def run_mc_sim(mean_returns, cov_matrix, stock_portfolio):
    simReturns = []
    simStDevs = []

    for _ in range(75000):
        fake_weights = np.random.random(len(stock_portfolio))
        fake_weights /= np. sum(fake_weights)
        np_mean_returns = list(mean_returns.values())
        np_mean_returns = np.array(np_mean_returns)
        simReturn = np.inner(fake_weights, np_mean_returns)
        simPortfolioStDev = np.sqrt((np.dot(np.dot(fake_weights, cov_matrix), fake_weights.T) * 252))
        simReturns.append(simReturn)
        simStDevs.append(simPortfolioStDev)

    return pd.DataFrame({'Return': simReturns, 'Volatility': simStDevs})

def format_as_percentage(x):
    return "{:.5%}".format(x)

def print_mvp_orp(mvp, mvp_performance, orp, orp_performance):
    # format for user readability
    read_mvp = mvp.applymap(format_as_percentage)
    read_orp = orp.applymap(format_as_percentage)

    print('Minimum Variance Portfolio\n--------------------------------------------------------------------------------------------------------------')
    print(tabulate(read_mvp, headers='keys', tablefmt = 'fancy_grid'))
    print('MVP Return:', '{:.5%}'.format(mvp_performance['Return']))
    print('MVP Standard Deviation:', '{:.5%}'.format(mvp_performance['Standard Deviation']))
    print('--------------------------------------------------------------------------------------------------------------\n\n')

    print('Optimal Risky Portfolio\n--------------------------------------------------------------------------------------------------------------')
    print(tabulate(read_orp, headers='keys', tablefmt = 'fancy_grid'))
    print('ORP Return:', '{:.5%}'.format(orp_performance['Return']))
    print('ORP Standard Deviation:', '{:.5%}'.format(orp_performance['Standard Deviation']))

# %%
stock_portfolio = collect_securities()
# stock_portfolio = ['VOO', 'CAT', 'NVDA', 'COST', 'NKE'] #example

if len(stock_portfolio) < 2:
    sys.exit('Portfolio needs to contain at least 2 securities')

test = pdr.get_data_yahoo(stock_portfolio, dt.datetime.today() - dt.timedelta(366), dt.datetime.today())

# %%
time_delta = get_historic_time_horizon()

# %%
returns = get_returns(stock_portfolio, time_delta)
mean_returns = calculate_mean_returns(returns, stock_portfolio)
std_devs = calculate_standard_deviation(returns)
weights = get_weights(stock_portfolio)
cov_matrix = get_covariance_matrix(returns, stock_portfolio)

# %%
current_port = create_portfolio_df(mean_returns, std_devs, stock_portfolio, weights)

# %%
orp, orp_performance = calc_orp(mean_returns, std_devs, stock_portfolio, cov_matrix)
mvp, mvp_performance = calc_mvp(mean_returns, std_devs, stock_portfolio, cov_matrix)

# %%
print_mvp_orp(mvp, mvp_performance, orp, orp_performance)

# %%
plot_ef(mean_returns, cov_matrix, stock_portfolio, mvp_performance, orp_performance)


