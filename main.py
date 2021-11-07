import pandas as pd
from pandas_datareader import data as pdr
import numpy as np
import datetime as dt
from scipy.stats import norm
from matplotlib import pyplot as plt

tickers = ['AAPL','IBM', 'GOOG', 'BP','XOM','COST','GS']
weights = np.array([.15, .20, .20, .15,.10,.15,.05])
initial_investment = 100

#pulling close prices from yahoo
data = pdr.get_data_yahoo(tickers, start="2016-01-01", end="2016-12-31")['Close']

#finding daily change of returns
returns = data.pct_change()
print(returns.tail())

#finding mean of returns of the distribution
avg_rets = returns.mean()
print(avg_rets)

#finding mean in % of portfolio returns
port_mean = avg_rets.dot(weights)
print(port_mean)

#finding cov matrix
cov_matrix = returns.cov()
print(cov_matrix)

#finding std deviation of portfolio
port_stdev = np.sqrt(weights.T.dot(cov_matrix).dot(weights))
print(port_stdev)

#finding absolute mean of portfolio w 100USD invested
mean_investment = (1+port_mean) * initial_investment

#finding std deviation of portfolio w 100USD invested
stdev_investment = initial_investment * port_stdev

#using alpha as 5%
conf_level1 = 0.05

#finding portfolio value at 5% left tail, ppf is percent point function, inverse of CDF 
cutoff1 = norm.ppf(conf_level1, mean_investment, stdev_investment)
print('Portfolio value at 5% left tail',cutoff1)

#finding Value at Risk daily
var_1d1 = initial_investment - cutoff1
print(f'The value at risk for 1-day period is {np.round(var_1d1,2)} at 95% confidence interval')

#plotting Value at Risk for 365days
var_array = []
num_days = 250
for x in range(1, num_days+1):    
    var_array.append(np.round(var_1d1 * np.sqrt(x),2)) #1 day VaR * sqrt of time, this is due to the fact that the standard deviation of stock returns tends to increase with the square root of time

#Value at Risk for 1 year period
print(f'The value at risk for 1 year period is {np.round(var_1d1 * np.sqrt(250),2)} at 95% confidence interval')

plt.xlabel("Trading Day")
plt.ylabel("Max portfolio loss (USD)")
plt.title("Max portfolio loss (VaR) over 1-year period")
plt.plot(var_array, "r")

num_portfolios = 30000

#generating random portfolios with different weights
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *250
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(250)
    return std, returns

def random_portfolios(num_portfolios, avg_rets, cov_matrix):
    #creating array for each portfolio std_dev, returns and sharpe ratio
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        #Generating random weights
        weights = np.random.random(7)
        
        #normalising weights
        weights /= np.sum(weights)
        
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, avg_rets, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = portfolio_return  / portfolio_std_dev
    return results, weights_record

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios):
    #Generating 30000 random weights, saving their returns, std dev, sharpe ratio
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix) 

    #finding index of max sharpe ratio
    max_sharpe_idx = np.argmax(results[2])
    #getting the returns and std dev of this particular portfolio
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    #allocation of weights with max sharpe ratio
    max_sharpe_allocation = weights[max_sharpe_idx]

    #finding index of min volatility/standard deviation
    min_vol_idx = np.argmin(results[0])
    #getting the returns and std dev of this particular portfolio
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    #allocation of weights with min volatility/standard deviation
    min_vol_allocation = weights[min_vol_idx]  
    
    #finding index of max returns
    max_returns_idx = np.argmax(results[1])
    #getting the returns and std dev of this particular portfolio
    sdp_max, rp_max = results[0,max_returns_idx], results[1,max_returns_idx]
    #allocation of weights with max returns
    max_returns_allocation = weights[max_returns_idx]
    print(tickers)
#     print(weights)
    print(f'''Max sharpe ratio portfolio return: {round(rp,2)}
Volatility: {round(sdp,2)}
{max_sharpe_allocation}''')
    print('-'*18)
    print(f'''Min volatility portfolio return: {round(rp_min,2)}
Volatility: {round(sdp_min,2)}
{min_vol_allocation}''')
    print('-'*18)
    print(f'''Max returns: {round(rp_max,2)}
Volatility: {round(sdp_max,2)}
{max_returns_allocation}''')
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.scatter(sdp_max,rp_max,marker='*',color='b',s=500, label='Maximum Returns')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)

display_simulated_ef_with_random(avg_rets, cov_matrix, num_portfolios)
