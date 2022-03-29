# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:57:38 2020

@author: indra
"""

# Importing Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
# %matplotlib inline
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
import warnings
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import lag_plot
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA
import itertools
from math import ceil
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LassoLars
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import r2_score


#%% Defining Functions

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. P-Values lesser than the significance level (0.05),
    implies   the Null Hypothesis can be rejected
       """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")   
        
def show_graph(train, test=None, pred=None, title=None):
    
    fig = plt.figure(figsize=(20, 5))

    # entire data
    ax1 = fig.add_subplot(121)
    ax1.set_xlabel('Dates')
    ax1.set_ylabel('Price')
    ax1.plot(train.index, train['Price'], color='green', label='Train price')
    if test is not None:
        ax1.plot(test.index, test['Price'], color='red', label='Test price')
    if pred is not None:
        if 'yhat' in pred.columns:
            ax1.plot(pred.index, pred['yhat'], color = 'blue', label = 'Predicted price')
            ax1.fill_between(pred.index, pred['yhat_lower'], pred['yhat_upper'], color='grey', label="Band Range")
        else:
            ax1.plot(pred.index, pred['Price'], color='blue', label='Predicted price')
    ax1.legend()
    if title is not None:
        plt.title(title + ' (Entire)')
    plt.grid(True)
   
    # zoom data
    period=50
    period=int(0.2*len(train))
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Dates')
    ax2.set_ylabel('Price')
    ax2.plot(train.index[-period:], train['Price'].tail(period), color='green', label='Train price')
    if test is not None:
        ax2.plot(test.index, test['Price'], color='red', label='Test price')
    if pred is not None:
        if 'yhat' in pred.columns:
            ax2.plot(pred.index, pred['yhat'], color = 'blue', label = 'Predicted price')
            ax2.fill_between(pred.index, pred['yhat_lower'], pred['yhat_upper'], color='grey', label="Band Range")
        else:
            ax2.plot(pred.index, pred['Price'], color='blue', label='Predicted price')
    ax2.legend()
    if title is not None:
        plt.title(title + ' (Recent ' + str(period) + ')')
    plt.grid(True)

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
def make_future_dates(last_date, period):
    prediction_dates=pd.date_range(last_date, periods=period+1, freq='MS')
    return prediction_dates[1:]

def evaluate_arima_model(train, test, order, maxlags=8, ic='aic'):
    # feature Scaling
    stdsc = StandardScaler()
    train_std = stdsc.fit_transform(train.values.reshape(-1, 1))
    test_std = stdsc.transform(test.values.reshape(-1, 1))
    # prepare training dataset
    history = [x for x in train_std]
    # make predictions
    predictions = list()
    # rolling forecasts
    for t in range(len(test_std)):
        # predict
        model = ARIMA(history, order=order)
        model_fit = model.fit(maxlags=maxlags, ic=ic, disp=0)
        yhat = model_fit.forecast()[0]
        # invert transformed prediction
        predictions.append(yhat)
        # observation
        history.append(test_std[t])
    # inverse transform
    predictions = stdsc.inverse_transform(np.array(predictions).reshape((-1)))
    # calculate mse
    mse = mean_squared_error(test, predictions)
    return predictions, mse

def evaluate_arima_models(train, test, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    pdq = list(itertools.product(p_values, d_values, q_values))
    for order in pdq:
        try:
            predictions, mse = evaluate_arima_model(train, test, order)
            if mse < best_score:
                best_score, best_cfg = mse, order
            print('Model(%s) mse=%.3f' % (order,mse))
        except:
            continue
    print('Best Model(%s) mse=%.3f' % (best_cfg, best_score)) 
    return best_cfg

def predict_arima_model(train, period, order, maxlags=8, ic='aic'):
    # Feature Scaling
    stdsc = StandardScaler()
    train_std = stdsc.fit_transform(train.values.reshape(-1, 1))
    # fit model
    model = ARIMA(train_std, order=order)
    model_fit = model.fit(maxlags=maxlags, ic=ic, disp=0)
    # make prediction
    yhat = model_fit.predict(len(train), len(train) + period -1, typ='levels')
    # inverse transform
    yhat = stdsc.inverse_transform(np.array(yhat).flatten())
    return yhat

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

def train_test_split_sorted(X, y, test_size, dates):
    """Splits X and Y into train and test sets according to a timestamp."""
    n_test = ceil(test_size * len(X))

    sorted_index = [x for _, x in sorted(zip(np.array(dates), np.arange(0, len(dates))), key=lambda pair: pair[0])]
    train_idx = sorted_index[:-n_test]
    test_idx = sorted_index[-n_test:]

    if isinstance(X, (pd.Series, pd.DataFrame)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
    else:
        y_train = y[train_idx]
        y_test = y[test_idx]

    return X_train, X_test, y_train, y_test

def plot_importances(model):
    importances = model.feature_importances_
    # std = np.std([model.feature_importances_ for tree in model.estimators_],
    #          axis=0)
    indices = np.argsort(importances)
    palette1 = itertools.cycle(sns.color_palette())
    # Store the feature ranking
    features_ranked=[]
    for f in range(X_train.shape[1]):
        features_ranked.append(X_train.columns[indices[f]])
        # Plot the feature importances of the forest

    plt.figure(figsize=(10,10), frameon=False, dpi=100)
    
    plt.title("Feature importances")
    plt.barh(range(X_train.shape[1]), importances[indices],
             color=[next(palette1)], align="center")
    plt.yticks(range(X_train.shape[1]), features_ranked)
    plt.ylabel('Features')
    plt.ylim([-1, X_train.shape[1]])
    plt.show()

def calculate_accuracy(y, y_hat, algorithm):
    mape = np.mean(np.abs(y - y_hat)/np.abs(y) * 100)              
    mae = mean_absolute_error(y, y_hat) 
    mse = mean_squared_error(y, y_hat)
    r_squared = r2_score(y, y_hat)
    rmse = sqrt(mean_squared_error(y, y_hat))  # RMSE
    rmspe = np.sqrt(np.mean(((y - y_hat) / y) ** 2))
    return({'Algorithm': algorithm, 'MAPE':mape, 'MAE':mae, 'MSE': mse, 
            'R_squared': r_squared, 'RMSE':rmse, 'RMSPE':rmspe})

#%% Evaluation metrics

EPSILON = 1e-10


def _error(actual: np.ndarray, predicted: np.ndarray):
    """ Simple error """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Percentage error
    Note: result is NOT multiplied by 100
    """
    return _error(actual, predicted) / (actual + EPSILON)


def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    """ Naive forecasting method which just repeats previous samples """
    return actual[:-seasonality]


def _relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark
        return _error(actual[seasonality:], predicted[seasonality:]) /\
               (_error(actual[seasonality:], _naive_forecasting(actual, seasonality)) + EPSILON)

    return _error(actual, predicted) / (_error(actual, benchmark) + EPSILON)


def _bounded_relative_error(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Bounded Relative Error """
    if benchmark is None or isinstance(benchmark, int):
        # If no benchmark prediction provided - use naive forecasting
        if not isinstance(benchmark, int):
            seasonality = 1
        else:
            seasonality = benchmark

        abs_err = np.abs(_error(actual[seasonality:], predicted[seasonality:]))
        abs_err_bench = np.abs(_error(actual[seasonality:], _naive_forecasting(actual, seasonality)))
    else:
        abs_err = np.abs(_error(actual, predicted))
        abs_err_bench = np.abs(_error(actual, benchmark))

    return abs_err / (abs_err + abs_err_bench + EPSILON)


def _geometric_mean(a, axis=0, dtype=None):
    """ Geometric mean """
    if not isinstance(a, np.ndarray):  # if not an ndarray object attempt to convert it
        log_a = np.log(np.array(a, dtype=dtype))
    elif dtype:  # Must change the default dtype allowing array type
        if isinstance(a, np.ma.MaskedArray):
            log_a = np.log(np.ma.asarray(a, dtype=dtype))
        else:
            log_a = np.log(np.asarray(a, dtype=dtype))
    else:
        log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))


def mse(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Squared Error """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return np.sqrt(mse(actual, predicted))


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted) / (actual.max() - actual.min())


def me(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Error """
    return np.mean(_error(actual, predicted))


def mae(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Absolute Error """
    return np.mean(np.abs(_error(actual, predicted)))


mad = mae  # Mean Absolute Deviation (it is the same as MAE)


def gmae(actual: np.ndarray, predicted: np.ndarray):
    """ Geometric Mean Absolute Error """
    return _geometric_mean(np.abs(_error(actual, predicted)))


def mdae(actual: np.ndarray, predicted: np.ndarray):
    """ Median Absolute Error """
    return np.median(np.abs(_error(actual, predicted)))


def mpe(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Percentage Error """
    return np.mean(_percentage_error(actual, predicted))


def mape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Absolute Percentage Error
    Properties:
        + Easy to interpret
        + Scale independent
        - Biased, not symmetric
        - Undefined when actual[t] == 0
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def mdape(actual: np.ndarray, predicted: np.ndarray):
    """
    Median Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.median(np.abs(_percentage_error(actual, predicted)))


def smape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Mean Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def smdape(actual: np.ndarray, predicted: np.ndarray):
    """
    Symmetric Median Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.median(2.0 * np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) + EPSILON))


def maape(actual: np.ndarray, predicted: np.ndarray):
    """
    Mean Arctangent Absolute Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.mean(np.arctan(np.abs((actual - predicted) / (actual + EPSILON))))


def mase(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """
    Mean Absolute Scaled Error
    Baseline (benchmark) is computed with naive forecasting (shifted by @seasonality)
    """
    return mae(actual, predicted) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))


def std_ae(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Error """
    __mae = mae(actual, predicted)
    return np.sqrt(np.sum(np.square(_error(actual, predicted) - __mae))/(len(actual) - 1))


def std_ape(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Percentage Error """
    __mape = mape(actual, predicted)
    return np.sqrt(np.sum(np.square(_percentage_error(actual, predicted) - __mape))/(len(actual) - 1))


def rmspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Mean Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.mean(np.square(_percentage_error(actual, predicted))))


def rmdspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Median Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def rmsse(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    """ Root Mean Squared Scaled Error """
    q = np.abs(_error(actual, predicted)) / mae(actual[seasonality:], _naive_forecasting(actual, seasonality))
    return np.sqrt(np.mean(np.square(q)))


def inrse(actual: np.ndarray, predicted: np.ndarray):
    """ Integral Normalized Root Squared Error """
    return np.sqrt(np.sum(np.square(_error(actual, predicted))) / np.sum(np.square(actual - np.mean(actual))))


def rrse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Relative Squared Error """
    return np.sqrt(np.sum(np.square(actual - predicted)) / np.sum(np.square(actual - np.mean(actual))))


def mre(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Error """
    return np.mean(_relative_error(actual, predicted, benchmark))


def rae(actual: np.ndarray, predicted: np.ndarray):
    """ Relative Absolute Error (aka Approximation Error) """
    return np.sum(np.abs(actual - predicted)) / (np.sum(np.abs(actual - np.mean(actual))) + EPSILON)


def mrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Relative Absolute Error """
    return np.mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mdrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Median Relative Absolute Error """
    return np.median(np.abs(_relative_error(actual, predicted, benchmark)))


def gmrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Geometric Mean Relative Absolute Error """
    return _geometric_mean(np.abs(_relative_error(actual, predicted, benchmark)))


def mbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Mean Bounded Relative Absolute Error """
    return np.mean(_bounded_relative_error(actual, predicted, benchmark))


def umbrae(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray = None):
    """ Unscaled Mean Bounded Relative Absolute Error """
    __mbrae = mbrae(actual, predicted, benchmark)
    return __mbrae / (1 - __mbrae)


def mda(actual: np.ndarray, predicted: np.ndarray):
    """ Mean Directional Accuracy """
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))

def r_squared(actual: np.ndarray, predicted: np.ndarray):
    """R-Squared"""
    return r2_score(actual, predicted)

METRICS = {
    'mse': mse,
    'rmse': rmse,
    'nrmse': nrmse,
    # 'me': me,
    'mae': mae,
    # 'mad': mad,
    # 'gmae': gmae,
    # 'mdae': mdae,
    # 'mpe': mpe,
    'mape': mape,
    # 'mdape': mdape,
    # 'smape': smape,
    # 'smdape': smdape,
    # 'maape': maape,
    'mase': mase,
    # 'std_ae': std_ae,
    # 'std_ape': std_ape,
    'rmspe': rmspe,
    # 'rmdspe': rmdspe,
    # 'rmsse': rmsse,
    # 'inrse': inrse,
    # 'rrse': rrse,
    # 'mre': mre,
    # 'rae': rae,
    # 'mrae': mrae,
    # 'mdrae': mdrae,
    # 'gmrae': gmrae,
    # 'mbrae': mbrae,
    # 'umbrae': umbrae,
    # 'mda': mda,
    'r2': r_squared,
}


def evaluate(actual: np.ndarray, predicted: np.ndarray, metrics=('mae', 'mse', 'smape', 'umbrae')):
    results = {}
    for name in metrics:
        try:
            results[name] = METRICS[name](actual, predicted)
        except Exception as err:
            results[name] = np.nan
            print('Unable to compute metric {0}: {1}'.format(name, err))
    return results


def evaluate_all(actual: np.ndarray, predicted: np.ndarray):
    return evaluate(actual, predicted, metrics=set(METRICS.keys()))

#%% Preparing Data

# Uploading data
oil_df = pd.read_csv("oil prices.csv")
futures_df = pd.read_csv("oil futures.csv")
imports_df = pd.read_csv("oil imports.csv")
exports_df = pd.read_csv("oil exports.csv")
productions_df = pd.read_csv("weekly oil production.csv")
reserves_df = pd.read_csv("oil reserves.csv")
sp500_df = pd.read_csv("s&p500 index.csv")
djia_df = pd.read_csv("dow jones industrial average index.csv")
nasdaq_df = pd.read_csv("nasdaq composite index.csv")
gold_df = pd.read_csv("gold.csv")

# Cleaning
futures_df = futures_df.drop(futures_df.columns[[5]], axis=1)
futures_df = futures_df.rename(columns={"Cushing, OK Crude Oil Future Contract 1 (Dollars per Barrel)":"Futures 1",
                                        "Cushing, OK Crude Oil Future Contract 2 (Dollars per Barrel)":"Futures 2",
                                        "Cushing, OK Crude Oil Future Contract 3 (Dollars per Barrel)":"Futures 3",
                                        "Cushing, OK Crude Oil Future Contract 4 (Dollars per Barrel)":"Futures 4"})

futures_df["Futures"] = futures_df.mean(axis=1)
futures_df = futures_df.drop(["Futures 1", "Futures 2", "Futures 3", "Futures 4"], axis=1).dropna()

imports_df = imports_df.rename(columns={"Weekly U.S. Imports of Crude Oil  (Thousand Barrels per Day)":"Imports"})

exports_df = exports_df.drop(exports_df.columns[[2]], axis=1)
exports_df = exports_df.rename(columns={"Weekly U.S. Exports of Crude Oil  (Thousand Barrels per Day)":"Exports"})

reserves_df = reserves_df.drop(reserves_df.columns[2], axis=1).drop(reserves_df.index[[119]])
reserves_df = reserves_df.rename(columns={"U.S. Crude Oil Proved Reserves (Million Barrels)":"US Oil Reserves"})

gold_df = gold_df.rename(columns={"Price":"Gold"})

sp500_df = sp500_df[["Date", "Close"]]
djia_df = djia_df[["Date", "Close"]]
nasdaq_df = nasdaq_df[["Date", "Close"]]

# Date format
oil_df["Date"] = pd.to_datetime(oil_df.Date, infer_datetime_format=True)
futures_df["Date"] = pd.to_datetime(futures_df.Date, infer_datetime_format=True)
imports_df["Date"] = pd.to_datetime(imports_df.Date, infer_datetime_format=True)
exports_df["Date"] = pd.to_datetime(exports_df.Date, infer_datetime_format=True)
productions_df["Date"] = pd.to_datetime(productions_df.Date, infer_datetime_format=True)
reserves_df["Date"] = pd.to_datetime(reserves_df.Date, format="%Y", infer_datetime_format=False)
gold_df["Date"] = pd.to_datetime(gold_df.Date, infer_datetime_format=True)
sp500_df["Date"] = pd.to_datetime(sp500_df.Date, infer_datetime_format=True)
djia_df["Date"] = pd.to_datetime(djia_df.Date, infer_datetime_format=True)
nasdaq_df["Date"] = pd.to_datetime(nasdaq_df.Date, infer_datetime_format=True)

# Combining market indexes
data_frames = [sp500_df, djia_df, nasdaq_df]
index_df = reduce(lambda  left,right: pd.merge(left,right,on=["Date"],
                                            how="outer"), data_frames)
index_df = index_df.rename(columns={"Close_x": "S&P500", "Close_y":"DJIA",
                                    "Close":"Nasdaq"})

# Resampling Monthly
oil_df = oil_df.set_index("Date").resample("M").mean()
futures_df = futures_df.set_index("Date").resample("M").mean()
imports_df = imports_df.set_index("Date").resample("M").mean()
exports_df = exports_df.set_index("Date").resample("M").mean()
productions_df = productions_df.set_index("Date").resample("M").mean()
# reserves_df = reserves_df.set_index("Date").resample("M").mean()
gold_df = gold_df.set_index("Date").resample("M").mean()
index_df = index_df.set_index("Date").resample("M").mean()

# # Resampling Weekly
# oil_df = oil_df.set_index("Date").resample("W").mean()
# futures_df = futures_df.set_index("Date").resample("W").mean()
# imports_df = imports_df.set_index("Date").resample("W").mean()
# exports_df = exports_df.set_index("Date").resample("W").mean()
# productions_df = productions_df.set_index("Date").resample("W").mean()
# # reserves_df = reserves_df.set_index("Date").resample("M").mean()
# gold_df = gold_df.set_index("Date").resample("W").mean()
# index_df = index_df.set_index("Date").resample("W").mean()

oil_df = oil_df.drop(oil_df.columns[[1]], axis=1) # drop Brent

# Merged dataset
data_frames = [oil_df, futures_df, imports_df, exports_df, productions_df,
                     gold_df, index_df]
merged_df = reduce(lambda  left,right: pd.merge(left,right,on=["Date"],
                                            how="outer"), data_frames)
merged_df = merged_df.sort_values(by="Date")
merged_df = merged_df.loc["1986-01-01":"2020-06-30"]

# Filling / Removing null values
merged_df = merged_df.dropna()

#%% 

# # # Descriptive Stats

# Correlation Analysis
corr = merged_df.corr()
mask = np.zeros_like(corr, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize = (15,10))
sns.heatmap(corr, mask = mask, square = True, ax = ax,
            cmap = "inferno", linewidths=0.5)

# STD and Variance
print(np.std(merged_df))
print(np.var(merged_df))

# kurtosis / skewness
for i in merged_df.columns:
    x = [stats.kurtosis(merged_df[i]), stats.skew(merged_df[i])]
    print(i, "Kurtosis of normal distribution: {}".format(stats.kurtosis(merged_df[i])))
    print(i, "Skewness of normal distribution: {}".format(stats.skew(merged_df[i])))

# 
summary = merged_df.describe()

# Median
for i in merged_df.columns:
    print(i, "Median: {}".format(np.median(merged_df[i]))) 
    
# Histogram
merged_df.hist()
np.log(merged_df).hist() # if log transformed

# Boxplot    
for column in merged_df:
    plt.figure()
    merged_df.boxplot([column])

for column in merged_df: # if log transformed
    plt.figure()
    np.log(merged_df).boxplot([column])

# Plots of original dataset
fig, axes = plt.subplots(nrows=3, ncols=3, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = merged_df[merged_df.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    ax.set_title(merged_df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
    plt.tight_layout()


# # # Statistical Analysis

# Granger-causality test 
maxlag=12
test = 'ssr_chi2test'
gran = grangers_causation_matrix(merged_df, variables = merged_df.columns)

# Cointegration test
cointegration_test(merged_df)

#%% Train and validation set

df_log = np.log1p(merged_df) #log transform
features = df_log.drop(["WTI", "Futures", "Gold"], axis=1) 
target = df_log[["WTI"]]

X_train, X_test, y_train, y_test = model_selection.train_test_split(features,
                                            target, test_size = 0.10, random_state = 42)

mods_sum = []
#%% Regression-based modelling

# Linear Regression 
model_name = "Linear Regression"

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
y_hat = linear_regression.predict(X_test)

# mods = calculate_accuracy(np.expm1(y_test.values), np.expm1(y_hat), model_name)
mods = evaluate_all(np.expm1(y_test.values), np.expm1(y_hat))
mods['algorithm'] = model_name
print(mods)
mods_sum.append(mods)

# Plot outputs
print(linear_regression.intercept_, linear_regression.coef_)

# Bayes Ridge regression 
model_name = "Bayes Regression"

bayesian_regression = BayesianRidge()
bayesian_regression.fit(X_train, y_train.values.ravel())
y_hat = bayesian_regression.predict(X_test)

# mods = calculate_accuracy(np.expm1(y_test.values), np.expm1(y_hat), model_name)
mods = evaluate_all(np.expm1(y_test.values), np.expm1(y_hat))
mods['algorithm'] = model_name
print(mods)
mods_sum.append(mods)

print(bayesian_regression.intercept_, bayesian_regression.coef_)


# LARS Lasso
model_name = "LARS Lasso"

lars_regression = LassoLars(alpha=0.3, fit_intercept=False, normalize=True)
lars_regression.fit(X_train, y_train)
y_hat = lars_regression.predict(X_test)

# mods = calculate_accuracy(np.expm1(y_test.values), np.expm1(y_hat), model_name)
mods = evaluate_all(np.expm1(y_test.values), np.expm1(y_hat))
mods['algorithm'] = model_name
print(mods)
mods_sum.append(mods)

print(lars_regression.intercept_, lars_regression.coef_)


# Decision Tree
model_name = "Decision Tree"

decision_tree = DecisionTreeRegressor(random_state=7)
decision_tree.fit(X_train, y_train)
y_hat = decision_tree.predict(X_test)

# mods = calculate_accuracy(np.expm1(y_test.values), np.expm1(y_hat), model_name)
mods = evaluate_all(np.expm1(y_test.values), np.expm1(y_hat))
mods['algorithm'] = model_name
print(mods)
mods_sum.append(mods)

plot_importances(decision_tree)

# Random Forest 
model_name = "Random Forest"

random_forest = RandomForestRegressor(n_estimators=29)
random_forest.fit(X_train, y_train.values.ravel())
y_hat = random_forest.predict(X_test)

mods = evaluate_all(np.expm1(y_test.values), np.expm1(y_hat))
mods['algorithm'] = model_name
print(mods)
mods_sum.append(mods)

plot_importances(random_forest)

# Adaptive Boosting 
model_name = "AdaBoost"

adaboost_tree = AdaBoostRegressor(DecisionTreeRegressor())
adaboost_tree.fit(X_train, y_train.values.ravel())
y_hat = adaboost_tree.predict(X_test)

mods = evaluate_all(np.expm1(y_test.values), np.expm1(y_hat))
mods['algorithm'] = model_name
print(mods)
mods_sum.append(mods)

plot_importances(adaboost_tree)

# KNN 
model_name = "KNN"

knn = KNeighborsRegressor(n_neighbors = 10, algorithm='auto', n_jobs=-1)
knn.fit(X_train, y_train)
y_hat = knn.predict(X_test)

mods = evaluate_all(np.expm1(y_test.values), np.expm1(y_hat))
mods['algorithm'] = model_name
print(mods)
mods_sum.append(mods)

# XGB
model_name = "Gradient Boosting"

xgboost_tree = xgb.XGBRegressor(n_jobs = -1, 
                                n_estimators = 1000,
                                eta = 0.05, #learning rate
                                max_depth = 6, 
                                min_child_weight = 3,
                                subsample = 0.8, 
                                colsample_bytree = 0.8,
                                tree_method = 'gpu_hist', 
                                reg_alpha = 0.05,
                                silent = 0, 
                                random_state = 1023)

xgboost_tree.fit(X_train, y_train, eval_set=[(X_train,y_train), (X_test, y_test)],
                 early_stopping_rounds = 100)
y_hat = xgboost_tree.predict(X_test)

mods = evaluate_all(np.expm1(y_test.values), np.expm1(y_hat))
mods['algorithm'] = model_name
print(mods)
mods_sum.append(mods)

# Neural Network 
model_name = "Neural Network"

neural_network = MLPRegressor(hidden_layer_sizes=(100, 50,),
                              early_stopping=True,
                              random_state=42,
                              verbose=True
                              )
neural_network.fit(X_train, y_train)
y_hat = neural_network.predict(X_test)

mods = evaluate_all(np.expm1(y_test.values), np.expm1(y_hat))
mods['algorithm'] = model_name
print(mods)
mods_sum.append(mods)

# Results
df_sum = pd.DataFrame(mods_sum)
df_sum.sort_values('mase', ascending=True)

#%%

ts_df = merged_df
ts_target = ts_df[["WTI"]]
ts_target = ts_target.rename(columns={"WTI":"Price"})

trainstart = "1991-02-01"
trainend = "2019-12-31"
teststart = "2020-01-01"

ts_train = ts_target[trainstart:trainend]
ts_test = ts_target[teststart:]

mods2_sum = []

#%%

# # AR -----------------------------------------------------------------------
model_name='AR Model'

# evaluate parameters
p_values = range(1, 4)
d_values = [0]
q_values = [0]
#evaluate_arima_models(train_data['Price'], test_data['Price'], p_values, d_values, q_values)

# predict test period with best parameter
predictions, mse1 = evaluate_arima_model(ts_train, ts_test,(1, 0, 0))
df_pred = pd.DataFrame({'Price':predictions},index=ts_test.index)

# calculate performance metrics
mods = evaluate_all(ts_test.values, predictions)
mods['algorithm'] = model_name
print(mods)
mods2_sum.append(mods)

# show result
show_graph(ts_train,ts_test,df_pred,title=model_name+'\nTest period prediction')

# # MA -----------------------------------------------------------------------
model_name='MA Model'

# evaluate parameters
p_values = [0]
d_values = [0]
q_values = range(1, 4)

# predict test period with best parameter
predictions, mse1 = evaluate_arima_model(ts_train['Price'], ts_test['Price'],(0, 0, 1))
df_pred = pd.DataFrame({'Price':predictions},index=ts_test.index)

# calculate performance metrics
mods = evaluate_all(ts_test.values, predictions)
mods['algorithm'] = model_name
print(mods)
mods2_sum.append(mods)

# show result
show_graph(ts_train, ts_test, df_pred, title=model_name + '\nTest period prediction')

# # ARIMA ----------------------------------------------------------------------
model_name='ARIMA Model'

# evaluate parameters
p_values = [1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(1, 3)

# predict test period with best parameter
predictions, mse1 = evaluate_arima_model(ts_train['Price'], ts_test['Price'],(2, 1, 1))
df_pred = pd.DataFrame({'Price':predictions},index=ts_test.index)

# calculate performance metrics
mods = evaluate_all(ts_test.values, predictions)
mods['algorithm'] = model_name
print(mods)
mods2_sum.append(mods)

# show result
show_graph(ts_train, ts_test, df_pred, title=model_name + '\nTest period prediction')

#%%
##### Multivariate ###############
ts_train = ts_df[trainstart:trainend]
ts_test = ts_df[teststart:]
nobs = len(ts_test)

ts_diff = ts_train.diff().dropna()
ts_diff = ts_diff.diff().dropna()
    
# selecting the order[p] of VAR model
model = VAR(ts_diff)
for i in [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')
    
x = model.select_order(maxlags=20)
print(x.summary())

# training model
model_fitted = model.fit(20)
print(model_fitted.summary())

lag_order = model_fitted.k_ar

# Input data for forecasting
forecast_input = ts_diff.values[-lag_order:]
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=ts_df.index[-nobs:], 
                           columns=ts_df.columns + '_2d')

df_results = invert_transformation(ts_test, df_forecast, second_diff=True)

model_name = "Multivariate Forecasting"
# mods = calculate_accuracy(df_results[["WTI_forecast"]].values, ts_test[["WTI"]].values, model_name)
mods = evaluate_all(df_results[["WTI_forecast"]].values, ts_test[["WTI"]].values)
mods['algorithm'] = model_name
print(mods)
mods2_sum.append(mods)

# Results
df2_sum = pd.DataFrame(mods2_sum)
df2_sum.sort_values('mase', ascending=True)

frames = [df_sum, df2_sum]
all_mods = pd.concat(frames)
all_mods = all_mods.sort_values('mase', ascending=True)

#%% 

# Log transformed plots
fig, axes = plt.subplots(nrows=3, ncols=3, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    d = df_log[df_log.columns[i]]
    ax.plot(d, color='red', linewidth=1)
    # Decorations
    ax.set_title(df_log.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_alpha(0)
    ax.tick_params(labelsize=6)
    plt.tight_layout();

# Differenced plots
fig, axes = plt.subplots(nrows=3, ncols=3, dpi=120, figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    d = ts_diff[ts_diff.columns[i]]
    ax.plot(d, color='red', linewidth=1)
    # Decorations
    ax.set_title(ts_diff.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_alpha(0)
    ax.tick_params(labelsize=6)
    plt.tight_layout();

# ACF plots
for i in ts_diff.columns:
    sm.graphics.tsa.plot_acf(ts_diff[i].values.squeeze(), lags=40, title=i+" Autocorrelation")
    plt.show()

# PACF plots
for i in ts_diff.columns:
    sm.graphics.tsa.plot_pacf(ts_diff[i].values.squeeze(), lags=40, title=i+" Partial Autocorrelation")
    plt.show()
 

# # # Statistical Analysis

# Ad-Fuller test
for name, column in merged_df.iteritems(): # BEFORE
    adfuller_test(column, name=column.name)
    print('\n')
    
for name, column in ts_diff.iteritems(): # AFTER
    adfuller_test(column, name=column.name)
    print('\n')

# Durbin-Watson Serial Correlation test
out = durbin_watson(model_fitted.resid)

for col, val in zip(ts_df.columns, out):
    def adjust(val, length= 6): return str(val).ljust(length)
    print(adjust(col), ':', round(val, 2))
