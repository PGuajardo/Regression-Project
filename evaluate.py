
'''
plot_residuals(y, yhat): creates a residual plot
'''

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import linregress

def plot_residuals(y, yhat):
    residuals = y - yhat
    plt.figure(figsize = (11,5))
    plt.hlines(0, y.min(), yhat.max(), ls = ':')
    plt.scatter(y, residuals)
    plt.axhline(y = 0, ls = ':')
    plt.xlabel('Actual')
    plt.ylabel('Residual')
    plt.title('Actual VS Residual')
    plt.show();


'''
regression_errors(y, yhat): returns the following values:
sum of squared errors (SSE)
explained sum of squares (ESS)
total sum of squares (TSS)
mean squared error (MSE)
root mean squared error (RMSE)
'''

def regression_errors(y, yhat):
    
    
    baseline = y.mean()
    basline_residual = y - baseline
    baseline_residual_2 = basline_residual**2

    # Total Sum of Squares = SSE for baseline
    TSS =  baseline_residual_2.sum()

    # Mean Squared Error
    MSE = mean_squared_error(y, yhat)

    # Sum of Squared Errors
    SSE = MSE * len(y)

    # Root Mean Squared Error
    RMSE = mean_squared_error(y, yhat, squared = False)

    # ESS - Explained sum of squares ('Explained Error')
    ESS = TSS - SSE

    return SSE, ESS, TSS, MSE, RMSE
    '''
    pd.Series({'sse' : SSE,
    'ess': ESS ,
    })
    '''


# baseline_mean_errors(y): computes the SSE, MSE, and RMSE for the baseline model

def baseline_mean_errors(y):
    baseline = y.mean()

    baseline_residuals = y - baseline
    baseline_residual_2 = baseline_residuals**2

    # Sum of Squared Errors BASLINE
    SSE_baseline = baseline_residual_2.sum()

    # Mean Squared Error BASELINE
    MSE_baseline = SSE_baseline / len(y)

    # Root Mean Squared Error BASELINE
    RMSE_baseline = sqrt(MSE_baseline)


    return SSE_baseline, MSE_baseline, RMSE_baseline
    '''
    {
        'sse' : SSE # etc
    }
    '''

# better_than_baseline(y, yhat): returns true if your model performs better than the baseline, otherwise false
def better_than_baseline(y, yhat):
    #results = regression_errors(y, yhat)
    #basline_results = baseline_mean_errors(y,yhat)
    # If the model is less than baseline then it is a better model, if not we dont need this model
    return y < yhat#results[4] < basline_results[2]


# Outputs R^2 value and P-value to test for correlation signifcance and if there is a relationship

def rval_p_significance(y, yhat):
    results = linregress(y, yhat)
    return f'P-Value : {results.pvalue} *****   R^2 Value: {(results.rvalue)**2}'
    '''
    pd.series({

        'P-Value' : results.pvale,
        'R-Value' : results.rvalue**2,
    })
    '''