# Regression-Project


# Goals defined by Telco!
- 1. Create A Regression ML model to predict property tax assessed values ("taxvaluedollarcnt") or single family properties

- 2. Find key drivers

- 3. Recommendations after creating ML model
# Project Goals:
The goal of this project is to create a Regression ML in predicting tax value of homes in california, finding key drivers, and creating a recommendation for zillow based of the findings. Using a step by step process to find the outcomes and their take aways!

# Project Description:
The company needs a model than can beat the current one (baseline) in predicting the tax value of homes in 2017, what state the data is located in, and the repsective counties. We need a quick MVP in finding the tax value of homes and the recommendations of the the finding afterwards!

# Initial Questions:
What are key drivers for zillow?
What area is this located in?

# Data Dictionary:

Use file `acquire.py` that will upload data to the final noteboolk.

--------------

idx  |Feature                           |Not null values |data type|
| --- | ---------------------------------|----------------|--------|  
| 0   |bedrooms                       | 45573 non-null  | float64  |
| 1   |bathrooms          | 45573 non-null  | float64  |
| 2   |area                  | 45573 non-null  | float64  |
| 3   |age                    | 45573 non-null  | float64  |
| 4   |county                            | 45573 non-null  | object  |
| 5   |tax_value                         | 45573 non-null | float64|
| 6   |garages                            | 45573 non-null  | float64  |
| 7   |garagetotalsqft                            | 45573 non-null  | float64  |
| 8   |pools                            | 45573 non-null  | int64  |
| 9   |roomcnt                            | 45573 non-null  | float64  |
| 10   |LA_County                            | 45573 non-null  | int64  |
| 11  |Orange_County                            | 45573 non-null  | int64  |
| 12  |Ventura_County                            | 45573 non-null  | int64  |

# Steps to reproduce:
You will need an env.py file taht contains the hostname, username and password of the mySQL database that contains the zillow.properties2017 , zillow.predictions2017, zillow.propertylandusetype tables. Store that env file locally in the repository

Clone the repo from github(including all the files acquire.py) (confirm .gitignore is hiding your env.py file) Libraries needed are pandas, matplotlib, seaborn, numby, sklearn After so you will be able to run zillow_report
# The Plan:
- Acquiring
- Pre Exploration
- Preperation of data
- Exploration 2.0
- Scaling
- Feature Look at
- Modeling
- Takeaways / Recommendations / and Next time

## Wrangle
Modules : acquire.py

acquire.py retrieves the zillow data base via sql scripyt, is also prepares using prep_zillow function which removes null and repalces some values with zeros.
It changes year_built to age, and also removes outliers all in prep_zillow function. It also splits the data into train, validate and test. Test is 20% of the original dataset, validate is .3 * .8 = 24% of the original dataset, and train is .7 * .80= 56% of the original dataset. The function returns, in this order, train, validate and test dataframes.

# Explore
Does being Orange County effect tax value of homes?
Does the age of the home effect the tax value?
Does the area of the home effect the tax value?
Does having a pool effect the tax value?
