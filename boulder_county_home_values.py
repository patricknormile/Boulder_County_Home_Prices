# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:53:38 2020

@author: patno_000
"""

"""

scrape boulder county home sale records

"""


from bs4 import BeautifulSoup as bs
import pandas as pd
import urllib
import re
import matplotlib.pyplot as plt
import numpy as np
import requests
""" first scrape all xlsx from boulder county residential sales """
url = "https://www.bouldercounty.org/property-and-land/assessor/sales/comps-2019/residential/"
page = requests.get(url)

soup = bs(page.content)

XLSX_links = [i.get('href') for i in soup.findAll('a', attrs={'href':re.compile("^https://assessor")}) if "XLSX" in i]

for i, url in enumerate(XLSX_links):
    if i < 10:
        doc = "0" + str(i)
    else:
        doc = str(i)
        
    name = "BC_XLSX_" + doc + ".xlsx"
    urllib.request.urlretrieve(url, name)

""" create a dict of all data frames to check if they have the same columns """
dfs = {}

for i in range(len(XLSX_links)):
    if i < 10:
        doc = "0" + str(i)
    else:
        doc = str(i)
    
    name = "df_" + doc
    name_old = "BC_XLSX_" + doc + ".xlsx"
    df =     pd.read_excel(name_old)
    dfs.update({name : df})
"""save the names because you'll need them"""
names = []
for i in range(len(XLSX_links)):
    if i < 10:
        doc = "0" + str(i)
    else:
        doc = str(i)
    
    name = "df_" + doc
    names.append(name)
    
for name in names:
    print(dfs[name].columns)

name_ref = set(dfs['df_00'].columns)

for name in names[1:]:
    print(set(dfs[name].columns).difference(name_ref))


name_ref2 = set(dfs['df_01'].columns)
for name in names[2:]:
    print(set(dfs[name].columns).difference(name_ref2), name_ref2.difference(set(dfs[name].columns)))
""" df_01 thru df_31 all have the same columns, potentially not in order"""

"""there are 3 sets of columns:
    df_00's columns (apartments), 
    df_01-df_31's columns
    df_32's columns (manufactured housing)
"""
print(XLSX_links[0])
print(XLSX_links[32])
""" I'll just use the ones where the columns match up, which include
    single family homes, townhomes, and condos
"""

main_df = dfs['df_01']
for name in names[2:len(names)-1]:
    main_df = main_df.append(dfs[name], ignore_index=True)

print(main_df.shape)
print(main_df.columns)

print(main_df.dtypes)
"""
Above Grd SF, Basemt Tot SF, Basemt Fin SF, Basemt Unf SF, Garage SF, Est Land SF, Sale Price, Time Adjust Sales Price
should all be numeric fields, not object
"""
print(main_df.describe())
main_df['Distrss Sale'].head()
cvrt_num = ['Above Grd SF', 'Basemt Tot SF', 'Basemt Fin SF', 'Basemt Unf SF', 'Garage SF', 'Est Land SF', 'Sale Price', 'Time Adjust Sales Price']

for name in cvrt_num:
    main_df[name] = pd.to_numeric(main_df[name].str.replace(",", "").str.replace("$",""))

print(main_df.dtypes)
print(main_df.describe())

main_df[['Sale Price','Time Adjust Sales Price']].plot(kind='box')
plt.show()
#### HUGE amount of outliers, very non-linear

main_df[['Location','Time Adjust Sales Price']].boxplot(by='Location',figsize=(20,20))
# many large outliers in boulder proper and unincorporated (no suprise)

main_df[['Above Grd SF','Time Adjust Sales Price']].plot(x='Above Grd SF',
       y='Time Adjust Sales Price', marker='.', kind='scatter')

main_df[['Basemt Tot SF','Time Adjust Sales Price']].plot(x='Basemt Tot SF',
       y='Time Adjust Sales Price', marker='.', kind='scatter')

print(
main_df[main_df['Basemt Tot SF'] > 0]['Basemt Tot SF'].count() / main_df['Basemt Tot SF'].count()
)
# ~64% of homes have a basement

print(main_df.isna().sum())
# most with no land are condos. OK to set nas in land sf to 0
print(main_df[main_df['Est Land SF'].isna()].groupby(['Property Type']).count())

main_df[main_df['Above Grd SF'].isna()]

main_df['Est Land SF'] = main_df['Est Land SF'].fillna(0)

print(main_df[main_df['Garage SF'].isna()].groupby(['Garage Type'])['Garage Type'].count())
# Garage with no value not always "None". I'll just forward fill but there is probably a better way to do it

# first update garage type None with NA sq f to 0 sq f
update_0 = (main_df['Garage Type'] == 'None') & (main_df['Garage SF'].isna())
main_df['Garage SF'] = np.where(update_0, 0, main_df['Garage SF'])
#forward fill the rest
main_df['Garage SF'] = main_df['Garage SF'].fillna(method='ffill')
print(main_df.isna().sum())


# Update na basement sf to zero
Basemt = ['Basemt Tot SF', 'Basemt Fin SF', 'Basemt Unf SF']
for b in Basemt:
    main_df[b] = main_df[b].fillna(0)


print(main_df.isna().sum())

# use only fields you want to move forward with
fields = ['Property Type','Location','Design','Quality','Eff Yr Built','Above Grd SF',
          'Basemt Tot SF','Basemt Fin SF','Basemt Unf SF','Garage Type',
          'Garage SF','Est Land SF','Distrss Sale','Sale Date Mon YR',
          'Market Area','Time Adjust Sales Price']

main_df = main_df[fields].dropna()
print(main_df.shape)
print(main_df.columns)

main_df['SaleYR'] = main_df['Sale Date Mon YR'].dt.year
main_df['SaleMO'] = main_df['Sale Date Mon YR'].dt.month
#one home shows negative 1 age so clip at zero
main_df['HomeAge'] = (main_df.SaleYR - main_df['Eff Yr Built']).clip(lower=0)

main_df.drop(['Eff Yr Built','Sale Date Mon YR'], axis=1,inplace=True)

######### one hot encode categorical variables
Dummies = ['Property Type','Location','Design','Quality',
           'Garage Type','Distrss Sale','Market Area','SaleYR',
           'SaleMO']

main_df[Dummies].dtypes
for dum in Dummies[-3:]:
    main_df[dum] = main_df[dum].astype(str)


one_hot = pd.get_dummies(main_df[Dummies])

main_df = pd.concat([main_df, one_hot], axis=1)
main_df.shape

main_df.drop(Dummies, axis=1, inplace=True)

""" now ready to split data into predictors and target """

target = main_df['Time Adjust Sales Price']
predictors = main_df.drop(['Time Adjust Sales Price'], axis=1)
target.shape
predictors.shape


"""


Neural Net to predict sale prices 


"""
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model


model_1 = Sequential()
n_cols = predictors.shape[1]
#first model start out with one hidden layer with 50 nodes
model_1.add(Dense(50, activation='relu', input_shape=(n_cols,)))
model_1.add(Dense(1))












