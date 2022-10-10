import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle 

import warnings
def ignore_warn(*arg, **kwargs):
    pass
warnings.warn=ignore_warn

data=pd.DataFrame({'Experience':[4,4,5,2,7,3,10,11],
                   'Test_score':[8,8,6,10,9,7,6,7],
                    'Interview_score':[9,6,7,10,6,10,7,8],
                    'Salary':[50000,45000,60000,65000,70000,62000,72000,80000]})

#seperating the x and y variables
x=data[['Experience','Test_score','Interview_score']]
y=data['Salary']


# Initiating the Linear Regression and fitting the data
LR_model=LinearRegression()

LR_model.fit(x,y)
pickle.dump(LR_model, open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
print("\n\n",model.predict([[2,10,10]]))