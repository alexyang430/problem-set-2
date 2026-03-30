'''
PART 4: Decision Trees
- Read in the dataframe(s) from PART 3
- Create a parameter grid called `param_grid_dt` containing three values for tree depth. (Note C has to be greater than zero) 
- Initialize the Decision Tree model. Assign this to a variable called `dt_model`. 
- Initialize the GridSearchCV using the logistic regression model you initialized and parameter grid you created. Do 5 fold crossvalidation. Assign this to a variable called `gs_cv_dt`. 
- Run the model 
- What was the optimal value for max_depth?  Did it have the most or least regularization? Or in the middle? 
- Now predict for the test set. Name this column `pred_dt` 
- Save dataframe(s) save as .csv('s) in `data/`
'''

# Import any further packages you may need for PART 4
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold_strat
from sklearn.tree import DecisionTreeClassifier as DTC

def run_decision_tree(df_train, df_test):

    features = ['current_charge_felony', 'num_fel_arrests_last_year']

    X_train = df_train[features].fillna(0)
    y_train = df_train['y']

    X_test = df_test[features].fillna(0)

    param_grid_dt = {'max_depth': [2, 5, 10]}

    dt_model = DecisionTreeClassifier(random_state=42)

    gs_cv_dt = GridSearchCV(dt_model, param_grid_dt, cv=5)
    gs_cv_dt.fit(X_train, y_train)

    best_depth = gs_cv_dt.best_params_['max_depth']
    print(f"Best max_depth: {best_depth}")

    if best_depth == min(param_grid_dt['max_depth']):
        print("Most regularization")
    elif best_depth == max(param_grid_dt['max_depth']):
        print("Least regularization")
    else:
        print("Middle regularization")

    df_test['pred_dt'] = gs_cv_dt.predict_proba(X_test)[:, 1]

    df_test.to_csv("data/test_final.csv", index=False)

    return df_test
