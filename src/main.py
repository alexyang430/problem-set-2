'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl.run_etl()
    # PART 2: Call functions/instanciate objects from preprocessing
    df_arrests = preprocessing.run_preprocessing()
    # PART 3: Call functions/instanciate objects from logistic_regression
    df_train, df_test = logistic_regression.run_logistic_regression(df_arrests)
    # PART 4: Call functions/instanciate objects from decision_tree
    df_test = decision_tree.run_decision_tree(df_train, df_test)
    # PART 5: Call functions/instanciate objects from calibration_plot
    calibration_plot.run_calibration(df_test)


if __name__ == "__main__":
    main()
