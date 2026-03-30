'''
PART 5: Calibration-light
- Read in data from `data/`
- Use `calibration_plot` function to create a calibration curve for the logistic regression model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Use `calibration_plot` function to create a calibration curve for the decision tree model. Set `n_bins` to 5. (The calibration plot may have less than 5 points, that's ok) 
- Which model is more calibrated? Print this question and your answer. 

Extra Credit
- Compute  PPV for the logistic regression model for arrestees ranked in the top 50 for predicted risk
- Compute  PPV for the decision tree model for arrestees ranked in the top 50 for predicted risk
- Compute AUC for the logistic regression model
- Compute AUC for the decision tree model
- Do both metrics agree that one model is more accurate than the other? Print this question and your answer. 
'''

# Import any further packages you may need for PART 5
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Calibration plot function 
def calibration_plot(y_true, y_prob, n_bins=10):
    """
    Create a calibration plot with a 45-degree dashed line.

    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_prob (array-like): Predicted probabilities for the positive class.
        n_bins (int): Number of bins to divide the data for calibration.

    Returns:
        None
    """
    #Calculate calibration values
    bin_means, prob_true = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    #Create the Seaborn plot
    sns.set(style="whitegrid")
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(prob_true, bin_means, marker='o', label="Model")
    
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend(loc="best")
    plt.show()

def run_calibration():

    # Read in data
    df = pd.read_csv("data/test_final.csv")

    y = df['y']

    # --- Calibration plots ---
    print("Generating calibration plot for Logistic Regression...")
    calibration_plot(y, df['pred_lr'], n_bins=5)

    print("Generating calibration plot for Decision Tree...")
    calibration_plot(y, df['pred_dt'], n_bins=5)

    # --- Calibration comparison ---
    print("\nWhich model is more calibrated?")
    print("Answer: Compare the plots — the model closer to the 45-degree line is better calibrated.")

    top_lr = df.nlargest(50, 'pred_lr')
    top_dt = df.nlargest(50, 'pred_dt')

    ppv_lr = top_lr['y'].mean()
    ppv_dt = top_dt['y'].mean()

    print(f"\nPPV (Top 50) - Logistic Regression: {ppv_lr}")
    print(f"PPV (Top 50) - Decision Tree: {ppv_dt}")

    # --- AUC ---
    auc_lr = roc_auc_score(y, df['pred_lr'])
    auc_dt = roc_auc_score(y, df['pred_dt'])

    print(f"\nAUC - Logistic Regression: {auc_lr}")
    print(f"AUC - Decision Tree: {auc_dt}")

    # --- Comparison ---
    print("\nDo both metrics agree on which model is more accurate?")

    better_auc = "Logistic Regression" if auc_lr > auc_dt else "Decision Tree"
    better_ppv = "Logistic Regression" if ppv_lr > ppv_dt else "Decision Tree"

    print(f"AUC favors: {better_auc}")
    print(f"PPV favors: {better_ppv}")

    if better_auc == better_ppv:
        print("Yes, both metrics agree.")
    else:
        print("No, the metrics do not agree.")
