import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from typing import Tuple
import os 
print(os.curdir())

file_path = "/Users/martin/Downloads/Pyrazinamide_data.csv"
def prepare_data(path: str = file_path)-> pd.DataFrame:
    """_summary_
    This Method load and process the Saliva dataset

    Args:
        path (str, optional): _description_. Defaults to file_path.
        
    ---------------------
    @Author: Martin Pius

    Returns:
        pd.DataFrame: _description_
    """
    data = pd.read_csv(path, header=1, sep = ";")
    data = data.iloc[:-1]  # Remove the last row (N row)
    data = data.apply(lambda x: x.str.replace(',', '.').astype(float) if x.dtype == 'object' else x)
    
    return data

def loocv_regression_logs() -> Tuple:
    """_summary_
    This method fit regression models based on LOOCV due to
    data scarcity to adress overfitting/underfitting
    
    --------------------
    @Author: Martin Pius
    
    Returns:
        Tuple: _description_
    """
    data = prepare_data()
    # Log-transformed variables
    data['log_Cmax_Saliva'] = np.log(data['Cmax_Saliva'])
    data['log_Cmax_Plasma'] = np.log(data['Cmax_Plasma'])
    data['log_AUC_TAU_Saliva'] = np.log(data['AUC_TAU_Saliva'])
    data['log_AUC_TAU_Plasma'] = np.log(data['AUC_TAU_Plasma'])

    # Define X (features) and y (target)
    X = data[['log_Cmax_Plasma', 'log_AUC_TAU_Plasma']].values  # Features: log-transformed plasma data
    y = data[['log_Cmax_Saliva', 'log_AUC_TAU_Saliva']].values  # Target: log-transformed saliva data

    # Number of samples
    n_samples = X.shape[0]

    # Initialize the model
    model = LinearRegression()

    # Initialize lists to store predictions and performance
    y_pred = np.zeros_like(y)
    mse = []

    # LOOCV (Leave-One-Out Cross-Validation)
    for i in range(n_samples):
        # Split data into training (all except the i-th sample) and test (the i-th sample)
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i, axis=0)
        X_test = X[i, :].reshape(1, -1)
        y_test = y[i, :].reshape(1, -1)

        # Train the model on the training data
        model.fit(X_train, y_train)

        # Predict on the test data (the i-th sample)
        y_pred[i, :] = model.predict(X_test)

        # Calculate MSE for this fold
        mse.append(mean_squared_error(y_test.flatten(), y_pred[i, :].flatten()))

    # Calculate the overall MSE for the model
    mean_mse = np.mean(mse)
    print(f"Mean Squared Error (MSE) for LOOCV: {mean_mse}")

    # Calculate conversion factors (mean and std)
    conversion_factors = y / X  # Saliva/Plasma ratios
    conversion_factor_mean = np.mean(conversion_factors, axis=0)
    conversion_factor_std = np.std(conversion_factors, axis=0)
    print(f"Conversion Factor (mean ± std): {conversion_factor_mean} ± {conversion_factor_std}")

    # Plot predicted vs actual values for AUC
    plt.figure(figsize=(8, 6))
    plt.scatter(y[:, 0], y_pred[:, 0], label='Cmax_Saliva', alpha=0.7)
    plt.scatter(y[:, 1], y_pred[:, 1], label='AUC_TAU_Saliva', alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.title('LOOCV: Predicted vs Actual Salivary AUCtau and Cmax -Logscaled')
    plt.savefig("saliva_log.png")
    
    return n_samples, X, y, model

def jackknife_method():
    """_summary_
    This method impliment the jackknife resampling for the 
    saliva/plasma conversion factor
    
    ---------------------
    @Author: Martin Pius
    
    """
    n_samples, X, y, model = loocv_regression_logs()
    jackknife_conversion_factors = []

    for i in range(n_samples):
        # Omit the i-th sample
        X_train_jk = np.delete(X, i, axis=0)
        y_train_jk = np.delete(y, i, axis=0)
        
        # Train the model again without the i-th sample
        model.fit(X_train_jk, y_train_jk)
        
        # Predict the omitted sample
        y_test_jk = y[i, :]
        _ = model.predict(X[i, :].reshape(1, -1))
        
        # Calculate conversion factor for this jackknife iteration
        jackknife_conversion_factors.append(y_test_jk / X[i, :])

    # Calculate mean and std of jackknife conversion factors
    jackknife_conversion_factors = np.array(jackknife_conversion_factors)
    jackknife_mean = np.mean(jackknife_conversion_factors, axis=0)
    jackknife_std = np.std(jackknife_conversion_factors, axis=0)

    print(f"Jackknife Conversion Factor (mean ± std): {jackknife_mean} ± {jackknife_std}")


if __name__ == "__main__":
    print(f"\n {160 * '*'}\n")
    n_samples, X, y, model = loocv_regression_logs()
    jackknife_method()
    print(f"\n {160 * '*'}\n")