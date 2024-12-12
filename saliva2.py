import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

file_path = "/Users/martin/Downloads/Pyrazinamide_data.csv"
def prepare_data(path: str = file_path) -> pd.DataFrame:
    """_summary_
    This method load and prepare the saliva - dataset
    
    ---------------------
    @Author: Martin Pius

    Args:
        path (str, optional): _description_. Defaults to file_path.

    Returns:
        pd.DataFrame: _description_
    """
  
    data = pd.read_csv(path, header=1, sep=";")
    data = data.iloc[:-1]  # Remove the last row (N row)
    # Replace the European decimals format i.e; "," with '.'
    data = data.apply(lambda x: x.str.replace(',', '.').astype(float) if x.dtype == 'object' else x)
    
    return data

def loocv_reg() -> None:
    
    """_summary_
    
    This method fit a LR models using LOOCV technique. We choose this
    approach due to small dataset to prevent overfiting.
    
    ---------------------
    @Author: Martin Pius
    Args:
       
    """
    data = prepare_data()
    # Define X (features) and y (target) without log-transformation
    X = data[['Cmax_Plasma', 'AUC_TAU_Plasma']].values  # Plasma concentrations
    y = data[['Cmax_Saliva', 'AUC_TAU_Saliva']].values  # Saliva concentrations

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

    # Plot predicted vs actual values for Cmax and AUC
    plt.figure(figsize=(8, 6))
    plt.scatter(y[:, 0], y_pred[:, 0], label='Cmax_Saliva', alpha=0.7)
    plt.scatter(y[:, 1], y_pred[:, 1], label='AUC_TAU_Saliva', alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.title('LOOCV: Predicted vs Actual Salivary AUCtau and Cmax')
    plt.savefig("saliva_6.png")
    
    return n_samples, model, X, y


def jackknife_method()->None:
    """_summary_
    
    This method implement the Jackknife procedure to 
    to estimate the plasma concentrations of all 15 patients (part 6)
    
    ---------------------
    @Author: Martin Pius
    
    """
    
    n_samples, model, X, y  = loocv_reg()

    # The Jackknife resampling for the saliva/plasma conversion factor
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
    n_samples, model, X, y  = loocv_reg()
    jackknife_method()
    