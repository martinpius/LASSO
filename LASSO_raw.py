import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def lasso_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_summary_
       This method create a toy dataset to  fit LASSO

    Returns:
        Tuple [np.ndarray]: _description_
    """
    np.random.seed(19376)
    n, p = 100, 10  # n: samples, p: predictors
    X = np.random.randn(n, p)  # Predictors matrix
    true_beta = np.array([5, -3, 0, 0, 0, 0, 2, 0, 0, 0])  # Sparse coefficients
    y = X @ true_beta + np.random.randn(n)  # Response variable with noise
    
    return X, true_beta, y 

def soft_thresholding(z: np.ndarray, lam: float) -> float:
    """_summary_
    This method implement the soft-thresholding operator
    to obtain LASSO parameters estimates.

    Args:
        z (np.ndarray): _description_ : Residuals
        lam (float): _description_ : penalty factor

    Returns:
        float: _description_ : LASSO estimate
    """
    # Implementing the soft-thresholing logic
    if z > lam:
        return z - lam
    elif z < -lam:
        return z + lam
    else:
        return 0
    
def lasso_coordinate_descent(X: np.ndarray,
                             y: np.ndarray,
                             lam: float, 
                             max_iter: int =1000, 
                             eps: float = 1e-6) -> np.ndarray:
    """_summary_
    This method implement an iterative approach called 
    coordinate descent to solve for Betas numerically

    Args:
        X (np.ndarray): _description_ : Covvariate matrix
        y (np.ndarray): _description_: Response vector
        lam (float): _description_: a floating point penalty
        max_iter (int, optional): _description_. Defaults to 1000.
        eps (float, optional): _description_. Defaults to 1e-6: For convergence creteria

    Returns:
        np.ndarray: _description_ : LASSO estimates
    """
    n, p = X.shape
    beta = np.zeros(p)  # Initialize coefficients
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            # Compute the partial residual
            residual = y - X @ beta + X[:, j] * beta[j]
            # Update beta[j] using soft-thresholding
            z_j = np.dot(X[:, j], residual) / n
            beta[j] = soft_thresholding(z_j, lam / n)
        # Check convergence
        if np.max(np.abs(beta - beta_old)) < eps:
            break
    return beta 

def lasso_trajectory(X: np.ndarray, 
                        y:np.ndarray,
                        lambdas: np.ndarray = np.logspace(-2, 10, 10))->None:
    """_summary_
    This Method 	
    Fit LASSO for different values of  \lambda and plot the trajectory
    for the estimates
	
	
    """
    # Test LASSO with different lambda values
    lasso_estimates = []

    for lam in lambdas:
        beta_hat = lasso_coordinate_descent(X, y, lam)
        lasso_estimates.append(beta_hat)

    # Convert to array for analysis
    lasso_estimates = np.array(lasso_estimates)

    # Plot results
    #p = X.shape[1]
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 8))
    for j in range(10):
        plt.plot(lambdas,
                 lasso_estimates[:, j],
                 label=f"Beta {j}" if true_beta[j] != 0 else None)
        
    plt.axhline(0,
                color="black", 
                linestyle="--", 
            linewidth=0.5)
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel("Coefficient Value")
    plt.title("LASSO Path (Coefficient Shrinkage)")
    plt.legend(loc = "best")
    plt.show()

def lasso_bias_variance(lambdas: np.ndarray = np.logspace(-3, 1, 10), num_sim=1000) -> None:
    """_summary_
    This Method:
    1. Compare estimated coefficients with true coefficients to evaluate bias.
    2. Simulate multiple datasets to compute variance of estimated coefficients.
    """
    # Generate synthetic data
    X, true_beta, y = lasso_data()

    # Initialize a container for LASSO estimates
    lasso_estimates = []

    for lam in lambdas:
        beta_hat = lasso_coordinate_descent(X, y, lam)
        lasso_estimates.append(beta_hat)

    # Convert to a 2D array for analysis
    lasso_estimates = np.array(lasso_estimates)

    # Compute bias
    bias = np.mean(lasso_estimates, axis=0) - true_beta
    print("Bias of LASSO estimates:", bias)

    # Simulate multiple datasets for variance estimation
    n, p = X.shape
    beta_variances = np.zeros((len(lambdas), p))

    for _ in range(num_sim):
        # Generate new data with the same true coefficients
        y_sim = X @ true_beta + np.random.randn(n)
        for i, lam in enumerate(lambdas):
            beta_hat = lasso_coordinate_descent(X, y_sim, lam)
            # Compute variance for each coefficient
            beta_variances[i, :] += (beta_hat - np.mean(lasso_estimates[i, :]))**2

    beta_variances /= num_sim

    # Plot variance for each coefficient
    plt.figure(figsize=(8, 6))
    for j in range(p):
        plt.plot(lambdas, beta_variances[:, j], label=f"Beta {j}" if true_beta[j] != 0 else None)
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel("Variance")
    plt.title("LASSO Coefficient Variance")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    X, true_beta, y = lasso_data()
    lam = 0.01
    betas = lasso_coordinate_descent(X = X,
                                     y = y,
                                     lam = lam)
   
    print(f"\n{160 * '*'}\n")
    print(f"\n>>>> True betas: {true_beta}\
        \n>>>> LASSO estimates: {betas}\n")  
    print(f"\n{160 * '*'}\n")
    lasso_trajectory(X = X, y = y)
    print(f"\n{160 * '*'}\n")
    lasso_bias_variance()
    print(f"\n{160 * '*'}\n")
    
      