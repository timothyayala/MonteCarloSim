import numpy as np
import pandas as pd
from scipy.stats import norm

# --- Configuration and Constants ---
# Set random seed for reproducibility in simulations
np.random.seed(42)

# --- Core Simulation Function ---
def monte_carlo_stock_simulation(
    initial_price: float,
    drift: float,            # Expected annual return (e.g., 0.08 for 8%)
    volatility: float,       # Annual standard deviation of returns (e.g., 0.20 for 20%)
    time_horizon_years: int, # Number of years for the simulation
    num_simulations: int,    # Number of simulation paths
    num_trading_days_per_year: int = 252 # Standard number of trading days in a year
) -> pd.DataFrame:
    """
    Performs Monte Carlo simulation for stock prices using Geometric Brownian Motion.

    Args:
        initial_price (float): The starting price of the stock.
        drift (float): The expected annual return (mean of the log returns).
        volatility (float): The annual standard deviation of the log returns.
        time_horizon_years (int): The total number of years for the simulation.
        num_simulations (int): The number of independent simulation paths to generate.
        num_trading_days_per_year (int): The number of trading days in a year.
                                         Defaults to 252.

    Returns:
        pd.DataFrame: A DataFrame where each column represents a simulation path
                      and rows represent daily prices over the time horizon.
    """
    # Calculate the number of total trading days
    num_days = time_horizon_years * num_trading_days_per_year

    # Calculate the daily drift and volatility
    dt = 1 / num_trading_days_per_year  # Time step (fraction of a year)
    daily_drift = drift - 0.5 * volatility**2  # Adjusted drift for log returns
    daily_volatility = volatility * np.sqrt(dt)

    # Generate random daily returns (standard normal distribution)
    # Shape: (num_days, num_simulations)
    random_shocks = np.random.standard_normal(size=(num_days, num_simulations))

    # Calculate daily log returns
    # dS/S = daily_drift * dt + daily_volatility * dW
    # ln(St/S0) = (drift - 0.5 * vol^2) * t + vol * Wt
    # Daily log return = daily_drift * dt + daily_volatility * random_shock
    daily_log_returns = daily_drift * dt + daily_volatility * random_shocks

    # Calculate cumulative log returns
    cumulative_log_returns = np.vstack([np.zeros(num_simulations), daily_log_returns]).cumsum(axis=0)

    # Calculate simulated prices
    # St = S0 * exp(cumulative_log_returns)
    simulated_prices = initial_price * np.exp(cumulative_log_returns)

    # Convert to DataFrame for easier handling and analysis
    price_paths_df = pd.DataFrame(simulated_prices)
    price_paths_df.columns = [f'Simulation_{i+1}' for i in range(num_simulations)]
    price_paths_df.index = pd.to_datetime(pd.date_range(start=pd.Timestamp.now().date(), periods=num_days + 1, freq='B'))

    return price_paths_df

# --- Performance Indicator Calculation Function ---
def calculate_performance_indicators(simulated_prices_df: pd.DataFrame, confidence_level: float = 0.95) -> dict:
    """
    Calculates various performance indicators from the simulated stock prices.

    Args:
        simulated_prices_df (pd.DataFrame): DataFrame of simulated stock price paths.
        confidence_level (float): The confidence level for VaR and CVaR calculations (e.g., 0.95 for 95%).

    Returns:
        dict: A dictionary containing calculated performance indicators.
    """
    # Get the final prices of all simulations
    final_prices = simulated_prices_df.iloc[-1]

    # Calculate basic statistics
    mean_final_price = final_prices.mean()
    median_final_price = final_prices.median()
    std_dev_final_price = final_prices.std()

    # Calculate Value at Risk (VaR)
    # VaR at a given confidence level is the maximum potential loss over a period
    # at that confidence level. Here, we calculate it as the price below which
    # (1 - confidence_level) of outcomes fall.
    var_price = final_prices.quantile(1 - confidence_level)
    # VaR as a loss from the initial price
    var_loss = initial_price - var_price

    # Calculate Conditional Value at Risk (CVaR) or Expected Shortfall
    # CVaR is the expected loss given that the loss exceeds the VaR.
    # It's the average of the worst (1 - confidence_level) outcomes.
    cvar_prices = final_prices[final_prices <= var_price]
    cvar_price = cvar_prices.mean()
    # CVaR as a loss from the initial price
    cvar_loss = initial_price - cvar_price

    # Calculate probability of price exceeding initial price
    prob_gain = (final_prices > simulated_prices_df.iloc[0, 0]).mean() # Compare to the initial price of the first simulation
    prob_loss = (final_prices < simulated_prices_df.iloc[0, 0]).mean()

    # Calculate min and max final prices
    min_final_price = final_prices.min()
    max_final_price = final_prices.max()

    indicators = {
        "mean_final_price": mean_final_price,
        "median_final_price": median_final_price,
        "std_dev_final_price": std_dev_final_price,
        f"VaR_{int(confidence_level*100)}%_price": var_price,
        f"VaR_{int(confidence_level*100)}%_loss": var_loss,
        f"CVaR_{int(confidence_level*100)}%_price": cvar_price,
        f"CVaR_{int(confidence_level*100)}%_loss": cvar_loss,
        "probability_of_gain": prob_gain,
        "probability_of_loss": prob_loss,
        "min_final_price": min_final_price,
        "max_final_price": max_final_price
    }
    return indicators

# --- Example Usage (for testing the Python script directly) ---
if __name__ == "__main__":
    print("--- Running Monte Carlo Stock Simulation Example ---")

    # Define simulation parameters
    initial_stock_price = 100.0
    expected_annual_return = 0.08  # 8%
    annual_volatility = 0.20     # 20%
    simulation_years = 5
    number_of_simulations = 1000

    # Run the simulation
    print(f"Simulating {number_of_simulations} paths over {simulation_years} years...")
    simulated_data = monte_carlo_stock_simulation(
        initial_price=initial_stock_price,
        drift=expected_annual_return,
        volatility=annual_volatility,
        time_horizon_years=simulation_years,
        num_simulations=number_of_simulations
    )

    print("\nFirst 5 rows of simulated price paths:")
    print(simulated_data.head())

    print("\nLast 5 rows of simulated price paths:")
    print(simulated_data.tail())

    # Calculate performance indicators
    print("\nCalculating performance indicators...")
    indicators = calculate_performance_indicators(simulated_data, confidence_level=0.99) # Using 99% VaR/CVaR

    print("\n--- Simulation Results ---")
    for key, value in indicators.items():
        if "prob" in key:
            print(f"{key.replace('_', ' ').title()}: {value:.2%}")
        elif "loss" in key:
            print(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
        else:
            print(f"{key.replace('_', ' ').title()}: ${value:,.2f}")

    print("\n--- End of Simulation Example ---")

    # To save the simulation data to a CSV file:
    # simulated_data.to_csv("monte_carlo_stock_paths.csv")
    # print("\nSimulated data saved to 'monte_carlo_stock_paths.csv'")
