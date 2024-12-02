import numpy as np
import pandas as pd
from gstools import vario_estimate, Spherical, Exponential, Gaussian
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = 'C:/Users/Asus/Desktop/CWRS/KED/sample_data.csv'
data = pd.read_csv(file_path)

# Extract coordinates and elevation
longitudes = data['Longitude'].values
latitudes = data['Latitude'].values

# Initialize dictionaries to store MSE for each model across all days
mse_totals = {'spherical': 0, 'exponential': 0, 'gaussian': 0}
mse_counts = {'spherical': 0, 'exponential': 0, 'gaussian': 0}

# Iterate over each day column (from 5th column onward for daily data)
for day in data.columns[4:]:
    day_data = data[day]
    known_values_mask = ~day_data.isna()
    known_values = day_data[known_values_mask].values

    # Ensure there are enough points and non-zero variance for variogram fitting
    if len(known_values) > 1:
        variance = np.var(known_values)
        
        # Skip if variance is zero
        if variance == 0:
            print(f"Skipping {day} due to zero variance in precipitation values.")
            continue

        # Extract coordinates for known values
        known_coords = np.column_stack((longitudes[known_values_mask], latitudes[known_values_mask]))

        # Calculate empirical variogram
        distances, semivariances = vario_estimate((known_coords[:, 0], known_coords[:, 1]), known_values)

        # Initial parameter estimates
        max_distance = np.max(distances)
        nugget = 0.1 * variance

        # Define variogram models with initial parameters
        models = {
            'spherical': Spherical(len_scale=max_distance / 3, var=variance, nugget=nugget),
            'exponential': Exponential(len_scale=max_distance / 3, var=variance, nugget=nugget),
            'gaussian': Gaussian(len_scale=max_distance / 3, var=variance, nugget=nugget)
        }

        # Calculate MSE for each model and add it to the total for each model
        for name, model in models.items():
            model_values = model.variogram(distances)
            mse = mean_squared_error(semivariances, model_values)
            mse_totals[name] += mse
            mse_counts[name] += 1

# Calculate the average MSE for each model
average_mse = {name: mse_totals[name] / mse_counts[name] for name in mse_totals if mse_counts[name] > 0}

# Find the model with the lowest average MSE
if average_mse:
    best_model_name = min(average_mse, key=average_mse.get)
    best_model_mse = average_mse[best_model_name]

    print("Overall Best Model:")
    print(f"Model: {best_model_name}")
    print(f"Average MSE: {best_model_mse}")

    # Display the MSE for each model
    print("\nAverage MSE for each model:")
    for name, mse in average_mse.items():
        print(f"{name.capitalize()} Model MSE: {mse}")
else:
    print("No model could be fitted due to zero variance in all days.")
