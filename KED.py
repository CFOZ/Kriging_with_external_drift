import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pykrige.uk import UniversalKriging
from scipy.spatial import cKDTree
from scipy.interpolate import NearestNDInterpolator

# Load the dataset
data = pd.read_csv('C:/Users/Asus/Desktop/CWRS/KED/sample_data.csv')

# Separate station data and precipitation data
station_data = data[['Station_ID', 'Longitude', 'Latitude', 'Elevation']]
precipitation_data = data.drop(columns=['Station_ID', 'Longitude', 'Latitude', 'Elevation'])

# Reshape the data for kriging
longitudes = station_data['Longitude'].values
latitudes = station_data['Latitude'].values
elevations = station_data['Elevation'].values

# Store results in a DataFrame for comparison
filled_precipitation_data = precipitation_data.copy()

# Initialize dictionaries to store RMSE and MAE for each station
station_rmse = {station: [] for station in station_data['Station_ID']}
station_mae = {station: [] for station in station_data['Station_ID']}

# Coordinates and elevation as array for nearest neighbors
coords = np.vstack((longitudes, latitudes)).T
tree = cKDTree(coords)

# Define a function for nearest-neighbor interpolation
def nearest_neighbor_interpolation(x, y, known_values, missing_x, missing_y, k=6):
    interpolator = NearestNDInterpolator(list(zip(x, y)), known_values)
    return interpolator(missing_x, missing_y)

# Apply Universal Kriging with elevation as external drift or fallback to nearest neighbor interpolation
for day in precipitation_data.columns:
    day_data = precipitation_data[day]
    known_values_mask = ~day_data.isna()
    known_values = day_data[known_values_mask].values

    if len(known_values) > 1:  # Ensure there are enough points for kriging
        try:
            # Try Universal Kriging with drift term using elevation
            kriging_model = UniversalKriging(
                longitudes[known_values_mask],
                latitudes[known_values_mask],
                known_values,
                drift_terms=['specified'],
                variogram_model='spherical',
                specified_drift=[np.array(elevations[known_values_mask])]  # Convert specified drift to NumPy array
            )
            # Interpolate missing values
            missing_values_mask = day_data.isna()
            if missing_values_mask.any():
                z_values, _ = kriging_model.execute(
                    'points',
                    longitudes[missing_values_mask],
                    latitudes[missing_values_mask],
                    specified_drift_arrays=[np.array(elevations[missing_values_mask])]  # Convert to NumPy array
                )
                filled_precipitation_data.loc[missing_values_mask, day] = z_values

        except Exception as e:
            print(f"Kriging failed for {day}. Error: {e}. Falling back to nearest-neighbor interpolation.")
            missing_values_mask = day_data.isna()
            if missing_values_mask.any():
                # Fallback to nearest 6 neighbors interpolation
                z_values = nearest_neighbor_interpolation(
                    longitudes[known_values_mask],
                    latitudes[known_values_mask],
                    known_values,
                    longitudes[missing_values_mask],
                    latitudes[missing_values_mask]
                )
                filled_precipitation_data.loc[missing_values_mask, day] = z_values

        # Calculate station-wise error metrics
        for idx, station_id in enumerate(station_data['Station_ID']):
            if known_values_mask.iloc[idx]:  # Only calculate error for known values
                true_value = day_data.iloc[idx]
                pred_value, _ = kriging_model.execute(
                    'points',
                    [longitudes[idx]],
                    [latitudes[idx]],
                    specified_drift_arrays=[np.array([elevations[idx]])]  # Ensure single drift value is NumPy array
                )
                # Calculate RMSE and MAE for the station
                rmse = mean_squared_error([true_value], [pred_value[0]], squared=False)
                mae = mean_absolute_error([true_value], [pred_value[0]])

                # Append errors to respective station lists
                station_rmse[station_id].append(rmse)
                station_mae[station_id].append(mae)

# Calculate overall station-wise RMSE and MAE, replacing NaN values with zero
overall_station_rmse = {station: np.nan_to_num(np.mean(errors), nan=0.0) for station, errors in station_rmse.items()}
overall_station_mae = {station: np.nan_to_num(np.mean(errors), nan=0.0) for station, errors in station_mae.items()}

print("Overall Station-wise RMSE:", overall_station_rmse)
print("Overall Station-wise MAE:", overall_station_mae)

# Calculate average RMSE and MAE across all stations, replacing NaN values with zero
average_rmse = np.nan_to_num(np.mean(list(overall_station_rmse.values())), nan=0.0)
average_mae = np.nan_to_num(np.mean(list(overall_station_mae.values())), nan=0.0)

print("Average RMSE across all stations:", average_rmse)
print("Average MAE across all stations:", average_mae)

# Optionally, save the filled data to a new CSV file
filled_precipitation_data.to_csv('C:/Users/Asus/Desktop/CWRS/KED/filled_precipitation_data_2.csv', index=False)
