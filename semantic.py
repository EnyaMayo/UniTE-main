## env: unite
## run: cd /home/pshao8/poi
#       python semantic.py
## output file: ./UniTE_h5_dataset/tky_trajectory_descriptions.csv
## parameters: NPOI=3, 过滤长度>=3, 删除了每条轨迹的最后一个poi点
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from datetime import datetime
import time
from joblib import Parallel, delayed
from tqdm import tqdm

# Record total start time
total_start = time.time()

# Define column names
columns = ['user_id', 'venue_id', 'venue_category_id', 'venue_category_name', 
           'lat', 'lng', 'Timezone_Offset', 'UTC_Time']

# Read data
start = time.time()
tky_df = pd.read_csv('./dataset_tsmc2014/dataset_TSMC2014_TKY.txt', sep='\t', header=None, 
                     names=columns, encoding='ISO-8859-1')
print(f"Read data: {time.time() - start:.2f} seconds")

# Convert to datetime and adjust to local time
start = time.time()
tky_df['time'] = pd.to_datetime(tky_df['UTC_Time'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
tky_df['local_time'] = tky_df['time'] + pd.to_timedelta(tky_df['Timezone_Offset'], unit='m')

# Drop invalid rows
invalid_times = tky_df['time'].isna().sum()
if invalid_times > 0:
    print(f"Warning: {invalid_times} rows have invalid UTC_Time. Dropping these rows.")
    tky_df = tky_df.dropna(subset=['time', 'local_time', 'venue_category_name', 'lat', 'lng'])
print("Initial rows:", len(tky_df))
print(f"Datetime processing: {time.time() - start:.2f} seconds")

# Create date and trip columns
start = time.time()
tky_df['date'] = tky_df['local_time'].dt.date
tky_df['trip'] = tky_df['user_id'].astype(str) + '_' + tky_df['date'].astype(str)

# Drop duplicates
tky_df = tky_df.drop_duplicates(subset=['user_id', 'local_time', 'venue_id'])
print(f"Trip creation: {time.time() - start:.2f} seconds")

# Sort by trip and generate sequence index
start = time.time()
tky_df = tky_df.sort_values(by=['trip', 'local_time'])
tky_df['seq_i'] = tky_df.groupby('trip').cumcount()

# Filter trips with length >= 3
trip_counts = tky_df.groupby('trip').size()
valid_trips = trip_counts[trip_counts >= 3].index
tky_df = tky_df[tky_df['trip'].isin(valid_trips)]

# Check if any trips remain
if tky_df.empty:
    raise ValueError("No trips with length >= 3 found.")
print("Trip count stats:", trip_counts.describe())
print("Trips with length >= 3:", len(valid_trips))
print("Sample trip counts:", trip_counts[trip_counts.index.str.contains('868_2012-04-04')])
print("Sample trip values:", tky_df['trip'].head())
print(f"Sort and filter trips: {time.time() - start:.2f} seconds")

# Exclude last record of each trip
start = time.time()
tky_df = tky_df.groupby('trip').head(-1).reset_index(drop=True)

# Verify trip column
if 'trip' not in tky_df.columns:
    raise ValueError("Trip column lost after head.")
print("tky_df columns after head:", tky_df.columns)
print(f"Exclude last point: {time.time() - start:.2f} seconds")

# Construct venue mapping
start = time.time()
venue_map = {vid: idx for idx, vid in enumerate(tky_df['venue_id'].unique())}
tky_df['road_id'] = tky_df['venue_id'].map(venue_map)
print(f"Venue mapping: {time.time() - start:.2f} seconds")

# Construct trips
start = time.time()
trips = tky_df[['trip', 'seq_i', 'local_time', 'lng', 'lat', 'road_id', 'venue_category_id', 'venue_category_name']].copy()
trips['road_prop'] = 0
trips.columns = ['trip', 'seq_i', 'time', 'lng', 'lat', 'road', 'level', 'road_prop', 'venue_category_name']

# Ensure trips is sorted
trips = trips.sort_values(['trip', 'seq_i'])

# Construct trip_info
trip_info = tky_df.groupby('trip').agg({
    'local_time': ['min', 'max'],
    'venue_id': 'count'
}).reset_index()
trip_info.columns = ['trip', 'start', 'end', 'length']
trip_info['driver'] = trip_info['trip'].str.split('_').str[0].astype('int32')

# Construct road_info
road_info = tky_df[['road_id', 'lng', 'lat', 'venue_category_id', 'venue_category_name']].drop_duplicates(subset=['road_id'])
road_info.columns = ['road', 'road_lng', 'road_lat', 'level', 'venue_category_name']

# Get last venue_id as label
last_venue = tky_df.groupby('trip').tail(1)[['trip', 'venue_id']].rename(columns={'venue_id': 'label'})
label_df = last_venue.reset_index(drop=True)
print(f"Construct datasets: {time.time() - start:.2f} seconds")

# Build BallTree
start = time.time()
road_info['lat_rad'] = np.radians(road_info['road_lat'])
road_info['lng_rad'] = np.radians(road_info['road_lng'])
poi_coords = road_info[['lat_rad', 'lng_rad']].values
ball_tree = BallTree(poi_coords, metric='haversine')
print(f"Build BallTree: {time.time() - start:.2f} seconds")

# Parameters
NPOI = 3
EARTH_RADIUS = 6371

# Batch query for start and end points
start = time.time()
start_points = trips.groupby('trip').first()[['lat', 'lng']]
end_points = trips.groupby('trip').last()[['lat', 'lng']]
print("start_points shape:", start_points.shape)
print("NaN in start_points:", start_points.isna().sum())
print("Sample start_points:\n", start_points.head())
start_points_rad = np.radians(start_points.values)
end_points_rad = np.radians(end_points.values)
distances_start, indices_start = ball_tree.query(start_points_rad, k=NPOI)
distances_end, indices_end = ball_tree.query(end_points_rad, k=NPOI)
trip_ids = start_points.index
print(f"Batch BallTree query (start + end): {time.time() - start:.2f} seconds")

# Verify trip_ids uniqueness
total_trips = len(trip_ids)
print(f"Total trips to process: {total_trips}")
print(f"Unique trips: {len(set(trip_ids))}")
print("Sample trip_ids:", trip_ids[:5].tolist())

# Function to process a single trip
def process_trip(trip_id, trip_data, trip_start_time, idx_start, idx_end, label, venue_map, road_info):
    dt = pd.to_datetime(trip_start_time)
    day = dt.strftime('%A')
    hour = dt.hour
    head_part = f"The trajectory happened on {day} at {hour} o’clock"
    
    # Start point: 3 nearest POIs, exclude self
    closest_pois_start = road_info.iloc[idx_start]
    if not trip_data.empty:
        start_point = trip_data.iloc[0][['lat', 'lng']]
        closest_pois_start = closest_pois_start[
            ~closest_pois_start[['road_lat', 'road_lng']].eq(start_point).all(axis=1)
        ]
        if len(closest_pois_start) < NPOI:
            return None
    start_poi_desc = ', '.join([f"{row['venue_category_name']} ({row['road_lat']}, {row['road_lng']})"
                                for _, row in closest_pois_start.iterrows() if pd.notna(row['venue_category_name'])])

    # End point: 3 nearest POIs, exclude self
    closest_pois_end = road_info.iloc[idx_end]
    if not trip_data.empty:
        end_point = trip_data.iloc[-1][['lat', 'lng']]
        closest_pois_end = closest_pois_end[
            ~closest_pois_end[['road_lat', 'road_lng']].eq(end_point).all(axis=1)
        ]
        if len(closest_pois_end) < NPOI:
            return None
    end_poi_desc = ', '.join([f"{row['venue_category_name']} ({row['road_lat']}, {row['road_lng']})"
                              for _, row in closest_pois_end.iterrows() if pd.notna(row['venue_category_name'])])

    if not start_poi_desc or not end_poi_desc:
        return None
    poi_part = f"starts near: {{{start_poi_desc}}}, ends near: {{{end_poi_desc}}}"
    
    # passes through: includes start, excludes end
    traj_points = ', '.join([f"({row['lat']}, {row['lng']}, {row['time'].strftime('%H:%M:%S')})"
                             for _, row in trip_data.iterrows()])
    traj_part = f"passes through {{{traj_points}}}"
    
    return {'trip_id': trip_id, 'description': f"{head_part}, {poi_part}, {traj_part}."}

# Prepare inputs with vectorized operations
start = time.time()
# Precompute trip_data, start_time, and labels
trip_data_groups = dict(tuple(trips.groupby('trip')))
start_time_map = trip_info.set_index('trip')['start'].to_dict()
label_map = label_df.set_index('trip')['label'].to_dict()

# Build inputs with tqdm
inputs = []
for i in tqdm(range(total_trips), desc="Preparing inputs"):
    trip_id = trip_ids[i]
    trip_data = trip_data_groups.get(trip_id, pd.DataFrame())
    if trip_data.empty:
        continue
    trip_start_time = start_time_map.get(trip_id)
    idx_start = indices_start[i]
    idx_end = indices_end[i]
    label = label_map.get(trip_id)
    if trip_start_time is None or label is None:
        continue
    inputs.append((trip_id, trip_data, trip_start_time, idx_start, idx_end, label, venue_map, road_info))
print(f"Prepare inputs: {time.time() - start:.2f} seconds")

# Run parallel processing
start = time.time()
trajectory_descriptions = Parallel(n_jobs=-1, verbose=10, batch_size=1000)(
    delayed(process_trip)(*args) for args in tqdm(inputs, desc="Processing trips")
)

# Filter out None results
trajectory_descriptions = [desc for desc in trajectory_descriptions if desc is not None]
print(f"Process trips: {time.time() - start:.2f} seconds")

# Save to CSV
start = time.time()
output_df = pd.DataFrame(trajectory_descriptions)
output_df.to_csv('./UniTE_h5_dataset/tky_trajectory_descriptions.csv', index=False)
print(f"Save CSV: {time.time() - start:.2f} seconds")

# Print total time and sample output
print(f"Total runtime: {(time.time() - total_start) / 60:.2f} minutes")
print("Sample Trajectory Descriptions:")
print(output_df.head())