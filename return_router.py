import pandas as pd
from math import radians, cos, sin, asin, sqrt

# Load the CSV files
returns = pd.read_csv("returns.csv")
inventory = pd.read_csv("store_inventory.csv")
demand = pd.read_csv("store_demand.csv")

# Merge inventory and demand data
store_data = pd.merge(inventory, demand, on=["store_id", "product_id"])

# Define the Haversine formula to calculate distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

# Store recommendations for each return
recommendations = []

# Go through each returned product
for _, ret in returns.iterrows():
    prod_id = ret["product_id"]
    ret_lat, ret_lng = ret["return_location_lat"], ret["return_location_lng"]

    # Find all stores with the same product and stock < 5
    candidates = store_data[
        (store_data["product_id"] == prod_id) &
        (store_data["current_stock"] < 5)
    ].copy()

    if candidates.empty:
        recommendations.append("No suitable store found")
        continue

    # Add distance to each candidate
    candidates["distance_km"] = candidates.apply(
        lambda row: haversine(ret_lat, ret_lng, row["lat"], row["lng"]), axis=1
    )

    # Sort by highest demand, then shortest distance
    candidates = candidates.sort_values(by=["past_week_sales", "distance_km"], ascending=[False, True])

    # Pick the best store
    best = candidates.iloc[0]
    recommendations.append(f"Route to {best['store_name']} ({best['distance_km']:.1f} km)")

# Add recommendations to the original returns DataFrame
returns["recommended_store"] = recommendations

# Show the result
print("\n🧠 Smart Routing Suggestions:\n")
print(returns[["return_id", "product_name", "recommended_store"]])
