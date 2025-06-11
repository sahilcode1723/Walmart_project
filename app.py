import streamlit as st
import pandas as pd
import pydeck as pdk
from math import radians, sin, cos, sqrt, atan2

# ---------- Haversine Formula ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ---------- Load CSVs ----------
returns_df = pd.read_csv("returns.csv")
inventory_df = pd.read_csv("store_inventory.csv")
demand_df = pd.read_csv("store_demand.csv")

# ---------- Merge inventory with demand ----------
inventory_df = pd.merge(inventory_df, demand_df, on=["store_id", "product_id"], how="left")
inventory_df["past_week_sales"].fillna(0, inplace=True)

# ---------- Recommendation Logic ----------
recommendations = []

for _, row in returns_df.iterrows():
    product_id = row["product_id"]
    product_name = row["product_name"]
    r_lat = row["return_location_lat"]
    r_lng = row["return_location_lng"]

    candidates = inventory_df[inventory_df["product_id"] == product_id].copy()
    candidates["distance_km"] = candidates.apply(
        lambda x: haversine(r_lat, r_lng, x["lat"], x["lng"]), axis=1
    )

    candidates.sort_values(by=["current_stock", "past_week_sales", "distance_km"],
                           ascending=[True, False, True], inplace=True)

    if not candidates.empty:
        best_store = candidates.iloc[0]
        recommendations.append({
            "return_id": row["return_id"],
            "product_id": product_id,
            "product_name": product_name,
            "return_lat": r_lat,
            "return_lng": r_lng,
            "store_name": best_store["store_name"],
            "store_lat": best_store["lat"],
            "store_lng": best_store["lng"],
            "distance_km": round(best_store["distance_km"], 2)
        })

rec_df = pd.DataFrame(recommendations)

# ---------- Red Dot Tooltip ----------
rec_df["tooltip"] = (
    "Product: " + rec_df["product_name"] + "<br/>" +
    "To Store: " + rec_df["store_name"] + "<br/>" +
    "Return Location: (" + rec_df["return_lat"].astype(str) + ", " + rec_df["return_lng"].astype(str) + ")"
)

# ---------- Green Dot Tooltip ----------
store_tooltip_df = rec_df.groupby(
    ["store_name", "store_lat", "store_lng"]
).agg({
    "product_name": lambda x: ', '.join(sorted(set(x)))
}).reset_index()

store_tooltip_df["tooltip"] = (
    "Store: " + store_tooltip_df["store_name"] + "<br/>" +
    "Products: " + store_tooltip_df["product_name"] + "<br/>" +
    "Coordinates: (" + store_tooltip_df["store_lat"].astype(str) + ", " + store_tooltip_df["store_lng"].astype(str) + ")"
)

# ---------- Apply Offset to Store Dots ----------
store_tooltip_df["offset_lat"] = store_tooltip_df["store_lat"] + 0.01
store_tooltip_df["offset_lng"] = store_tooltip_df["store_lng"] + 0.01

# ---------- Streamlit UI ----------
st.title("📍 Smart Product Return Map")
st.markdown("🔴 Red = Return Point | 🟢 Green = Recommended Store (slightly offset to avoid overlap)")

# ---------- Map Layers ----------
return_layer = pdk.Layer(
    "ScatterplotLayer",
    data=rec_df,
    get_position='[return_lng, return_lat]',
    get_fill_color='[255, 0, 0, 160]',
    get_radius=10000,
    pickable=True,
)

store_layer = pdk.Layer(
    "ScatterplotLayer",
    data=store_tooltip_df,
    get_position='[offset_lng, offset_lat]',
    get_fill_color='[0, 200, 0, 160]',
    get_radius=10000,
    pickable=True,
)

line_layer = pdk.Layer(
    "LineLayer",
    data=rec_df,
    get_source_position='[return_lng, return_lat]',
    get_target_position='[store_lng, store_lat]',
    get_color=[0, 0, 0],
    get_width=2
)

view_state = pdk.ViewState(
    latitude=rec_df["return_lat"].mean(),
    longitude=rec_df["return_lng"].mean(),
    zoom=4
)

# ---------- Map ----------
st.pydeck_chart(
    pdk.Deck(
        layers=[line_layer, return_layer, store_layer],
        initial_view_state=view_state,
        tooltip={
            "html": "{tooltip}",
            "style": {"backgroundColor": "black", "color": "white"}
        }
    )
)

# ---------- Table ----------
st.subheader("📊 Routing Summary")
st.dataframe(rec_df[[
    "return_id", "product_name", "store_name", "distance_km"
]])
