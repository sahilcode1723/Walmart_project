import streamlit as st
import pandas as pd
import pydeck as pdk
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import MinMaxScaler

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

    if candidates.empty:
        continue

    # Normalize values
    scaler = MinMaxScaler()

    stock_score = 1 - scaler.fit_transform(candidates[["current_stock"]])  # lower stock = higher score
    sales_score = scaler.fit_transform(candidates[["past_week_sales"]])    # higher sales = higher score
    distance_score = 1 - scaler.fit_transform(candidates[["distance_km"]]) # closer = higher score

    # Final score
    candidates["score"] = (
        0.5 * stock_score.flatten() +
        0.3 * sales_score.flatten() +
        0.2 * distance_score.flatten()
    )

    best_store = candidates.loc[candidates["score"].idxmax()]

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
    "<b>Product:</b> " + rec_df["product_name"] + "<br/>" +
    "<b>To Store:</b> " + rec_df["store_name"] + "<br/>" +
    "<b>Location:</b> (" + rec_df["return_lat"].astype(str) + ", " + rec_df["return_lng"].astype(str) + ")"
)

# ---------- Green Dot Tooltip ----------
store_tooltip_df = rec_df.groupby(
    ["store_name", "store_lat", "store_lng"]
).agg({
    "product_name": lambda x: ', '.join(sorted(set(x)))
}).reset_index()

store_tooltip_df["tooltip"] = (
    "<b>Store:</b> " + store_tooltip_df["store_name"] + "<br/>" +
    "<b>Products:</b> " + store_tooltip_df["product_name"] + "<br/>" +
    "<b>Coordinates:</b> (" + store_tooltip_df["store_lat"].astype(str) + ", " + store_tooltip_df["store_lng"].astype(str) + ")"
)

# ---------- Apply Offset to Store Dots ----------
store_tooltip_df["offset_lat"] = store_tooltip_df["store_lat"] + 0.01
store_tooltip_df["offset_lng"] = store_tooltip_df["store_lng"] + 0.01

# ---------- Streamlit UI ----------
st.title("📍 Smart Product Return Map")
st.markdown("🔴 Red = Return Point | 🟢 Green = Recommended Store (slightly offset to avoid overlap)")

# ---------- Enhancements Start ----------

# Filters
product_filter = st.multiselect("🔎 Filter by Product", rec_df["product_name"].unique())
store_filter = st.multiselect("🏪 Filter by Store", rec_df["store_name"].unique())
search_input = st.text_input("🔍 Search by Return ID or Product Name")

filtered_df = rec_df.copy()
if product_filter:
    filtered_df = filtered_df[filtered_df["product_name"].isin(product_filter)]
if store_filter:
    filtered_df = filtered_df[filtered_df["store_name"].isin(store_filter)]
if search_input:
    filtered_df = filtered_df[
        filtered_df["product_name"].str.contains(search_input, case=False) |
        filtered_df["return_id"].astype(str).str.contains(search_input)
    ]

# ---------- Interactive Selection ----------
selected_return = st.selectbox("🖱️ Select a Return ID to View Details", filtered_df["return_id"].unique() if not filtered_df.empty else [])

if selected_return:
    selected_row = filtered_df[filtered_df["return_id"] == selected_return].iloc[0]
    st.markdown("### 🔍 Selected Return Details")
    st.write(f"**Product:** {selected_row['product_name']}")
    st.write(f"**Return Location:** ({selected_row['return_lat']}, {selected_row['return_lng']})")
    st.write(f"**Recommended Store:** {selected_row['store_name']}")
    st.write(f"**Store Location:** ({selected_row['store_lat']}, {selected_row['store_lng']})")
    st.write(f"**Distance (km):** {selected_row['distance_km']}")

# Summary
st.markdown("### 📈 Summary")
st.write(f"**Total Returns:** {len(filtered_df)}")
st.write(f"**Unique Stores Recommended:** {filtered_df['store_name'].nunique()}")
st.write(f"**Average Distance (km):** {round(filtered_df['distance_km'].mean(), 2)}")

# CSV Export
st.download_button("⬇️ Download Filtered Recommendations", filtered_df.to_csv(index=False), file_name="filtered_recommendations.csv", mime="text/csv")

# ---------- Enhancements End ----------

# ---------- Map Layers ----------
return_layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_df,
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
    data=filtered_df,
    get_source_position='[return_lng, return_lat]',
    get_target_position='[store_lng, store_lat]',
    get_color=[0, 0, 0],
    get_width=2
)

if not filtered_df.empty:
    view_state = pdk.ViewState(
        latitude=filtered_df["return_lat"].mean(),
        longitude=filtered_df["return_lng"].mean(),
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
else:
    st.warning("⚠️ No data to display on the map. Please adjust your filters.")


# ---------- Table ----------
st.subheader("📊 Routing Summary")
st.dataframe(filtered_df[[
    "return_id", "product_name", "store_name", "distance_km"
]])
