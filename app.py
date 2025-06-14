import streamlit as st
import pandas as pd
import pydeck as pdk
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np

# ---------- Haversine ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ---------- Caching CSVs ----------
@st.cache_data
def load_default_data():
    returns = pd.read_csv("returns.csv")
    inventory = pd.read_csv("store_inventory.csv")
    demand = pd.read_csv("store_demand.csv")
    return returns, inventory, demand

# ---------- Genetic Algorithm ----------
def genetic_algorithm(returns_df, inventory_df, generations=20, population_size=10):
    def fitness(weights):
        w_stock, w_sales, w_dist = weights
        total = w_stock + w_sales + w_dist
        if total == 0:
            return float('inf')
        w_stock /= total
        w_sales /= total
        w_dist /= total

        total_score = 0
        for product_id, group in inventory_df.groupby("product_id"):
            group = group.copy()
            group["stock_score"] = 1 - MinMaxScaler().fit_transform(group[["current_stock"]])
            group["sales_score"] = MinMaxScaler().fit_transform(group[["past_week_sales"]])
            for _, row in returns_df[returns_df["product_id"] == product_id].iterrows():
                r_lat = row["return_location_lat"]
                r_lng = row["return_location_lng"]
                group["distance_km"] = group.apply(lambda x: haversine(r_lat, r_lng, x["lat"], x["lng"]), axis=1)
                group["distance_score"] = 1 - MinMaxScaler().fit_transform(group[["distance_km"]])
                group["score"] = (
                    w_stock * group["stock_score"] +
                    w_sales * group["sales_score"] +
                    w_dist * group["distance_score"]
                )
                best = group.loc[group["score"].idxmax()]
                total_score += best["distance_km"]
        return total_score

    def mutate(weights):
        i = random.randint(0, 2)
        weights[i] += random.uniform(-0.1, 0.1)
        weights = [max(0.0, min(1.0, w)) for w in weights]
        return weights

    population = [np.random.dirichlet(np.ones(3), size=1)[0].tolist() for _ in range(population_size)]
    for _ in range(generations):
        population.sort(key=fitness)
        next_gen = population[:population_size // 2]
        while len(next_gen) < population_size:
            parent = random.choice(next_gen)
            child = mutate(parent[:])
            next_gen.append(child)
        population = next_gen
    best = population[0]
    return best

def main():
    st.set_page_config(page_title="Smart Return Routing", layout="wide")

    st.sidebar.header("📂 Upload Your Data (Optional)")

    uploaded_returns = st.sidebar.file_uploader("Upload Returns CSV", type=["csv"])
    uploaded_inventory = st.sidebar.file_uploader("Upload Inventory CSV", type=["csv"])
    uploaded_demand = st.sidebar.file_uploader("Upload Demand CSV", type=["csv"])

    if uploaded_returns and uploaded_inventory and uploaded_demand:
        returns_df = pd.read_csv(uploaded_returns)
        inventory_df = pd.read_csv(uploaded_inventory)
        demand_df = pd.read_csv(uploaded_demand)
    else:
        st.sidebar.info("Using default data files.")
        returns_df, inventory_df, demand_df = load_default_data()

    inventory_df = pd.merge(inventory_df, demand_df, on=["store_id", "product_id"], how="left")
    inventory_df["past_week_sales"].fillna(0, inplace=True)

    recommendations = []

    # ---------- GA Button ----------
    if st.sidebar.button("🧠 Optimize Weights (GA)"):
        w_stock, w_sales, w_dist = genetic_algorithm(returns_df, inventory_df)
        st.session_state["stock_weight"] = float(round(w_stock, 2))
        st.session_state["sales_weight"] = float(round(w_sales, 2))
        st.session_state["distance_weight"] = float(round(w_dist, 2))
        st.sidebar.success("Optimized weights applied!")

    # Sidebar sliders
    stock_weight = st.sidebar.slider("📉 Stock Weight", 0.0, 1.0, st.session_state.get("stock_weight", 0.5), key="stock_weight")
    sales_weight = st.sidebar.slider("📈 Sales Weight", 0.0, 1.0, st.session_state.get("sales_weight", 0.3), key="sales_weight")
    distance_weight = st.sidebar.slider("🧭 Distance Weight", 0.0, 1.0, st.session_state.get("distance_weight", 0.2), key="distance_weight")

    total_weight = stock_weight + sales_weight + distance_weight

    if total_weight == 0:
        st.error("⚠️ Please adjust at least one weight above 0 to compute recommendations.")
        return

    stock_weight /= total_weight
    sales_weight /= total_weight
    distance_weight /= total_weight

    use_boost = sum([stock_weight > 0, sales_weight > 0, distance_weight > 0]) > 1

    st.markdown("### ⚖️ Scoring Weights Breakdown")
    st.write(f"Normalized Weights Applied:")
    st.write(f"- 📉 **Stock Weight:** {round(stock_weight, 2)}")
    st.write(f"- 📈 **Sales Weight:** {round(sales_weight, 2)}")
    st.write(f"- 🧭 **Distance Weight:** {round(distance_weight, 2)}")
    st.caption("The final score = (stock × weight) + (sales × weight) + (distance × weight) + optional boost")

    normalized_data = {}
    for product_id, group in inventory_df.groupby("product_id"):
        group = group.copy()
        group["stock_score"] = 1 - MinMaxScaler().fit_transform(group[["current_stock"]])
        group["sales_score"] = MinMaxScaler().fit_transform(group[["past_week_sales"]])
        normalized_data[product_id] = group

    for _, row in returns_df.iterrows():
        product_id = row["product_id"]
        product_name = row["product_name"]
        r_lat = row["return_location_lat"]
        r_lng = row["return_location_lng"]

        if product_id not in normalized_data:
            continue

        candidates = normalized_data[product_id].copy()
        candidates["distance_km"] = candidates.apply(
            lambda x: haversine(r_lat, r_lng, x["lat"], x["lng"]), axis=1
        )

        candidates["distance_score"] = 1 - MinMaxScaler().fit_transform(candidates[["distance_km"]])
        sales_scaled = MinMaxScaler().fit_transform(candidates[["past_week_sales"]])
        dist_scaled = 1 - MinMaxScaler().fit_transform(candidates[["distance_km"]])
        candidates["boost"] = 0.5 * sales_scaled.flatten() + 0.5 * dist_scaled.flatten()

        candidates["score"] = (
            stock_weight * candidates["stock_score"] +
            sales_weight * candidates["sales_score"] +
            distance_weight * candidates["distance_score"]
        )
        if use_boost:
            candidates["score"] += 0.1 * candidates["boost"]

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
            "distance_km": round(best_store["distance_km"], 2),
            "boost_score": round(best_store["boost"], 3)
        })

    rec_df = pd.DataFrame(recommendations)

    rec_df["tooltip"] = (
        "<b>Product:</b> " + rec_df["product_name"] + "<br/>" +
        "<b>To Store:</b> " + rec_df["store_name"] + "<br/>" +
        "<b>Location:</b> (" + rec_df["return_lat"].astype(str) + ", " + rec_df["return_lng"].astype(str) + ")"
    )

    store_tooltip_df = rec_df.groupby(["store_name", "store_lat", "store_lng"]).agg({
        "product_name": lambda x: ', '.join(sorted(set(x)))
    }).reset_index()
    store_tooltip_df["tooltip"] = (
        "<b>Store:</b> " + store_tooltip_df["store_name"] + "<br/>" +
        "<b>Products:</b> " + store_tooltip_df["product_name"] + "<br/>" +
        "<b>Coordinates:</b> (" + store_tooltip_df["store_lat"].astype(str) + ", " + store_tooltip_df["store_lng"].astype(str) + ")"
    )
    store_tooltip_df["offset_lat"] = store_tooltip_df["store_lat"] + 0.01
    store_tooltip_df["offset_lng"] = store_tooltip_df["store_lng"] + 0.01

    st.title("📍 Smart Product Return Map")
    st.markdown("🔴 Red = Return Point | 🟢 Green = Recommended Store")

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

    selected_return = st.selectbox("🖱️ Select a Return ID to View Details", filtered_df["return_id"].unique() if not filtered_df.empty else [])
    if selected_return:
        selected_row = filtered_df[filtered_df["return_id"] == selected_return].iloc[0]
        st.markdown("### 🔍 Selected Return Details")
        st.write(f"**Product:** {selected_row['product_name']}")
        st.write(f"**Return Location:** ({selected_row['return_lat']}, {selected_row['return_lng']})")
        st.write(f"**Recommended Store:** {selected_row['store_name']}")
        st.write(f"**Store Location:** ({selected_row['store_lat']}, {selected_row['store_lng']})")
        st.write(f"**Distance (km):** {selected_row['distance_km']}")
        st.write(f"**Boost Score:** {selected_row['boost_score']}")

    st.markdown("### 📈 Summary")
    st.write(f"**Total Returns:** {len(filtered_df)}")
    st.write(f"**Unique Stores Recommended:** {filtered_df['store_name'].nunique()}")
    st.write(f"**Average Distance (km):** {round(filtered_df['distance_km'].mean(), 2)}")
    st.download_button("⬇️ Download Filtered Recommendations", filtered_df.to_csv(index=False), file_name="filtered_recommendations.csv", mime="text/csv")

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
        st.pydeck_chart(pdk.Deck(
            layers=[line_layer, return_layer, store_layer],
            initial_view_state=view_state,
            tooltip={"html": "{tooltip}", "style": {"backgroundColor": "black", "color": "white"}}
        ))
    else:
        st.warning("⚠️ No data to display on the map. Adjust your filters.")

    st.subheader("📊 Routing Summary")
    st.dataframe(filtered_df[["return_id", "product_name", "store_name", "distance_km", "boost_score"]])

if __name__ == "__main__":
    main()
