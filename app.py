"""
Streamlit dashboard for the landslide susceptibility + rainfall alert prototype.

Run locally:   streamlit run app.py
Deploy:        push this repo (with susceptibility_grid.csv) to GitHub,
                then deploy on Streamlit Community Cloud (share.streamlit.io)

Needs susceptibility_grid.csv (from 03_generate_grid.py) in the same folder.
"""
import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Landslide Early Warning – India", layout="wide")

GRID_CSV = "susceptibility_grid.csv"
RAIN_ALERT_THRESHOLD_MM = 50  # 24h rainfall considered "heavy"
HIGH_SUSCEPTIBILITY = 0.5


@st.cache_data
def load_grid():
    return pd.read_csv(GRID_CSV)


def susceptibility_color(score):
    if score >= 0.7:
        return "#B4001B"
    if score >= HIGH_SUSCEPTIBILITY:
        return "#E8890C"
    if score >= 0.3:
        return "#E8C90C"
    return "#2E8B57"


def nearest_grid_point(df, lat, lon):
    d2 = (df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2
    return df.loc[d2.idxmin()]


@st.cache_data(ttl=1800)  # cache live rainfall for 30 min
def fetch_rainfall(lat, lon):
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=precipitation_sum&past_days=1&forecast_days=2&timezone=auto"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    dates = data["daily"]["time"]
    rain = data["daily"]["precipitation_sum"]
    return dict(zip(dates, rain))


def advisory_text(susceptibility, rain_today, alert):
    if alert:
        en = (
            f"HIGH RISK: This area has high landslide susceptibility "
            f"({susceptibility:.0%}) and recent rainfall of {rain_today:.0f} mm, "
            f"above the {RAIN_ALERT_THRESHOLD_MM} mm alert threshold. "
            f"Avoid travel on steep slopes, watch for cracks in the ground or "
            f"tilting trees, and follow local authority guidance."
        )
        hi = (
            f"उच्च जोखिम: इस क्षेत्र में भूस्खलन की संभावना अधिक है "
            f"({susceptibility:.0%}) और हाल की वर्षा {rain_today:.0f} मिमी रही है, "
            f"जो {RAIN_ALERT_THRESHOLD_MM} मिमी की सीमा से अधिक है। "
            f"ढलानों की यात्रा से बचें और स्थानीय प्रशासन के निर्देशों का पालन करें।"
        )
    else:
        en = (
            f"No immediate alert. Susceptibility is {susceptibility:.0%} and "
            f"recent rainfall is {rain_today:.0f} mm, below the alert threshold. "
            f"Continue to monitor during heavy rain spells."
        )
        hi = (
            f"फिलहाल कोई चेतावनी नहीं। संभावना {susceptibility:.0%} है और हाल की "
            f"वर्षा {rain_today:.0f} मिमी रही, जो सीमा से कम है। भारी बारिश के "
            f"दौरान निगरानी जारी रखें।"
        )
    return en, hi


def main():
    st.title("AI-Based Landslide Early Warning & Risk Monitoring — India")
    st.caption(
        "Prototype susceptibility model (XGBoost, terrain + rainfall + land cover) "
        "combined with live rainfall to flag high-risk zones. Not a validated "
        "operational warning system — see limitations in the project report."
    )

    try:
        df = load_grid()
    except FileNotFoundError:
        st.error(f"'{GRID_CSV}' not found. Run 03_generate_grid.py first.")
        return

    with st.sidebar:
        st.header("View options")
        show_ner_only = st.checkbox("Zoom to North-East Region (NER)", value=False)
        min_score = st.slider("Show susceptibility above", 0.0, 1.0, 0.0, 0.05)

        st.header("Check a location")
        lat_in = st.number_input("Latitude", value=27.33, format="%.4f")
        lon_in = st.number_input("Longitude", value=88.61, format="%.4f")
        check = st.button("Check alert for this location")

    view_df = df[df["susceptibility"] >= min_score]
    if show_ner_only:
        view_df = view_df[view_df["is_ner"] == 1]
        center, zoom = [26.5, 92.5], 6
    else:
        center, zoom = [22.5, 82.0], 5

    col_map, col_panel = st.columns([2.2, 1])

    with col_map:
       m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
        # keep the map light: sample down if too many points
        plot_df = view_df.sample(min(len(view_df), 6000), random_state=0)
        for _, r in plot_df.iterrows():
            folium.CircleMarker(
                location=[r["lat"], r["lon"]],
                radius=2.2,
                color=susceptibility_color(r["susceptibility"]),
                fill=True,
                fill_opacity=0.7,
                stroke=False,
            ).add_to(m)
        if check:
            folium.Marker(
                [lat_in, lon_in], tooltip="Selected location",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)
        st_folium(m, height=560, use_container_width=True)
        st.caption("Green -> Yellow -> Orange -> Red = increasing landslide susceptibility")

    with col_panel:
        st.subheader("Location check")

        if check:
            pt = nearest_grid_point(df, lat_in, lon_in)
            susceptibility = float(pt["susceptibility"])
            try:
                rain = fetch_rainfall(lat_in, lon_in)
                dates = sorted(rain.keys())
                rain_today = rain[dates[-2]] if len(dates) >= 2 else list(rain.values())[0]
                rain_tomorrow = rain[dates[-1]] if len(dates) >= 2 else None
            except Exception as e:
                st.warning(f"Could not fetch live rainfall: {e}")
                rain_today, rain_tomorrow = 0.0, None

            alert = susceptibility >= HIGH_SUSCEPTIBILITY and rain_today >= RAIN_ALERT_THRESHOLD_MM
            st.session_state["result"] = {
                "lat": lat_in, "lon": lon_in,
                "susceptibility": susceptibility,
                "rain_today": rain_today,
                "rain_tomorrow": rain_tomorrow,
                "alert": alert,
            }

        result = st.session_state.get("result")
        if result:
            st.metric("Susceptibility (nearest grid cell)", f"{result['susceptibility']:.0%}")
            st.metric("Rainfall (last 24h)", f"{result['rain_today']:.0f} mm")
            if result["rain_tomorrow"] is not None:
                st.metric("Forecast (next 24h)", f"{result['rain_tomorrow']:.0f} mm")

            if result["alert"]:
                st.error("⚠️ HIGH RISK ALERT")
            else:
                st.success("No alert")

            en, hi = advisory_text(result["susceptibility"], result["rain_today"], result["alert"])
            st.markdown("**Advisory (English)**")
            st.write(en)
            st.markdown("**सलाह (हिंदी)**")
            st.write(hi)
        else:
            st.info("Enter coordinates in the sidebar and click 'Check alert for this location'.")

        st.divider()
        st.subheader("Grid summary")
        st.write(f"Total hilly grid points: {len(df)}")
        st.write(f"NER grid points: {int(df['is_ner'].sum())}")
        st.write(f"High susceptibility (>= {HIGH_SUSCEPTIBILITY:.0%}): {int((df['susceptibility'] >= HIGH_SUSCEPTIBILITY).sum())}")


if __name__ == "__main__":
    main()
