import streamlit as st
import requests
import openai
import json
from zoning import get_zoning_info
from layout_utils import plot_layout
from gpt_functions import generate_building_options

# -- Streamlit Page Setup --
st.set_page_config(page_title="TerraIQ - Parcel Analyzer", layout="wide")
st.title("🧠 TerraIQ | Loudoun County AI Parcel Analyzer")

# -- API Key Setup --
openai.api_key = st.secrets["OPENAI_API_KEY"]

# -- Location Input --
st.subheader("1. Locate Your Parcel")
col1, col2 = st.columns([2, 1])

with col1:
    use_location = st.checkbox("Use My Current Location (Browser Permission Required)")
    address_input = st.text_input("Or Enter Parcel Address in Loudoun County")

with col2:
    parcel_size_acres = st.number_input("Parcel Size (acres)", min_value=0.01, value=0.25)

coords = None
if use_location:
    st.info("Using browser geolocation is supported only with custom JS. For now, enter coordinates manually.")
    lat = st.number_input("Latitude", value=39.0851)
    lon = st.number_input("Longitude", value=-77.6454)
    coords = (lat, lon)
elif address_input:
    geocode_url = f"https://nominatim.openstreetmap.org/search?q={address_input},Loudoun+County,VA&format=json&limit=1"
    r = requests.get(geocode_url).json()
    if r:
        coords = (float(r[0]['lat']), float(r[0]['lon']))
        st.success(f"Found location: {coords}")
    else:
        st.error("Could not find the address.")

# -- Fetch Zoning and Analyze --
if coords:
    zoning_info = get_zoning_info(coords[0], coords[1])
    st.subheader("2. Zoning Information")
    st.json(zoning_info)

    st.subheader("3. AI-Powered Concept Plans")
    parcel_size_sqft = round(parcel_size_acres * 43560, 2)
    response = generate_building_options(zoning_info, parcel_size_sqft)

    if response:
        try:
            concept = json.loads(response)
            for option in concept.get("options", []):
                st.markdown(f"### 🏗️ {option['option_name']}")
                st.write(f"- Building Area: {option['building_area_sft']} sqft")
                st.write(f"- Floors: {option['floors']}, Units/Floor: {option['units_per_floor']}")
                st.plotly_chart(plot_layout(option['layout']), use_container_width=True)

            with st.expander("Show Raw JSON Output"):
                st.json(concept)
        except json.JSONDecodeError:
            st.error("GPT returned invalid JSON. Please try again.")
    else:
        st.warning("No concept plans were generated.")

    st.info("This is an AI-generated conceptual analysis. All outputs are based on zoning inputs fetched in real time.")
