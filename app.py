import streamlit as st
import requests
import json
import plotly.graph_objects as go
from zoning import get_zoning_info
from layout_utils import plot_layout
from gpt_functions import generate_building_options

# -- Streamlit Page Setup --
st.set_page_config(page_title="TerraIQ - Parcel Analyzer", layout="wide")
st.title("🧠 TerraIQ | Loudoun County AI Parcel Analyzer")

# Streamlit reruns this whole script on every widget interaction anywhere
# on the page - not just ones related to geocoding/zoning/concepts. Without
# caching, adjusting an unrelated input (e.g. the acres spinner) would
# re-fire a live Nominatim/zoning/Groq call every time, quickly tripping
# Nominatim's strict 1 req/sec rate limit and burning Groq API quota for
# no reason. Caching by input keeps each external call to once per unique
# address/coordinate/zoning combination.


@st.cache_data(ttl=1800, show_spinner="Geocoding address...")
def geocode_address(address: str):
    response = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": f"{address}, Loudoun County, VA", "format": "json", "limit": 1},
        # Nominatim's usage policy requires a descriptive User-Agent;
        # requests without one are frequently rejected with a non-JSON
        # response, which crashes a bare .json() call.
        headers={"User-Agent": "TerraIQ-UrbanAI/1.0 (https://github.com/AkshitRampershad/UrbanAI)"},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=1800, show_spinner="Looking up zoning...")
def cached_zoning_info(lat: float, lon: float):
    return get_zoning_info(lat, lon)


@st.cache_data(ttl=1800, show_spinner="Generating concept plans...")
def cached_concept_options(zoning_info: dict, parcel_size_sqft: float):
    return generate_building_options(zoning_info, parcel_size_sqft)

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
    try:
        r = geocode_address(address_input)
    except requests.RequestException as e:
        st.error(f"Geocoding request failed: {e}")
        r = None
    except ValueError:
        st.error("Geocoding service returned an unexpected (non-JSON) response. Please try again.")
        r = None

    if r:
        coords = (float(r[0]['lat']), float(r[0]['lon']))
        st.success(f"Found location: {coords}")
    elif r is not None:
        st.error("Could not find the address.")

# -- Fetch Zoning and Analyze --
if coords:
    zoning_info = cached_zoning_info(coords[0], coords[1])
    st.subheader("2. Zoning Information")
    if "error" in zoning_info:
        st.error(f"Zoning API Error: {zoning_info['error']}")
    else:
        st.json(zoning_info)

        st.subheader("3. AI-Powered Concept Plans")
        parcel_size_sqft = round(parcel_size_acres * 43560, 2)
        concept_response = cached_concept_options(zoning_info, parcel_size_sqft)

        try:
            concept = json.loads(concept_response)
        except json.JSONDecodeError:
            st.error("Failed to parse concept plans response.")
            concept = {}

        if "error" in concept:
            st.error(f"Concept Plan Error: {concept['error']}")
        else:
            for option in concept.get("options", []):
                st.markdown(f"### 🏗️ {option['option_name']}")
                st.write(f"- Building Area: {option['building_area_sft']} sqft")
                st.write(f"- Floors: {option['floors']}, Units/Floor: {option['units_per_floor']}")
                st.plotly_chart(plot_layout(option['layout']), use_container_width=True)

            with st.expander("Show Raw JSON Output"):
                st.json(concept)

    st.info("This is an AI-generated conceptual analysis. All outputs are based on zoning inputs fetched in real time.")
