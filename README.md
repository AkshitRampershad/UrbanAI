# UrbanAI — TerraIQ Parcel Analyzer

An AI agent for urban development and planning. TerraIQ looks up a parcel's real zoning constraints in Loudoun County, VA, then uses an LLM to propose concept building plans that maximize investment potential while staying within those constraints — visualized as an interactive building footprint.

## How It Works
1. **Locate the parcel** — enter an address in Loudoun County (geocoded via [Nominatim/OpenStreetMap](https://nominatim.openstreetmap.org/)), or provide latitude/longitude directly.
2. **Fetch zoning data** — the app queries [Loudoun County's public zoning ArcGIS service](https://maps.loudoun.gov/) for the parcel's zoning attributes in real time.
3. **Generate concept plans** — the zoning info and parcel size (in acres) are sent to an LLM (Groq-hosted Mixtral 8x7B) with a prompt asking for 2–3 building concepts that maximize investment potential without violating the zoning rules, each with a floor count, units per floor, and a structured footprint.
4. **Visualize the layout** — each concept's footprint (and stairs, if returned) is rendered as an interactive Plotly building outline, with raw JSON available for inspection.

## Features
- Address or coordinate-based parcel lookup for Loudoun County, VA
- Live zoning lookup against the county's public GIS API
- AI-generated concept building plans constrained by real zoning data
- Interactive Plotly visualization of proposed building footprints
- Runs as a single-page Streamlit app, ready for GitHub Codespaces

## Tech Stack
- **[Streamlit](https://streamlit.io/)** — UI and app framework
- **[Groq API](https://groq.com/)** (Mixtral 8x7B) — concept plan generation
- **[Plotly](https://plotly.com/python/)** — building layout visualization
- **Loudoun County ArcGIS REST API** — zoning data source
- **Nominatim (OpenStreetMap)** — address geocoding

## Project Structure
| File | Purpose |
| --- | --- |
| `app.py` | Streamlit UI — parcel input, orchestrates zoning lookup, concept generation, and rendering |
| `zoning.py` | Queries Loudoun County's zoning ArcGIS endpoint for a given lat/lon |
| `gpt_functions.py` | Calls the Groq API to generate zoning-constrained building concepts |
| `layout_utils.py` | Renders a concept's footprint/stairs as a Plotly figure |
| `requirements.txt` | Python dependencies |
| `.devcontainer/` | GitHub Codespaces config — auto-installs dependencies and launches the app |

## Getting Started

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com/keys)

### Setup
```bash
git clone https://github.com/AkshitRampershad/UrbanAI.git
cd UrbanAI
pip install -r requirements.txt
```

Add your Groq API key as a Streamlit secret:
```bash
mkdir -p .streamlit
echo 'GROQ_API_KEY = "your-groq-api-key"' > .streamlit/secrets.toml
```

Run the app:
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Using GitHub Codespaces
This repo includes a `.devcontainer` configuration — opening it in a Codespace installs dependencies and launches Streamlit on port 8501 automatically. You'll still need to add `GROQ_API_KEY` as a Codespaces secret before the concept-generation step will work.

## Scope & Limitations
- Zoning lookups are specific to **Loudoun County, VA** — the ArcGIS endpoint and geocoding query are hardcoded to that jurisdiction.
- Concept plans are AI-generated from zoning metadata and are meant as a starting point for exploration, not a substitute for a licensed architect, surveyor, or zoning attorney.
- Browser geolocation is not yet wired up; coordinates must currently be entered manually or resolved from an address.
