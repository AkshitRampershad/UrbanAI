import requests

def get_zoning_info(lat, lon):
    """
    Query Loudoun County's Zoning API (ArcGIS REST) for zoning info at the given coordinates.
    Returns basic zoning and overlay details.
    """
    base_url = "https://logis.loudoun.gov/arcgis/rest/services/COL/Zoning/MapServer/3/query"
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "outFields": "*",
        "returnGeometry": "false",
        "f": "json"
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if data.get("features"):
            attr = data["features"][0]["attributes"]
            return {
                "Zoning_District": attr.get("ZONING"),
                "Overlay_District": attr.get("OVERLAY"),
                "Label": attr.get("LABEL"),
                "Notes": attr.get("NOTES")
            }
        else:
            return {"error": "No zoning data found for this location."}

    except Exception as e:
        return {"error": str(e)}
