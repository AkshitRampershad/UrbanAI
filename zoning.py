import requests

# Loudoun County retired maps.loudoun.gov and moved its GIS REST services to
# logis.loudoun.gov. The zoning polygon layer also moved from
# Public/Zoning/MapServer/7 to COL/Zoning/MapServer/3 ("Zoning", the
# current official zoning map, fields include ZO_ZONE/ZO_ORDINANCE/etc).
ZONING_QUERY_URL = "https://logis.loudoun.gov/gis/rest/services/COL/Zoning/MapServer/3/query"


def get_zoning_info(lat, lon):
    try:
        params = {
            "f": "json",
            "geometryType": "esriGeometryPoint",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "geometry": f"{lon},{lat}",
            "outFields": "*",
        }

        response = requests.get(ZONING_QUERY_URL, params=params, timeout=15)

        if response.status_code != 200:
            return {"error": f"Loudoun API returned status {response.status_code}"}

        try:
            data = response.json()
        except ValueError:
            return {"error": "Invalid JSON from Loudoun County API"}

        if not data.get("features"):
            return {"error": "No zoning info found for this location."}

        return data["features"][0]["attributes"]

    except Exception as e:
        return {"error": str(e)}
