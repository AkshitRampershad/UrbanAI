import requests

def get_zoning_info(lat, lon):
    try:
        url = f"https://maps.loudoun.gov/arcgis/rest/services/Public/Zoning/MapServer/7/query?f=json&geometryType=esriGeometryPoint&inSR=4326&spatialRel=esriSpatialRelIntersects&geometry={lon},{lat}&outFields=*"

        response = requests.get(url)

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
