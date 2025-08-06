import requests
import json

GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "mixtral-8x7b-32768"

def generate_building_options(zoning_info, parcel_sqft):
    """
    Use Groq + Mixtral to generate building plan options.
    """
    prompt = f"""
    You're an expert urban planner and architect. A user has a parcel of land with the following zoning constraints:

    Zoning Info: {json.dumps(zoning_info)}
    Parcel Size: {parcel_sqft} square feet

    Propose 2-3 concept building plans that maximize investment potential while staying within zoning regulations.
    For each concept, return:
    - option_name
    - building_area_sft
    - floors
    - units_per_floor
    - layout (structured JSON with keys: footprint [[x,y],...], stairs [[x,y],...])

    Respond with a JSON object named "options" that holds a list of concept options.
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful urban planning assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=body)
        try:
            result = response.json()
        except json.JSONDecodeError:
            return json.dumps({"error": "Groq API returned invalid JSON"})

        if "choices" not in result:
            return json.dumps({"error": "Groq API did not return 'choices'. Response: " + str(result)})

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return json.dumps({"error": str(e)})
