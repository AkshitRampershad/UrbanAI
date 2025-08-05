import openai
import json
from openai import OpenAI

# Initialize OpenAI client using your secret key from streamlit
client = OpenAI(api_key=openai.api_key)

def generate_building_options(zoning_info, parcel_sqft):
    """
    Use OpenAI GPT-4o to generate concept building plans based on zoning and parcel size.
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

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful urban planning assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )

        return response.choices[0].message.content
    except Exception as e:
        return json.dumps({"error": str(e)})
