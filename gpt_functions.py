import openai
import json

def generate_building_options(zoning_info, parcel_sqft):
    """
    Use OpenAI GPT to generate high-level building concept options based on zoning and parcel size.
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
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message['content']
    except Exception as e:
        return json.dumps({"error": str(e)})
