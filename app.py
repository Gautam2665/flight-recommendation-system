from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import random
import requests
from datetime import datetime, date, timedelta

# ---------------- APP INIT ----------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOAD MODELS ----------------
base_model = joblib.load(os.path.join(BASE_DIR, "models", "base_model.pkl"))
holiday_model = joblib.load(os.path.join(BASE_DIR, "models", "holiday_model.pkl"))

# ---------------- LOAD DATA ----------------
df = pd.read_csv(os.path.join(BASE_DIR, "data", "Clean_flight_data.csv"))
df["days_left"] = df["days_left"].astype(int)
df["class"] = df["class"].str.capitalize()

# ---------------- GOOGLE HOLIDAY API ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HOLIDAY_CALENDAR_ID = "en.indian#holiday@group.v.calendar.google.com"

def get_holidays_api():
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not set, skipping holidays")
        return {}

    now = datetime.utcnow().isoformat() + "Z"
    end_date = (datetime.utcnow() + timedelta(days=365)).isoformat() + "Z"

    url = (
        f"https://www.googleapis.com/calendar/v3/calendars/"
        f"{HOLIDAY_CALENDAR_ID}/events"
        f"?timeMin={now}&timeMax={end_date}"
        f"&singleEvents=true&orderBy=startTime"
        f"&key={GOOGLE_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        events = response.json().get("items", [])
        return {
            event["start"]["date"]: event["summary"]
            for event in events
            if "date" in event["start"]
        }
    except Exception as e:
        print("Holiday API error:", e)
        return {}

holidays = get_holidays_api()

# ---------------- HELPERS ----------------
def enrich_features(df, user_date=None, holidays_map=None):
    df = df.copy()
    if user_date:
        travel_date_obj = datetime.strptime(user_date, "%Y-%m-%d").date()
        df["day_of_week"] = travel_date_obj.weekday()
        df["is_holiday"] = 1 if travel_date_obj.strftime("%Y-%m-%d") in holidays_map else 0
    else:
        df["day_of_week"] = 0
        df["is_holiday"] = 0
    return df

AIRLINE_LOGOS = {
    "Air India": "/static/logos/air-india.png",
    "Indigo": "/static/logos/indigo.png",
    "SpiceJet": "/static/logos/spicejet.png",
    "Vistara": "/static/logos/vistara.png",
    "GO FIRST": "/static/logos/goair.png",
    "AirAsia": "/static/logos/airasia.png",
    "Trujet": "/static/logos/truejet.png",
    "StarAir": "/static/logos/starair.png"
}

airport_lookup = {
    "Mumbai": {"code": "BOM", "name": "Chhatrapati Shivaji Maharaj International Airport"},
    "Delhi": {"code": "IGI", "name": "Indira Gandhi International Airport"},
    "Chennai": {"code": "MAA", "name": "Chennai International Airport"},
    "Bangalore": {"code": "BLR", "name": "Kempegowda International Airport"},
    "Hyderabad": {"code": "HYD", "name": "Rajiv Gandhi International Airport"},
    "Kolkata": {"code": "CCU", "name": "Netaji Subhash Chandra Bose International Airport"},
}

TIME_SLOT_LABELS = {
    "early_morning": "Early Morning (3AM - 6AM)",
    "morning": "Morning (6AM - 12PM)",
    "afternoon": "Afternoon (12PM - 6PM)",
    "evening": "Evening (6PM - 8PM)",
    "late_night": "Late Night (12AM - 3AM)",
    "night": "Night (8PM - 12AM)"
}

def get_time_slot(time_str):
    if not time_str:
        return "unknown"
    if ":" in time_str:
        try:
            hour = int(time_str.split(":")[0])
            if 0 <= hour < 6:
                return "early_morning"
            elif 6 <= hour < 12:
                return "morning"
            elif 12 <= hour < 18:
                return "afternoon"
            elif 18 <= hour <= 23:
                return "evening"
        except:
            pass
    return "unknown"

# ---------------- CORE LOGIC ----------------
def recommend_flights(source, destination, flight_class, travel_date, holidays, sort_by="cheap", top_n=200):
    today = date.today()
    travel_date_obj = datetime.strptime(travel_date, "%Y-%m-%d").date()
    days_left = (travel_date_obj - today).days

    filtered = df[
        (df["source_city"].str.lower() == source.lower()) &
        (df["destination_city"].str.lower() == destination.lower()) &
        (df["class"].str.lower() == flight_class.lower()) &
        (df["days_left"] == days_left)
    ].copy()

    if filtered.empty:
        return []

    filtered = enrich_features(filtered, travel_date, holidays)

    features = [
        "source_city", "destination_city", "airline",
        "departure_time", "arrival_time", "stops",
        "class", "days_left", "day_of_week", "is_holiday"
    ]

    filtered["predicted_price"] = base_model.predict(filtered[features])

    if filtered["is_holiday"].iloc[0] == 1:
        holiday_adjustment = holiday_model.predict(filtered[features])
        filtered["predicted_price"] += (holiday_adjustment - filtered["predicted_price"]) * 0.75

    filtered["predicted_price"] = filtered["predicted_price"].round(2)
    filtered["airline_logo"] = filtered["airline"].map(AIRLINE_LOGOS)

    if sort_by == "best":
        filtered = filtered.sort_values(by=["stops", "departure_time", "predicted_price"])
    else:
        filtered = filtered.sort_values(by=["predicted_price", "days_left"])

    return filtered.head(top_n).to_dict(orient="records")

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/flight-prices")
def flight_prices():
    source = request.args.get("source")
    destination = request.args.get("destination")
    flight_class = request.args.get("class")
    travel_date = request.args.get("date")

    flights = recommend_flights(source, destination, flight_class, travel_date, holidays)
    return jsonify(flights)

@app.route("/flight-details")
def flight_details():
    source = request.args.get("source")
    destination = request.args.get("destination")
    flight_class = request.args.get("class")
    travel_date = request.args.get("date")
    sort_by = request.args.get("sort_by", "cheap")

    flights = recommend_flights(source, destination, flight_class, travel_date, holidays, sort_by)
    return render_template("flight-details.html", flights=flights)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)