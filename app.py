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

# ---------------- GOOGLE HOLIDAY API (SAFE) ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HOLIDAY_CALENDAR_ID = "en.indian#holiday@group.v.calendar.google.com"

def get_holidays_api():
    if not GOOGLE_API_KEY:
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
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return {}

        events = r.json().get("items", [])
        return {
            e["start"]["date"]: e["summary"]
            for e in events
            if "date" in e.get("start", {})
        }
    except Exception:
        return {}

# ✅ GLOBAL HOLIDAYS (IMPORTANT)
holidays = get_holidays_api()

# ---------------- JINJA FILTER ----------------
@app.template_filter("format_date")
def format_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%d %B %Y")
    except Exception:
        return value

# ---------------- HELPERS ----------------
def enrich_features(df, user_date, holidays_map):
    df = df.copy()
    travel_date = datetime.strptime(user_date, "%Y-%m-%d").date()
    df["day_of_week"] = travel_date.weekday()
    df["is_holiday"] = 1 if travel_date.strftime("%Y-%m-%d") in holidays_map else 0
    return df

AIRLINE_LOGOS = {
    "Air India": "/static/logos/air-india.png",
    "Indigo": "/static/logos/indigo.png",
    "SpiceJet": "/static/logos/spicejet.png",
    "Vistara": "/static/logos/vistara.png",
    "GO FIRST": "/static/logos/goair.png",
    "AirAsia": "/static/logos/airasia.png",
    "Trujet": "/static/logos/truejet.png",
    "StarAir": "/static/logos/starair.png",
}

airport_lookup = {
    "Mumbai": {"code": "BOM"},
    "Delhi": {"code": "IGI"},
    "Chennai": {"code": "MAA"},
    "Bangalore": {"code": "BLR"},
    "Hyderabad": {"code": "HYD"},
    "Kolkata": {"code": "CCU"},
}

# ---------------- CORE LOGIC ----------------
def recommend_flights(source, destination, flight_class, travel_date, holidays, sort_by="cheap"):
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
        holiday_pred = holiday_model.predict(filtered[features])
        filtered["predicted_price"] += (holiday_pred - filtered["predicted_price"]) * 0.75

    filtered["predicted_price"] = filtered["predicted_price"].round(2)
    filtered["airline_logo"] = filtered["airline"].map(AIRLINE_LOGOS)

    if sort_by == "best":
        filtered = filtered.sort_values(by=["stops", "departure_time", "predicted_price"])
    else:
        filtered = filtered.sort_values(by=["predicted_price", "days_left"])

    return filtered.to_dict(orient="records")

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/destinations")
def destinations():
    return render_template("destinations.html", available_cities=list(airport_lookup.keys()))

@app.route("/searchflight")
def searchflight():
    return render_template("flight-details.html")


@app.route("/suggest-airport")
def suggest_airport():
    query = request.args.get("q", "").lower().strip()
    if not query:
        return jsonify([])

    suggestions = []
    for city, info in airport_lookup.items():
        if city.lower().startswith(query) or info["code"].lower().startswith(query):
            suggestions.append({
                "label": f"{city} ({info['code']})",
                "value": info["code"]
            })
    return jsonify(suggestions)

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

    return render_template(
        "flight-details.html",
        flights=flights,
        source=source,
        destination=destination,
        travel_date=travel_date,
        flight_class=flight_class,
        sort_by=sort_by
    )

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
