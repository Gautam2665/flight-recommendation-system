from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
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

# ---------------- AIRPORT LOOKUP ----------------
airport_lookup = {
    "Mumbai": {"code": "BOM"},
    "Delhi": {"code": "IGI"},
    "Chennai": {"code": "MAA"},
    "Bangalore": {"code": "BLR"},
    "Hyderabad": {"code": "HYD"},
    "Kolkata": {"code": "CCU"},
}

reverse_airport_lookup = {v["code"]: k for k, v in airport_lookup.items()}

# ---------------- SAFE HELPERS ----------------
def clean_param(value):
    """Return None if value is empty/null/undefined"""
    if not value or value.lower() in ["null", "undefined"]:
        return None
    return value

def extract_city(value):
    """Extract city from 'Chennai (MAA)' or return as-is"""
    if not value:
        return None
    if "(" in value:
        return value.split("(")[0].strip()
    return value

# ---------------- GOOGLE HOLIDAY API (SAFE) ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_holidays_api():
    if not GOOGLE_API_KEY:
        return {}

    now = datetime.utcnow().isoformat() + "Z"
    end_date = (datetime.utcnow() + timedelta(days=365)).isoformat() + "Z"

    url = (
        "https://www.googleapis.com/calendar/v3/calendars/"
        "en.indian#holiday@group.v.calendar.google.com/events"
        f"?timeMin={now}&timeMax={end_date}&singleEvents=true&orderBy=startTime&key={GOOGLE_API_KEY}"
    )

    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return {}
        events = r.json().get("items", [])
        return {e["start"]["date"]: e["summary"] for e in events if "date" in e["start"]}
    except Exception:
        return {}

holidays = get_holidays_api()

# ---------------- JINJA FILTER ----------------
@app.template_filter("format_date")
def format_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%d %B %Y")
    except Exception:
        return value

# ---------------- FEATURE ENGINEERING ----------------
def enrich_features(df, travel_date, holidays_map):
    df = df.copy()
    travel_date_obj = datetime.strptime(travel_date, "%Y-%m-%d").date()
    df["day_of_week"] = travel_date_obj.weekday()
    df["is_holiday"] = 1 if travel_date in holidays_map else 0
    return df

# ---------------- CORE ML LOGIC ----------------
def recommend_flights(source, destination, flight_class, travel_date, holidays, sort_by="cheap"):
    if not all([source, destination, flight_class, travel_date]):
        return []

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

    if sort_by == "best":
        filtered = filtered.sort_values(by=["stops", "departure_time", "predicted_price"])
    else:
        filtered = filtered.sort_values(by=["predicted_price", "days_left"])

    return filtered.to_dict(orient="records")

# ---------------- ROUTES ----------------
@app.route("/")
@app.route("/home")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/destinations")
def destinations():
    return render_template("destinations.html")

@app.route("/searchflight")
def searchflight():
    return render_template("flight-details.html")

# ---------------- AUTOCOMPLETE ----------------
@app.route("/suggest-airport")
def suggest_airport():
    q = request.args.get("q", "").lower()
    results = []
    for city, info in airport_lookup.items():
        if city.lower().startswith(q):
            results.append({
                "label": f"{city} ({info['code']})",
                "value": f"{city} ({info['code']})"
            })
    return jsonify(results)

# ---------------- FILTER API (CRASH-PROOF) ----------------
@app.route("/get-filters")
def get_filters():
    source = extract_city(clean_param(request.args.get("source")))
    destination = extract_city(clean_param(request.args.get("destination")))
    flight_class = clean_param(request.args.get("class"))
    travel_date = clean_param(request.args.get("date"))

    if not all([source, destination, flight_class, travel_date]):
        return jsonify({
            "airlines": [],
            "min_price": 0,
            "max_price": 0,
            "stops": [],
            "departure_times": [],
            "arrival_times": []
        })

    flights = recommend_flights(
        source, destination, flight_class, travel_date, holidays
    )

    if not flights:
        return jsonify({
            "airlines": [],
            "min_price": 0,
            "max_price": 0,
            "stops": [],
            "departure_times": [],
            "arrival_times": []
        })

    prices = [f["predicted_price"] for f in flights]

    return jsonify({
        "airlines": sorted(set(f["airline"] for f in flights)),
        "min_price": int(min(prices)),
        "max_price": int(max(prices)),
        "stops": sorted(set(f["stops"] for f in flights)),
        "departure_times": [],
        "arrival_times": []
    })

# ---------------- FLIGHT RESULTS ----------------
@app.route("/flight-details")
def flight_details():
    source = extract_city(request.args.get("source"))
    destination = extract_city(request.args.get("destination"))
    flight_class = request.args.get("class")
    travel_date = request.args.get("date")
    sort_by = request.args.get("sort_by", "cheap")

    flights = recommend_flights(
        source, destination, flight_class, travel_date, holidays, sort_by
    )

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
