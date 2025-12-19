from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import requests
import random
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
    "Mumbai": {"code": "BOM", "name": "Chhatrapati Shivaji Maharaj International Airport"},
    "Delhi": {"code": "IGI", "name": "Indira Gandhi International Airport"},
    "Chennai": {"code": "MAA", "name": "Chennai International Airport"},
    "Bangalore": {"code": "BLR", "name": "Kempegowda International Airport"},
    "Hyderabad": {"code": "HYD", "name": "Rajiv Gandhi International Airport"},
    "Kolkata": {"code": "CCU", "name": "Netaji Subhas Chandra Bose International Airport"},
}

# ---------------- AIRLINE LOGOS ----------------
AIRLINE_LOGOS = {
    "Air India": "/static/logos/air-india.png",
    "IndiGo": "/static/logos/indigo.png",
    "Vistara": "/static/logos/vistara.png",
    "SpiceJet": "/static/logos/spicejet.png",
    "AirAsia": "/static/logos/airasia.png",
    "Go First": "/static/logos/gofirst.png",
}

def get_airline_logo(airline):
    return AIRLINE_LOGOS.get(airline, "/static/logos/default.png")

# ---------------- TIME SLOTS ----------------
TIME_SLOT_LABELS = {
    "early_morning": "Early Morning (3AM - 6AM)",
    "morning": "Morning (6AM - 12PM)",
    "afternoon": "Afternoon (12PM - 6PM)",
    "evening": "Evening (6PM - 8PM)",
    "night": "Night (8PM - 12AM)",
    "late_night": "Late Night (12AM - 3AM)",
}

def get_time_slot(time_str):
    if not time_str or ":" not in time_str:
        return "unknown"
    try:
        hour = int(time_str.split(":")[0])
        if 0 <= hour < 6: return "early_morning"
        if 6 <= hour < 12: return "morning"
        if 12 <= hour < 18: return "afternoon"
        if 18 <= hour < 20: return "evening"
        if 20 <= hour <= 23: return "night"
    except:
        pass
    return "unknown"

# ---------------- SAFE HELPERS ----------------
def clean_param(value):
    if not value or value.lower() in ["null", "undefined"]:
        return None
    return value

def extract_city(value):
    if not value:
        return None
    if "(" in value:
        return value.split("(")[0].strip()
    return value

# ---------------- GOOGLE HOLIDAYS (SAFE) ----------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_holidays_api():
    if not GOOGLE_API_KEY:
        return {}
    try:
        now = datetime.utcnow().isoformat() + "Z"
        end = (datetime.utcnow() + timedelta(days=365)).isoformat() + "Z"
        url = (
            "https://www.googleapis.com/calendar/v3/calendars/"
            "en.indian#holiday@group.v.calendar.google.com/events"
            f"?timeMin={now}&timeMax={end}&singleEvents=true&orderBy=startTime&key={GOOGLE_API_KEY}"
        )
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return {}
        events = r.json().get("items", [])
        return {e["start"]["date"]: e["summary"] for e in events if "date" in e["start"]}
    except:
        return {}

holidays = get_holidays_api()

# ---------------- JINJA FILTER ----------------
@app.template_filter("format_date")
def format_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%d %B %Y")
    except:
        return value

# ---------------- FEATURE ENGINEERING ----------------
def enrich_features(df, travel_date):
    df = df.copy()
    d = datetime.strptime(travel_date, "%Y-%m-%d").date()
    df["day_of_week"] = d.weekday()
    df["is_holiday"] = 1 if travel_date in holidays else 0
    return df

# ---------------- CORE ML ENGINE ----------------
def recommend_flights(source, destination, flight_class, travel_date, sort_by="cheap"):
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

    filtered = enrich_features(filtered, travel_date)

    features = [
        "source_city", "destination_city", "airline",
        "departure_time", "arrival_time", "stops",
        "class", "days_left", "day_of_week", "is_holiday"
    ]

    filtered["predicted_price"] = base_model.predict(filtered[features])

    if filtered["is_holiday"].iloc[0] == 1:
        holiday_pred = holiday_model.predict(filtered[features])
        filtered["predicted_price"] += (holiday_pred - filtered["predicted_price"]) * 0.75

    # ---------- DISPLAY ENRICHMENT ----------
    filtered["airline_logo"] = filtered["airline"].apply(get_airline_logo)
    filtered["source_code"] = filtered["source_city"].map(lambda c: airport_lookup[c]["code"])
    filtered["destination_code"] = filtered["destination_city"].map(lambda c: airport_lookup[c]["code"])
    filtered["source_airport"] = filtered["source_city"].map(lambda c: airport_lookup[c]["name"])
    filtered["destination_airport"] = filtered["destination_city"].map(lambda c: airport_lookup[c]["name"])

    filtered["aircraft"] = filtered["airline"].apply(
        lambda a: random.choice(["A350", "B787"]) if a in ["Vistara", "Air India"] else "A320"
    )

    filtered["depart_terminal"] = "T1"
    filtered["arrival_terminal"] = "T1"

    filtered["meals"] = filtered["airline"].apply(
        lambda a: "Complimentary Meals" if a in ["Vistara", "Air India"] else "Buy Onboard Meals"
    )

    filtered["usb"] = filtered["airline"].apply(
        lambda a: "Yes" if a in ["Vistara", "Air India", "IndiGo"] else "No"
    )

    filtered["beverages"] = filtered["airline"].apply(
        lambda a: "Complimentary Beverages" if a in ["Vistara", "Air India"] else "Buy Onboard Beverages"
    )

    filtered["baggage"] = filtered["class"].apply(
        lambda c: "20kg Check-in + 7kg Cabin" if c == "Economy" else "30kg Check-in + 10kg Cabin"
    )

    # ---------- SORT ----------
    if sort_by == "best":
        filtered = filtered.sort_values(by=["stops", "departure_time", "predicted_price"])
    else:
        filtered = filtered.sort_values(by=["predicted_price"])

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
    return jsonify([
        {"label": f"{city} ({info['code']})", "value": f"{city} ({info['code']})"}
        for city, info in airport_lookup.items()
        if city.lower().startswith(q)
    ])

# ---------------- FILTER API ----------------
@app.route("/get-filters")
def get_filters():
    source = extract_city(clean_param(request.args.get("source")))
    destination = extract_city(clean_param(request.args.get("destination")))
    flight_class = clean_param(request.args.get("class"))
    travel_date = clean_param(request.args.get("date"))

    flights = recommend_flights(source, destination, flight_class, travel_date)

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

    stop_prices = {}
    for f in flights:
        stop_prices.setdefault(f["stops"], []).append(f["predicted_price"])

    return jsonify({
        "airlines": sorted(set(f["airline"] for f in flights)),
        "min_price": int(min(prices)),
        "max_price": int(max(prices)),
        "stops": [
            {
                "label": "Non Stop" if s == 0 else f"{s} Stop",
                "value": s,
                "min_price": int(min(p))
            }
            for s, p in stop_prices.items()
        ],
        "departure_times": [
            {"label": TIME_SLOT_LABELS[t], "value": t}
            for t in sorted({get_time_slot(f["departure_time"]) for f in flights})
            if t != "unknown"
        ],
        "arrival_times": [
            {"label": TIME_SLOT_LABELS[t], "value": t}
            for t in sorted({get_time_slot(f["arrival_time"]) for f in flights})
            if t != "unknown"
        ]
    })

# ---------------- FLIGHT DETAILS ----------------
@app.route("/flight-details")
def flight_details():
    source = extract_city(request.args.get("source"))
    destination = extract_city(request.args.get("destination"))
    flight_class = request.args.get("class")
    travel_date = request.args.get("date")
    sort_by = request.args.get("sort_by", "cheap")
    travellers = int(request.args.get("travellers", 1))

    flights = recommend_flights(source, destination, flight_class, travel_date, sort_by)

    return render_template(
        "flight-details.html",
        flights=flights,
        source=source,
        destination=destination,
        travel_date=travel_date,
        flight_class=flight_class,
        sort_by=sort_by,
        travellers=travellers
    )

# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
