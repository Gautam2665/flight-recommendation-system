from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import random
from datetime import datetime
from datetime import datetime, date, timedelta
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

app = Flask(__name__)


base_model = joblib.load("base_model.pkl")
holiday_model = joblib.load("holiday_model.pkl")

# Google Calendar API Setup
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
HOLIDAY_CALENDAR_ID = "en.indian#holiday@group.v.calendar.google.com"

df = pd.read_csv("Clean_flight_data.csv")
df["days_left"] = df["days_left"].astype(int)
df["class"] = df["class"].str.capitalize()


def get_holidays():
    """Fetch upcoming public holidays from Google Calendar."""
    creds = None
    token_file = "token.json"

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)

        with open(token_file, "w") as token:
            token.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds)
    now = datetime.now().isoformat() + "Z"
    end_date = (datetime.now() + timedelta(days=365)).isoformat() + "Z"

    events_result = service.events().list(
        calendarId=HOLIDAY_CALENDAR_ID,
        timeMin=now,
        timeMax=end_date,
        maxResults=50,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    return {event["start"]["date"]: event["summary"] for event in events_result.get("items", [])}

holidays = get_holidays()  

def enrich_features(df, user_date=None, holidays_map=None):
    df = df.copy()
    if user_date:
        
        travel_date_obj = datetime.strptime(user_date, "%Y-%m-%d").date()
        day_of_week = travel_date_obj.weekday()
        is_holiday = 1 if travel_date_obj.strftime("%Y-%m-%d") in holidays_map else 0
        df["day_of_week"] = day_of_week
        df["is_holiday"] = is_holiday
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
    "late_night": "Late Night (12PM-3AM)",
    "night": "Night (8PM - 12PM)"
}

def get_time_slot(time_str):
    if not time_str:
        return "unknown"

    time_str = time_str.strip()

    
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
            else:
                return "unknown"
        except Exception:
            pass

    
    normalized = time_str.lower().replace(" ", "_")
    valid_labels = {"early_morning", "morning", "afternoon", "evening", "night", "late_night"}
    return normalized if normalized in valid_labels else "unknown"


def recommend_flights(source, destination, flight_class, travel_date, holidays,sort_by="cheap", top_n=200):
    """Recommend flights with ML-based holiday price adjustment based on user input."""

    today_date = date.today()

    try:
        travel_date_obj = datetime.strptime(travel_date, "%Y-%m-%d").date()
    except ValueError:
        print("Invalid date format received:", travel_date)
        return []  # Invalid date

    days_left = (travel_date_obj - today_date).days
    print(f"Filtering flights for: {source} -> {destination}, Class: {flight_class}, Travel Date: {travel_date} ({days_left} days left) , Filter:{sort_by}")

    df["days_left"] = df["days_left"].astype(int)
    source = source.lower().strip().split("(")[0]  
    destination = destination.lower().strip().split("(")[0]

    
    filtered_flights = df[
        (df["source_city"].str.lower() == source.lower()) &
        (df["destination_city"].str.lower() == destination.lower()) &
        (df["class"].str.lower() == flight_class.lower()) &
        (df["days_left"] == days_left)
    ].copy()

    if filtered_flights.empty:
        print("No matching flights found.")
        return []

    
    filtered_flights = enrich_features(filtered_flights, user_date=travel_date, holidays_map=holidays)

    features = [
        "source_city", "destination_city", "airline", "departure_time",
        "arrival_time", "stops", "class", "days_left", "day_of_week", "is_holiday"
    ]

    # Predict the base price
    filtered_flights["predicted_price"] = base_model.predict(filtered_flights[features])

    
    if filtered_flights["is_holiday"].iloc[0] == 1:
        holiday_price_adjustment = holiday_model.predict(filtered_flights[features])
        filtered_flights["predicted_price"] = (
            filtered_flights["predicted_price"] +
            (holiday_price_adjustment - filtered_flights["predicted_price"]) * 0.75
        )
    filtered_flights["holiday"] = filtered_flights["is_holiday"].map(
    lambda x: "Holiday Pricing" if x == 1 else "No Holiday"
    )
    
    filtered_flights["airline_logo"] = filtered_flights["airline"].map(AIRLINE_LOGOS)
    filtered_flights["source_code"] = filtered_flights["source_city"].map(lambda x: airport_lookup.get(x, {}).get("code", ""))
    filtered_flights["destination_code"] = filtered_flights["destination_city"].map(lambda x: airport_lookup.get(x, {}).get("code", ""))
    filtered_flights["source_airport"] = filtered_flights["source_city"].map(lambda x: airport_lookup.get(x, {}).get("name", ""))
    filtered_flights["destination_airport"] = filtered_flights["destination_city"].map(lambda x: airport_lookup.get(x, {}).get("name", ""))
    filtered_flights["predicted_price"] = filtered_flights["predicted_price"].round(2)

    
    filtered_flights["aircraft"] = filtered_flights["airline"].map(
        lambda airline: random.choice(["A321", "A350", "B787"]) if airline in ["Vistara", "Air India"]
        else "B737" if airline == "SpiceJet"
        else "A320"
    )

    
    filtered_flights["depart_terminal"] = filtered_flights["source_city"].map(
    lambda city: random.choice(["T1", "T2", "T3"]) if city == "Delhi"
    else random.choice(["T1", "T2"]) if city in ["Mumbai", "Bangalore"]
    else "T1"
    )
    
    
    filtered_flights["arrival_terminal"] = filtered_flights["destination_city"].map(
        lambda city: random.choice(["T1", "T2", "T3"]) if city == "Delhi"
        else random.choice(["T1", "T2"]) if city in ["Mumbai", "Bangalore"]
        else "T1"
    )

    # Meal and baggage info
    filtered_flights["meals"] = filtered_flights["airline"].map(
        lambda airline: "Complimentary Meals" if airline in ["Vistara", "Air India"] else "Buy Meals"
    )

    filtered_flights["usb"] = filtered_flights["airline"].map(
    lambda airline: "Yes" if airline in ["Vistara", "Air India", "Indigo"] else "No"
    )

    # Beverages availability
    filtered_flights["beverages"] = filtered_flights["airline"].map(
        lambda airline: "Complimentary Beverages" if airline in ["Vistara", "Air India"] else "Buy Onboard Beverages"
    )

    filtered_flights["baggage"] = filtered_flights["class"].map(
        lambda c: "20kg Check-in + 7kg Cabin" if c == "Economy" else "30kg Check-in + 10kg Cabin"
    )
    # Sort AFTER assigning all extra display info
    if sort_by == "best":
        sorted_flights = filtered_flights.sort_values(
            by=["stops", "departure_time", "predicted_price"]
        )
    else:  # Default: sort by cheapest
        sorted_flights = filtered_flights.sort_values(
            by=["predicted_price", "days_left"]
        )

    return sorted_flights.head(top_n).to_dict(orient="records")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/searchflight")
def searchflight():
    return render_template("flight-details.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.template_filter('format_date')
def format_date(value):
    try:
        date_obj = datetime.strptime(value, "%Y-%m-%d")
        return date_obj.strftime("%d %B")
    except:
        return value  # fallback if date parsing fails


@app.route("/flight-prices", methods=["GET"])
def flight_prices():
    source_code = request.args.get("source", "").split("(")[0].strip()
    destination_code = request.args.get("destination", "").split("(")[0].strip()
    flight_class = request.args.get("class")
    travel_date = request.args.get("date")

    source_city = next((city for city, info in airport_lookup.items() if info['code'] == source_code), source_code)
    destination_city = next((city for city, info in airport_lookup.items() if info['code'] == destination_code), destination_code)

    flights = recommend_flights(source_city, destination_city, flight_class, travel_date, holidays, sort_by="cheap")

    return jsonify(flights)  


@app.route("/suggest-airport", methods=["GET"])
def suggest_airport():
    query = request.args.get('q', '').lower().strip()
    if not query:
        return jsonify([])  # Return empty list if no query

    # Filter suggestions to match cities in airport_lookup
    suggestions = []
    for city, info in airport_lookup.items():
        if city.lower().startswith(query) or info['code'].lower().startswith(query):  # Only return suggestions that match the query
            suggestions.append({
                "label": f"{city.title()} ({info['code']})",  # Example: Mumbai (BOM)
                "value": city  # You can use this to reference the city internally
            })

    return jsonify(suggestions)

@app.route("/flight-details", methods=["GET"])
def flight_details():
    source_code = request.args.get("source", "").split("(")[0].strip()
    destination_code = request.args.get("destination", "").split("(")[0].strip()
    flight_class = request.args.get("class")
    travel_date = request.args.get("date")
    filter_type = request.args.get("sort_by", "cheap")

    # Reverse lookup city names from codes
    source_city = next((city for city, info in airport_lookup.items() if info['code'] == source_code), source_code)
    destination_city = next((city for city, info in airport_lookup.items() if info['code'] == destination_code), destination_code)

    # Get all flights
    flights = recommend_flights(source_city, destination_city, flight_class, travel_date, holidays, sort_by=filter_type)

    # Add baggage info
    for flight in flights:
        flight['baggage'] = "20kg Check-in + 7kg Cabin" if flight['class'] == "Economy" else "30kg Check-in + 10kg Cabin"

    # Apply airline filter
    selected_airline = request.args.get("airline")
    if selected_airline:
        selected_airlines = [a.strip() for a in selected_airline.split(",")]
        flights = [f for f in flights if f["airline"] in selected_airlines]

    # Apply stops filter
    selected_stops = request.args.getlist("stops")
    if selected_stops:
        flights = [f for f in flights if str(f["stops"]) in selected_stops]

    # Apply max price filter
    max_price = request.args.get("max_price")
    if max_price:
        try:
            max_price_val = int(max_price)
            flights = [f for f in flights if f["predicted_price"] <= max_price_val]
        except ValueError:
            pass

    # Apply departure time filter using get_time_slot
    selected_dep_times = request.args.get("departure_time")
    if selected_dep_times:
        allowed_slots = selected_dep_times.split(",")
        flights = [f for f in flights if get_time_slot(f["departure_time"]) in allowed_slots]

    # Apply arrival time filter using get_time_slot
    selected_arr_times = request.args.get("arrival_time")
    if selected_arr_times:
        allowed_slots = selected_arr_times.split(",")
        flights = [f for f in flights if get_time_slot(f["arrival_time"]) in allowed_slots]
   
    travellers = int(request.args.get("travellers", 1))
    print(f"Flights found: {len(flights)}")  # This will now show filtered count


    return render_template(
        "flight-details.html",
        flights=flights,
        source=source_city,
        destination=destination_city,
        travel_date=travel_date,
        flight_class=flight_class,
        sort_by=filter_type,
        travellers=travellers
    )

@app.route("/destinations")
def destinations():
    # Get a list of all cities from the airport_lookup
    available_cities = list(airport_lookup.keys())
    
    # Render the HTML template and pass the available cities to it
    return render_template("destinations.html", available_cities=available_cities)



@app.route("/get-filters")
def get_filters():
    source_code = request.args.get("source", "").split("(")[0].strip()
    destination_code = request.args.get("destination", "").split("(")[0].strip()
    flight_class = request.args.get("class")
    travel_date = request.args.get("date")
    filter_type = request.args.get("sort_by", "cheap")

    if not travel_date:
        return jsonify({"error": "Missing travel date"}), 400

    source_city = next((city for city, info in airport_lookup.items() if info['code'] == source_code), source_code)
    destination_city = next((city for city, info in airport_lookup.items() if info['code'] == destination_code), destination_code)

    flights = recommend_flights(source_city, destination_city, flight_class, travel_date, holidays, sort_by=filter_type)

    airlines = sorted(set(f["airline"] for f in flights))
    prices = [f["predicted_price"] for f in flights]
    min_price = int(min(prices)) if prices else 0
    max_price = int(max(prices)) if prices else 0

    stop_prices = {}
    for f in flights:
        stops = f.get("stops", "Unknown")
        stop_prices.setdefault(stops, []).append(f["predicted_price"])
    stops = [
        {
            "label": "Non Stop" if stop == "0" else f"{stop} Stop",
            "value": stop,
            "min_price": int(min(stop_prices[stop]))
        }
        for stop in stop_prices
    ]

    # Departure time slots
    dep_slot_set = set()
    for f in flights:
        slot = get_time_slot(f.get("departure_time", ""))
        if slot != "unknown":
            dep_slot_set.add(slot)
    departure_times = [
        {"label": TIME_SLOT_LABELS[slot], "value": slot}
        for slot in sorted(dep_slot_set)
    ]

    # Arrival time slots
    arr_slot_set = set()
    for f in flights:
        slot = get_time_slot(f.get("arrival_time", ""))
        if slot != "unknown":
            arr_slot_set.add(slot)
    arrival_times = [
        {"label": TIME_SLOT_LABELS[slot], "value": slot}
        for slot in sorted(arr_slot_set)
    ]

    return jsonify({
        "airlines": airlines,
        "min_price": min_price,
        "max_price": max_price,
        "stops": stops,
        "departure_times": departure_times,
        "arrival_times": arrival_times
    })



if __name__ == "__main__":
    app.run(debug=True)