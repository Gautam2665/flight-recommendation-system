from flask import Flask, render_template, request, jsonify
from datetime import datetime, date
import random

app = Flask(__name__)

# ======================================================
# AIRLINE LOGOS (FIXES 404 ISSUE)
# ======================================================

AIRLINE_LOGOS = {
    "Air India": "/static/logos/air-india.png",
    "Indigo": "/static/logos/indigo.png",
    "SpiceJet": "/static/logos/spicejet.png",
    "Vistara": "/static/logos/vistara.png",
    "GO FIRST": "/static/logos/goair.png",
    "AirAsia": "/static/logos/airasia.png",
    "Trujet": "/static/logos/truejet.png"
}

DEFAULT_LOGO = "/static/logos/default.png"

# ======================================================
# TIME SLOT LOGIC (FIXES FILTERS + JS CRASH)
# ======================================================

TIME_SLOT_LABELS = {
    "early_morning": "Early Morning (3AM - 6AM)",
    "morning": "Morning (6AM - 12PM)",
    "afternoon": "Afternoon (12PM - 6PM)",
    "evening": "Evening (6PM - 8PM)",
    "night": "Night (8PM - 12AM)",
    "late_night": "Late Night (12AM - 3AM)"
}

def get_time_slot(time_str):
    if not time_str:
        return "unknown"

    try:
        hour = int(time_str.split(":")[0])
        if 3 <= hour < 6:
            return "early_morning"
        elif 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 20:
            return "evening"
        elif 20 <= hour <= 23:
            return "night"
        elif 0 <= hour < 3:
            return "late_night"
    except:
        pass

    return "unknown"

# ======================================================
# SAFE DATE PARSER
# ======================================================

def safe_date(date_str):
    if not date_str or date_str == "null":
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except:
        return None

# ======================================================
# FLIGHT GENERATOR (REPLACES ML SAFELY)
# ======================================================

def recommend_flights(source, destination, flight_class, travel_date, sort_by="cheap"):
    airlines = list(AIRLINE_LOGOS.keys())
    flights = []

    for _ in range(40):
        airline = random.choice(airlines)
        dep_hour = random.randint(0, 23)
        arr_hour = (dep_hour + random.randint(1, 3)) % 24

        dep_time = f"{dep_hour:02d}:{random.choice(['00','15','30','45'])}"
        arr_time = f"{arr_hour:02d}:{random.choice(['00','15','30','45'])}"

        price = random.randint(2500, 9000)

        flights.append({
            "airline": airline,
            "airline_logo": AIRLINE_LOGOS.get(airline, DEFAULT_LOGO),
            "flight": f"{airline[:2].upper()}-{random.randint(100,999)}",
            "aircraft": random.choice(["A320", "A321", "B737"]),
            "source_city": source,
            "destination_city": destination,
            "source_airport": source,
            "destination_airport": destination,
            "source_code": source[-4:-1] if "(" in source else "",
            "destination_code": destination[-4:-1] if "(" in destination else "",
            "departure_time": dep_time,
            "arrival_time": arr_time,
            "actual_dep_time": dep_time,
            "actual_arr_time": arr_time,
            "duration": f"{random.randint(1,3)}h {random.randint(0,55)}m",
            "stops": random.choice([0, 1]),
            "predicted_price": price,
            "holiday": "Standard Pricing",
            "baggage": "20kg Check-in + 7kg Cabin" if flight_class == "Economy" else "30kg + 10kg",
            "meals": "Complimentary Meals" if airline in ["Air India", "Vistara"] else "Buy Meals",
            "beverages": "Complimentary Beverages" if airline in ["Air India", "Vistara"] else "Buy Beverages",
            "usb": "Yes" if airline in ["Indigo", "Air India", "Vistara"] else "No",
            "depart_terminal": random.choice(["T1", "T2"]),
            "arrival_terminal": random.choice(["T1", "T2"]),
            "travel_date": travel_date
        })

    if sort_by == "best":
        flights.sort(key=lambda x: (x["stops"], x["predicted_price"]))
    else:
        flights.sort(key=lambda x: x["predicted_price"])

    return flights

# ======================================================
# ROUTES
# ======================================================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/searchflight")
def searchflight():
    return render_template("searchflight.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/deals")
def deals():
    return render_template("deals.html")

@app.route("/destinations")
def destinations():
    return render_template("destinations.html")

@app.route("/flight-details")
def flight_details():
    source = request.args.get("source")
    destination = request.args.get("destination")
    flight_class = request.args.get("class")
    travel_date = request.args.get("date")
    sort_by = request.args.get("sort_by", "cheap")
    travellers = int(request.args.get("travellers", 1))

    flights = recommend_flights(source, destination, flight_class, travel_date, sort_by)

    # FILTERS
    if request.args.get("airline"):
        allowed = request.args.get("airline").split(",")
        flights = [f for f in flights if f["airline"] in allowed]

    if request.args.get("stops"):
        allowed = request.args.get("stops").split(",")
        flights = [f for f in flights if str(f["stops"]) in allowed]

    if request.args.get("departure_time"):
        allowed = request.args.get("departure_time").split(",")
        flights = [f for f in flights if get_time_slot(f["departure_time"]) in allowed]

    if request.args.get("arrival_time"):
        allowed = request.args.get("arrival_time").split(",")
        flights = [f for f in flights if get_time_slot(f["arrival_time"]) in allowed]

    if request.args.get("max_price"):
        try:
            max_p = int(request.args.get("max_price"))
            flights = [f for f in flights if f["predicted_price"] <= max_p]
        except:
            pass

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

# ======================================================
# FILTER DATA API (FIXED)
# ======================================================

@app.route("/get-filters")
def get_filters():
    source = request.args.get("source")
    destination = request.args.get("destination")
    flight_class = request.args.get("class")
    travel_date = request.args.get("date")
    sort_by = request.args.get("sort_by", "cheap")

    flights = recommend_flights(source, destination, flight_class, travel_date, sort_by)

    airlines = sorted(set(f["airline"] for f in flights))
    prices = [f["predicted_price"] for f in flights]

    stop_map = {}
    for f in flights:
        stop_map.setdefault(f["stops"], []).append(f["predicted_price"])

    stops = [{
        "label": "Direct" if k == 0 else f"{k} Stop",
        "value": str(k),
        "min_price": min(v)
    } for k, v in stop_map.items()]

    dep_slots = sorted({get_time_slot(f["departure_time"]) for f in flights if get_time_slot(f["departure_time"]) != "unknown"})
    arr_slots = sorted({get_time_slot(f["arrival_time"]) for f in flights if get_time_slot(f["arrival_time"]) != "unknown"})

    return jsonify({
        "airlines": airlines,
        "min_price": min(prices) if prices else 0,
        "max_price": max(prices) if prices else 0,
        "stops": stops,
        "departure_times": [{"label": TIME_SLOT_LABELS[s], "value": s} for s in dep_slots],
        "arrival_times": [{"label": TIME_SLOT_LABELS[s], "value": s} for s in arr_slots]
    })

# ======================================================
# AUTOCOMPLETE
# ======================================================

@app.route("/suggest-airport")
def suggest_airport():
    q = request.args.get("q", "").lower()
    airports = ["Chennai (MAA)", "Mumbai (BOM)", "Delhi (DEL)", "Bangalore (BLR)", "Hyderabad (HYD)"]
    return jsonify([{"label": a} for a in airports if q in a.lower()])


if __name__ == "__main__":
    app.run(debug=True)