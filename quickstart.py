import datetime
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error


# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

# Public Holiday Calendar ID (Example: India)
HOLIDAY_CALENDAR_ID = "en.indian#holiday@group.v.calendar.google.com"  # Change as needed

def get_holidays(days=364):
    """Fetch upcoming public holidays from Google Calendar."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build("calendar", "v3", credentials=creds)

        # Fetch upcoming holidays within the specified range
        now = datetime.datetime.now().isoformat() + "Z"
        end_date = (datetime.datetime.now() + datetime.timedelta(days=days)).isoformat() + "Z"

        events_result = (
            service.events()
            .list(
                calendarId=HOLIDAY_CALENDAR_ID,
                timeMin=now,
                timeMax=end_date,
                maxResults=20,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        events = events_result.get("items", [])
        holiday_list = [(event["summary"], event["start"].get("date")) for event in events]

        return holiday_list

    except HttpError as error:
        print(f"An error occurred: {error}")
        return []

# Fetch holidays for the next 30 days
holidays = get_holidays(364)
print("Upcoming Holidays:", holidays)

def get_demand_factor(travel_date, holidays):
    """Adjust demand factor based on holidays near the travel date."""
    event_factor = 1.0  # Default factor
    
    # Define high & medium demand holidays
    high_demand_events = ["diwali", "new year", "christmas", "eid", "bakrid", "muharram"]
    medium_demand_events = ["long weekend", "festival", "holi", "raksha bandhan", "janmashtami", 
                            "good friday", "republic day", "independence day", "rath yatra"]
    
    for holiday_name, holiday_date in holidays:
        if holiday_date == travel_date:  # Exact match
            event_name = holiday_name.lower()
            
            # High-demand events (30% increase)
            if any(keyword in event_name for keyword in high_demand_events):
                event_factor = 1.3  
                break  # No need to check further
            
            # Medium-demand events (20% increase)
            elif any(keyword in event_name for keyword in medium_demand_events):
                event_factor = 1.2  

    return event_factor

# ✅ Fetch upcoming holidays
holidays = get_holidays(300)  # Fetch for the next 300 days

# Example Usage
travel_date = "2025-08-15"  # Janmashtami
demand_factor = get_demand_factor(travel_date, holidays)
print(f"Demand Factor for {travel_date}: {demand_factor}")

df = pd.read_csv("Clean_flight_data.csv")

df["days_left"] = df["days_left"].astype(int)
df["class"] = df["class"].str.capitalize()

features = ["source_city", "destination_city", "class", "days_left"]
target = "price"

X = df[features]
y = df[target]

y = y.clip(upper=30000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ["source_city", "destination_city", "class"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1)) # Increased depth for better learning
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")

# Function to recommend cheapest flights
def recommend_flights(source, destination, flight_class, days_left, df, model, top_n=10):
    """Recommend flights based on price and real-time event demand."""
    
    # Filter available flights
    filtered_flights = df[(df["source_city"] == source) & 
                          (df["destination_city"] == destination) & 
                          (df["class"] == flight_class)].copy()

    if filtered_flights.empty:
        return "No matching flights found. Try adjusting your input."

    # Predict base price using ML model
    filtered_flights["predicted_price"] = model.predict(filtered_flights[["source_city", "destination_city", "class", "days_left"]])

    # Fetch real-time event data
    events = get_upcoming_events(30)

    # Adjust prices based on demand
    filtered_flights["event_impact"] = filtered_flights["days_left"].apply(lambda x: get_demand_factor(x, events))
    filtered_flights["final_price"] = filtered_flights["predicted_price"] * filtered_flights["event_impact"]

    # Sort by best price
    sorted_flights = filtered_flights.sort_values(by=["days_left","final_price"]).head(top_n)

    return sorted_flights[["airline", "flight", "departure_time", "stops", "arrival_time","days_left", "final_price"]]

source_input = "Delhi"
destination_input = "Mumbai"
class_input = "Economy"
days_left_input = 1
recommended_flights = recommend_flights(source_input, destination_input, class_input, days_left_input, df,model)
print(recommended_flights)
