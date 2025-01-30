from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import requests

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        # Mock data for demonstration
        data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "traffic_volume": np.random.randint(50, 500),
            "vehicle_speed": np.random.randint(10, 100),
            "congestion_level": np.random.randint(1, 10),
            "vehicle_count": np.random.randint(0, 1000)
        }

        # Preprocess data
        features, target = self.preprocess_data(pd.DataFrame([data]))

        # Predict traffic
        model, predictions = self.train_and_predict(features, target)

        # Optimize signal timings
        signal_timings = self.optimize_signals(predictions)

        # Generate route recommendations
        routes = self.generate_routes(predictions)

        # Return results
        response = {
            "data": data,
            "predictions": predictions.tolist(),
            "signal_timings": signal_timings,
            "routes": routes
        }
        self.wfile.write(json.dumps(response).encode())

    def preprocess_data(self, data):
        try:
            data["traffic_volume"] = data["traffic_volume"] / 100
            data["vehicle_speed"] = data["vehicle_speed"] / 100
            data["vehicle_count"] = data["vehicle_count"] / 1000
            features = data[["traffic_volume", "vehicle_speed", "vehicle_count"]]
            target = data["congestion_level"]
            return features, target
        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            return None, None

    def train_and_predict(self, features, target):
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            return model, predictions
        except Exception as e:
            print(f"Error during traffic prediction: {e}")
            return None, None

    def optimize_signals(self, predictions):
        try:
            signal_timings = []
            for congestion in predictions:
                if congestion < 3:
                    signal_timings.append(30)
                elif congestion < 7:
                    signal_timings.append(60)
                else:
                    signal_timings.append(90)
            return signal_timings
        except Exception as e:
            print(f"Error during signal optimization: {e}")
            return None

    def generate_routes(self, predictions):
        try:
            routes = []
            for congestion in predictions:
                if congestion >= 7:
                    routes.append("Alternative Route A")
                else:
                    routes.append("Main Route")
            return routes
        except Exception as e:
            print(f"Error during route generation: {e}")
            return None