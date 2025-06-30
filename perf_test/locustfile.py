# In perf_test/locustfile.py

import os
import random
import math
import pandas as pd
from locust import HttpUser, task, between

# --- START: LOAD DATA ONCE ---
# Load the test data at the module level. This happens only one time when Locust starts.
print("--- Loading test data ---")
try:
    PARQUET_PATH = os.path.join(os.path.dirname(__file__), "request.parquet")
    TEST_DATA = pd.read_parquet(PARQUET_PATH).to_dict("records")
    print(f"--- Test data loaded: {len(TEST_DATA)} records ---")
except FileNotFoundError:
    print(f"!!! ERROR: request.parquet not found at {PARQUET_PATH}. Exiting. !!!")
    TEST_DATA = []
# --- END: LOAD DATA ONCE ---


class RecommendationUser(HttpUser):
    # Set a host if not provided on the command line
    host = "http://localhost:8000"
    wait_time = between(0.5, 2.0)
    
    # The on_start method is now removed, as it's no longer needed.

    @task
    def get_recommendations(self):
        """
        Simulates a user requesting recommendations.
        """
        if not TEST_DATA:
            # If data loading failed, stop the user from making requests.
            self.environment.runner.quit()
            return

        # Use the globally loaded TEST_DATA
        params = random.choice(TEST_DATA).copy()
        
        # Clean NaN values from the chosen parameters
        params_to_send = {}
        for key, value in params.items():
            if not (isinstance(value, float) and math.isnan(value)):
                params_to_send[key] = value
        
        user_id = params_to_send.pop("user_id", None)
        if not user_id:
            return

        url = f"/recommend/{user_id}"
        
        self.client.get(url, params=params_to_send, name="/recommend/[user_id]")