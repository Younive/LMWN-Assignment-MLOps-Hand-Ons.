# sequential_test.py

import os
from time import time

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "") # Assuming request.parquet is in perf_test


def test_perf(reqs: list) -> tuple[float, list[float], int]:
    """Test performance of the API"""
    total_time = 0
    time_all = []
    failed_requests = 0
    for req in tqdm(reqs, desc="Sending Requests"):
        req_copy = req.copy()
        user_id = req_copy.pop("user_id")

        try:
            s = time()
            # IMPORTANT: Ensure this port matches your docker-compose.yml (e.g., 8000)
            response = requests.get(f"http://127.0.0.1:8000/recommend/{user_id}", params=req_copy, timeout=10)
            time_use = (time() - s) * 1000  # Convert to ms

            if response.status_code == 200:
                total_time += time_use
                time_all.append(time_use)
            else:
                failed_requests += 1
                print(f"Request failed for user {user_id} with status {response.status_code}: {response.text[:100]}")

        except requests.exceptions.RequestException as e:
            failed_requests += 1
            print(f"Request failed for user {user_id} with exception: {e}")


    return total_time, time_all, failed_requests


def load_request() -> list:
    """Load request data from parquet file"""
    df = pd.read_parquet(os.path.join(DATA_DIR, "request.parquet"))
    reqs = df.to_dict(orient="records")
    return reqs


def process_request(reqs: list):
    """Pop sort_dis and max_dis if it is nan"""
    for req in reqs:
        if pd.isna(req.get("sort_dis")):
            req.pop("sort_dis", None)
        if pd.isna(req.get("max_dis")):
            req.pop("max_dis", None)


def print_result(req_size: int, total_time: float, time_all: list[float], failed_count: int):
    """Print result of the test"""
    successful_requests = len(time_all)
    if successful_requests == 0:
        print("No successful requests were made.")
        return

    percentile_value = np.percentile(time_all, 90)
    # Calculate RPS based only on the time spent on successful requests
    req_per_sec = successful_requests / (total_time / 1000) if total_time > 0 else 0

    print("\n--- Performance Test Results ---")
    print(f"Total Requests Sent: {req_size}")
    print(f"Successful Requests: {successful_requests}")
    print(f"Failed Requests: {failed_count}")
    print("----------------------------------")
    print(f"90th Percentile Latency: {percentile_value:.2f} ms")
    print(f"Requests Per Second (RPS): {req_per_sec:.2f}")
    print(f"Average Latency: {(total_time / successful_requests):.2f} ms")
    print(f"Min Latency: {min(time_all):.2f} ms")
    print(f"Max Latency: {max(time_all):.2f} ms")
    print("----------------------------------")


if __name__ == "__main__":
    reqs = load_request()
    req_size = len(reqs)
    process_request(reqs)

    total_time, time_all, failed_count = test_perf(reqs)
    print_result(req_size, total_time, time_all, failed_count)