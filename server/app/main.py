from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
# from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
import joblib
import os
from dotenv import load_dotenv
from haversine import haversine, Unit
import pandas as pd
from .database import engine, users_table, restaurants_table
import h3
import time
import numpy as np
import warnings
import redis
import pickle

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
# --- REDIS CACHE SETUP ---

# Create a Redis client connection. It will connect using the environment variable.
try:
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
    # Ping the server to test the connection
    redis_client.ping()
    print("--- Connected to Redis successfully ---")
except redis.exceptions.ConnectionError as e:
    print(f"--- Could not connect to Redis: {e} ---")
    redis_client = None

# --- Model Loading ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    Loads the ML model on startup and cleans up on shutdown.
    """
    # Startup: Load the model
    print("--- Loading ML model ---")
    
    # automatically pick up the MODEL_PATH from .env (in case not running on docker)
    model_path = os.getenv("MODEL_PATH", "/app/model/model.pkl") 
    
    # The model.pkl is the binary of the Scikit-Learn NearestNeighbors object
    app.state.model = joblib.load(model_path) 
    print("--- Model loaded successfully ---")
    
    yield  # The application runs while the lifespan context is active

    # Shutdown: Clean up resources
    print("--- Cleaning up resources ---")
    app.state.model = None

# --- App Initialization ---

# Register the lifespan manager with the FastAPI app
app = FastAPI(
    lifespan=lifespan,
    title="Restaurant Recommendation API",
    description="API for serving restaurant recommendations using a Scikit-Learn model.",
    version="1.0.0"
)

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Restaurant Recommendation API"}

# Define the endpoint for recommendations
@app.get("/recommend/{user_id}")
@app.post("/recommend/{user_id}")
def get_recommendations(
    user_id: str,
    latitude: float,
    longitude: float,
    size: int = Query(20, gt=0),
    max_dis: int = Query(5000, gt=0),
    sort_dis: int = Query(0, ge=0, le=1)
):
    # --- Get User Features (with Lazy-Loading Cache) ---
    features = None
    if redis_client:
        cached_features = redis_client.get(user_id)
    else:
        cached_features = None

    if cached_features:
        print(f"DEBUG: Cache HIT for user_id: {user_id}")
        features = pickle.loads(cached_features)
    else:
        print(f"DEBUG: Cache MISS for user_id: {user_id}")
        with engine.connect() as connection:
            query = select(users_table.c.feature_0, *[users_table.c[f'feature_{i}'] for i in range(1, 1000)]).where(users_table.c.user_id == user_id)
            user_data = connection.execute(query).first()
        
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        features = np.array([user_data]).reshape(1, -1)
        if redis_client:
            redis_client.set(user_id, pickle.dumps(features), ex=3600)

    # --- Run Model Prediction ---
    n_neighbors_to_query = 200
    distances, indices = app.state.model.kneighbors(features, n_neighbors=n_neighbors_to_query)
    model_indices_list = [int(i) for i in indices[0]]
    if not model_indices_list:
        return {"restaurants": []}

    # --- Fetch Restaurant Data ---
    with engine.connect() as connection:
        user_h3_index = h3.latlng_to_cell(latitude, longitude, 9)
        search_area_h3_indices = list(h3.grid_disk(user_h3_index, k=4))
        if not search_area_h3_indices:
            return {"restaurants": []}

        query = select(restaurants_table).where(
            restaurants_table.c.index.in_(model_indices_list),
            restaurants_table.c.h3_index.in_(search_area_h3_indices)
        )
        restaurant_results = [row._asdict() for row in connection.execute(query).fetchall()]
    
    if not restaurant_results:
        return {"restaurants": []}
    
    # --- Final Processing ---
    restaurant_map_by_index = {r['index']: r for r in restaurant_results}
    user_coords = (latitude, longitude)
    results = []
    model_distances = distances[0]
    for i, model_index in enumerate(model_indices_list):
        restaurant_data = restaurant_map_by_index.get(model_index)
        if not restaurant_data: continue
        try:
            restaurant_coords = (float(restaurant_data['latitude']), float(restaurant_data['longitude']))
        except (ValueError, TypeError, KeyError): continue
        displacement_meters = haversine(user_coords, restaurant_coords, unit=Unit.METERS)
        results.append({"id": restaurant_data['restaurant_id'], "difference": model_distances[i], "displacement": round(displacement_meters)})

    # --- Filtering and Sorting ---
    filtered_results = [r for r in results if r['displacement'] <= int(max_dis)]
    if int(sort_dis) == 1:
        sorted_results = sorted(filtered_results, key=lambda r: r['displacement'])
    else:
        sorted_results = sorted(filtered_results, key=lambda r: r['difference'])

    final_results = sorted_results[:int(size)]
    return {"restaurants": final_results}