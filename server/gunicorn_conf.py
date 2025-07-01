import os
import pickle
import numpy as np
import redis
from sqlalchemy import select

# Import your database engine and table definitions
# Make sure your database.py can be imported like this
from app.database import engine, users_table

def on_starting(server):
    """
    Gunicorn master process hook, runs only once before workers are started.
    Perfect for pre-warming the cache.
    """
    print("--- GUNICORN MASTER: Starting hook `on_starting` ---")
    try:
        # Establish a Redis connection just for this task
        redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, db=0)
        redis_client.ping()
        print("--- GUNICORN MASTER: Connected to Redis successfully for pre-warming. ---")
    except redis.exceptions.ConnectionError as e:
        print(f"!!! GUNICORN MASTER: Could not connect to Redis: {e}. Skipping cache pre-warming. !!!")
        return

    # Best-Effort Cache Pre-warming
    try:
        print("--- GUNICORN MASTER: Pre-loading user features into Redis cache... ---")
        with engine.connect() as connection:
            query = select(users_table)
            result_proxy = connection.execute(query)
            
            total_loaded_count = 0
            while True:
                batch = result_proxy.fetchmany(5000)
                if not batch:
                    break

                with redis_client.pipeline() as pipe:
                    for user in batch:
                        user_id = user.user_id
                        features = np.array(user[1:]).reshape(1, -1)
                        pipe.set(user_id, pickle.dumps(features), ex=3600)
                    pipe.execute()
                
                total_loaded_count += len(batch)
                print(f"--- GUNICORN MASTER: Loaded {total_loaded_count} users into cache... ---")

        print(f"--- GUNICORN MASTER: Successfully pre-loaded a total of {total_loaded_count} users into cache. ---")
    except Exception as e:
        print(f"!!! GUNICORN MASTER: Cache pre-warming failed: {e}. Workers will use lazy-loading. !!!")