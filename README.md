# Restaurant Recommendation API

> *__Note:__ This project is a comprehensive solution to the take-home assignment for the LINE MAN Wongnai Junior Machine Learning Engineer position.

## Description

This project provides a high-performance HTTP API server for a restaurant recommendation system. The core of the recommendation logic is powered by a pre-trained Scikit-Learn `NearestNeighbors` model.

The primary goal of this project was to build a stable and scalable service capable of meeting strict performance requirements: serving 30 requests per second with a 90th percentile response time within 100 milliseconds.

## Features & Architecture

The final implementation is a robust, production-ready system featuring several key optimizations to ensure stability and performance under concurrent load.

* **Web Server**: A multi-process **Gunicorn** server managing **Uvicorn** workers, providing both stability and true parallelism to handle high request volumes.
* **API Framework**: **FastAPI** for its high performance and automatic data validation.
* **Database**: **PostgreSQL** to store user feature data and restaurant location data.
* **Cache Pre-warming**: A **Gunicorn `on_starting` server hook** is used to run a one-time, memory-efficient batch process that pre-loads all user data into Redis. This ensures the cache is fully "warm" when the workers start, guaranteeing maximum performance.
* **Centralized Caching**: **Redis** is used as a high-performance, in-memory cache. It is shared by all Gunicorn worker processes, significantly boosting performance and reducing database load.
* **Database Indexing**: Critical database indexes were added to `users(user_id)` and `restaurants(h3_index)` to ensure millisecond-level query times.
* **Geospatial Pre-filtering**: Implemented an H3-based pre-filtering strategy to dramatically reduce the search space for nearby restaurants. 
* **Memory Optimization**: The request-response cycle was optimized to use lightweight NumPy arrays instead of pandas DataFrames, solving Out of Memory errors.

## Prerequisites

* Docker
* Docker Compose

## Project Structure

```
├── .env                    # Environment variables for local setup (DB credentials, etc.)
├── .gitignore              # Files and directories to ignore in version control
├── README.md               # This file
├── docker-compose.yml      # Defines and runs the multi-container application (API, DB, Cache)
|
├── data/
│   ├── restaurant.parquet
│   ├── user.small.parquet
|   └──user.parquet
|
├── model/
│   └── model.pkl           # The pre-trained Scikit-Learn NearestNeighbors model
│
├── perf_test/
│   ├── locustfile.py       # Locust script for load testing the API
│   └── request.parquet     # Request parameters for the performance test
│
├── scripts/
│   ├── create_db.py        # Script to create the database schema, indexes, and load data
|   └── inference.py        # Inference Script for inspect model output
│
└── server/
    ├── Dockerfile          # Instructions to build the API server Docker image
    ├── requirements.txt    # Python dependencies for the server
    ├── gunicorn_conf.py    # guvicorn config file
    └── app/
        ├── database.py     # SQLAlchemy engine and table metadata setup
        └── main.py         # Main FastAPI application, endpoints, and caching logic
```

## Setup & Installation

Follow these steps precisely to set up and run the application.

**1. Create Environment File**

The script to populate the database runs on your local machine and requires database credentials. Create a file named `.env` in the root of the project directory.

**.env**
* POSTGRES_USER=user
* POSTGRES_PASSWORD=password
* POSTGRES_DB=restaurants_db
* POSTGRES_HOST=127.0.0.1
* POSTGRES_PORT=5432
* REDIS_HOST=localhost

**2. Clean Previous Docker State (Important)**

To ensure a completely fresh start and that the new database schema is applied, run this command to remove any old containers and volumes.

```bash
docker-compose down -v
```
**3. Start the Database and Cache Services**

Start the PostgreSQL and Redis containers.

```bash
docker-compose up -d db redis
```

**4. Create Schema and Load Data**

Run the database creation script. This will create the tables, add the necessary performance indexes, and load the data from the 
`user.parquet` and `restaurant.parquet` files.

```bash
python scripts/create_db.py
```

**5. Start the Full Application**

Once the database is ready, build and start the API server, which will connect to the other running services.

```bash
docker-compose up --build -d
```

API should now be running and available on `http://localhost:8000`

## Usage

### API Endpoint

 * URL: `/recommend/{user_id}`
 * Method: `GET` or `POST`

### Path Parameters

 * `user_id` (string, required): The ID of the user for whom to generate recommendations (e.g., `u00000`).

### Query Parameters

 * `atitude` (float, required): User's latitude. 

 * `longitude` (float, required): User's longitude. 

 * `size` (integer, optional, default: 20): The number of recommended restaurants to return. 

 * `max_dis` (integer, optional, default: 5000): The maximum geodesic displacement in meters. Restaurants further than this are considered irrelevant. 

 * `sort_dis` (integer, optional, default: 0): A flag to control sorting. 
 `1`: Sort results by geodesic displacement. 
 `0` or not provided: Sort results by the model's Euclidean distance.

### Example Request

You can test the running API with `curl`:

```bash
curl "http://localhost:8000/recommend/u40099?latitude=13.988&longitude=100.432&size=5&max_dis=20000"
```

## Performance Testing

The project includes a `perf_test` folder with a Locust script to simulate load and verify the performance requirements.

**1. Install Locust:**

```bash
pip install locust
```

**2. Run the Test::**

Execute the following command from the project's root directory.

```bash
locust -f perf_test/locustfile.py
```

**3. Start new load test:**

- Open your web browser to `http://localhost:8089`

- Enter the number of users to simulate and a spawn rate.

- Click "Start" and observe the statistics to verify the RPS and latency targets.

## Key-Learning

This project served as a comprehensive, end-to-end exercise in deploying a machine learning model as a high-performance, production-ready service. The journey from a functional prototype to a stable, optimized system revealed several critical engineering lessons:

**1. Architecture First:** An application's concurrency model is fundamental to its stability. Initial attempts using a purely `async` architecture led to complex deadlocks under load. The solution was to pivot to a simpler, more robust multi-process architecture using Gunicorn, which provided stability and true parallelism.

**2. The Database is the First Bottleneck:** The single greatest performance gain came from adding an index to the `users.user_id` column. This simple change reduced a key query's latency by over 90%, proving that **database optimization is a mandatory first step** before attempting code-level optimizations.

**3. Memory is a Critical Resource:** Initial load tests resulted in "Out of Memory" crashes. This was traced to the high memory footprint of using the pandas library within the request-response cycle. By **replacing DataFrames with lightweight NumPy arrays**, the memory pressure was eliminated, allowing the server to remain stable under high concurrency.

**4. The ML Model Is a System Component:** Final analysis showed that the ultimate performance limit was the CPU time of the `model.kneighbors()` inference itself. This highlights that an ML Engineer's job includes considering the inference performance of a model, not just its accuracy, and may require using specialized libraries like Faiss or Annoy to meet production latency targets.

Summary of those key architectural decisions we made during the optimization process.

**1. From pandas DataFrame to Pure NumPy**
* **Problem:** Under concurrent load, the API server was crashing. The logs showed workers being killed with **"Out of Memory" (OOM)** errors.

* **Solution:** Our investigation found that creating a pandas DataFrame for user features in every request consumed a large amount of memory. We replaced this by converting the data from the database directly into a lightweight **NumPy array**.

* **Reasoning:** NumPy arrays are significantly more memory-efficient for purely numerical data than pandas DataFrames because they don't have the extra overhead of an index and other metadata. This change solved the OOM crashes and made the server stable.

**2. From Raw SQL to SQLAlchemy Table Objects**
* **Problem:** Building complex SQL queries using Python f-strings was becoming error-prone, especially when trying to pass parameters for the `WHERE ... IN (...)` clauses, which led to multiple database syntax errors.

* **Solution:** We defined our database tables in `database.py` using **SQLAlchemy's `Table` construct**. This allowed us to build queries programmatically (e.g., `select(users_table).where(...)`).

* **Reasoning:** Using SQLAlchemy's objects provides a safer, clearer, and more maintainable way to write queries in Python. It helps prevent syntax errors and abstracts away the raw SQL, making the code more robust.

**3. From async/Thread Pool to Multi-Process Gunicorn**
* **Problem:** The initial async def version of the API, which used run_in_threadpool to handle blocking calls, suffered from severe freezes and connection timeouts under concurrent load. This was due to complex deadlocks between the asyncio event loop, the thread pool, and the database connection pool.

* **Solution:** We removed all async/await and run_in_threadpool logic and switched the server architecture to Gunicorn with synchronous Uvicorn workers.

* **Reasoning:** Gunicorn achieves concurrency by running multiple, independent worker processes. Each worker handles one request at a time from start to finish. This model is simpler and more stable for this workload because it eliminates the shared resource contention that was causing the async version to deadlock. It provides true parallelism and is the industry-standard for deploying robust Python web applications.

---