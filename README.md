# Restaurant Recommendation API

## Description

This project provides a high-performance HTTP API server for a restaurant recommendation system, developed as part of the LINE MAN Wongnai Machine Learning Engineer Hands-on assignment. The core of the recommendation logic is powered by a pre-trained Scikit-Learn `NearestNeighbors` model.

The primary goal of this project was to build a stable and scalable service capable of meeting strict performance requirements: serving 30 requests per second with a 90th percentile response time within 100 milliseconds.

## Features & Architecture

The final implementation is a robust, production-ready system featuring several key optimizations:

* **Web Server**: A multi-process Gunicorn server managing Uvicorn workers, providing both stability and high concurrency to handle load without deadlocking.
* **API Framework**: FastAPI for its high performance and automatic data validation.
* **Database**: PostgreSQL to store user feature data and restaurant location data.
* **Centralized Caching**: Redis is used as a high-performance, in-memory cache for user features. This provides a shared cache for all Gunicorn worker processes, significantly boosting performance and reducing database load under concurrent requests.
* **Database Indexing**: Critical database indexes were added to `users(user_id)` and `restaurants(h3_index)` to ensure millisecond-level query times under load.
* **Geospatial Pre-filtering**: Implemented an H3-based pre-filtering strategy to dramatically reduce the search space for nearby restaurants.
* **Memory Optimization**: The request-response cycle was optimized to use lightweight NumPy arrays instead of pandas DataFrames, solving Out of Memory errors under concurrent load.

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
    └── app/
        ├── database.py     # SQLAlchemy engine and table metadata setup
        └── main.py         # Main FastAPI application, endpoints, and caching logic
```

## Setup & Installation

Follow these steps precisely to set up and run the application.

**1. Create Environment File**

The script to populate the database runs on your local machine and requires database credentials. Create a file named `.env` in the root of the project directory.

.env
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

- Click "Start swarming" and observe the statistics to verify the RPS and latency targets.

## Key-Learning

---