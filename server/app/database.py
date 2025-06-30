# In server/database.py
import os
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Index

def get_database_engine():
    """Creates and returns a SQLAlchemy engine for PostgreSQL."""
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB")

    if not all([db_user, db_password, db_host, db_name]):
        raise ValueError("Database environment variables are not set.")

    # This URL configures the connection pool.
    # pool_size=10, max_overflow=5 are good starting points.
    db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url, pool_size=10, max_overflow=5)
    return engine

engine = get_database_engine()
metadata = MetaData()

# Define the 'users' table using SQLAlchemy Core
# This allows us to build queries without writing raw SQL strings
users_table = Table('users', metadata,
    Column('user_id', String, primary_key=True),
    *[Column(f'feature_{i}', Float) for i in range(1000)]
)

# Define the 'restaurants' table
restaurants_table = Table('restaurants', metadata,
    Column('restaurant_id', String, primary_key=True),
    Column('index', Integer, index=True),
    Column('latitude', Float),
    Column('longitude', Float),
    Column('h3_index', String, index=True)
)