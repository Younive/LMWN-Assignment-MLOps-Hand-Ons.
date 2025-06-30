from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import pandas as pd
import os
from dotenv import load_dotenv
import h3

load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def get_db_engine() -> Engine:
    """Creates and returns a SQLAlchemy engine for PostgreSQL."""
    db_user = os.getenv("POSTGRES_USER")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB")
    db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url)
    return engine

def read_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read data from parquet files into pandas DataFrames.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the user DataFrame and the restaurant DataFrame.
    """
    user_df = pd.read_parquet(os.path.join(DATA_DIR, "user.parquet"))
    restaurant_df = pd.read_parquet(os.path.join(DATA_DIR, "restaurant.parquet"))
    restaurant_df['latitude'] = pd.to_numeric(restaurant_df['latitude'], errors='coerce')
    restaurant_df['longitude'] = pd.to_numeric(restaurant_df['longitude'], errors='coerce')
    restaurant_df.dropna(subset=['latitude', 'longitude'], inplace=True)
    print("Calculating H3 indexes for restaurants...")
    restaurant_df['h3_index'] = restaurant_df.apply(
        lambda row: h3.latlng_to_cell(row['latitude'], row['longitude'], 9),
        axis=1
    )
    return user_df, restaurant_df

def create_tables(engine: Engine, reset: bool = True):
    """Create database tables for users and restaurants.

    Args:
        engine (Engine): The SQLAlchemy engine.
        reset (bool, optional): Drop tables before creating them. Defaults to True.
    """
    with engine.connect() as connection:
        if reset:
            print("Dropping existing tables...")
            connection.execute(text("DROP TABLE IF EXISTS restaurants;"))
            connection.execute(text("DROP TABLE IF EXISTS users;"))
        
        print("Creating new tables...")
        
        # Create restaurants table
        # [cite_start]Columns are based on restaurant.parquet from the problem description [cite: 12]
        connection.execute(text("""
            CREATE TABLE restaurants (
                restaurant_id TEXT PRIMARY KEY,
                "index" INTEGER,
                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,
                h3_index TEXT
            );
            CREATE INDEX idx_restaurants_h3 ON restaurants(h3_index);
        """))
        
        # Create users table
        # Dynamically create feature columns based on a sample
        feature_columns = [f'feature_{i} DOUBLE PRECISION' for i in range(1000)]
        create_users_table_sql = f"""
            CREATE TABLE users (
                user_id TEXT PRIMARY KEY,
                {', '.join(feature_columns)}
            );
        """
        connection.execute(text(create_users_table_sql))

        print("Creating database indexes...")        

        connection.execute(text("CREATE UNIQUE INDEX idx_users_user_id ON users(user_id);"))
        connection.commit()

        print("Indexes created successfully.")

        print("Tables created successfully.")


def insert_data(user_df: pd.DataFrame, restaurant_df: pd.DataFrame, engine: Engine):
    """Insert data from DataFrames into the PostgreSQL database.

    Args:
        user_df (pd.DataFrame): DataFrame containing user data.
        restaurant_df (pd.DataFrame): DataFrame containing restaurant data.
        engine (Engine): The SQLAlchemy engine.
    """
    print("Inserting restaurant data to database...")
    restaurant_df.to_sql("restaurants", engine, if_exists="append", index=False)
    
    print("Inserting user data to database...")
    user_df.to_sql("users", engine, if_exists="append", index=False, chunksize=1000) # Use chunksize for large data
    
    print("Data insertion complete.")

if __name__ == "__main__":
    
    # Establish database connection
    db_engine = get_db_engine()
    
    # Create tables (and drop if they exist)
    create_tables(db_engine, reset=True)
    
    # Read data from source files
    user_df, restaurant_df = read_data()
    
    # Insert the data into the tables
    insert_data(user_df, restaurant_df, db_engine)