import os
from sqlalchemy import create_engine

def get_engine():
    user = os.getenv("POSTGRES_USER", "ipcam")
    pwd = os.getenv("POSTGRES_PASSWORD", "ipcam")
    db  = os.getenv("POSTGRES_DB", "ipcam")
    host = os.getenv("POSTGRES_HOST", "db")
    port = os.getenv("POSTGRES_PORT", "5432")
    url = f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
    engine = create_engine(url, pool_pre_ping=True, future=True)
    return engine
