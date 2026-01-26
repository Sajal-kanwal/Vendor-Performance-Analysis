# db_connect.py
"""
Centralized, secure MySQL connection module for EDA notebooks and scripts.

Usage in notebook:
    from db_connect import get_engine, get_connection, query_to_df

    engine = get_engine()
    df = query_to_df("SELECT * FROM sales LIMIT 1000")
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

import logging

# ────────────────────────────────────────────────
# Logging (simple console + optional file)
# ────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def _load_env() -> None:
    """Try to load .env from common locations."""
    env_found = False
    
    # Look in current directory, parent, grandparent, etc.
    for depth in range(5):
        env_path = Path.cwd() / (('../' * depth) + '.env')
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            logger.info(f"Loaded .env from: {env_path.resolve()}")
            env_found = True
            break
    
    if not env_found:
        logger.warning("No .env file found in current or parent directories")


# Load env variables at module import (only once)
_load_env()


def get_db_credentials() -> dict:
    """
    Collect MySQL credentials from environment variables.
    All fields are required except PORT (defaults to 3306).
    """
    required = ["MYSQL_HOST", "MYSQL_USER", "MYSQL_PASSWORD", "MYSQL_DATABASE"]
    creds = {}

    for key in required:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Missing required environment variable: {key}")
        creds[key] = value

    creds["MYSQL_PORT"] = os.getenv("MYSQL_PORT", "3306")

    return creds


def get_engine(
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_recycle: int = 1800,
) -> Engine:
    """
    Create and return a SQLAlchemy engine using credentials from .env.

    Returns a pooled engine suitable for repeated queries in notebooks.
    """
    creds = get_db_credentials()

    connection_string = (
        f"mysql+mysqlconnector://"
        f"{creds['MYSQL_USER']}:{creds['MYSQL_PASSWORD']}@"
        f"{creds['MYSQL_HOST']}:{creds['MYSQL_PORT']}/"
        f"{creds['MYSQL_DATABASE']}"
        "?charset=utf8mb4"
    )

    try:
        engine = create_engine(
            connection_string,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,           # helps detect stale connections
            echo=echo,                    # set True only when debugging
        )
        logger.info(f"Engine created for database: {creds['MYSQL_DATABASE']}")
        return engine
    except SQLAlchemyError as e:
        logger.error(f"Failed to create engine → {str(e)}")
        raise


# ────────────────────────────────────────────────
# Convenience functions for notebooks
# ────────────────────────────────────────────────

_engine: Optional[Engine] = None


def get_connection():
    """Get a connection from the (cached) engine — mostly for manual use."""
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine.connect()


def query_to_df(
    query: str,
    params=None,
    engine: Optional[Engine] = None,
) -> pd.DataFrame:
    """
    Execute SQL query and return result as pandas DataFrame.

    Examples:
        df = query_to_df("SELECT * FROM orders WHERE date >= '2025-01-01'")
        df = query_to_df("SELECT * FROM customers WHERE city = :city", {"city": "Jaipur"})
    """
    global _engine
    if engine is None:
        if _engine is None:
            _engine = get_engine()
        engine = _engine

    try:
        with engine.connect() as conn:
            if params:
                result = pd.read_sql(text(query), conn, params=params)
            else:
                result = pd.read_sql(query, conn)
        logger.info(f"Query executed → {len(result):,} rows returned")
        return result
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        logger.debug(f"Query was: {query}")
        raise


def test_connection() -> bool:
    """Quick connection test — useful in notebooks."""
    try:
        with get_connection() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Connection test → SUCCESS")
        return True
    except Exception as e:
        logger.error(f"Connection test failed → {str(e)}")
        return False


if __name__ == "__main__":
    # Quick test when running the file directly
    print("Testing connection...")
    success = test_connection()
    print("Connection OK" if success else "Connection FAILED")