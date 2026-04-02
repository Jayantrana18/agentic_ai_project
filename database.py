import os
import time
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# -------------------------------------------------------------------------
# DATABASE CONFIGURATION
# -------------------------------------------------------------------------
# For local testing, we use SQLite because it works out-of-the-box without setup.
# In a real deployed environment, you would set the DATABASE_URL environment 
# variable to your PostgreSQL connection string, e.g.:
# "postgresql://username:password@localhost:5432/medrag_db"

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///medrag_logs.db")

# Streamlit uses multiple threads. By default, SQLite blocks cross-thread connections.
# We need to tell SQLite it's safe to share the connection across Streamlit's threads.
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# -------------------------------------------------------------------------
# DATABASE SCHEMA 
# -------------------------------------------------------------------------
class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    question = Column(String, index=True)
    answer = Column(String)
    sources_used = Column(Integer)  # Number of documents retrieved
    response_time = Column(Float)   # Time taken to generate the answer in seconds

def init_db():
    """Create the database tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

def log_interaction(question: str, answer: str, sources_used: int, response_time: float):
    """Log a user query and the system's response into the database."""
    db = SessionLocal()
    try:
        log_entry = QueryLog(
            question=question,
            answer=answer,
            sources_used=sources_used,
            response_time=response_time
        )
        db.add(log_entry)
        db.commit()
    except Exception as e:
        print(f"Database logging error: {e}")
    finally:
        db.close()
