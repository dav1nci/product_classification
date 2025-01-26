from config import Settings
from serving.model_loader import ModelLoader
from sql_utils.database import init_db
from minio_utils.connection import init_minio_filestructure


settings = Settings()

init_minio_filestructure()

engine, SessionLocal = init_db(settings)

def get_db_session():
    """Dependency to get the database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

model_loader = ModelLoader()
