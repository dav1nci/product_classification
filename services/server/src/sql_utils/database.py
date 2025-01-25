import os

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from config import Settings





def create_jobs_table(engine):
    query = """
    CREATE TABLE jobs (
        job_id CHAR(36) NOT NULL PRIMARY KEY, -- UUIDv4 stored as a CHAR(36)
        start_time DATETIME NOT NULL,     
        end_time DATETIME,                
        job_status VARCHAR(255) NOT NULL, 
        input_file_s3 TEXT,
        output_file_s3 TEXT
    );
    """
    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()


def create_auto_train_table(engine):
    query = """
    CREATE TABLE auto_training (
        id CHAR(36) NOT NULL PRIMARY KEY,                    
        training_date TIMESTAMP NOT NULL,          
        dataset_hash VARCHAR(255) NOT NULL,
        dataset_path VARCHAR(255) NOT NULL,
        run_id VARCHAR(255) ,              
        best_checkpoint VARCHAR(255),              
        f1 INT ,                           
        min_f1 INT ,                       
        weights_s3 VARCHAR(255),    
        status VARCHAR(50) NOT NULL
    );

    """
    with engine.connect() as conn:
        conn.execute(text(query))
        conn.commit()


def table_exists(engine, table_name):
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    if table_name in tables:
        return True
    else:
        return False


def get_database_connection(user, password, host, port, database_name):
    try:
        database_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database_name}"

        # engine = create_engine(database_uri)
        engine = create_engine(database_uri, echo=True, future=True)
        SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

        with engine.connect() as connection:
            print(f"Connected to the database '{database_name}' successfully.")

        return engine, SessionLocal

    except SQLAlchemyError as e:
        print(f"Failed to connect to the database. Error: {e}")
        return None




def init_db(settings: Settings):
    engine, SessionLocal = get_database_connection(settings.MYSQL_JOB_DB_USER, settings.MYSQL_JOB_DB_PASSWORD, 'db',
                                     settings.MYSQL_PORT, settings.MYSQL_JOB_DB)
    if not table_exists(engine, 'jobs'):
        print("table jobs didn't exist, creating new one")
        create_jobs_table(engine)
    else:
        print("table jobs exists")

    if not table_exists(engine, 'auto_training'):
        print("table auto_training didn't exist, creating new one")
        create_auto_train_table(engine)
    else:
        print("table auto_training exists")

    return engine, SessionLocal



if __name__ == "__main__":
    pass
