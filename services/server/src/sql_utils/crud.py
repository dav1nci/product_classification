from sqlalchemy.orm import Session
from sqlalchemy import text, insert
from sqlalchemy.exc import OperationalError


from fastapi import HTTPException


def job_exists_in_db(engine, job_id):
    q = f"select job_id from jobs where job_id = {job_id};"
    with engine.connect() as conn:
        cur = conn.execute(text(q))
        if cur.rowcount == 0:
            return False
        return True

        # conn.rollback()


def add_inference_job(session, job_id, start_time, job_status, input_file_s3):
    q = f"""
    INSERT INTO
        jobs (job_id, start_time, job_status, input_file_s3)
    VALUES
        (:job_id, :start_time, :job_status, :input_file_s3);
    """
    try:
        session.execute(text(q), {'job_id': job_id,
                                        'start_time': start_time,
                                        'job_status': job_status,
                                        'input_file_s3': input_file_s3
                                        })
        session.commit()

    except OperationalError as e:
        print("Database connection lost:", e)
        raise HTTPException(status_code=500, detail="Database connection error")

# INSERT INTO auto_training (training_date, dataset_hash, status) VALUES (:training_date, :dataset_hash, :status) RETURNING id;
def add_training_job(session, train_job_id, training_date, dataset_hash, dataset_path, status):
    """
    id SERIAL PRIMARY KEY,                    
        training_date TIMESTAMP NOT NULL,          
        dataset_hash VARCHAR(255) NOT NULL,        
        run_id VARCHAR(255) ,              
        best_checkpoint VARCHAR(255),              
        f1 INT ,                           
        min_f1 INT ,                       
        weights_s3 VARCHAR(255),    
        status VARCHAR(50) NOT NULL
    :param session: 
    :return: 
    """
    q = f"""
        INSERT INTO
            auto_training (id, training_date, dataset_hash, dataset_path, status)
        VALUES
            (:train_job_id, :training_date, :dataset_hash, :dataset_path, :status);
        """
    try:
        session.execute(text(q), {'train_job_id': train_job_id,
                                           'training_date': training_date,
                                  'dataset_hash': dataset_hash,
                                  'dataset_path': dataset_path,
                                  'status': status,
                                  })

        # generated_id = session.fetchone()[0]
        session.commit()

    except OperationalError as e:
        print("Database connection lost:", e)
        raise HTTPException(status_code=500, detail="Database connection error")


def finalize_training_in_db(session, train_job_id, mlflow_run_id, best_checkpoint, f1, min_f1, weights_s3):
    q = """
    UPDATE auto_training SET run_id = :mlflow_run_id, best_checkpoint = :best_checkpoint, f1 = :f1, 
        min_f1 = :min_f1, weights_s3 = :weights_s3, status = 'FINISHED'
    WHERE id = :train_job_id;
    """
    try:
        session.execute(text(q), {'train_job_id': train_job_id,
                                  'mlflow_run_id': mlflow_run_id,
                                  'best_checkpoint': best_checkpoint,
                                  'f1': f1,
                                  'min_f1': min_f1,
                                  'weights_s3': weights_s3
                                  })
    except OperationalError as e:
        print("Database connection lost:", e)
        raise HTTPException(status_code=500, detail="Database connection error")


def get_job_status(engine, job_id):
    q = """
    SELECT job_status FROM jobs WHERE job_id = :job_id;
    """
    with engine.connect() as conn:
        result = conn.execute(text(q), {'job_id': job_id})
        job_status = result.one_or_none()
        if job_status is None:
            raise HTTPException(status_code=404, detail=f"Status for job {job_id} was not found in database")
        # print(f"JOB STATUS: {job_status}")
        return job_status[0]


def update_job_status(engine, job_id, job_status):
    q = """
    UPDATE jobs SET job_status = :job_status WHERE job_id = :job_id;
    """
    with engine.connect() as conn:
        result = conn.execute(text(q), {'job_status': job_status,
                                        'job_id': job_id
                                        })
        conn.commit()


def update_job_progress(engine, job_id, job_progress):
    q = """
    UPDATE jobs SET job_progress = :job_progress WHERE job_id = :job_id;
    """
    with engine.connect() as conn:
        result = conn.execute(text(q), {'job_progress': job_progress,
                                        'job_id': job_id
                                        })
        conn.commit()


def update_output_file_s3(engine, job_id, minio_url):
    q = """
    UPDATE jobs SET s3_blob = :s3_blob WHERE job_id = :job_id;
    """
    with engine.connect() as conn:
        result = conn.execute(text(q), {'s3_blob': minio_url,
                                        'job_id': job_id
                                        })
        conn.commit()


def get_minio_objectpath(engine, job_id):
    q = """
    SELECT s3_blob FROM jobs WHERE job_id = :job_id;
    """
    with engine.connect() as conn:
        result = conn.execute(text(q), {'job_id': job_id})
        s3_blob = result.one_or_none()
        if s3_blob is None:
            raise HTTPException(status_code=404, detail=f"s3_blob for job {job_id} was not found in database")
        # print(f"JOB STATUS: {job_status}")
        return s3_blob[0]
