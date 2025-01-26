from sqlalchemy.orm import Session
from sqlalchemy import text, insert
from sqlalchemy.exc import OperationalError


from fastapi import HTTPException

def add_inference_job(session, job_id, start_time, job_status, input_file_s3):
    """
    job_id CHAR(36) NOT NULL PRIMARY KEY, -- UUIDv4 stored as a CHAR(36)
        start_time DATETIME NOT NULL,
        end_time DATETIME,
        job_status VARCHAR(255) NOT NULL,
        input_file_s3 TEXT,
        output_file_s3 TEXT
    :param session:
    :param job_id:
    :param start_time:
    :param job_status:
    :param input_file_s3:
    :return:
    """
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


def finalize_inference_job_in_db(session, job_id, end_time, output_file_s3):
    q = """
        UPDATE jobs SET end_time = :end_time, output_file_s3 = :output_file_s3, job_status = 'FINISHED'
        WHERE job_id = :job_id;
        """
    try:
        session.execute(text(q), {'end_time': end_time,
                                  'output_file_s3': output_file_s3,
                                  'job_id': job_id
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
        session.commit()
    except OperationalError as e:
        print("Database connection lost:", e)
        raise HTTPException(status_code=500, detail="Database connection error")


def fetch_recent_model(session):
    q = """
    SELECT run_id, best_checkpoint 
    FROM auto_training 
    WHERE status = 'FINISHED' 
    ORDER BY training_date 
    DESC 
    LIMIT 1;
    """
    try:
        result = session.execute(text(q), {}).all()
        # result = session.all()
        if result:
            print("Most recent 'FINISHED' row:", result)
            return result
        else:
            print("No rows with status 'FINISHED' found.")
            return None

    except OperationalError as err:
        print("Error:", err)


def report_model_metrics_to_db(session, associated_job_id, timestamp, model_name,
                               f1, min_f1, total_predictions, latency_avg_ms):
    """
            CREATE TABLE model_metrics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            associated_job_id
            timestamp DATETIME NOT NULL,
            model_name VARCHAR(100),
            f1 FLOAT ,
            min_f1 FLOAT ,
            total_predictions INT,
            latency_avg_ms FLOAT
        );
            :return:
            """
    q = f"""
        INSERT INTO
            model_metrics (associated_job_id, timestamp, model_name, f1, min_f1, total_predictions, latency_avg_ms)
        VALUES
            (:associated_job_id, :timestamp, :model_name, :f1, :min_f1, :total_predictions, :latency_avg_ms);
        """
    try:
        session.execute(text(q), {'associated_job_id': associated_job_id,
                                  'timestamp': timestamp,
                                  'model_name': model_name,
                                  'f1': f1,
                                  'min_f1': min_f1,
                                  'total_predictions': total_predictions,
                                  'latency_avg_ms': latency_avg_ms,
                                  })

        # generated_id = session.fetchone()[0]
        session.commit()

    except OperationalError as e:
        print("Database connection lost:", e)
        raise HTTPException(status_code=500, detail="Database connection error")

