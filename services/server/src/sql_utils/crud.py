from sqlalchemy import text
from sqlalchemy.exc import OperationalError


from fastapi import HTTPException

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

        session.commit()

    except OperationalError as e:
        print("Database connection lost:", e)
        raise HTTPException(status_code=500, detail="Database connection error")

