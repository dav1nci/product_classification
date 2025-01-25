from sql_utils.crud import finalize_training_in_db
from training.model_trainer import AutomaticModelTrainer
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from dependencies import get_db_session



def worker(trainer: AutomaticModelTrainer):
    trainer.run()
    db_session_generator = get_db_session()  # Initialize the generator
    db_session = next(db_session_generator)  # Get the session instance

    # Use the session for database operations
    print(type(db_session))  # Should print <class 'sqlalchemy.orm.session.Session'>

    # When done, remember to close the generator to clean up resources

    # with get_db_session() as db_session:
    finalize_training_in_db(db_session,
                            trainer.db_record_id,
                            trainer.run_id,
                            trainer.best_step_info['best_checkpoint'],
                            trainer.best_step_info['best_f1'],
                            trainer.best_step_info['best_f1_min'],
                            'weights_s3_test')

    db_session_generator.close()



# "best_checkpoint": f"checkpoint-{metric_history[max_metric_index].step}",
#         "best_f1": f1_history[max_metric_index].value,
#         "best_f1_min":