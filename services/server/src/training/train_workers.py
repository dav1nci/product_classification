from sql_utils.crud import finalize_training_in_db
from training.model_trainer import AutomaticModelTrainer
from dependencies import get_db_session, logger


def regular_train_worker(trainer: AutomaticModelTrainer):
    try:
        trainer.run()
        db_session_generator = get_db_session()
        db_session = next(get_db_session())

        finalize_training_in_db(db_session,
                                trainer.db_record_id,
                                trainer.run_id,
                                trainer.best_step_info['best_checkpoint'],
                                trainer.best_step_info['best_f1'],
                                trainer.best_step_info['best_f1_min'],
                                'weights_s3_test')
        db_session_generator.close()

    except Exception as e:
        logger.error(f"Error during training process: {e}")


