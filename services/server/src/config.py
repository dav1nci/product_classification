from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Inference Model config
    INFERENCE_BATCH_SIZE: int
    SERVING_MODEL_ID: str
    TOKEN_MAX_LENGTH: int

    # DB config settings
    MYSQL_JOB_DB: str
    MYSQL_JOB_DB_USER: str
    MYSQL_JOB_DB_PASSWORD: str
    MYSQL_PORT: int

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True


# settings = Settings()
