CREATE DATABASE IF NOT EXISTS job_db;
CREATE USER IF NOT EXISTS 'job_db_user'@'%' IDENTIFIED BY 'job_db_password';
GRANT ALL PRIVILEGES ON job_db.* TO 'job_db_user'@'%';
FLUSH PRIVILEGES;