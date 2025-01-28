## Approach overview

This repository contains an implementation of products classification task. This solution consists of :
* minio ()
* 

## Installation

The solution is tested on UBUNTU 22.04 with NVIDIA gtx 3060-ti. nvidia docker plugin has to be installed, so as docker compose
```commandline
# clone repository
git clone https://github.com/dav1nci/product_classification.git
# enter the services folder
cd product_classification/services/
docker compose up --build
```
It takes some time for all services to spin up, apply migrations, up to 5 minutes.

### Part 1. Model Training and Fine-Tuning

The experiments notebook is located https://github.com/dav1nci/product_classification/blob/master/experiments/notebooks/berf-funetuning.ipynb

I found class imbalance in training dataset, and I applied class weights for training, however it didn't help much, so the others ways should be considered. I'd start with oversampling. I went with bert-base-uncased model from huggingface, fine-tuning this model on given dataset. For experiment tracking I used MLFlow that is running in separate docker container. For evaluation metric I decided to use f1 weighted among all classes, and min_f1 which is a minimum f1 calculated separately for each class. min_f1 gives an intuition when model is being affected by class imbalance, when this metric goes down on training plots.

### Part 2. Model Serving Library

Model serving and model retraining are wrapped up into REST API, using FastAPI python library, with 3 endpoints:

```json
POST http://localhost:8000/v1/predict/csv

// Example input json
{
    "input_file_s3": "s3://dataset/human_feedback/Query_and_Validation_Data.csv"
}
```
```json
POST http://localhost:8000/v1/train/csv

// Example input json
{
    "train_file_s3": "s3://dataset/train/Training_Data.csv",
    "epoch_num": 2,
    "best_model_metric": "eval_f1_min"
}
```
```json
POST http://localhost:8000/v1/train/human_feedback
        
// Example input json
{
    "train_dir_s3": "s3://dataset/train/",
    "epoch_num": 2,
    "best_model_metric": "eval_f1_min"
}
```

the `s3` locations mentioned in input jsons are referring to local minio instance, that is a separate service described in docker compose yaml.

The overall approach of a solution is that each time inference endpoint is triggered, system creates a job as an instance of InferencePipeline class, defined [here](https://github.com/dav1nci/product_classification/blob/master/services/server/src/serving/inference_pipeline.py#L19). It checks if there's any recent model available in database, and if there's any, it pulls it locally, and updates the inference model weights with newer model. 

The same with model retraining, the job class for model retraining defined [here](https://github.com/dav1nci/product_classification/blob/master/services/server/src/training/model_trainer.py#L17)

The inference predictions are stored in local s3(minio). Inference job information is stored at mysql database, that is also part of docker compose services. 

When training endpoint is triggered, metrics and model weights are logged to docker compose instance of MLFlow. 

The log file is located at `services/log` directory, that is being mapped as docker volume when container spins up.

### Part 3. Model Monitoring and Feedback Integration

I used grafana for model monitoring, that is included in this solution as separate service in docker compose. I was looking for the ways to fully automate grafana setup, it can be automated, but there's one thing that can't be automated, which is generating access token. In order to setup visualizations you need 2 steps:
1. Going to grafana UI and setting up the API token
2. Running `init_grafana.sh` script at `services/server/`

#### 3.1 Generating token in grafana UI

1. Go to http://localhost:3000/login
2. Log in using username `admin` and password `admin` credentials
3. Press `Skip` in the following dialog
4. On the left panel go to `Administration` -> `Users and access` -> `Service accounts` and click on `Add service account`
5. In the `Display name` field type in `test`, and in `Role` choose `Admin` and click `Create`
6. Click `Add service account token` button, type in some name in `Display Name` field, and then `Generate Token` in the pop up dialogue
7. Click `Copy to clipboard` and close the pop up window.

#### 3.2 Run `init_grafana.sh`

1. In console, and execute 
```bash
docker exec -ti services-server-1 bash
/server/init_grafana.sh TOKEN_ID
```

Check http://localhost:3000/dashboards, there has to be `Model Monitoring Dashboard` created, and if you click on it you can see visualization of `f1` and `min_f1` metrics over time

When it comes to human feedback integration, I would approach it this way. Each time `/predict/csv` endpoint is triggered, and if `HUMAN_VERIFIED_Category` is present in that csv, human verified subset of that input csv is copied to s3 location. After some time, a day, or a week, or a month, all these human verified subsets are accumulated into training dataset, that can be used for model retraining. Ideally, there has to be CronJob scheduled with some time interval that is running finetuning continiously after some period. 
### Part 4. Containerizing

The solution is containerized into docker image, and locally deployed using docker compose.

### Part 5. Deployment Design Flowchart

Flowcharts can be found at `flowchart` folder. `kubernetes_components.png` represents my vision on how I would approach configuring this solution in kubernetes. `human_feedback_loop.png` shows how I would approach integrating human feedback into training process.

## Further improvements

- [ ] Better handle class imbalance. Class weights implemented, but didn't do much of a difference. Test oversampling and other techniques
- [ ] Consider more advanced pipeline management frameworks like Airflow for more complicated pipelines management
- [ ] Better production model handling. Add support for models versioning and adding model to model registry at MLFlow

