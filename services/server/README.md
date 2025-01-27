## Approach overview

This repository contains an implementation of products classification task. This solution consists of :
* minio ()


### Part 1. Model Training and Fine-Tuning



```sql
mysql> select *  from auto_training;
+--------------------------------------+---------------------+------------------------------------------+--------------------------------------+----------------------------------+-----------------+----------+--------+-----------------+----------+
| id                                   | training_date       | dataset_hash                             | dataset_path                         | run_id                           | best_checkpoint | f1       | min_f1 | weights_s3      | status   |
+--------------------------------------+---------------------+------------------------------------------+--------------------------------------+----------------------------------+-----------------+----------+--------+-----------------+----------+
| 20713186-7661-4883-9ea6-4b6624d11a75 | 2025-01-25 21:12:47 | 06d184b35e4a3ba4a6c9f7143157997fa3e7e13f | s3://dataset/train/Training_Data.csv | 99ce50ef3b8147c1a1e7739bfe86d5c6 | checkpoint-25   | 0.330085 |      0 | weights_s3_test | FINISHED |
+--------------------------------------+---------------------+------------------------------------------+--------------------------------------+----------------------------------+-----------------+----------+--------+-----------------+----------+

```

### Part 3. Model Monitoring and Feedback Integration

I used grafana for model monitoring, that is included in this solution as separate service. I was looking for the ways to fully automate grafana setup, it can be automated, but there's one thing that can't be automated, which is generating access token. In order to setup visualizations you need 2 steps:
1. Going to grafana UI and setting up the API token
2. Running `init_grafana.sh` script wit 

#### 3.1 Generating token in grafana UI

1. Go to http://localhost:3000/login
2. Log in using username `admin` and password `admin` credentials
3. Press `Skip` in the following dialog
4. On the left panel go to `Administration` -> `Users and access` -> `Service accounts` and click on `Add service account`
5. In the `Display name` field type in `test`, and in `Role` choose `Admin` and click `Create`
6. Click `Add service account token` button, type in some name in `Display Name` field, and then `Generate Token` in the pop up dialogue
7. Click `Copy to clickboard` and close the pop up window.

#### 3.2 Run `init_grafana.sh`

1. In console, and execute 
```bash
docker exec -ti services-server-1 bash
/server/init_grafana.sh TOKEN_ID
```

Check http://localhost:3000/dashboards, there has to be `Model Monitoring Dashboard` created, and if you click on it you can see visualization of `f1` and `min_f1` metrics over time

### Part 4. Containerizing

```bash
cd services
docker compose up --build
```

## Further improvements

- [ ] Better handle class imbalance. Class weights implemented, but didn't do much of a difference. Test oversampling
- [ ] Consider more advanced pipeline management frameworks like Airflow for more complicated pipelines management
- [ ] Better model handling. Add support for models versioning and adding to model registry at MLFlow