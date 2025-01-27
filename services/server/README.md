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

Grafana dashboard sql for f1:
```sql
SELECT timestamp, f1 FROM model_metrics ORDER BY timestamp;
```
Grafana dashboard sql for f1_min:
```sql
SELECT timestamp, min_f1 FROM model_metrics ORDER BY timestamp;
```

## Further improvements

- [ ] Better handle class imbalance. Class weights implemented, but didn't do much of a difference. Test oversampling
- [ ] Consider more advanced pipeline management frameworks like Airflow for more complicated pipelines management
- [ ] Better model handling. Add support for models versioning and adding to model registry at MLFlow