#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Missing API_TOKEN parameter."
    echo "Usage: $0 <API_TOKEN>"
    echo "Example: $0 abc123xyz"
    exit 1
fi

API_TOKEN=$1
echo ${API_TOKEN}

curl -X POST http://grafana:3000/api/datasources \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MySQL Data Source",
    "type": "mysql",
    "access": "proxy",
    "url": "db:3306",
    "user": "job_db_user",
    "database": "job_db",
    "secureJsonData": {
      "password": "job_db_password"
    },
    "isDefault": true
  }'

curl -X POST http://grafana:3000/api/dashboards/db \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "dashboard": {
      "id": null,
      "title": "Model Monitoring Dashboard",
      "panels": [
        {
          "type": "timeseries",
          "title": "F1 Over Time",
          "targets": [
            {
              "refId": "A",
              "rawSql": "SELECT timestamp, f1 FROM model_metrics ORDER BY timestamp;",
              "format": "table"
            },
            {
              "refId": "B",
              "rawSql": "SELECT timestamp, min_f1 FROM model_metrics ORDER BY timestamp;",
              "format": "table"
            }
          ],
          "gridPos": { "x": 0, "y": 0, "w": 12, "h": 8 }
        }
      ]
    },
    "overwrite": true
  }'