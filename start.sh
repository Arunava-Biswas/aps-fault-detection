#!bin/sh
# to use airflow as a schedular, and 2nd one will provide the UI
nohup airflow scheduler &
airflow webserver