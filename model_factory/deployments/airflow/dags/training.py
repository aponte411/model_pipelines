import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.contrib.operators.kubernetes_pod_operator import \
    KubernetesPodOperator
from airflow.operators.bash_operator import BashOperator

default_args = {
    'owner': 'model-factory',
    'depends_on_past': False,
    'start_date': datetime(2015, 1, 1),
    'email': ['email@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=2)
}

dag = DAG('model-factory-pipeline',
          default_args=default_args,
          catchup=False,
          schedule_interval=None)

start_task = BashOperator(
    task_id='start-task',
    bash_command='echo {{ ds }}',
    dag=dag,
)

train_model = KubernetesPodOperator(namespace='default',
                                    image=os.environ.get("IMAGE_NAME"),
                                    image_pull_policy="Always",
                                    cmds=['python', 'training_pipeline.py'],
                                    name='training',
                                    task_id='training_models',
                                    get_logs=True,
                                    in_cluster=True,
                                    is_delete_operator_pod=True,
                                    resources={
                                        "request_cpu": 8,
                                        "limit_cpu": 12,
                                        "request_memory": "10000",
                                        "limit_memory": "30000",
                                    },
                                    dag=dag)

end_task = BashOperator(
    task_id='end-task',
    bash_command='echo {{ ds }}',
    dag=dag,
)

start_task >> train_model >> end_task
