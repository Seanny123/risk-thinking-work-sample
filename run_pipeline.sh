prefect server start &
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
python train/folder_pipeline.py
