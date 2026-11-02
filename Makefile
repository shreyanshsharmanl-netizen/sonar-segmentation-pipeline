check:
	ruff check . && pytest
pipeline:
	PYTHONPATH=src python -m sonarseg.pipeline
experiments:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
serve:
	PYTHONPATH=src uvicorn sonarseg.api:app --reload
