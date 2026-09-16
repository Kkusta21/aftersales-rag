.PHONY: install test eval api ui mcp docker

install:
	pip install -e ".[dev,ui]"

test:
	pytest -q

eval:
	python eval/run_eval.py --write

api:
	uvicorn aftersales_rag.api:app --reload

ui:
	streamlit run app/streamlit_app.py

mcp:
	aftersales-mcp

docker:
	docker build -t aftersales-rag . && docker run --rm -p 8000:8000 --env-file .env aftersales-rag
