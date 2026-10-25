PYTHON := uv run python

.PHONY: eval api

sync:
	uv sync --extra embeddings

api:
	uv run --extra embeddings uvicorn rmt_assignment3_project.api.api:app --reload

eval:
	uv run --extra embeddings python eval/run_eval.py answers --questions eval/questions_answerable.csv --runs 1 --pause 3

parse_ozon_products:
	uv run python .\src\rmt_assignment3_project\parsers\ozon_products.py https://ozon.kz/seller/dameli-3283894/ --max-pages 50 --delay 0

parse_ozon_questions_from_db:
	uv run python .\src\rmt_assignment3_project\parsers\ozon_questions.py --from-db --delay 0

rag_index:
	uv run python -m rmt_assignment3_project.rag.index

rag_index_e5:
	uv run --extra embeddings python -m rmt_assignment3_project.rag.index --embedder e5 --out rag_index_e5

compare_search:
	uv run --extra embeddings python eval/compare_search.py
