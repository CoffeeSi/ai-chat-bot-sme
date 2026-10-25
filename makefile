PYTHON := uv run python

.PHONY: eval api

eval:
	uv run python eval/run_eval.py answers --questions eval/questions.csv --runs 1 --pause 3

api:
	uv run uvicorn rmt_assignment3_project.api.api:app --reload

parse_ozon_products:
	uv run python .\src\rmt_assignment3_project\parsers\ozon_products.py https://ozon.kz/seller/dameli-3283894/ --max-pages 50 --delay 0

parse_ozon_questions_from_db:
	uv run python .\src\rmt_assignment3_project\parsers\ozon_questions.py --from-db --delay 0