# SkillMap

AI-enabled labour market intelligence and skill demand-supply forecasting engine (SIH 2026, PS 26246, MSDE).

Architecture: notebook -> lmis_bundle.pkl -> load_db.py -> MySQL -> FastAPI (api.py) -> Streamlit (app.py) and the SkillMap Assistant.

## Run
1. Start MySQL 8 (or `docker compose up -d`) and create the database/user from `.env` (defaults: db `skillmap`, user `skillmap`, password `skillmap_pw`).
2. `pip install -r requirements.txt`
3. Run `notebook/LMIS_Part1_Part2.ipynb` (about 3 minutes); it writes `lmis_bundle.pkl`. A ready bundle is included at the project root.

No MySQL yet? `streamlit run app.py` alone works: if the API is not running, app.py reads lmis_bundle.pkl directly (Dashboard and Recommendation Engine). The Assistant also works: it builds a local SQLite copy (skillmap_local.db) from the pkl on first use. MySQL + API are optional and only needed for the full backend setup.
4. `python load_db.py`   (creates tables from schema.sql and fills them)
5. `python run_all.py`   (API on :8000, app on :8501). Or run separately: `uvicorn api:app --port 8000` and `streamlit run app.py`.

API docs: http://localhost:8000/docs

## Assistant
Questions are answered from MySQL (state, district, skill, sector, horizon, ranking, comparison, method). If `GEMINI_API_KEY` is set in `.env`, Gemini only rephrases facts fetched from the database; otherwise a built-in answer engine is used. Chats are stored in table `chat_log`.

## Note on data
Demand comes from the supplied NCS-based dataset. No supply registry was provided, so supply is simulated (anchored to PMKVY district volumes). Results demonstrate the method.
