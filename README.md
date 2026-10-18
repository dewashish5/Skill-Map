<div align="center">

<img src="assets/logo_lockup.png" alt="SkillMap" width="320">

### AI-Enabled Labour Market Intelligence and Skill Demand-Supply Forecasting Engine
**Smart India Hackathon 2026 | Problem Statement 26246 | Ministry of Skill Development and Entrepreneurship**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?logo=mysql&logoColor=white)](https://www.mysql.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-XGBoost-2F9E44)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-425B9A)](https://shap.readthedocs.io/)
[![Gemini](https://img.shields.io/badge/Gemini-optional-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Languages](https://img.shields.io/badge/UI-7_Indian_languages-76C0EC)](#multi-language-and-voice)

*District-level skill forecasting | Shortage and oversupply risk | Seat-allocation recommendations | A data-grounded assistant that speaks seven languages*

</div>

---

## The Problem

India trains millions of people every year, but training seats are planned without a district-level view of what employers will need. Some districts train too few people for fast-growing trades while others keep filling seats for trades that already have more certified people than vacancies. The result is both unfilled jobs and unemployed trainees.

PS 26246 asks for an engine that forecasts skill demand and supply, flags where they will diverge, and tells planners what to do about it. SkillMap does this for every district and skill, at 3, 6 and 12 month horizons, and explains each recommendation.

---

## System Architecture and Workflow

```
 Notebook (feature engineering, models, SHAP, back-tests)
        |
        v
 lmis_bundle.pkl  --load_db.py-->  MySQL 8  <--- SQLAlchemy ---  FastAPI (api.py)
        |                                                            |
        |   (no MySQL? app reads the pkl directly,                   |  REST + /docs
        |    assistant uses a local SQLite copy)                     v
        +--------------------------------------------------->  Streamlit (app.py)
                                                                Dashboard | Recommendation Engine | Assistant
                                                                          |
                                                                optional: Gemini rephrases facts
                                                                fetched from the database
```

### How It Works

1. **Data preparation** - The supplied NCS-based demand dataset is cleaned (duplicates, spellings, dates, missing months, logical inconsistencies, outliers) and every fix is logged in a data-quality audit that is shown in the app.
2. **Forecasting** - For every district x skill series, demand and supply are forecast at 3, 6 and 12 months. Linear regression, LightGBM, XGBoost, an ensemble, ARIMA and naive baselines are compared; the best model per horizon is selected on a rolling-origin back-test.
3. **Gap and risk** - Forecast demand minus forecast supply gives the skill gap. A gap index and policy bands convert it into a risk class: Low, Medium, High, Critical (shortage) or oversupply.
4. **Explainability** - SHAP drivers show why a series is forecast to rise or fall, and an XGBoost classifier gives early-warning probabilities of each risk class.
5. **Recommendations** - A policy layer turns risk into a seat action: increase capacity, fill existing seats first, reduce, or maintain. It respects a target utilisation and expansion/reduction limits, and suggests same-district reallocation from oversupplied trades to short ones.
6. **Assistant** - Questions are answered from the database (state, district, skill, sector, horizon, ranking, comparison, method). Optional Gemini only rephrases the facts that were fetched.

---

## Application Screenshots

### Dashboard Overview
> Filters for state, sector, skill, horizon and demand scenario. KPI cards, key insights (biggest shortage, biggest oversupply, most pressured state, seats to move) and a quick-start guide.

![Dashboard overview](docs/screenshots/dashboard_overview.png)

---

### Map and Drill-down
> India tile map coloured by gap index, high and critical share, or net gap. Click a state to open its districts and skills.

![Map and drill-down](docs/screenshots/map_drilldown.png)

---

### Gap and Risk
> Where demand and supply diverge, ranked by risk class, with the series that need action first.

![Gap and risk](docs/screenshots/gap_risk.png)

---

### Explainability
> SHAP drivers behind the forecasts and early-warning probabilities for each risk class.

![Explainability](docs/screenshots/explainability.png)

---

### Accuracy
> Rolling-origin back-test of every model against naive baselines, classifier scores and the data-quality audit.

![Accuracy](docs/screenshots/accuracy.png)

---

### Recommendation Engine - Action Plan
> Seat decisions for every district and skill, ranked by priority (P1 act now, onwards), with the number of seats to add or release.

![Recommendation action plan](docs/screenshots/recommendation_action_plan.png)

---

### What-if Simulator
> Add or remove seats, or shock demand, and see the new gap and risk instantly before committing to a decision.

![What-if simulator](docs/screenshots/what_if.png)

---

### SkillMap Assistant
> A chatbot that knows every number on the site. Ask in English, Hinglish or by voice. Answers come from the database, with a data table behind each one. Chat history is stored in SQL, with search, new chat and clear history.

![SkillMap Assistant](docs/screenshots/assistant.png)

---

## Feature Summary

| Area | What it does |
|------|--------------|
| **Dashboard** | Overview, map and drill-down, demand and supply, gap and risk, explainability, accuracy, methodology |
| **Recommendation Engine** | Action plan, district planner, what-if simulator, budget optimiser, export |
| **Targets export** | Next-cycle seat targets per district and trade as Excel, CSV and a planner-brief PDF |
| **Assistant** | Data-grounded chat, follow-up suggestions, data tables, voice input and spoken replies, SQL chat history |
| **Accessibility** | Text size, high contrast and reduced-motion options from the "Aa" menu |
| **Languages** | English, Hindi, Marathi, Tamil, Telugu, Bengali, Gujarati |
| **Speed** | Cached queries, gzip, lazy tab rendering, loading animation while data updates |

---

## Forecasting Results

Back-test on 9,072 rolling-origin forecasts (63 districts x 12 skills series over the history Apr 2022 to Mar 2026). Full tables are in the Accuracy tab.

| Item | Result |
|------|--------|
| Demand model selected | LightGBM at 3 and 6 months, Ensemble (LightGBM + XGBoost) at 12 months |
| Demand WAPE (fixed split, 756 series) | about 7.3 to 8.5 percent, against 11 to 15 percent for naive baselines |
| Risk classifier (XGBoost) | Accuracy 0.61, macro F1 0.59, against 0.58 and 0.56 for "risk stays as today" |
| Within one risk class | about 91 percent |
| Demand scenarios | Low, Base, High (growth of about -8.6, +7.3 and +35.9 percent) |

---

## Multi-language and Voice

- Switch the whole interface between seven languages from the navigation bar. Fonts for every script are loaded automatically.
- Voice input uses the browser Web Speech API with a full-screen listening overlay. Replies can be read aloud, with voice settings in the assistant sidebar.
- The assistant understands English and Hinglish. Questions typed or spoken in Tamil, Telugu, Bengali or Gujarati are best answered when `GEMINI_API_KEY` is set.
- Translations were written by the team and should be reviewed by native speakers before public release.

---

## Quick Start

> **Requirements:** Python 3.9+ (3.10 or newer recommended). MySQL is optional.

### Option A - Frontend only (no MySQL)

```sh
git clone <your-repo-url>
cd skillmap
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**. If the API is not running, the app reads `lmis_bundle.pkl` directly. The Assistant uses a local SQLite copy (`skillmap_local.db`) built on first use.

### Option B - Full stack (MySQL + FastAPI + Streamlit)

```sh
# 1. Start MySQL 8 (or use Docker)
docker compose up -d

# 2. Configure environment
cp .env.example .env        # edit values if needed

# 3. Install and load the database
pip install -r requirements.txt
python load_db.py           # creates tables from schema.sql and loads lmis_bundle.pkl

# 4. Run API and app together
python run_all.py
```

Or run them separately:

```sh
uvicorn api:app --port 8000     # API docs at http://localhost:8000/docs
streamlit run app.py            # app at http://localhost:8501
```

On Windows use `run.bat`; on Linux and macOS use `./run.sh`.

### Rebuild the forecasts (optional)

Run `notebook/LMIS_Part1_Part2.ipynb` (about 3 minutes). It writes a fresh `lmis_bundle.pkl`; then run `python load_db.py` again. A ready bundle is already included.

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | SQLAlchemy URL of the MySQL database | `mysql+pymysql://skillmap:skillmap_pw@localhost:3306/skillmap` |
| `API_URL` | Where the Streamlit app finds the API | `http://127.0.0.1:8000` |
| `GEMINI_API_KEY` | Optional. Lets Gemini phrase assistant answers from database facts | empty (built-in answer engine) |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.0-flash` |
| `SKILLMAP_AUTOSTART` | Set to `1` to let the app start a local API itself | off |

> **Without a Gemini key:** the assistant still works fully with the built-in answer engine. Gemini is never used to invent numbers; it only rewrites facts already fetched from the database.

---

## API Endpoints

Interactive docs at `http://localhost:8000/docs`.

| Endpoint | Purpose |
|----------|---------|
| `GET /health`, `/meta`, `/areas`, `/summary` | Service status, metadata, available states, districts and skills, national summary |
| `GET /forecast` | Final output (brief format) for one district x skill |
| `GET /district/{district}` | All skills of a district |
| `GET /series`, `/data` | Filterable series list and the complete table for dashboards and export |
| `GET /history` | Monthly history and forecast points for a filter |
| `GET /high-risk` | Series at or above a risk level |
| `GET /recommendations` | Seat decisions ranked by priority |
| `POST /whatif` | Gap and risk after adding or removing seats, or shocking demand |
| `GET /explain` | SHAP drivers and early-warning probabilities |
| `GET /evaluation` | Back-test results, classifier scores, data-quality audit |
| `GET /targets` | Next-cycle seat targets per district and trade (JSON or CSV) |
| `GET /brief` | Area brief for a state, district, skill or district x skill |
| `POST /chat` | Ask the SkillMap Assistant |
| `GET /kb`, `/chat-log` | Knowledge-base documents, latest questions |
| `GET/PUT/DELETE /conversations` | Saved chat history |

---

## Database Schema

```
dim_state(state_id, state_name)
dim_district(district_id, state_id, district_name)
dim_skill(skill_id, skill_name, sector)
fact_series(series_id, state_id, district_id, skill_id, current_demand, forecast_3m/6m/12m + bands,
            supply, gap, gap_index and risk per horizon, recommendation, ...)
fact_history(series_id, month_date, demand_raw, demand_const, supply_certified, supply_enrolled, supply_placed)
kb_document(doc_id, category, title, content, keywords)     -- assistant knowledge base
app_meta(meta_key, meta_value)                               -- model info, policy, evaluation
chat_log(...)                                                -- every question asked to the assistant
chat_conversation(conv_id, title, day_key, msgs, updated_ms) -- saved chat history
v_series                                                     -- view joining facts with names
```

The full definition is in `schema.sql`.

---

## Engineering Principles

| Principle | Implementation |
|-----------|----------------|
| **Grounded answers** | The assistant retrieves facts with SQL and cites a data table; the LLM never produces numbers |
| **Parameterised SQL** | Queries go through SQLAlchemy with bound parameters |
| **Honest data labelling** | The footer and methodology state that supply is simulated |
| **Graceful degradation** | No API: read the pkl. No MySQL: local SQLite. No Gemini: built-in engine |
| **Back-tested, not assumed** | Every model is compared with naive baselines on rolling-origin splits |
| **Secrets stay in `.env`** | `.env` is git-ignored; `.env.example` holds safe defaults |

---

## Project Structure

```
skillmap/
|-- app.py                  # Streamlit frontend: Dashboard, Recommendation Engine, Assistant, voice, 7 languages
|-- api.py                  # FastAPI backend and the data-grounded assistant engine
|-- load_db.py              # Creates tables and loads lmis_bundle.pkl into MySQL
|-- local_db.py             # Builds the local SQLite copy used when MySQL is absent
|-- schema.sql              # MySQL / MariaDB schema
|-- run_all.py              # One-command start for API and app
|-- run.sh, run.bat         # Install, load and run helpers
|-- docker-compose.yml      # MySQL 8 container
|-- lmis_bundle.pkl         # Forecasts, SHAP, evaluation and model (output of the notebook)
|-- requirements.txt
|-- .env.example            # Safe credential template
|-- .streamlit/config.toml  # Theme
|-- assets/                 # SkillMap logos and favicon
|-- notebook/
|   |-- LMIS_Part1_Part2.ipynb   # Cleaning, features, models, SHAP, back-tests
|   `-- skillmap_dataset.csv     # Supplied NCS-based demand dataset
`-- docs/screenshots/       # Screenshots used in this README
```

---

## Honest Status

> **Supply data is simulated.** Demand comes from the supplied NCS-based dataset. No district-level supply registry was provided, so supply is simulated and anchored to real PMKVY FY2021-22 district volumes. Results demonstrate the method; they are not official statistics. Plugging in a real training-centre registry replaces the simulated series without changing the pipeline.

> **Translations** were produced by the team and need native-speaker review.

> **Voice** relies on the browser Web Speech API (best in Chrome and Edge) and depends on the device microphone and installed voices.

---

## Built With

| Technology | Role |
|-----------|------|
| [Streamlit](https://streamlit.io/) | Frontend, custom CSS, voice bridge |
| [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | REST API and assistant engine |
| [MySQL 8](https://www.mysql.com/) / SQLite | Primary database and local fallback |
| [SQLAlchemy](https://www.sqlalchemy.org/) + PyMySQL | Database access |
| [LightGBM](https://lightgbm.readthedocs.io/), [XGBoost](https://xgboost.readthedocs.io/), scikit-learn, statsmodels | Forecasting and risk classification |
| [SHAP](https://shap.readthedocs.io/) | Explainability |
| [Plotly](https://plotly.com/python/) | Charts and the India tile map |
| openpyxl, ReportLab | Excel and PDF target exports |
| [Google Gemini API](https://ai.google.dev/) | Optional phrasing of assistant answers |
| IBM Plex Sans / Mono, Noto Sans | Typography across all supported scripts |

---

<div align="center">

**SkillMap - the right skills, in the right district, at the right time.**

*SIH 2026 | PS 26246 | MSDE*

</div>