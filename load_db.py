"""
SkillMap - load the notebook bundle (lmis_bundle.pkl) into MySQL.

    python load_db.py                       # uses DATABASE_URL from .env / environment
    python load_db.py --bundle lmis_bundle.pkl --url "mysql+pymysql://user:pass@localhost:3306/skillmap"

Creates (or re-creates) all tables from schema.sql and fills them:
dim_state, dim_district, dim_skill, fact_series, fact_history, kb_document, app_meta.
Only open a bundle that you created yourself (it is a pickle file).
"""
import argparse
import json
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

HERE = Path(__file__).resolve().parent


def load_env():
    f = HERE / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def run_schema(conn, path):
    sql = Path(path).read_text(encoding="utf-8")
    sql = re.sub(r"^--.*$", "", sql, flags=re.M)
    for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
        if stmt.upper().startswith(("CREATE DATABASE", "USE ")):
            continue                                   # the database is selected by DATABASE_URL
        conn.execute(text(stmt))


def build_kb(B):
    """Knowledge-base chunks the assistant retrieves from. Numbers are taken from the bundle itself."""
    M, E = B["meta"], B["evaluation"]
    pol = B["policy"]
    fx = pd.DataFrame(E["demand_fixed"]); sfx = pd.DataFrame(E["supply_fixed"])
    sel_d, sel_s = M["demand_selected"], M["supply_selected"]
    name2code = {v: k for k, v in M["model_names"].items()}

    def mape(df, h, label):
        code = name2code.get(label, label)
        r = df[(df["Horizon"] == int(h)) & (df["Model"] == code)]
        return float(r["MAPE"].iloc[0]) if len(r) else float("nan")

    d_acc = ", ".join(f"{h}M {mape(fx, h, sel_d[h]):.1f}%" for h in ("3", "6", "12"))
    s_acc = ", ".join(f"{h}M {mape(sfx, h, sel_s[h]):.1f}%" for h in ("3", "6", "12"))
    clf = pd.DataFrame(E["classifier"]).set_index("model")
    ca = float(clf.loc["XGBoost classifier", "Accuracy"]); cp = float(clf.loc["Persistence (risk stays as today)", "Accuracy"])
    sh, ov = pol["shortage_bands"], pol["oversupply_bands"]
    sim = M.get("supply_is_simulated", True)
    tm = M["target_months"]
    docs = [
        ("overview", "What is SkillMap", "SkillMap is an AI-enabled labour market intelligence platform for India (Smart India Hackathon 2026, problem statement 26246, Ministry of Skill Development and Entrepreneurship). It forecasts skill demand and training supply for every district and skill, calculates the skill gap, classifies risk as Low, Medium, High or Critical, and recommends whether to increase, reduce or maintain training seats.",
         "skillmap platform about what is overview purpose sih msde"),
        ("overview", "Coverage of the platform", f"The current bundle covers {M['n_series']} district x skill series: {M['n_states']} states, {M['n_districts']} districts and {M['n_skills']} skills. Forecasts start from {M['last_data_month']} and target {tm['3']} (3 months), {tm['6']} (6 months) and {tm['12']} (12 months).",
         "coverage states districts skills series horizon months forecast origin"),
        ("data", "Demand data", "Demand is the monthly flow of job vacancies per district and skill, taken from the National Career Service based dataset provided for the hackathon (48 months, April 2022 to March 2026). Every April the NCS re-bases its counts, so SkillMap normalises history to constant latest-financial-year units before modelling (anchor normalisation).",
         "demand data vacancies ncs anchor normalisation rebase source dataset"),
        ("data", "Supply data", ("Supply is the monthly flow of newly certified, job-ready candidates produced by the training system of a district for a skill. IMPORTANT: no seat, enrolment, certification or placement registry was provided, so the supply panel in this version is SIMULATED. It is anchored to the real PMKVY trained volume of each district (FY2021-22) and driven by documented assumptions (course length, completion rate, growth). A real registry file can be plugged in without changing any code. Numbers about supply, gaps, risk and seats therefore demonstrate the method; they are not findings about real districts." if sim else "Supply is the monthly flow of newly certified, job-ready candidates produced by the training system of a district for a skill, taken from the supplied registry."),
         "supply data simulated real registry certified candidates pmkvy assumptions seats enrolment placement"),
        ("method", "Demand forecasting method", f"Demand is forecast 3, 6 and 12 months ahead per district and skill with a direct multi-horizon model on the log-ratio to the recent 3-month average. Candidates: Linear Regression baseline, LightGBM, XGBoost, their ensemble, ARIMA and naive benchmarks. The model per horizon is chosen by rolling-origin validation. Selected: 3M {sel_d['3']}, 6M {sel_d['6']}, 12M {sel_d['12']}.",
         "demand forecast model lightgbm xgboost arima linear regression ensemble rolling validation selected"),
        ("method", "Supply forecasting method", f"Supply uses the same discipline with pipeline-aware features: enrolment already in training (a leading indicator), seat utilisation, placement rate and market tightness. Selected models: 3M {sel_s['3']}, 6M {sel_s['6']}, 12M {sel_s['12']}.",
         "supply forecast model features enrolment pipeline utilisation placement"),
        ("method", "Skill gap and gap index", "Skill gap = forecast demand minus forecast supply, in people per month. A positive gap is a shortage, a negative gap is oversupply. The gap index = (demand - supply) / max(demand, supply, 1) ranges from -1 (pure oversupply) to +1 (pure shortage) and makes small and large districts comparable.",
         "skill gap formula gap index shortage oversupply demand minus supply"),
        ("method", "Risk classes", f"Risk is Low, Medium, High or Critical. For shortages the gap index thresholds are Medium at {sh[0]:.2f}, High at {sh[1]:.2f} and Critical at {sh[2]:.2f}. Oversupply is tolerated more (graduates can be re-skilled): Medium at {ov[0]:.2f}, High at {ov[1]:.2f}, Critical at {ov[2]:.2f}. Series with fewer than {pol['low_volume']} people per month are capped at Medium because small numbers are noisy. These bands are policy assumptions and can be changed in the Recommendation Engine.",
         "risk low medium high critical bands thresholds policy classification"),
        ("method", "Early-warning risk classifier", f"An XGBoost classifier predicts the risk class that will be realised 12 months ahead from demand, supply, pipeline and placement signals. On the hold-out year its accuracy was {ca:.0%} versus {cp:.0%} for the persistence baseline (risk stays as today). It is an early-warning aid next to the transparent rule-based headline risk.",
         "xgboost classifier early warning ml risk accuracy persistence baseline"),
        ("method", "Recommendation engine", f"The recommendation engine converts the 12-month gap into a seat decision. Shortage with unused seats (utilisation below {pol['fill_first_util']:.0%}): fill existing seats first. Shortage with full seats: increase training capacity by the seats needed to close the policy share of the gap (Medium {pol['close_fraction']['Medium']:.0%}, High {pol['close_fraction']['High']:.0%}, Critical {pol['close_fraction']['Critical']:.0%}), capped at +{pol['max_expansion']:.0%} per year. Oversupply: reduce seats by the policy share of the surplus (capped at -{pol['max_reduction']:.0%}). Balanced: maintain capacity. Freed seats are offered to shortage skills in the same district.",
         "recommendation engine seats increase reduce maintain fill existing seats reallocation cap policy"),
        ("method", "Seat formula and priority", f"Annual seats needed = extra certified people per month x 12 / (completion rate x target utilisation of {pol['target_utilisation']:.0%}). Priority = risk weight (Low 0.25, Medium 1, High 2, Critical 4) x absolute gap. P1 is the top 10 percent, P2 the next 20 percent.",
         "seat formula priority p1 p2 p3 completion rate utilisation weight"),
        ("method", "Confidence and prediction intervals", "Every forecast carries an 80 percent prediction interval calibrated on a hold-out year, by volume tier and volatility. Confidence = 1 / (1 + relative half-width), so low-volume and volatile series get lower confidence.",
         "confidence interval prediction uncertainty 80 percent calibrated"),
        ("method", "Demand scenarios", f"The next-year NCS anchor growth is unknowable, so SkillMap offers three demand scenarios: Low ({M['scenarios']['low']:+.1%}), Base ({M['scenarios']['base']:+.1%}) and High ({M['scenarios']['high']:+.1%}). Only demand is scaled; supply is not anchor-dependent.",
         "scenario low base high anchor growth ncs next year"),
        ("accuracy", "Forecast accuracy", f"Hold-out results on the last full financial year (fixed origin, MAPE): demand {d_acc}; supply {s_acc}. Supply accuracy is measured on the simulated registry and is therefore optimistic compared with what real data would show.",
         "accuracy mape error hold-out validation performance metrics mae rmse"),
        ("method", "Explainability with SHAP", "SHAP values show which factors push each supply forecast and each risk class up or down: recent certified volume, enrolment pipeline, seat utilisation, placement rate, demand pressure and seasonality. Contributions are additive: base value plus contributions equals the model output.",
         "shap explainability drivers factors why explain feature importance"),
        ("limits", "Limitations", "1) Supply data is simulated in this version. 2) Risk bands, close and reduce fractions and seat caps are policy assumptions, not estimated from data. 3) Demand forecasts carry the base NCS anchor assumption. 4) Recommendations are district-level seat decisions and ignore cost, trainer availability and trainee mobility. 5) The risk classifier is an aid, not a replacement for the rule-based risk.",
         "limitations assumptions caveats simulated honest"),
        ("usage", "Dashboard guide", "The Dashboard has Overview, Demand, Supply, Gap and Risk, Explainability, Model accuracy and Data and Method views, all filterable by state, skill, horizon and demand scenario. The Recommendation Engine is a separate workspace with Action plan, District planner, What-if simulator, Budget optimiser and Export. The Assistant answers questions about any state, district or skill.",
         "dashboard guide how to use workspace tabs filters assistant"),
        ("usage", "API guide", "The FastAPI backend reads from MySQL. Endpoints: /health, /meta, /summary, /areas, /forecast, /district/{name}, /series, /high-risk, /recommendations, /explain, /history, /brief, /whatif (POST) and /chat (POST). Interactive documentation is at /docs.",
         "api backend fastapi endpoints mysql database docs"),
        ("usage", "Architecture", "SkillMap connects five parts: the notebook (data cleaning, models, evaluation) writes one bundle file lmis_bundle.pkl; load_db.py loads it into MySQL; the FastAPI backend serves forecasts, risk, recommendations and the assistant from MySQL; the Streamlit frontend shows the dashboard, the recommendation engine and the assistant.",
         "architecture notebook pkl bundle mysql fastapi streamlit components flow"),
        ("glossary", "Glossary", "Demand: vacancies per month. Supply: newly certified candidates per month. Training capacity: monthly certification capacity = annual seats x completion rate / 12. Seat utilisation: enrolled / seats sanctioned. Placement rate: placed / certified. Shortage: demand above supply. Oversupply: supply above demand. P1: act now.",
         "glossary definitions terms training capacity utilisation placement rate"),
    ]
    return pd.DataFrame(docs, columns=["category", "title", "content", "keywords"])


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(HERE / "lmis_bundle.pkl") if (HERE / "lmis_bundle.pkl").exists() else str(HERE / "notebook" / "lmis_bundle.pkl"))
    ap.add_argument("--url", default=os.environ.get("DATABASE_URL", "mysql+pymysql://skillmap:skillmap_pw@localhost:3306/skillmap"))
    a = ap.parse_args()
    if not Path(a.bundle).exists():
        sys.exit(f"Bundle not found: {a.bundle}\nRun the notebook first (its last cell writes lmis_bundle.pkl).")
    with open(a.bundle, "rb") as fh:
        B = pickle.load(fh)
    S = pd.DataFrame(B["series"])
    H = B["history"]
    print(f"Bundle: {len(S)} series, created {B['meta']['created']}")

    eng = create_engine(a.url, pool_pre_ping=True)
    with eng.begin() as conn:
        run_schema(conn, HERE / "schema.sql")
        states = pd.DataFrame({"state_name": sorted(S["state"].unique())})
        states.to_sql("dim_state", conn, if_exists="append", index=False)
        sid = pd.read_sql(text("SELECT state_id, state_name FROM dim_state"), conn).set_index("state_name")["state_id"]
        d = S[["state", "district"]].drop_duplicates().sort_values(["state", "district"])
        pd.DataFrame({"state_id": d["state"].map(sid), "district_name": d["district"]}).to_sql("dim_district", conn, if_exists="append", index=False)
        k = S[["skill", "sector"]].drop_duplicates().sort_values("skill")
        pd.DataFrame({"skill_name": k["skill"], "sector": k["sector"]}).to_sql("dim_skill", conn, if_exists="append", index=False)
        did = pd.read_sql(text("SELECT district_id, district_name FROM dim_district"), conn).set_index("district_name")["district_id"]
        kid = pd.read_sql(text("SELECT skill_id, skill_name FROM dim_skill"), conn).set_index("skill_name")["skill_id"]

        F = S.drop(columns=["state", "district", "skill", "sector"]).copy()
        F.insert(1, "state_id", S["state"].map(sid).to_numpy())
        F.insert(2, "district_id", S["district"].map(did).to_numpy())
        F.insert(3, "skill_id", S["skill"].map(kid).to_numpy())
        F.to_sql("fact_series", conn, if_exists="append", index=False, chunksize=200)

        dates = pd.to_datetime([m + "-01" for m in H["dates"]])
        ids = list(H["series_ids"])
        arr = {k2: np.asarray(H[k2], float) for k2 in ["demand_nominal", "demand_const", "supply_certified", "supply_enrolled", "supply_placed"]}
        hist = pd.DataFrame({"series_id": np.tile(ids, len(dates)), "month_date": np.repeat(dates, len(ids)),
                             "demand_raw": arr["demand_nominal"].ravel(), "demand_const": arr["demand_const"].ravel(),
                             "supply_certified": arr["supply_certified"].ravel(), "supply_enrolled": arr["supply_enrolled"].ravel(),
                             "supply_placed": arr["supply_placed"].ravel()})
        hist.to_sql("fact_history", conn, if_exists="append", index=False, chunksize=5000)

        build_kb(B).to_sql("kb_document", conn, if_exists="append", index=False)
        meta = {k2: json.dumps(v, default=str) for k2, v in {"meta": B["meta"], "policy": B["policy"], "risk_order": B["risk_order"], "brief_fields": B["brief_fields"],
                                                "final_schema": B["final_schema"], "scenario_risk": B["scenario_risk"], "realloc": B["realloc"], "evaluation": {k: v for k, v in B["evaluation"].items() if k in ("demand_rolling", "supply_rolling", "demand_fixed", "supply_fixed", "classifier", "classifier_confusion", "demand_audit", "supply_audit")}}.items()}
        pd.DataFrame({"meta_key": list(meta), "meta_value": list(meta.values())}).to_sql("app_meta", conn, if_exists="append", index=False)

        for t_ in ["dim_state", "dim_district", "dim_skill", "fact_series", "fact_history", "kb_document", "app_meta"]:
            print(f"  {t_:14s} {conn.execute(text(f'SELECT COUNT(*) FROM {t_}')).scalar():>8,} rows")
    print("Done. Start the backend with:  uvicorn api:app --port 8000")


if __name__ == "__main__":
    main()