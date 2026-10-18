"""
Builds a local SQLite copy of the SkillMap database from lmis_bundle.pkl, so the Assistant works without MySQL.
Same tables and view as schema.sql; api.py runs unchanged on top of it.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

HERE = Path(__file__).resolve().parent


def build(bundle_path, db_path):
    from load_db import build_kb
    with open(bundle_path, "rb") as fh:          # created by this project's notebook
        B = pickle.load(fh)
    S = pd.DataFrame(B["series"])
    H = B["history"]
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    eng = create_engine(f"sqlite:///{db_path}")
    with eng.begin() as c:
        states = pd.DataFrame({"state_id": range(1, S["state"].nunique() + 1), "state_name": sorted(S["state"].unique())})
        states.to_sql("dim_state", c, index=False)
        sid = states.set_index("state_name")["state_id"]
        d = S[["state", "district"]].drop_duplicates().sort_values(["state", "district"]).reset_index(drop=True)
        dd = pd.DataFrame({"district_id": range(1, len(d) + 1), "state_id": d["state"].map(sid), "district_name": d["district"]})
        dd.to_sql("dim_district", c, index=False)
        did = dd.set_index("district_name")["district_id"]
        k = S[["skill", "sector"]].drop_duplicates().sort_values("skill").reset_index(drop=True)
        kk = pd.DataFrame({"skill_id": range(1, len(k) + 1), "skill_name": k["skill"], "sector": k["sector"]})
        kk.to_sql("dim_skill", c, index=False)
        kid = kk.set_index("skill_name")["skill_id"]
        F = S.drop(columns=["state", "district", "skill", "sector"]).copy()
        F.insert(1, "state_id", S["state"].map(sid).to_numpy())
        F.insert(2, "district_id", S["district"].map(did).to_numpy())
        F.insert(3, "skill_id", S["skill"].map(kid).to_numpy())
        F.to_sql("fact_series", c, index=False)
        dates = pd.to_datetime([str(m)[:7] + "-01" for m in H["dates"]])
        ids = list(H["series_ids"])
        arr = {x: np.asarray(H[x], float) for x in ["demand_nominal", "demand_const", "supply_certified", "supply_enrolled", "supply_placed"]}
        pd.DataFrame({"series_id": np.tile(ids, len(dates)), "month_date": np.repeat(dates, len(ids)).astype(str),
                      "demand_raw": arr["demand_nominal"].ravel(), "demand_const": arr["demand_const"].ravel(),
                      "supply_certified": arr["supply_certified"].ravel(), "supply_enrolled": arr["supply_enrolled"].ravel(),
                      "supply_placed": arr["supply_placed"].ravel()}).to_sql("fact_history", c, index=False)
        kb = build_kb(B)
        kb.insert(0, "doc_id", range(1, len(kb) + 1))
        kb.to_sql("kb_document", c, index=False)
        meta = {x: json.dumps(B[x]) for x in ["meta", "policy", "risk_order", "brief_fields", "final_schema", "scenario_risk", "realloc"] if x in B}
        meta["evaluation"] = json.dumps({k: v for k, v in B["evaluation"].items() if k in ("demand_rolling", "supply_rolling", "demand_fixed", "supply_fixed", "classifier", "classifier_confusion", "demand_audit", "supply_audit")}, default=str)
        pd.DataFrame({"meta_key": list(meta), "meta_value": list(meta.values())}).to_sql("app_meta", c, index=False)
        c.execute(text("CREATE TABLE chat_log (chat_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, question TEXT NOT NULL, intent TEXT, answer TEXT, used_llm INTEGER NOT NULL DEFAULT 0, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"))
        c.execute(text("CREATE VIEW v_series AS SELECT f.*, s.state_name AS state, d.district_name AS district, k.skill_name AS skill, k.sector AS sector "
                       "FROM fact_series f JOIN dim_state s ON s.state_id=f.state_id JOIN dim_district d ON d.district_id=f.district_id JOIN dim_skill k ON k.skill_id=f.skill_id"))
    return db_path