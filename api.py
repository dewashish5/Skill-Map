"""
SkillMap - FastAPI backend (reads MySQL) with the data-grounded SkillMap Assistant.

    uvicorn api:app --host 0.0.0.0 --port 8000        # docs: http://localhost:8000/docs

Configuration (environment or .env file next to this file):
    DATABASE_URL    mysql+pymysql://user:password@localhost:3306/skillmap
    GEMINI_API_KEY  optional - if set, the assistant lets Gemini phrase the answer, strictly from facts fetched from MySQL
    GEMINI_MODEL    optional - default gemini-2.0-flash
"""
import difflib
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text

HERE = Path(__file__).resolve().parent


def _load_env():
    f = HERE / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()
DATABASE_URL = os.environ.get("DATABASE_URL", "mysql+pymysql://skillmap:skillmap_pw@localhost:3306/skillmap")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
RISK = ["Low", "Medium", "High", "Critical"]
HZ = [3, 6, 12]

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=1800)


_QC: Dict[Any, Any] = {}          # small read cache: forecast tables are static between loads, chat tables are never cached
_QC_TTL = 300.0


def q(sql: str, **p) -> List[Dict[str, Any]]:
    cacheable = "chat_" not in sql
    if cacheable:
        key = (sql, tuple(sorted(p.items())))
        hit = _QC.get(key)
        if hit and time.time() - hit[0] < _QC_TTL:
            return [dict(r) for r in hit[1]]
    with engine.connect() as c:
        rows = [dict(r) for r in c.execute(text(sql), p).mappings().all()]
    if cacheable:
        if len(_QC) > 800:
            _QC.clear()
        _QC[key] = (time.time(), rows)
        return [dict(r) for r in rows]
    return rows


def q1(sql: str, **p) -> Optional[Dict[str, Any]]:
    r = q(sql, **p)
    return r[0] if r else None


def num(x, nd=0):
    try:
        return round(float(x), nd) if nd else int(round(float(x)))
    except Exception:
        return x


def fmt(n) -> str:
    """Indian digit grouping"""
    try:
        n = int(round(float(n)))
    except Exception:
        return str(n)
    s, neg = str(abs(n)), n < 0
    if len(s) > 3:
        head, tail, parts = s[:-3], s[-3:], []
        while len(head) > 2:
            parts.insert(0, head[-2:]); head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if neg else "") + s


def sgn(n) -> str:
    return f"+{fmt(n)}" if float(n) > 0 else fmt(n)


app = FastAPI(title="SkillMap API", version="2.0",
              description="Labour-market intelligence for India: demand and supply forecasts, skill gap, risk, recommendations and the SkillMap Assistant. Data source: MySQL.")
from fastapi.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ═════════════════════════════════════════════════════════════════════════════════════════════
#  catalog + meta (cached)
# ═════════════════════════════════════════════════════════════════════════════════════════════
_CAT: Dict[str, Any] = {}


def catalog(force=False) -> Dict[str, Any]:
    if _CAT and not force:
        return _CAT
    import json
    states = [r["state_name"] for r in q("SELECT state_name FROM dim_state ORDER BY state_name")]
    districts = q("SELECT d.district_name AS district, s.state_name AS state FROM dim_district d JOIN dim_state s ON s.state_id=d.state_id ORDER BY d.district_name")
    skills = q("SELECT skill_name AS skill, sector FROM dim_skill ORDER BY skill_name")
    meta = {r["meta_key"]: json.loads(r["meta_value"]) for r in q("SELECT meta_key, meta_value FROM app_meta WHERE meta_key <> 'evaluation'")}
    _CAT.update({"states": states, "districts": districts, "skills": skills, "meta": meta})
    return _CAT


def target_label(h: int) -> str:
    return catalog()["meta"]["meta"]["target_months"][str(h)]


# ═════════════════════════════════════════════════════════════════════════════════════════════
#  core queries
# ═════════════════════════════════════════════════════════════════════════════════════════════
BRIEF = ["district", "skill", "forecast_supply_3m", "forecast_supply_6m", "forecast_supply_12m", "training_capacity", "skill_gap", "risk", "recommendation"]
BRIEF_SQL = ("district, skill, supply_3m AS forecast_supply_3m, supply_6m AS forecast_supply_6m, supply_12m AS forecast_supply_12m, "
             "training_capacity, skill_gap, risk, recommendation")


def where(state=None, district=None, skill=None, risk=None, extra=None):
    cl, p = [], {}
    for col, val, key in (("state", state, "st"), ("district", district, "di"), ("skill", skill, "sk")):
        if val:
            cl.append(f"LOWER({col}) = LOWER(:{key})"); p[key] = val
    if risk:
        cl.append("risk = :rk"); p["rk"] = risk
    if extra:
        cl.append(extra)
    return (" WHERE " + " AND ".join(cl)) if cl else "", p


def clean(rows):
    out = []
    for r in rows:
        out.append({k: (float(v) if hasattr(v, "as_integer_ratio") and not isinstance(v, (int, bool)) else v) for k, v in r.items()})
    return out


@app.get("/health", tags=["info"])
def health():
    try:
        n = q1("SELECT COUNT(*) AS n FROM fact_series")["n"]
        return {"status": "ok", "database": "connected", "series": int(n), "llm": bool(GEMINI_KEY)}
    except Exception as e:
        raise HTTPException(503, f"database not reachable: {e}")


@app.get("/meta", tags=["info"])
def meta():
    return catalog()["meta"]


@app.get("/areas", tags=["info"], summary="States, districts and skills available")
def areas():
    c = catalog()
    return {"states": c["states"], "districts": c["districts"], "skills": c["skills"]}


@app.get("/summary", tags=["info"])
def summary(state: Optional[str] = None, skill: Optional[str] = None, district: Optional[str] = None):
    w, p = where(state, district, skill)
    r = q1(f"SELECT COUNT(*) n, SUM(forecast_12m) d12, SUM(supply_12m) s12, SUM(CASE WHEN seat_change>0 THEN seat_change ELSE 0 END) add_s, "
           f"SUM(CASE WHEN seat_change<0 THEN -seat_change ELSE 0 END) cut_s FROM v_series{w}", **p)
    rk = {x["risk"]: int(x["n"]) for x in q(f"SELECT risk, COUNT(*) n FROM v_series{w} GROUP BY risk", **p)}
    d, s = float(r["d12"] or 0), float(r["s12"] or 0)
    return {"series": int(r["n"]), "forecast_demand_12m": d, "forecast_supply_12m": s, "national_gap_12m": d - s,
            "gap_index": (d - s) / max(d, s, 1), "risk_counts": {k: rk.get(k, 0) for k in RISK},
            "seats_to_add": int(r["add_s"] or 0), "seats_to_reduce": int(r["cut_s"] or 0), "target_months": catalog()["meta"]["meta"]["target_months"]}


@app.get("/forecast", tags=["forecast"], summary="Final output (brief format) for one district x skill")
def forecast(district: str, skill: str, state: Optional[str] = None):
    w, p = where(state, district, skill)
    rows = q(f"SELECT {BRIEF_SQL}, state, current_demand, forecast_3m, forecast_6m, forecast_12m, demand_growth, confidence FROM v_series{w}", **p)
    if not rows:
        raise HTTPException(404, "unknown district / skill")
    return clean(rows) if len(rows) > 1 else clean(rows)[0]


@app.get("/district/{district}", tags=["forecast"], summary="All skills of a district")
def district_all(district: str, state: Optional[str] = None):
    w, p = where(state, district)
    rows = q(f"SELECT {BRIEF_SQL}, state, priority FROM v_series{w} ORDER BY priority_rank", **p)
    if not rows:
        raise HTTPException(404, "unknown district")
    return clean(rows)


@app.get("/series", tags=["forecast"], summary="Filterable list of series")
def series(state: Optional[str] = None, district: Optional[str] = None, skill: Optional[str] = None, risk: Optional[str] = None, limit: int = 100):
    if risk and risk.title() not in RISK:
        raise HTTPException(400, f"risk must be one of {RISK}")
    w, p = where(state, district, skill, risk.title() if risk else None)
    return clean(q(f"SELECT {BRIEF_SQL}, state FROM v_series{w} ORDER BY priority_rank LIMIT :lim", lim=max(1, min(limit, 1000)), **p))


@app.get("/high-risk", tags=["risk"], summary="Series at or above a risk level")
def high_risk(min_risk: str = "High", limit: int = 100, state: Optional[str] = None, skill: Optional[str] = None):
    if min_risk.title() not in RISK:
        raise HTTPException(400, f"min_risk must be one of {RISK}")
    lv = RISK.index(min_risk.title())
    w, p = where(state, None, skill, None, extra="risk IN (" + ",".join(f"'{r}'" for r in RISK[lv:]) + ")")
    rows = q(f"SELECT {BRIEF_SQL}, state, priority FROM v_series{w} ORDER BY priority_rank LIMIT :lim", lim=max(1, min(limit, 1000)), **p)
    total = q1(f"SELECT COUNT(*) n FROM v_series{w}", **p)["n"]
    return {"count": int(total), "items": clean(rows)}


@app.get("/recommendations", tags=["recommendation"], summary="Seat decisions ranked by priority")
def recommendations(state: Optional[str] = None, district: Optional[str] = None, skill: Optional[str] = None, priority: Optional[str] = None, limit: int = 100):
    w, p = where(state, district, skill)
    if priority:
        w += (" AND " if w else " WHERE ") + "priority LIKE :pr"; p["pr"] = priority.upper() + "%"
    return clean(q("SELECT state, district, skill, risk, recommendation, seat_change, priority, priority_rank, recommendation_detail "
                   f"FROM v_series{w} ORDER BY priority_rank LIMIT :lim", lim=max(1, min(limit, 1000)), **p))


@app.get("/explain", tags=["explainability"], summary="Main SHAP drivers and early-warning probabilities")
def explain(district: str, skill: str, state: Optional[str] = None):
    w, p = where(state, district, skill)
    r = q1(f"SELECT state, district, skill, supply_drivers_12m, risk_drivers, ml_risk, ml_prob_low, ml_prob_medium, ml_prob_high, ml_prob_critical FROM v_series{w}", **p)
    if not r:
        raise HTTPException(404, "unknown district / skill")
    return {"state": r["state"], "district": r["district"], "skill": r["skill"], "supply_drivers_12m": r["supply_drivers_12m"], "risk_drivers": r["risk_drivers"],
            "ml_risk": r["ml_risk"], "ml_probabilities": {k: float(r[f"ml_prob_{k.lower()}"]) for k in RISK}}


@app.get("/history", tags=["forecast"], summary="Monthly history and forecast points, aggregated over the filter")
def history(state: Optional[str] = None, district: Optional[str] = None, skill: Optional[str] = None):
    w, p = where(state, district, skill)
    w_v = re.sub(r"LOWER\((state|district|skill)\)", r"LOWER(v.\1)", w)
    rows = q(f"SELECT h.month_date, SUM(h.demand_const) demand, SUM(h.supply_certified) supply, SUM(h.supply_enrolled) enrolled "
             f"FROM fact_history h JOIN v_series v ON v.series_id=h.series_id{w_v} GROUP BY h.month_date ORDER BY h.month_date", **p) if w else \
        q("SELECT h.month_date, SUM(h.demand_const) demand, SUM(h.supply_certified) supply, SUM(h.supply_enrolled) enrolled FROM fact_history h GROUP BY h.month_date ORDER BY h.month_date")
    if not rows:
        raise HTTPException(404, "no data for this filter")
    f = q1(f"SELECT SUM(forecast_3m) d3, SUM(forecast_6m) d6, SUM(forecast_12m) d12, SUM(supply_3m) s3, SUM(supply_6m) s6, SUM(supply_12m) s12 FROM v_series{w}", **p)
    for r_ in rows:
        if isinstance(r_["month_date"], str):
            r_["month_date"] = datetime.fromisoformat(r_["month_date"][:10])
    last = rows[-1]["month_date"]
    fx = []
    for h in HZ:
        m = last.month - 1 + h
        fx.append(f"{last.year + m // 12}-{m % 12 + 1:02d}")
    return {"months": [r["month_date"].strftime("%Y-%m") for r in rows], "demand": [round(float(r["demand"]), 1) for r in rows],
            "supply": [round(float(r["supply"]), 1) for r in rows], "enrolled": [round(float(r["enrolled"]), 1) for r in rows],
            "forecast_months": fx, "forecast_demand": [float(f[f"d{h}"]) for h in HZ], "forecast_supply": [float(f[f"s{h}"]) for h in HZ]}


class WhatIf(BaseModel):
    district: str
    skill: str
    state: Optional[str] = None
    seats_delta: int = 0
    demand_shock_pct: float = 0.0


@app.post("/whatif", tags=["recommendation"], summary="Gap and risk after adding / removing seats or shocking demand")
def whatif(qy: WhatIf):
    w, p = where(qy.state, qy.district, qy.skill)
    r = q1(f"SELECT forecast_12m, supply_12m, completion_rate FROM v_series{w}", **p)
    if not r:
        raise HTTPException(404, "unknown district / skill")
    pol = catalog()["meta"]["policy"]
    comp = max(float(r["completion_rate"]), 0.3)
    D0, S0 = float(r["forecast_12m"]), float(r["supply_12m"])
    D1, S1 = D0 * (1 + qy.demand_shock_pct / 100.0), S0 + qy.seats_delta * comp * pol["target_utilisation"] / 12.0

    def risk(D, S):
        gi = (D - S) / max(D, S, 1.0)
        b = pol["shortage_bands"] if gi > 0 else pol["oversupply_bands"]
        a = abs(gi)
        lvl = 3 if a >= b[2] else 2 if a >= b[1] else 1 if a >= b[0] else 0
        if max(D, S) < pol["low_volume"]:
            lvl = min(lvl, 1)
        return gi, RISK[lvl]
    g0, r0 = risk(D0, S0); g1, r1 = risk(D1, S1)
    return {"before": {"demand": D0, "supply": S0, "gap": D0 - S0, "gap_index": g0, "risk": r0}, "after": {"demand": D1, "supply": S1, "gap": D1 - S1, "gap_index": g1, "risk": r1}}


# ═════════════════════════════════════════════════════════════════════════════════════════════
#  profiles (shared by /brief and the assistant)
# ═════════════════════════════════════════════════════════════════════════════════════════════
def kpi(label, value, sub=""):
    return {"label": label, "value": value, "sub": sub}


def risk_counts(w, p):
    rk = {x["risk"]: int(x["n"]) for x in q(f"SELECT risk, COUNT(*) n FROM v_series{w} GROUP BY risk", **p)}
    return {k: rk.get(k, 0) for k in RISK}


def chart_for(state=None, district=None, skill=None, title=""):
    try:
        h = history(state, district, skill)
    except HTTPException:
        return None
    return {"kind": "area", "title": title, **h}


def table_rows(rows, cols):
    return [{c: (float(r[c]) if isinstance(r[c], float) else r[c]) for c in cols} for r in rows]


def series_profile(r: Dict[str, Any]) -> Dict[str, Any]:
    gi = float(r["gap_index"])
    kind = "shortage" if r["skill_gap"] > 0 else "oversupply" if r["skill_gap"] < 0 else "balance"
    rows = []
    for h in HZ:
        rows.append({"Horizon": f"{h} months ({target_label(h)})", "Demand": int(r[f"forecast_{h}m"]), "Supply": int(r[f"supply_{h}m"]), "Gap": int(r[f"gap_{h}m"]), "Risk": r[f"risk_{h}m"]})
    md = (f"**{r['skill']} in {r['district']}, {r['state']}** ({r['sector']} sector)\n\n"
          f"- **Demand:** {fmt(r['current_demand'])} vacancies per month now; forecast {fmt(r['forecast_3m'])} (3M), {fmt(r['forecast_6m'])} (6M), {fmt(r['forecast_12m'])} (12M), "
          f"a {r['demand_growth']:+.0f}% change over 12 months (confidence {r['confidence']:.2f}).\n"
          f"- **Supply:** {fmt(r['current_supply'])} certified per month now; forecast {fmt(r['supply_3m'])} (3M), {fmt(r['supply_6m'])} (6M), {fmt(r['supply_12m'])} (12M). "
          f"Training capacity {fmt(r['training_capacity'])} per month, seat utilisation {r['seat_utilisation']:.0%}, placement rate {r['placement_rate']:.0%}.\n"
          f"- **Gap at 12 months:** {sgn(r['skill_gap'])} per month ({kind}), gap index {gi:+.2f}, risk **{r['risk']}**. Early-warning classifier: {r['ml_risk']} "
          f"({float(r['ml_prob_' + r['ml_risk'].lower()]):.0%}).\n"
          f"- **Recommendation:** {r['recommendation']}"
          + (f" ({sgn(r['seat_change'])} annual seats)" if r["seat_change"] else "") + f", priority {r['priority']}. {r['recommendation_detail']}\n"
          f"- **Main drivers:** supply forecast - {r['supply_drivers_12m']}; risk - {r['risk_drivers']}.")
    return {"title": f"{r['skill']} - {r['district']}", "markdown": md,
            "kpis": [kpi("Demand 12M", fmt(r["forecast_12m"]), "per month"), kpi("Supply 12M", fmt(r["supply_12m"]), "per month"),
                     kpi("Skill gap", sgn(r["skill_gap"]), kind), kpi("Risk", r["risk"], r["priority"])],
            "table": rows, "chart": chart_for(r["state"], r["district"], r["skill"], f"{r['skill']} in {r['district']}"),
            "facts": {"series": {k: (v if not hasattr(v, "isoformat") else str(v)) for k, v in r.items()}}}


def aggregate(w, p):
    return q1(f"SELECT COUNT(*) n, SUM(current_demand) cd, SUM(forecast_3m) d3, SUM(forecast_6m) d6, SUM(forecast_12m) d12, SUM(current_supply) cs, SUM(supply_3m) s3, SUM(supply_6m) s6, "
              f"SUM(supply_12m) s12, SUM(training_capacity) cap, SUM(CASE WHEN seat_change>0 THEN seat_change ELSE 0 END) add_s, SUM(CASE WHEN seat_change<0 THEN -seat_change ELSE 0 END) cut_s FROM v_series{w}", **p)


def gap_idx(d, s):
    return (d - s) / max(d, s, 1.0)


def area_profile(state=None, district=None, skill=None) -> Dict[str, Any]:
    w, p = where(state, district, skill)
    a = aggregate(w, p)
    if not a or not a["n"]:
        raise HTTPException(404, "no data for this area")
    d12, s12 = float(a["d12"]), float(a["s12"])
    gi = gap_idx(d12, s12)
    rc = risk_counts(w, p)
    if district:
        where_txt = f"{district} district" + (f" ({state})" if state else "")
        group_col = "skill"
    elif skill and state:
        where_txt, group_col = f"{skill} in {state}", "district"
    elif skill:
        where_txt, group_col = f"{skill} across India", "state"
    else:
        where_txt, group_col = f"{state}", "skill"
    if district and skill:
        where_txt, group_col = f"{skill} in {district}", "skill"
    top_short = q(f"SELECT state, district, skill, forecast_12m, supply_12m, skill_gap, risk, recommendation, seat_change FROM v_series{w}{' AND' if w else ' WHERE'} skill_gap>0 ORDER BY priority_score DESC LIMIT 3", **p)
    top_over = q(f"SELECT state, district, skill, forecast_12m, supply_12m, skill_gap, risk, recommendation, seat_change FROM v_series{w}{' AND' if w else ' WHERE'} skill_gap<0 ORDER BY priority_score DESC LIMIT 3", **p)
    grp = q(f"SELECT {group_col} AS name, SUM(forecast_12m) demand_12m, SUM(supply_12m) supply_12m, SUM(skill_gap) gap_12m, "
            f"SUM(CASE WHEN risk IN ('High','Critical') THEN 1 ELSE 0 END) high_risk, SUM(seat_change) net_seats FROM v_series{w} GROUP BY {group_col} ORDER BY SUM(skill_gap) DESC", **p)
    kind = "shortage" if d12 > s12 else "oversupply"
    lines = [f"**{where_txt}** - {int(a['n'])} district-skill series, forecast to {target_label(12)}.", "",
             f"- **Demand:** {fmt(a['cd'])} vacancies per month now; forecast {fmt(a['d3'])} (3M), {fmt(a['d6'])} (6M), {fmt(a['d12'])} (12M).",
             f"- **Supply:** {fmt(a['cs'])} certified per month now; forecast {fmt(a['s3'])} (3M), {fmt(a['s6'])} (6M), {fmt(a['s12'])} (12M); training capacity {fmt(a['cap'])} per month.",
             f"- **Net gap at 12 months:** {sgn(d12 - s12)} per month ({kind}), gap index {gi:+.2f}.",
             f"- **Risk mix:** " + ", ".join(f"{rc[k]} {k}" for k in RISK) + ".",
             f"- **Seat actions:** +{fmt(a['add_s'])} annual seats where short, -{fmt(a['cut_s'])} where over-supplied."]
    if top_short:
        lines += ["", "**Largest shortages**"] + [f"- {x['skill'] if group_col == 'skill' else x['skill'] + ', ' + x['district']}: gap {sgn(x['skill_gap'])} ({x['risk']}), {x['recommendation'].lower()}"
                                                     + (f", {sgn(x['seat_change'])} seats" if x["seat_change"] else "") for x in top_short]
    if top_over:
        lines += ["", "**Largest surpluses**"] + [f"- {x['skill'] if group_col == 'skill' else x['skill'] + ', ' + x['district']}: gap {sgn(x['skill_gap'])} ({x['risk']}), {x['recommendation'].lower()}"
                                                    + (f", {sgn(x['seat_change'])} seats" if x["seat_change"] else "") for x in top_over]
    table = [{group_col.title(): g["name"], "Demand 12M": int(g["demand_12m"]), "Supply 12M": int(g["supply_12m"]), "Gap 12M": int(g["gap_12m"]),
              "High / Critical": int(g["high_risk"]), "Net seats": int(g["net_seats"])} for g in grp]
    return {"title": where_txt, "markdown": "\n".join(lines),
            "kpis": [kpi("Demand 12M", fmt(d12), "per month"), kpi("Supply 12M", fmt(s12), "per month"), kpi("Net gap", sgn(d12 - s12), kind),
                     kpi("High / Critical", str(rc["High"] + rc["Critical"]), f"of {int(a['n'])} series")],
            "table": table, "chart": chart_for(state, district, skill, where_txt),
            "facts": {"area": where_txt, "aggregate": {k: float(v) for k, v in a.items() if v is not None}, "risk_counts": rc, "top_shortages": clean(top_short),
                      "top_surpluses": clean(top_over), "by_" + group_col: clean(grp)}}


@app.get("/brief", tags=["assistant"], summary="Area brief: details of a state / district / skill / district x skill")
def brief(state: Optional[str] = None, district: Optional[str] = None, skill: Optional[str] = None):
    if not (state or district or skill):
        raise HTTPException(400, "give at least one of state, district, skill")
    if district and skill:
        w, p = where(state, district, skill)
        r = q1(f"SELECT * FROM v_series{w}", **p)
        if not r:
            raise HTTPException(404, "unknown district / skill")
        return series_profile(r)
    return area_profile(state, district, skill)


# ═════════════════════════════════════════════════════════════════════════════════════════════
#  the SkillMap Assistant: entity extraction -> intent -> SQL facts -> answer (optionally phrased by Gemini)
# ═════════════════════════════════════════════════════════════════════════════════════════════
SKILL_ALIAS = {
    "EV Technician": ["ev", "electric vehicle", "electric vehicles", "ev technician"],
    "Solar PV Technician": ["solar", "solar pv", "photovoltaic"],
    "Data Analyst": ["data analyst", "data analytics", "analyst"],
    "General Duty Assistant": ["general duty", "gda", "nursing assistant", "nursing", "healthcare assistant", "health assistant"],
    "Lab Technician": ["lab technician", "lab tech", "laboratory", "pathology"],
    "Automotive Service Technician": ["automotive", "auto mechanic", "mechanic", "car service", "automobile"],
    "Electrician": ["electrician", "electricians", "wiring"],
    "Plumber": ["plumber", "plumbers", "plumbing"],
    "Welder": ["welder", "welders", "welding"],
    "Electrical Fitter": ["electrical fitter", "fitter", "fitters"],
    "Mason": ["mason", "masons", "masonry", "bricklayer"],
    "Data Entry Operator": ["data entry", "deo", "typist", "data operator"],
}
STATE_ALIAS = {"Delhi": ["delhi", "new delhi", "ncr"], "Uttar Pradesh": ["uttar pradesh", "up"], "Madhya Pradesh": ["madhya pradesh", "mp"],
               "Tamil Nadu": ["tamil nadu", "tn"], "West Bengal": ["west bengal", "bengal", "wb"], "Maharashtra": ["maharashtra"], "Karnataka": ["karnataka"]}
RANK_RE = re.compile(r"\b(top|highest|most|worst|biggest|largest|lowest|least|rank|ranking|list|which|where|who|best|leading|bottom)\b")
METRICS = [("oversupply", r"\b(oversupply|over supply|over-supplied|surplus|excess|too many)\b"),
           ("growth", r"\b(growth|growing|fastest|rising|increase in demand)\b"),
           ("shortage", r"\b(shortage|short|gap|deficit|critical|crisis|risk|risky|high risk|scarcity|unmet)\b"),
           ("priority", r"\b(recommend|recommendation|action|actions|priority|priorities|seats|invest|plan)\b"),
           ("demand", r"\bdemand\b"), ("supply", r"\bsupply\b")]
DATA_WORDS = re.compile(r"\b(demand|supply|gap|risk|forecast|detail|details|overview|tell|show|status|profile|summary|report|recommend|recommendation|action|seats|info|information|kitni|kya|batao|brief|data)\b")
DEFINE_RE = re.compile(r"\b(what is|what's|what are|define|meaning of|explain|how is|how are|how do|how does|why is|why are|methodology|method|formula|simulated|source|accuracy|shap|mape|what does)\b")
FOLLOW_RE = re.compile(r"\b(there|that district|that skill|same|it|this district|this skill|what about|and for|also)\b")
SESSIONS: Dict[str, Dict[str, Any]] = {}

SCOPE_MSG = ("I can only answer questions about the SkillMap platform and its data: demand, supply, skill gaps, risk and seat recommendations for Indian states, districts and skills, "
             "plus how the methodology works. Try: 'Electrician demand in Patna', 'top critical shortages in Bihar', 'details of Pune' or 'how is risk calculated'.")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s.lower())).strip()


def find_entities(msg: str) -> Dict[str, Any]:
    c = catalog()
    m = " " + norm(msg) + " "
    ent: Dict[str, Any] = {"state": None, "district": None, "skill": None}

    def has(token: str) -> bool:
        return f" {norm(token)} " in m
    # districts: longest match first (some names are common words: require whole-word match)
    for d in sorted(c["districts"], key=lambda x: -len(x["district"])):
        if len(d["district"]) >= 3 and has(d["district"]):
            ent["district"], ent["state"] = d["district"], d["state"]
            break
    if not ent["state"]:
        for s in sorted(c["states"], key=len, reverse=True):
            al = STATE_ALIAS.get(s, [s.lower()])
            if any(has(a) for a in al + [s]):
                ent["state"] = s
                break
    # skills: aliases, longest first
    best = None
    for sk, al in SKILL_ALIAS.items():
        for a in al + [sk.lower()]:
            if has(a) and (best is None or len(a) > best[1]):
                best = (sk, len(a))
    if best:
        ent["skill"] = best[0]
    # typo tolerance on single tokens / bigrams
    toks = norm(msg).split()
    grams = toks + [" ".join(toks[i:i + 2]) for i in range(len(toks) - 1)]
    if not ent["district"]:
        names = {norm(d["district"]): d for d in c["districts"] if len(d["district"]) >= 5}
        for g in grams:
            if len(g) >= 5:
                mt = difflib.get_close_matches(g, list(names), n=1, cutoff=0.86)
                if mt:
                    ent["district"], ent["state"] = names[mt[0]]["district"], names[mt[0]]["state"]
                    break
    if not ent["skill"]:
        allal = {norm(a): sk for sk, al in SKILL_ALIAS.items() for a in al if len(a) >= 5}
        for g in grams:
            if len(g) >= 5:
                mt = difflib.get_close_matches(g, list(allal), n=1, cutoff=0.86)
                if mt:
                    ent["skill"] = allal[mt[0]]
                    break
    return ent


def horizon_of(msg: str) -> int:
    m = norm(msg)
    if re.search(r"\b3 month|three month|quarter|next quarter\b", m): return 3
    if re.search(r"\b6 month|six month|half year|half yearly\b", m): return 6
    return 12


def limit_of(msg: str, default=5) -> int:
    mm = re.search(r"\b(?:top|first|bottom|best|worst)\s+(\d{1,2})\b", msg.lower()) or re.search(r"\b(\d{1,2})\s+(?:districts|skills|series|areas|states|results)\b", msg.lower())
    return max(1, min(int(mm.group(1)), 20)) if mm else default


def kb_search(msg: str, k=2):
    docs = q("SELECT doc_id, category, title, content, keywords FROM kb_document")
    qt = set(t for t in norm(msg).split() if len(t) > 2 and t not in {"what", "the", "and", "for", "how", "are", "does", "this", "that", "with", "about", "from", "tell", "can", "you", "your", "skillmap"})
    syn = {"accurate": "accuracy", "accurately": "accuracy", "precise": "accuracy", "reliable": "accuracy", "errors": "error", "trust": "accuracy", "models": "model", "forecasts": "forecast"}
    qt = qt | {syn[t] for t in qt if t in syn}
    scored = []
    for d in docs:
        text_ = norm(d["title"] + " " + d["content"] + " " + (d["keywords"] or ""))
        words = set(text_.split())
        title_w = set(norm(d["title"]).split())
        sc = sum(1.0 for t in qt if t in words) + 1.5 * sum(1.0 for t in qt if t in title_w) + sum(0.5 for t in qt if t in (d["keywords"] or "").lower())
        scored.append((sc, d))
    scored.sort(key=lambda x: -x[0])
    return [(s, d) for s, d in scored[:k] if s >= 1.5]


def ranking(ent, msg, metric, risk_only=None):
    h = horizon_of(msg)
    n = limit_of(msg)
    w, p = where(ent["state"], ent["district"], ent["skill"], risk_only)
    gcol, scol = f"gap_{h}m", f"supply_{h}m"
    dcol = f"forecast_{h}m"
    order, title, extra = "priority_score DESC", "Highest-priority actions", None
    if metric == "oversupply":
        extra, order, title = f"{gcol} < 0", f"{gcol} ASC", "Largest oversupply"
    elif metric == "shortage":
        extra, order, title = f"{gcol} > 0", f"priority_score DESC" if not risk_only else f"{gcol} DESC", "Largest shortages" if not risk_only else f"Largest {risk_only} shortages"
        if risk_only:
            extra = f"{gcol} > 0"
    elif metric == "growth":
        order, title = "demand_growth DESC", "Fastest-growing demand (12M)"
        extra = "low_volume_flag = 0"
    elif metric == "demand":
        order, title = f"{dcol} DESC", f"Highest demand ({h}M)"
    elif metric == "supply":
        order, title = f"{scol} DESC", f"Highest supply ({h}M)"
    if extra:
        w += (" AND " if w else " WHERE ") + extra
    rows = q(f"SELECT state, district, skill, {dcol} AS demand, {scol} AS supply, {gcol} AS gap, risk, recommendation, seat_change, priority FROM v_series{w} ORDER BY {order} LIMIT :lim", lim=n, **p)
    scope = ", ".join(x for x in [ent["skill"], ent["district"], ent["state"]] if x) or "India"
    if not rows:
        return {"title": title, "markdown": f"No series match that request for {scope}.", "table": [], "kpis": [], "chart": None, "facts": {}}
    md = [f"**{title} - {scope}** (horizon {h} months, {target_label(h)})", ""]
    for i, r in enumerate(rows, 1):
        md.append(f"{i}. **{r['skill']}, {r['district']} ({r['state']})** - demand {fmt(r['demand'])}, supply {fmt(r['supply'])}, gap {sgn(r['gap'])}, risk {r['risk']}; "
                  f"{r['recommendation'].lower()}" + (f" ({sgn(r['seat_change'])} seats)" if r["seat_change"] else "") + f", {r['priority']}.")
    table = [{"State": r["state"], "District": r["district"], "Skill": r["skill"], "Demand": int(r["demand"]), "Supply": int(r["supply"]), "Gap": int(r["gap"]),
              "Risk": r["risk"], "Action": r["recommendation"], "Seats": int(r["seat_change"])} for r in rows]
    return {"title": title, "markdown": "\n".join(md), "table": table, "kpis": [], "chart": None, "facts": {"ranking": clean(rows), "horizon": h, "scope": scope}}


_LLM: Dict[str, Any] = {"off_until": 0.0, "cache": {}, "http": None}


def llm_phrase(question: str, facts_md: str, facts: Dict[str, Any]) -> Optional[str]:
    """Let Gemini phrase the answer, strictly from facts fetched from MySQL. Returns None on any failure."""
    if not GEMINI_KEY:
        return None
    if time.time() < _LLM["off_until"]:
        return None                                # recent failure: answer from the database immediately instead of waiting again
    ck = hash((question.strip().lower(), facts_md))
    if ck in _LLM["cache"]:
        return _LLM["cache"][ck]
    prompt = ("You are the SkillMap Assistant for an Indian labour-market intelligence platform. Answer the user's question using ONLY the FACTS below "
              "(they were fetched from the platform database). Do not invent numbers. If the facts do not contain the answer, say so briefly. "
              "Reply in the same language style as the question (English, Hindi in Roman script, or Hinglish). Keep it concise and professional, use short paragraphs or bullet points in markdown, no emojis. Mention that supply data is simulated when you discuss supply, gap or risk.\n\n"
              f"QUESTION: {question}\n\nFACTS (markdown summary):\n{facts_md}\n\nFACTS (json):\n{json.dumps(facts, default=str)[:12000]}")
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 700}}
    # Keys that start with "AQ." are Vertex AI express-mode keys; other keys are Google AI Studio keys. Try the matching endpoint first.
    vertex = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{GEMINI_MODEL}:generateContent"
    studio = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    urls = [vertex, studio] if GEMINI_KEY.startswith("AQ.") else [studio, vertex]
    if _LLM["http"] is None:
        _LLM["http"] = requests.Session()
    for u in urls:
        try:
            r = _LLM["http"].post(u, headers={"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"}, json=body, timeout=(3, 8))
            if r.status_code != 200:
                continue
            parts = r.json()["candidates"][0]["content"]["parts"]
            out = "".join(x.get("text", "") for x in parts).strip()
            if out:
                if len(_LLM["cache"]) > 200:
                    _LLM["cache"].clear()
                _LLM["cache"][ck] = out
                return out
        except Exception:
            continue
    _LLM["off_until"] = time.time() + 120          # back off for two minutes
    return None


class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_llm: bool = True


def compare_areas(msg: str):
    """'compare X and Y' for two states or two districts: side-by-side table from the database."""
    c = catalog()
    m = " " + norm(msg) + " "
    found = []
    for s_ in c["states"]:
        for a in STATE_ALIAS.get(s_, [s_.lower()]) + [s_]:
            if f" {norm(a)} " in m:
                found.append(("state", s_)); break
    if len(found) < 2:
        found = []
        for d in sorted(c["districts"], key=lambda x: -len(x["district"])):
            if len(d["district"]) >= 3 and f" {norm(d['district'])} " in m and not any(f[1] == d["district"] for f in found):
                found.append(("district", d["district"]))
    if len(found) < 2:
        return None
    rows, parts = [], []
    for kind, name in found[:3]:
        p = area_profile(name if kind == "state" else None, name if kind == "district" else None, None)
        k = {x["label"]: x["value"] for x in p["kpis"]}
        rows.append({"Area": name, **k})
        parts.append(p["markdown"])
    return {"title": "Comparison: " + " vs ".join(n for _, n in found[:3]), "markdown": "**Side-by-side comparison** (12-month horizon, base scenario)\n\n" + "\n\n---\n\n".join(parts),
            "table": rows, "kpis": [], "chart": None, "facts": {"compared": [n for _, n in found[:3]]}}



# ---------------------------------------------------------------------------------------------
#  Hinglish / WhatsApp-style understanding: normalise the message to plain English keywords
# ---------------------------------------------------------------------------------------------
_PHRASES = [
    (r"\bkya (haal|hal)\b|\bkaise ho\b|\bkese ho\b|\bkaisa hai\b|\bwhat'?s up\b|\bsup\b", "hello"),
    (r"\bkya kar (sakte|sakta|skte|skta) (ho|hai|h)\b|\bkya kya kar\b|\bmadad\b|\bhelp me\b", "help"),
    (r"\b(tum|aap|tu) kaun (ho|hai|h)\b|\bwho are you\b", "who are you"),
    (r"\bsabse (zyada|jyada|jada|zada|bada|badi|bade)\b|\bsabse most\b", "highest"),
    (r"\bsabse (kam|kum|chota|choti)\b", "lowest"),
    (r"\b(zyada|jyada|jada) supply\b|\bsupply (zyada|jyada|jada)\b|\bextra supply\b", "oversupply"),
    (r"\bdemand (zyada|jyada|jada)\b|\b(zyada|jyada|jada) demand\b", "highest demand"),
    (r"\b(agle|next|aane wale|aane wala) (ek|1) saal\b|\b(ek|1) saal\b|\bagla saal\b|\bagle saal\b", "12 month"),
    (r"\b(agle|next)? ?(teen|3) (mahine|mahina|mhine|month)\b", "3 month"),
    (r"\b(agle|next)? ?(chhe|chah|che|6) (mahine|mahina|mhine|month)\b", "6 month"),
    (r"\bkya (karna|krna) chahiye\b|\bkya (karein|kare|krein)\b", "recommendation"),
    (r"\bkitni? (seats|seat)\b", "seats"),
    (r"\bkitn[aie]\b", "how much"),
    (r"\b(kaha|kahan|kidhar|kaunsa|kaunsi|kaunse|kaun sa|kaun si|kon sa|kon si|konsa|konsi|kaun|kon)\b", "where which"),
]
_TOKENS = {
    "dilli": "delhi", "bangal": "bengal", "tamilnadu": "tamil nadu", "gujrat": "gujarat", "maharastra": "maharashtra", "karnatak": "karnataka", "kerela": "kerala",
    "hii": "hello", "hiii": "hello", "hlo": "hello", "hlw": "hello", "heyy": "hello", "hey": "hello", "namaste": "hello", "namaskar": "hello", "hello": "hello", "hi": "hello",
    "thanks": "thanks", "thx": "thanks", "thankyou": "thanks", "shukriya": "thanks", "dhanyavad": "thanks", "dhanyawad": "thanks",
    "bye": "bye", "alvida": "bye", "byee": "bye",
    "bata": "show", "batao": "show", "bataiye": "show", "btao": "show", "bta": "show", "btana": "show", "bataye": "show", "dikhao": "show", "dikha": "show", "dikhana": "show", "nikalo": "show", "nikal": "show",
    "jankari": "details", "jaankari": "details", "haal": "details", "halat": "details", "sthiti": "status",
    "maang": "demand", "mang": "demand", "maangh": "demand",
    "aapurti": "supply", "kami": "shortage", "kmi": "shortage", "khatra": "risk", "jokhim": "risk",
    "zyada": "most", "jyada": "most", "jada": "most", "zada": "most", "kam": "least", "kum": "least",
    "sabse": "most", "naukri": "demand", "naukriyan": "demand",
    "ka": "", "ki": "", "ke": "", "ko": "", "mein": "", "me": "", "mei": "", "main": "", "h": "", "hai": "", "hain": "", "he": "", "ye": "", "yeh": "", "ys": "", "voh": "", "woh": "", "wo": "",
    "iska": "", "uska": "", "baare": "", "bare": "", "bhai": "", "bro": "", "yaar": "", "yrr": "", "yar": "", "plz": "", "pls": "", "please": "", "kr": "", "kar": "", "karo": "", "krdo": "", "do": "",
    "aur": "and", "or": "and", "tulna": "compare", "tulana": "compare", "vs": "vs",
}
_HING_MARK = set("ka ki ke ko mein me mei h hai hain kitni kitna kitne kya batao bata bta btao dikhao sabse zyada jyada kaha kahan kaun kaunsa bhai yaar yrr kr kar karo haal maang kami khatra agle mahine saal aur nikalo ye yeh voh wo dilli".split())


def to_english(msg: str):
    """Returns (english_keywords_text, is_hinglish). Entity names (states, districts, skills) are left untouched."""
    low = msg.lower()
    toks = re.findall(r"[a-z0-9']+", low)
    hing = sum(1 for t in toks if t in _HING_MARK) >= 1 and any(t in _HING_MARK for t in toks if t not in ("me", "main", "h", "do", "or"))
    for pat, rep in _PHRASES:
        low = re.sub(pat, " " + rep + " ", low)
    out = []
    for t in re.findall(r"[a-z0-9']+", low):
        if t in _TOKENS:
            r = _TOKENS[t]
            if r:
                out.append(r)
        else:
            out.append(t)
    text = re.sub(r"\s+", " ", " ".join(out)).strip()
    return (text if text else msg), hing




def answer(message: str, session_id: str, use_llm=True) -> Dict[str, Any]:
    original = message.strip()
    msg, hing = to_english(original)          # understands Hinglish / WhatsApp-style messages
    low = msg.lower()
    sess = SESSIONS.setdefault(session_id, {"ent": {"state": None, "district": None, "skill": None}})
    ent = find_entities(msg)
    has_ent = any(ent.values())
    follow = bool(FOLLOW_RE.search(low)) and not has_ent and any(sess["ent"].values())
    if follow:
        ent = dict(sess["ent"])
        has_ent = True
    metric = next((name for name, rx in METRICS if re.search(rx, low)), None)
    risk_only = "Critical" if re.search(r"\bcritical\b", low) else ("High" if re.search(r"\bhigh risk\b", low) else None)
    is_rank = (bool(RANK_RE.search(low)) and metric is not None) or (metric == "priority" and bool(re.search(r"\bseats?\b", low)) and not (ent["district"] and ent["skill"]))
    intent, res = "scope", None
    if not has_ent and re.search(r"\bthanks\b", low):
        intent = "smalltalk"
        res = {"title": "You are welcome", "markdown": ("Koi baat nahi. Aur kuch poochna ho to bata dena, jaise kisi district ya skill ki demand." if hing else "You are welcome. Ask me about any state, district or skill whenever you need."), "table": [], "kpis": [], "chart": None, "facts": {}}
    elif not has_ent and re.search(r"\bbye\b", low):
        intent = "smalltalk"
        res = {"title": "Goodbye", "markdown": ("Theek hai, milte hain. SkillMap par wapas aate rehna." if hing else "Goodbye. Come back to SkillMap any time."), "table": [], "kpis": [], "chart": None, "facts": {}}
    elif not has_ent and re.search(r"\bwho are you\b", low):
        intent = "smalltalk"
        res = {"title": "SkillMap Assistant", "markdown": ("Main SkillMap Assistant hoon. Main India ke states, districts aur skills ki demand, supply, gap, risk aur seat recommendation bata sakta hoon." if hing else "I am the SkillMap Assistant. I can tell you the demand, supply, gap, risk and seat recommendation for Indian states, districts and skills."), "table": [], "kpis": [], "chart": None, "facts": {}}
    elif re.match(r"^(hello|good (morning|afternoon|evening))\b", low) and not has_ent:
        intent = "greeting"
        if hing:
            greet = ("Namaste. Main SkillMap Assistant hoon. Aap English ya Hinglish, dono mein pooch sakte ho.\n\nJaise: *Pune me welder ki demand kitni hai*, *Bihar me sabse zyada shortage kaha hai*, "
                     "*Delhi ka haal batao*, *risk kaise calculate hota hai*.")
        else:
            greet = ("Hello. I am the SkillMap Assistant. Give me an area, a skill or both and I will fetch the demand, supply, gap, risk and seat recommendation. You can write in English or Hinglish.\n\n"
                     "Examples: *Electrician in Patna*, *details of Pune*, *top critical shortages in Bihar*, *oversupply of Data Entry Operators*, *how is risk calculated?*")
        res = {"title": "SkillMap Assistant", "markdown": greet, "table": [], "kpis": [], "chart": None, "facts": {}}
    elif re.search(r"\b(help|what can you do|capabilities|how to use you)\b", low) and not has_ent and hing:
        intent = "help"
        res = {"title": "Main kya kar sakta hoon", "markdown": "Main SkillMap database se jawab deta hoon:\n\n- **Area ki details:** koi state, district, skill, ya district me skill.\n- **Ranking:** sabse zyada shortage, oversupply, demand growth, priority actions (state ya skill se filter).\n- **Recommendation:** seats ke baare me kya karna chahiye aur kyu.\n- **Method:** gap index, risk bands, accuracy, data source, limitations.\n\nAap English ya Hinglish dono me pooch sakte ho.", "table": [], "kpis": [], "chart": None, "facts": {}}
    elif re.search(r"\b(help|what can you do|capabilities|how to use you)\b", low) and not has_ent:
        intent = "help"
        res = {"title": "What I can do", "markdown": "I answer from the SkillMap database:\n\n- **Area details:** a state, a district, a skill, or a skill in a district.\n- **Rankings:** top shortages, oversupply, fastest-growing demand, highest-priority actions (filter by state or skill).\n"
                      "- **Recommendations:** what to do with seats and why.\n- **Method questions:** gap index, risk bands, accuracy, data sources, limitations.\n\nI only answer about SkillMap data; I do not browse the internet.", "table": [], "kpis": [], "chart": None, "facts": {}}
    elif DEFINE_RE.search(low) and not (ent["district"] or ent["skill"]) and not is_rank:
        hits = kb_search(msg)
        if hits:
            intent = "method"
            res = {"title": hits[0][1]["title"], "markdown": "\n\n".join(f"**{d['title']}**\n\n{d['content']}" for _, d in hits), "table": [], "kpis": [], "chart": None, "facts": {"kb": [d["title"] for _, d in hits]}}
    if res is None and re.search(r"\b(compare|comparison|versus|vs)\b", low):
        res = compare_areas(msg)
        intent = "compare" if res else "scope"
    if res is None and is_rank:
        intent = "ranking"
        mname = metric if metric != "priority" else "priority"
        res = ranking(ent, msg, mname, risk_only)
    elif res is None and has_ent and (DATA_WORDS.search(low) or True):
        if ent["district"] and ent["skill"]:
            intent = "series"
            w, p = where(ent["state"], ent["district"], ent["skill"])
            r = q1(f"SELECT * FROM v_series{w}", **p)
            res = series_profile(r) if r else {"title": "Not found", "markdown": "No such district-skill combination exists in the database.", "table": [], "kpis": [], "chart": None, "facts": {}}
        elif ent["district"]:
            intent = "district"
            res = area_profile(ent["state"], ent["district"], None)
        elif ent["skill"] and ent["state"]:
            intent = "skill_state"
            res = area_profile(ent["state"], None, ent["skill"])
        elif ent["skill"]:
            intent = "skill"
            res = area_profile(None, None, ent["skill"])
        else:
            intent = "state"
            res = area_profile(ent["state"], None, None)
    if res is None:
        hits = kb_search(msg)
        if hits:
            intent = "method"
            res = {"title": hits[0][1]["title"], "markdown": "\n\n".join(f"**{d['title']}**\n\n{d['content']}" for _, d in hits), "table": [], "kpis": [], "chart": None, "facts": {"kb": [d["title"] for _, d in hits]}}
        else:
            res = {"title": "Outside scope", "markdown": ("Main sirf SkillMap ke data ke baare mein jawab de sakta hoon: states, districts aur skills ki demand, supply, gap, risk aur seat recommendation, aur ye sab kaise nikalte hain. "
                                                            "Try karo: 'Patna me electrician ki demand', 'Bihar me sabse zyada shortage kaha hai', 'Pune ka haal batao'." if hing else SCOPE_MSG), "table": [], "kpis": [], "chart": None, "facts": {}}
    if has_ent:
        sess["ent"] = {k: v for k, v in ent.items()}
    used_llm = False
    text_out = res["markdown"]
    if intent in ("series", "district", "state", "skill", "skill_state", "ranking", "compare"):
        text_out = ("Ye raha aapka jawab.\n\n" if hing else "Here is what I found.\n\n") + text_out
    if use_llm and GEMINI_KEY and intent not in ("scope", "greeting", "help"):
        phrased = llm_phrase(original, res["markdown"], res.get("facts", {}))
        if phrased:
            text_out, used_llm = phrased, True
    follow_ups = []
    if intent in ("district", "state", "skill", "skill_state"):
        follow_ups = [f"Top critical shortages in {ent['state'] or ent['district'] or 'India'}", "How is risk calculated?"]
    elif intent == "series":
        follow_ups = [f"Details of {ent['district']}", f"{ent['skill']} across India"]
    elif intent == "ranking":
        follow_ups = ["What is the gap index?", "Show details of the first district"]
    else:
        follow_ups = ["Top critical shortages in India", "How accurate are the forecasts?"]
    return {"session_id": session_id, "intent": intent, "entities": {k: v for k, v in ent.items() if v}, "answer": text_out, "answer_rule_based": res["markdown"], "used_llm": used_llm,
            "title": res["title"], "kpis": res.get("kpis", []), "table": res.get("table", []), "chart": res.get("chart"), "follow_ups": follow_ups,
            "sources": ["fact_series", "fact_history", "kb_document"] if intent != "scope" else []}


@app.post("/chat", tags=["assistant"], summary="Ask the SkillMap Assistant about any area, skill or the method")
def chat(body: ChatIn):
    if not body.message or not body.message.strip():
        raise HTTPException(400, "message is empty")
    sid = body.session_id or uuid.uuid4().hex[:12]
    try:
        out = answer(body.message, sid, body.use_llm)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"assistant error: {e}")
    try:
        with engine.begin() as c:
            c.execute(text("INSERT INTO chat_log (session_id, question, intent, answer, used_llm) VALUES (:s,:q,:i,:a,:l)"),
                      {"s": sid, "q": body.message[:2000], "i": out["intent"], "a": out["answer"], "l": int(out["used_llm"])})
    except Exception:
        pass
    return out


@app.get("/data", tags=["forecast"], summary="Complete series table (one row per district x skill) for dashboards and export")
def data():
    from fastapi.responses import Response
    hit = _QC.get("data_json")
    if hit and time.time() - hit[0] < _QC_TTL:
        return Response(content=hit[1], media_type="application/json")
    body = json.dumps(clean(q("SELECT * FROM v_series ORDER BY priority_rank")), default=str).encode()
    _QC["data_json"] = (time.time(), body)
    return Response(content=body, media_type="application/json")


@app.get("/kb", tags=["assistant"], summary="Knowledge-base documents the assistant retrieves from")
def kb():
    return q("SELECT category, title, content FROM kb_document ORDER BY doc_id")


@app.get("/chat-log", tags=["assistant"], summary="Latest questions asked to the assistant (stored in MySQL)")
def chat_log(session_id: Optional[str] = None, limit: int = 20):
    if session_id:
        rows = q("SELECT chat_id, session_id, question, intent, used_llm, created_at FROM chat_log WHERE session_id=:s ORDER BY chat_id DESC LIMIT :l", s=session_id, l=limit)
    else:
        rows = q("SELECT chat_id, session_id, question, intent, used_llm, created_at FROM chat_log ORDER BY chat_id DESC LIMIT :l", l=limit)
    for r in rows:
        r["created_at"] = str(r["created_at"])
    return rows


# ----------------------------------------------------------------------------------------------- conversations (chat history in SQL)
class ConvIn(BaseModel):
    title: str
    ts: str = ""
    d: str = ""
    msgs: List[Any] = []


_conv_ready = False


def _conv_table():
    """chat_conversation holds one row per chat (messages as JSON). Created on first use, works on MySQL and SQLite."""
    global _conv_ready
    if _conv_ready:
        return
    big = "LONGTEXT" if engine.dialect.name == "mysql" else "TEXT"
    with engine.begin() as c:
        c.execute(text(f"CREATE TABLE IF NOT EXISTS chat_conversation (conv_id VARCHAR(32) PRIMARY KEY, title VARCHAR(200) NOT NULL, day_key VARCHAR(12), "
                       f"updated_label VARCHAR(40), msgs {big} NOT NULL, updated_ms BIGINT NOT NULL)"))
    _conv_ready = True


@app.get("/conversations", tags=["assistant"], summary="Saved chats (history), newest first")
def conv_list(limit: int = 60):
    _conv_table()
    rows = q("SELECT conv_id, title, day_key, updated_label, msgs FROM chat_conversation ORDER BY updated_ms DESC LIMIT :l", l=limit)
    return [{"id": r["conv_id"], "title": r["title"], "d": r["day_key"] or "", "ts": r["updated_label"] or "", "msgs": json.loads(r["msgs"] or "[]")} for r in rows]


@app.put("/conversations/{cid}", tags=["assistant"], summary="Create or update one chat")
def conv_save(cid: str, body: ConvIn):
    _conv_table()
    with engine.begin() as c:
        c.execute(text("DELETE FROM chat_conversation WHERE conv_id=:i"), {"i": cid})
        c.execute(text("INSERT INTO chat_conversation (conv_id, title, day_key, updated_label, msgs, updated_ms) VALUES (:i,:t,:d,:u,:m,:ms)"),
                  {"i": cid, "t": body.title[:200], "d": body.d, "u": body.ts, "m": json.dumps(body.msgs, ensure_ascii=False, default=str), "ms": int(time.time() * 1000)})
    return {"ok": True}


@app.delete("/conversations/{cid}", tags=["assistant"], summary="Delete one chat")
def conv_delete(cid: str):
    _conv_table()
    with engine.begin() as c:
        c.execute(text("DELETE FROM chat_conversation WHERE conv_id=:i"), {"i": cid})
    return {"ok": True}


@app.delete("/conversations", tags=["assistant"], summary="Delete all chats")
def conv_clear():
    _conv_table()
    with engine.begin() as c:
        c.execute(text("DELETE FROM chat_conversation"))
    return {"ok": True}


# ----------------------------------------------------------------------------------------------- evaluation and target file (for dashboards and scheme workflows)
@app.get("/evaluation", tags=["explainability"], summary="Back-test results (rolling origin), classifier scores and data-quality audit")
def evaluation():
    from fastapi.responses import JSONResponse
    row = q("SELECT meta_value FROM app_meta WHERE meta_key='evaluation'")
    if row:
        return JSONResponse(json.loads(row[0]["meta_value"]))
    import pickle
    for p in (Path(__file__).resolve().parent / "lmis_bundle.pkl", Path(__file__).resolve().parent / "notebook" / "lmis_bundle.pkl"):
        if p.exists():
            with open(p, "rb") as fh:
                B = pickle.load(fh)
            keep = {k: v for k, v in B["evaluation"].items() if k in EVAL_KEYS}
            return JSONResponse(json.loads(json.dumps(keep, default=str)))
    raise HTTPException(404, "evaluation not loaded: run load_db.py again or keep lmis_bundle.pkl next to api.py")


EVAL_KEYS = ["demand_rolling", "supply_rolling", "demand_fixed", "supply_fixed", "classifier", "classifier_confusion", "demand_audit", "supply_audit"]


@app.get("/targets", tags=["recommendation"], summary="Next-cycle seat targets per district and trade (JSON or CSV) for target-setting workflows")
def targets(state: Optional[str] = None, risk: Optional[str] = None, fmt: str = "json"):
    from fastapi.responses import Response
    where, p = ["1=1"], {}
    if state:
        where.append("LOWER(state)=LOWER(:st)"); p["st"] = state
    if risk:
        where.append("LOWER(risk)=LOWER(:rk)"); p["rk"] = risk
    rows = q("SELECT priority_rank, priority, state, district, skill, sector, risk, skill_gap, annual_seats, seat_change, recommendation, recommendation_detail "
             f"FROM v_series WHERE {' AND '.join(where)} ORDER BY priority_rank", **p)
    out = []
    for r in clean(rows):
        r["target_seats"] = int(round((r["annual_seats"] or 0) + (r["seat_change"] or 0)))
        out.append(r)
    if fmt.lower() == "csv":
        import io, csv
        buf = io.StringIO()
        if out:
            w = csv.DictWriter(buf, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        return Response(content=buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=skillmap_targets.csv"})
    return out


@app.get("/", tags=["info"])
def root():
    return {"service": "SkillMap API", "docs": "/docs", "assistant": "POST /chat", "llm_enabled": bool(GEMINI_KEY),
            "endpoints": ["/health", "/meta", "/areas", "/summary", "/forecast", "/district/{district}", "/series", "/high-risk", "/recommendations", "/explain", "/history", "/brief", "/whatif (POST)", "/chat (POST)"]}