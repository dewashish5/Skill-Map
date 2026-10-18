# pyright: reportMissingImports=false, reportMissingModuleSource=false, reportAttributeAccessIssue=false, reportArgumentType=false, reportCallIssue=false, reportOptionalSubscript=false, reportOptionalMemberAccess=false, reportOperatorIssue=false, reportIndexIssue=false, reportGeneralTypeIssues=false
"""
SkillMap - Streamlit frontend.

    streamlit run app.py

Talks only to the FastAPI backend (API_URL, default http://127.0.0.1:8000), which reads MySQL.
Workspaces: Dashboard, Recommendation Engine, SkillMap Assistant.
"""
import io
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from requests.adapters import HTTPAdapter
import streamlit as st

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def _env():
    f = HERE / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_env()
API = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")

# ----------------------------------------------------------------------------------------------- palette (from the logo)
NAVY, BLUE, CYAN, TEAL, GREEN = "#2B3A6B", "#425B9A", "#76C0EC", "#FF95A5", "#FFC4CD"   # palette: #425B9A #76C0EC #FFF6DC #FF95A5
MIST, MIST2, CARD, LINE = "#ECEEFA", "#DADFF3", "#FBFBFF", "#D3D8EE"
INK, MUTED = "#2B3A6B", "#6A7699"
RISK_C = {"Low": "#76C0EC", "Medium": "#F2C879", "High": "#FF95A5", "Critical": "#D6455D"}
RISKS = ["Low", "Medium", "High", "Critical"]
ACT_C = {"Increase training capacity": "#D6455D", "Fill existing seats first": "#F2C879", "Reduce seats": "#425B9A", "Maintain capacity": "#76C0EC"}

# ----------------------------------------------------------------------------------------------- Streamlit version compatibility
# Newer Streamlit uses width="stretch"; older releases only know use_container_width=True. Translate once, so the app runs on both.
def _streamlit_is_new():
    try:
        parts = [int("".join(ch for ch in p if ch.isdigit()) or 0) for p in st.__version__.split(".")[:2]]
        return tuple(parts) >= (1, 50)
    except Exception:
        return False


if os.environ.get("SKILLMAP_LEGACY_ST") == "1" or not _streamlit_is_new():
    from streamlit.delta_generator import DeltaGenerator as _DG

    def _compat(fn):
        def inner(*a, **k):
            if k.get("width") == "stretch":
                k.pop("width")
                k["use_container_width"] = True
            return fn(*a, **k)
        return inner

    for _n in ("plotly_chart", "dataframe", "button", "download_button", "image"):
        if hasattr(_DG, _n):
            setattr(_DG, _n, _compat(getattr(_DG, _n)))
        if hasattr(st, _n):
            setattr(st, _n, _compat(getattr(st, _n)))


st.set_page_config(page_title="SkillMap", page_icon=str(ASSETS / "favicon.png"), layout="wide", initial_sidebar_state="collapsed")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Noto+Sans+Devanagari:wght@400;500;600;700&family=Noto+Sans+Tamil:wght@400;600&family=Noto+Sans+Telugu:wght@400;600&family=Noto+Sans+Bengali:wght@400;600&family=Noto+Sans+Gujarati:wght@400;600&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans','Noto Sans Devanagari','Nirmala UI','Mangal','Inter','Segoe UI',system-ui,sans-serif; color:{INK}; }}
.stApp {{ background: radial-gradient(1100px 520px at 12% -8%,rgba(118,192,236,.22),transparent 60%), radial-gradient(900px 480px at 100% 0%,rgba(255,149,165,.18),transparent 55%), linear-gradient(180deg,{MIST} 0%,#F4F1FA 60%,{MIST} 100%); }}
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {{ display:none !important; }}
header[data-testid="stHeader"] {{ display:none; }}
#MainMenu, footer {{ visibility:hidden; }}
.block-container {{ padding-top:.9rem; padding-bottom:2rem; max-width:1500px; }}
[data-testid="stVerticalBlock"] {{ gap:.9rem; }}
section[data-testid="stSidebar"] {{ background:{CARD}; border-right:1px solid {LINE}; }}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {{ border-radius:12px; margin:2px 0; color:{INK}; font-weight:600; }}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {{ background:linear-gradient(90deg,{BLUE},{CYAN}); color:#F6FAFE; }}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color:#F6FAFE !important; }}
h1,h2,h3,h4 {{ color:{NAVY}; letter-spacing:-0.01em; }}
.hero {{ background:linear-gradient(105deg,{NAVY} 0%,{BLUE} 62%,{CYAN} 135%); border-radius:16px; padding:18px 28px 16px 28px; margin-bottom:12px; color:#EAF3FF; box-shadow:0 8px 22px rgba(43,58,107,.14); display:flex; align-items:flex-end; justify-content:space-between; gap:24px; flex-wrap:wrap; }}
.hero .l {{ min-width:300px; }}
.eyebrow {{ font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; text-transform:uppercase; letter-spacing:.2em; font-size:.7rem; color:#9AD3F3; font-weight:500; }}
.hero h1 {{ color:#F7FAFE; margin:4px 0 0 0; font-size:1.75rem; font-weight:600; letter-spacing:-0.015em; line-height:1.15; padding:0; }}
.hero p {{ margin:6px 0 0 0; color:#D3E3F7; font-size:.92rem; max-width:760px; line-height:1.5; }}
.hero .tags {{ display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }}
.hero .tag {{ display:inline-block; font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; text-transform:uppercase; letter-spacing:.1em; background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.22); border-radius:4px; padding:3px 9px; font-size:.64rem; color:#EAF3FF; white-space:nowrap; }}
.kpi {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:11px 15px 12px 15px; box-shadow:0 2px 10px rgba(66,91,154,.08); height:100%; }}
.kpi .l {{ color:{MUTED}; font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.7rem; font-weight:500; text-transform:uppercase; letter-spacing:.12em; }}
.kpi .v {{ color:{NAVY}; font-size:1.55rem; font-weight:600; line-height:1.2; margin-top:1px; font-variant-numeric:tabular-nums; letter-spacing:-0.01em; }}
.kpi .s {{ color:{MUTED}; font-size:.78rem; margin-top:2px; }}
.kpi .bar {{ height:3px; border-radius:3px; margin-top:8px; background:linear-gradient(90deg,var(--c1),var(--c2)); }}
.panel {{ background:transparent; border:none; border-left:3px solid {CYAN}; padding:0 0 0 11px; margin:10px 0 2px 0; }}
.panel h4 {{ margin:0; font-family:'IBM Plex Sans',sans-serif; font-size:1rem; font-weight:600; letter-spacing:-0.005em; color:{NAVY}; line-height:1.3; }}
.panel .sub {{ color:{MUTED}; font-size:.78rem; margin:1px 0 0 0; line-height:1.35; }}
.note {{ background:#FFF4DC; border:1px solid #F2D58F; color:#5E4310; border-radius:12px; padding:9px 14px; font-size:.84rem; margin-bottom:12px; }}
.pill {{ display:inline-block; border-radius:999px; padding:2px 10px; font-size:.74rem; font-weight:700; color:#F6FAFE; }}
div[data-baseweb="tab-list"] {{ gap:6px; margin-top:6px; margin-bottom:6px; }}
button[data-baseweb="tab"] {{ background:{CARD}; border:1px solid {LINE}; border-radius:12px; padding:6px 16px; color:{INK}; font-weight:600; }}
button[data-baseweb="tab"][aria-selected="true"] {{ background:linear-gradient(90deg,{BLUE},{CYAN}); color:#F6FAFE; border-color:transparent; }}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{ display:none; }}
button[data-testid="stTab"] {{ background:{CARD}; border:1px solid {LINE}; border-radius:12px; padding:6px 16px; margin-right:6px; }}
button[data-testid="stTab"] p {{ color:{INK}; font-weight:600; }}
button[data-testid="stTab"][aria-selected="true"] {{ background:linear-gradient(90deg,{BLUE},{CYAN}); border-color:transparent; }}
button[data-testid="stTab"][aria-selected="true"] p {{ color:#F6FAFE; }}
[data-testid="stToolbar"], [data-testid="stDeployButton"] {{ display:none; }}
button[role="tab"] {{ background:{CARD} !important; border:1px solid {LINE} !important; border-radius:12px !important; padding:6px 16px !important; margin-right:6px; }}
button[role="tab"] p {{ color:{INK} !important; font-weight:600; }}
button[role="tab"][aria-selected="true"] {{ background:linear-gradient(90deg,{BLUE},{CYAN}) !important; border-color:transparent !important; }}
button[role="tab"][aria-selected="true"] p {{ color:#F6FAFE !important; }}
[data-testid="stBottom"], [data-testid="stBottom"] > div {{ background:transparent !important; }}
.stButton>button, .stDownloadButton>button {{ border-radius:12px; border:1px solid {LINE}; background:{CARD}; color:{INK}; font-weight:600; }}
.stButton>button[kind="primary"], button[data-testid="stBaseButton-primary"] {{ background:linear-gradient(90deg,{BLUE},{CYAN}); border:none; box-shadow:0 4px 14px rgba(66,91,154,.28); }}
.stButton>button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p {{ color:#F7FAFE !important; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) {{ position:sticky; top:10px; z-index:1000; background:rgba(251,251,255,.72); backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); border:1px solid rgba(66,91,154,.20); border-radius:999px; padding:2px 18px; box-shadow:0 6px 22px rgba(43,58,107,.10); margin-bottom:14px; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) .stButton>button {{ background:transparent; border:1px solid transparent; box-shadow:none; border-radius:999px; min-height:2rem; padding:0 12px; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) .stButton>button p {{ white-space:nowrap; font-size:.74rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:{INK}; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) .stButton>button:hover {{ background:rgba(66,91,154,.10); border-color:transparent; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) .stButton>button[kind="primary"], .block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) button[data-testid="stBaseButton-primary"] {{ background:linear-gradient(90deg,{BLUE},{CYAN}); box-shadow:0 4px 14px rgba(66,91,154,.30); }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) .stButton>button[kind="primary"] p, .block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) button[data-testid="stBaseButton-primary"] p {{ color:#F7FAFE !important; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) [data-testid="stHorizontalBlock"] {{ align-items:center; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) [data-testid="stHorizontalBlock"] > div {{ align-self:center; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) [data-testid="stImage"] img {{ display:block; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) [data-testid="stMarkdownContainer"] p {{ margin:0; }}
.nav-tag {{ margin:0; text-align:right; font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; color:{MUTED}; }}
.chat-head {{ display:flex; align-items:center; gap:12px; margin-bottom:2px; }}
.chat-head .dot {{ width:9px; height:9px; border-radius:50%; background:{CYAN}; box-shadow:0 0 0 4px rgba(118,192,236,.28); }}
.chat-head .t {{ font-weight:700; font-size:1.15rem; color:{NAVY}; }}
[data-testid="stElementContainer"]:has(.nav-marker), .element-container:has(.nav-marker), [data-testid="stElementContainer"]:has(.u-msg), .element-container:has(.u-msg), [data-testid="stElementContainer"]:has(.a-msg), .element-container:has(.a-msg) {{ display:none; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) [data-testid="stHorizontalBlock"] {{ align-items:center !important; min-height:46px; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) [data-testid="stHorizontalBlock"] > div {{ align-self:center !important; display:flex; flex-direction:column; justify-content:center; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) [data-testid="stVerticalBlock"] {{ gap:0 !important; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) [data-testid="stImage"] {{ line-height:0; }}
[data-testid="stWidgetLabel"] p {{ font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.66rem !important; text-transform:uppercase; letter-spacing:.14em; color:{MUTED}; font-weight:500; }}
div[data-baseweb="tab-list"] {{ gap:12px !important; margin:30px 0 16px 0 !important; }}
[data-testid="stChatInput"] {{ border-radius:999px; }}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.chat-head) {{ background:linear-gradient(180deg,rgba(247,250,254,.92),rgba(231,241,251,.92)); border:1px solid {LINE}; border-radius:22px; box-shadow:0 10px 30px rgba(43,58,107,.10); padding:6px 8px; }}
.chat-head {{ gap:12px; }}
.chat-head img {{ width:40px; height:40px; border-radius:12px; border:1px solid {LINE}; background:#FFF6DC; padding:4px; }}
.chat-head .t {{ font-weight:600; font-size:1.1rem; letter-spacing:-0.005em; }}
.chat-head .on {{ font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.64rem; letter-spacing:.14em; text-transform:uppercase; color:#2E9E6B; display:flex; align-items:center; gap:6px; }}
.chat-head .on i {{ width:7px; height:7px; border-radius:50%; background:#3CC38A; display:inline-block; box-shadow:0 0 0 3px rgba(60,195,138,.22); }}
div[data-testid="stChatMessage"]:has(.u-msg) {{ flex-direction:row-reverse; margin-left:auto; width:fit-content; max-width:76%; background:linear-gradient(135deg,{BLUE},#5B83C4); border:none; border-radius:20px 20px 5px 20px; box-shadow:0 4px 14px rgba(66,91,154,.25); padding:.65rem 1.05rem 1.3rem !important; overflow:visible !important; min-height:0 !important; }}
div[data-testid="stChatMessage"]:has(.u-msg) [data-testid="stChatMessageContent"] {{ overflow:visible; }}
div[data-testid="stChatMessage"]:has(.u-msg) p, div[data-testid="stChatMessage"]:has(.u-msg) span, div[data-testid="stChatMessage"]:has(.u-msg) div {{ color:#F7FAFE; }}
div[data-testid="stChatMessage"]:has(.u-msg) [data-testid^="stChatMessageAvatar"] {{ display:none; }}
div[data-testid="stChatMessage"]:has(.a-msg) {{ max-width:92%; background:{CARD}; border:1px solid {LINE}; border-radius:20px 20px 20px 5px; }}
.msg-time {{ font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.62rem; letter-spacing:.08em; color:{MUTED}; margin-top:2px; }}
div[data-testid="stChatMessage"]:has(.u-msg) .msg-time {{ color:#D3E3F7; text-align:right; }}
.typing {{ display:inline-flex; gap:5px; padding:8px 4px; }}
.typing span {{ width:8px; height:8px; border-radius:50%; background:{BLUE}; opacity:.35; animation:skm-blink 1.2s infinite ease-in-out; }}
.typing span:nth-child(2) {{ animation-delay:.2s; }} .typing span:nth-child(3) {{ animation-delay:.4s; }}
@keyframes skm-blink {{ 0%,80%,100% {{ opacity:.25; transform:translateY(0); }} 40% {{ opacity:1; transform:translateY(-3px); }} }}
.chat-line {{ border:none; height:1px; margin:2px 0 6px 0; background:linear-gradient(90deg,{LINE},transparent); }}
.chat-sub {{ color:{MUTED}; font-size:.86rem; margin:0 0 10px 21px; }}
.stButton>button p {{ white-space:normal; }}
.topbar-line {{ height:1px; border:none; margin:6px 0 18px 0; background:linear-gradient(90deg,transparent,{LINE} 12%,{LINE} 88%,transparent); }}
.stButton>button:hover, .stDownloadButton>button:hover {{ border-color:{BLUE}; color:{BLUE}; }}
.stButton>button[kind="primary"] {{ background:linear-gradient(90deg,{BLUE},{CYAN}); color:#F6FAFE; border:none; }}
div[data-testid="stChatMessage"] {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px; }}
div[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:12px; }}
/* ---------- v8 refinements ---------- */
html {{ scroll-behavior:smooth; }}
.block-container {{ max-width:1320px; padding-top:1.1rem; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) {{ padding:8px 26px !important; top:12px; margin-bottom:22px; border-radius:999px; box-shadow:0 10px 30px rgba(43,58,107,.12); }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) [data-testid="stHorizontalBlock"] {{ min-height:60px !important; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) .stButton>button {{ min-height:2.7rem !important; padding:0 18px !important; }}
.block-container > [data-testid="stVerticalBlock"] > *:has(.nav-marker) .stButton>button p {{ font-size:.78rem !important; }}
/* filter card */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.filter-marker) {{ background:{CARD}; border:1px solid {LINE}; border-radius:18px; box-shadow:0 4px 18px rgba(66,91,154,.08); padding:6px 10px 10px 10px; }}
.f-title {{ font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.68rem; letter-spacing:.2em; text-transform:uppercase; color:{BLUE}; font-weight:600; margin:2px 0 4px 2px; }}
[data-testid="stWidgetLabel"] p {{ font-family:'IBM Plex Sans',sans-serif !important; font-size:.78rem !important; text-transform:none !important; letter-spacing:0 !important; color:{INK} !important; font-weight:600 !important; }}
div[data-baseweb="select"] > div {{ background:#fff !important; border:1px solid {LINE} !important; border-radius:12px !important; min-height:2.6rem; box-shadow:none; }}
div[data-baseweb="select"] > div:hover {{ border-color:{CYAN} !important; }}
div[data-baseweb="select"] div[value], div[data-baseweb="select"] span {{ font-size:.92rem; }}
/* tabs: one segmented track */
[role="tablist"] {{ border-bottom:1px solid {LINE} !important; box-shadow:none; gap:4px !important; margin:26px 0 22px 0 !important; background:{CARD}; border:1px solid {LINE}; border-radius:16px; padding:5px; width:fit-content; max-width:100%; box-shadow:0 3px 12px rgba(66,91,154,.07); }}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"], [role="tablist"] > div:not([role="tab"]) {{ display:none !important; }}
[role="tab"] {{ background:transparent !important; border:none !important; border-radius:12px !important; padding:8px 18px !important; margin:0 !important; }}
[role="tab"] p {{ font-size:.88rem; font-weight:600; letter-spacing:.005em; }}
[role="tab"]:hover {{ background:rgba(66,91,154,.08) !important; }}
[role="tab"][aria-selected="true"] {{ background:linear-gradient(90deg,{BLUE},{CYAN}) !important; box-shadow:0 4px 12px rgba(66,91,154,.28); }}
[role="tab"][aria-selected="true"] p {{ color:#FBFBFF !important; }}
/* every chart is its own card */
.panel {{ background:{CARD}; border:1px solid {LINE}; border-left:1px solid {LINE}; border-radius:16px; padding:14px 18px; margin:0; box-shadow:0 4px 16px rgba(66,91,154,.08); }}
.panel.hd {{ border-radius:16px 16px 0 0; border-bottom:none; padding:16px 20px 20px 20px; box-shadow:none; position:relative; }}
.panel.hd::before {{ content:""; position:absolute; left:20px; top:0; width:44px; height:3px; border-radius:0 0 4px 4px; background:linear-gradient(90deg,{BLUE},{CYAN}); }}
[data-testid="stElementContainer"]:has(.panel.hd), .element-container:has(.panel.hd) {{ margin-bottom:-.9rem; margin-top:10px; }}
[data-testid="stElementContainer"]:has(.panel.hd) + [data-testid="stElementContainer"], .element-container:has(.panel.hd) + .element-container {{ background:{CARD}; border:1px solid {LINE}; border-top:none; border-radius:0 0 16px 16px; padding:2px 14px 12px 14px; box-shadow:0 8px 20px rgba(66,91,154,.09); }}
.kpi {{ transition:transform .25s ease, box-shadow .25s ease; }}
.kpi:hover {{ transform:translateY(-3px); box-shadow:0 10px 24px rgba(66,91,154,.16); }}
.stButton>button:hover {{ transform:translateY(-1px); }}
/* entrance animations */
@keyframes skm-up {{ from {{ opacity:0; transform:translateY(14px); }} to {{ opacity:1; transform:none; }} }}
@keyframes skm-grow {{ from {{ transform:scaleX(0); }} to {{ transform:scaleX(1); }} }}
.hero, .kpi, div[data-testid="stVerticalBlockBorderWrapper"]:has(.filter-marker) {{ animation:skm-up .4s cubic-bezier(.2,.7,.2,1) both; }}
.kpi .bar {{ transform-origin:left; animation:skm-grow 1.1s .3s cubic-bezier(.2,.7,.2,1) both; }}
[data-testid="stHorizontalBlock"] > div:nth-child(2) .kpi {{ animation-delay:.07s; }} [data-testid="stHorizontalBlock"] > div:nth-child(3) .kpi {{ animation-delay:.14s; }}
[data-testid="stHorizontalBlock"] > div:nth-child(4) .kpi {{ animation-delay:.21s; }} [data-testid="stHorizontalBlock"] > div:nth-child(5) .kpi {{ animation-delay:.28s; }}
/* splash */
.skm-splash {{ position:fixed; inset:0; z-index:99999; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:22px; background:radial-gradient(900px 500px at 30% 20%,rgba(118,192,236,.35),transparent 60%), radial-gradient(800px 480px at 80% 80%,rgba(255,149,165,.30),transparent 60%), linear-gradient(135deg,#2B3A6B,#425B9A); animation:skm-splash 2.6s ease forwards; pointer-events:none; }}
.skm-splash img {{ width:96px; height:96px; border-radius:26px; background:#FFF6DC; padding:12px; box-shadow:0 18px 50px rgba(0,0,0,.35); animation:skm-pop 1s cubic-bezier(.2,.9,.3,1.2) both; }}
.skm-splash .w {{ color:#F7FAFE; font-weight:600; font-size:2rem; letter-spacing:-0.01em; animation:skm-up .8s .25s both; }}
.skm-splash .w b {{ color:#76C0EC; font-weight:600; }}
.skm-splash .t {{ color:#BFD0F0; font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.72rem; letter-spacing:.24em; text-transform:uppercase; margin-top:-14px; animation:skm-up .8s .45s both; }}
.skm-splash .ld {{ width:220px; height:4px; border-radius:4px; background:rgba(255,255,255,.18); overflow:hidden; }}
.skm-splash .ld i {{ display:block; height:100%; width:100%; background:linear-gradient(90deg,#76C0EC,#FF95A5); transform-origin:left; animation:skm-grow 1.8s .2s ease both; }}
@keyframes skm-pop {{ from {{ opacity:0; transform:scale(.6) rotate(-8deg); }} to {{ opacity:1; transform:none; }} }}
@keyframes skm-splash {{ 0%,78% {{ opacity:1; visibility:visible; }} 100% {{ opacity:0; visibility:hidden; }} }}
/* ---------- assistant (chat workspace) ---------- */
.as-side {{ background:{CARD}; border:1px solid {LINE}; border-radius:18px; padding:14px; margin-bottom:12px; box-shadow:0 4px 16px rgba(66,91,154,.07); }}
.as-brand {{ display:flex; align-items:center; gap:10px; }}
.as-brand img {{ width:40px; height:40px; border-radius:12px; background:#FFF6DC; padding:5px; border:1px solid {LINE}; }}
.as-brand b {{ display:block; font-size:.98rem; color:{NAVY}; }} .as-brand span {{ font-size:.74rem; color:{MUTED}; }}
.as-eng {{ display:flex; gap:10px; align-items:flex-start; padding-bottom:12px; border-bottom:1px solid {LINE}; }}
.as-eng i {{ width:8px; height:8px; border-radius:50%; background:#3CC38A; margin-top:6px; box-shadow:0 0 0 3px rgba(60,195,138,.22); flex:none; }}
.as-eng b {{ display:block; font-size:.86rem; }} .as-eng span {{ font-size:.74rem; color:{MUTED}; }}
.as-cap {{ font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.64rem; letter-spacing:.18em; text-transform:uppercase; color:{BLUE}; font-weight:600; margin:12px 0 8px 0; }}
.as-mic {{ background:linear-gradient(135deg,rgba(118,192,236,.18),rgba(255,149,165,.14)); border:1px solid {LINE}; border-radius:14px; padding:12px; }}
.as-mic .mi {{ width:26px; height:26px; border-radius:8px; background:{BLUE}; margin-bottom:8px; -webkit-mask:none; position:relative; }}
.as-mic .mi::after {{ content:""; position:absolute; left:9px; top:5px; width:8px; height:11px; border:2px solid #FBFBFF; border-radius:5px; box-sizing:border-box; }}
.as-mic b {{ display:block; font-size:.88rem; }} .as-mic span {{ font-size:.76rem; color:{MUTED}; }}
.as-foot span {{ font-size:.74rem; color:{MUTED}; line-height:1.45; display:block; }}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.voice-marker) {{ background:{CARD}; border:1px solid {LINE}; border-radius:18px; box-shadow:0 4px 16px rgba(66,91,154,.07); margin-bottom:12px; }}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.as-marker) {{ background:linear-gradient(180deg,rgba(251,251,255,.96),rgba(240,242,252,.96)); border:1px solid {LINE}; border-radius:22px; box-shadow:0 12px 34px rgba(43,58,107,.10); padding:8px 14px 10px 14px; }}
.as-top {{ display:flex; align-items:center; justify-content:space-between; padding:6px 4px 12px 4px; border-bottom:1px solid {LINE}; margin-bottom:6px; }}
.as-title {{ font-size:1.12rem; font-weight:600; color:{NAVY}; }} .as-sub {{ font-size:.8rem; color:{MUTED}; margin-top:1px; }}
.as-pill {{ display:flex; align-items:center; gap:7px; font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.66rem; letter-spacing:.12em; text-transform:uppercase; color:{BLUE}; background:{CARD}; border:1px solid {LINE}; padding:6px 12px; border-radius:999px; }}
.as-pill i {{ width:6px; height:6px; border-radius:50%; background:#3CC38A; }}
.as-hero {{ text-align:center; padding:18px 0 6px 0; animation:skm-up .7s cubic-bezier(.2,.7,.2,1) both; }}
.as-hero .ic {{ width:54px; height:54px; margin:0 auto 12px auto; border-radius:16px; background:linear-gradient(135deg,{BLUE},{CYAN}); display:flex; align-items:center; justify-content:center; box-shadow:0 10px 26px rgba(66,91,154,.30); }}
.as-hero .ic img {{ width:34px; height:34px; background:#FFF6DC; border-radius:10px; padding:3px; }}
.as-hero .e {{ font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.66rem; letter-spacing:.22em; text-transform:uppercase; color:{MUTED}; }}
.as-hero h2 {{ font-size:1.9rem; font-weight:600; letter-spacing:-0.015em; color:{NAVY}; margin:4px 0 6px 0; padding:0; }}
.as-hero p {{ color:{MUTED}; font-size:.9rem; margin:0 0 6px 0; }}
div[data-testid="stVerticalBlock"]:has(> div > div > .as-cards) .stButton>button, div:has(> .as-cards) .stButton>button {{ text-align:left; justify-content:flex-start; border-radius:14px; padding:12px 16px; min-height:4rem; background:{CARD}; border:1px solid {LINE}; box-shadow:0 2px 8px rgba(66,91,154,.06); }}
div[data-testid="stVerticalBlock"]:has(.as-cards) .stButton>button:hover {{ transform:translateY(-2px); border-color:{CYAN}; box-shadow:0 10px 22px rgba(66,91,154,.14); }}
div[data-testid="stVerticalBlock"]:has(.as-cards) .stButton>button p {{ text-align:left; font-size:.82rem; font-weight:400; color:{MUTED}; line-height:1.35; margin:0; }}
div[data-testid="stVerticalBlock"]:has(.as-cards) .stButton>button p strong {{ display:block; font-size:.9rem; font-weight:600; color:{NAVY}; }}
[class*="st-key-st"] button {{ text-align:left !important; justify-content:flex-start !important; align-items:flex-start !important; min-height:4.1rem; padding:12px 16px !important; border-radius:14px; background:{CARD}; border:1px solid {LINE}; box-shadow:0 2px 8px rgba(66,91,154,.06); }}
[class*="st-key-st"] button > div {{ width:100%; display:flex; flex-direction:column; align-items:flex-start !important; justify-content:flex-start !important; }}
[class*="st-key-st"] button [data-testid="stMarkdownContainer"] {{ width:100%; text-align:left !important; }}
[class*="st-key-st"] button p {{ text-align:left !important; margin:0 0 2px 0 !important; font-size:.82rem; font-weight:400; color:{MUTED}; line-height:1.35; }}
[class*="st-key-st"] button p strong {{ display:block; font-size:.92rem; font-weight:600; color:{NAVY}; }}
[class*="st-key-st"] button:hover {{ transform:translateY(-2px); border-color:{CYAN}; box-shadow:0 10px 22px rgba(66,91,154,.14); }}
.as-lbl {{ font-size:.8rem; font-weight:600; padding-top:8px; color:{INK}; text-align:right; }}
.as-hint {{ display:flex; justify-content:space-between; font-size:.7rem; color:{MUTED}; padding:2px 6px 0 6px; }}
.as-disc {{ text-align:center; font-size:.68rem; color:{MUTED}; padding:6px 0 2px 0; }}
[data-testid="stChatInput"] {{ border-radius:18px !important; background:#fff; border:1px solid {LINE}; box-shadow:0 6px 20px rgba(66,91,154,.10); }}
/* ---------- speed and loading feedback ---------- */
[data-testid="stStatusWidget"] {{ display:none !important; }}
.stApp[data-test-script-state="running"]::after {{ content:""; position:fixed; left:0; top:0; height:3px; width:100%; z-index:100001; background:linear-gradient(90deg,transparent,{CYAN},{TEAL},{BLUE},transparent); background-size:50% 100%; background-repeat:no-repeat; animation:skm-load 1.1s ease-in-out infinite; }}
@keyframes skm-load {{ from {{ background-position:-60% 0; }} to {{ background-position:160% 0; }} }}
.stApp[data-test-script-state="running"]::before {{ content:"Updating"; position:fixed; right:22px; bottom:22px; z-index:100000; padding:9px 16px 9px 34px; border-radius:999px; background:{NAVY}; color:#F7FAFE; font:600 .76rem 'IBM Plex Sans',sans-serif; letter-spacing:.04em; box-shadow:0 10px 26px rgba(43,58,107,.32); background-image:radial-gradient(circle at 17px 50%,{CYAN} 0 5px,transparent 6px); animation:skm-chip 1s ease-in-out infinite alternate; }}
@keyframes skm-chip {{ from {{ box-shadow:0 10px 26px rgba(43,58,107,.30); }} to {{ box-shadow:0 10px 34px rgba(118,192,236,.55); }} }}
[data-stale="true"] {{ animation:skm-stale 1s ease-in-out infinite alternate; }}
@keyframes skm-stale {{ from {{ opacity:.55; }} to {{ opacity:.8; }} }}
/* segmented section tabs (lazy) */
[data-testid="stElementContainer"]:has(.seg-marker), .element-container:has(.seg-marker), [data-testid="stElementContainer"]:has(.hist-marker), .element-container:has(.hist-marker) {{ display:none; }}
/* history */
.hist-title {{ font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.64rem; letter-spacing:.18em; text-transform:uppercase; color:{BLUE}; font-weight:600; margin:2px 2px 6px 2px; display:flex; justify-content:space-between; }}
.hist-empty {{ font-size:.78rem; color:{MUTED}; padding:6px 4px; }}
[class*="st-key-hs_"] button {{ text-align:left !important; justify-content:flex-start !important; padding:8px 10px !important; min-height:2.5rem; border-radius:12px; background:transparent; border:1px solid transparent; box-shadow:none; }}
[class*="st-key-hs_"] button > div, [class*="st-key-hs_"] button [data-testid="stMarkdownContainer"] {{ width:100%; display:flex; justify-content:flex-start !important; text-align:left !important; }}
[class*="st-key-hs_"] button p {{ text-align:left !important; font-size:.82rem; font-weight:500; margin:0; color:{INK}; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:100%; }}
[class*="st-key-hs_"] button:hover {{ background:rgba(66,91,154,.08); }}
[class*="st-key-hs_"] button[kind="primary"], [class*="st-key-hs_"] button[data-testid="stBaseButton-primary"] {{ background:linear-gradient(90deg,rgba(66,91,154,.14),rgba(118,192,236,.18)); border-color:{LINE}; }}
[class*="st-key-hs_"] button[kind="primary"] p, [class*="st-key-hs_"] button[data-testid="stBaseButton-primary"] p {{ color:{NAVY} !important; font-weight:600; }}
[class*="st-key-hd_"] button {{ min-height:2.5rem; padding:0 !important; border-radius:10px; background:transparent; border:1px solid transparent; color:{MUTED}; box-shadow:none; }}
[class*="st-key-hd_"] button:hover {{ background:rgba(214,69,93,.10); color:#D6455D; }}
[class*="st-key-tog"] button {{ border-radius:12px; font-size:.8rem; padding:4px 12px; min-height:2.3rem; }}
/* assistant v3 */
div[data-testid="stVerticalBlockBorderWrapper"]:has(.rail-marker) {{ background:linear-gradient(180deg,{CARD},#F1F2FB); border:1px solid {LINE}; border-radius:22px; box-shadow:0 10px 30px rgba(43,58,107,.09); padding:8px 6px; }}
.as-brand {{ padding:4px 2px 10px 2px; }}
[class*="st-key-newc"] button {{ background:linear-gradient(90deg,{BLUE},{CYAN}) !important; border:none !important; border-radius:14px; min-height:2.7rem; box-shadow:0 8px 20px rgba(66,91,154,.28); }}
[class*="st-key-newc"] button p {{ color:#FBFBFF !important; font-weight:600; letter-spacing:.02em; }}
[class*="st-key-newc"] button:hover {{ transform:translateY(-1px); box-shadow:0 12px 26px rgba(66,91,154,.36); }}
[class*="st-key-hist_q"] input {{ border-radius:12px; background:#fff; font-size:.86rem; }}
.hist-grp {{ font-size:.68rem; font-weight:600; color:{MUTED}; text-transform:uppercase; letter-spacing:.12em; margin:8px 4px 2px 4px; }}
.rail-foot {{ display:flex; align-items:center; gap:8px; font-size:.74rem; color:{MUTED}; padding:6px 4px 2px 4px; border-top:1px solid {LINE}; margin-top:6px; }}
.rail-foot i {{ width:7px; height:7px; border-radius:50%; background:#3CC38A; box-shadow:0 0 0 3px rgba(60,195,138,.2); }}
[class*="st-key-clrall"] button {{ background:transparent; border:1px dashed {LINE}; color:{MUTED}; font-size:.78rem; min-height:2.2rem; box-shadow:none; }}
[class*="st-key-clrall"] button:hover {{ color:#D6455D; border-color:#D6455D; }}
[class*="st-key-tog_hist"] button {{ min-height:2.4rem; padding:0 !important; border-radius:12px; font-size:1.05rem; }}
.chat-line {{ border:none; border-top:1px solid {LINE}; margin:8px 0 4px 0; }}
.as-title {{ font-size:1.1rem; max-width:560px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
div[data-testid="stExpander"] {{ border:1px solid {LINE}; border-radius:12px; background:#fff; }}
div[data-testid="stExpander"] summary p {{ font-size:.82rem; font-weight:600; }}
/* ---------- methodology flow, breadcrumb, language ---------- */
.flow {{ display:flex; align-items:stretch; gap:0; flex-wrap:wrap; margin:6px 0; }}
.flow .st {{ flex:1 1 120px; background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:10px 12px; box-shadow:0 3px 12px rgba(66,91,154,.07); animation:skm-up .5s both; }}
.flow .st b {{ display:inline-flex; width:22px; height:22px; border-radius:50%; align-items:center; justify-content:center; background:linear-gradient(135deg,{BLUE},{CYAN}); color:#fff; font-size:.72rem; font-family:'IBM Plex Mono',monospace; }}
.flow .st h5 {{ margin:6px 0 2px 0; font-size:.88rem; color:{NAVY}; }} .flow .st p {{ margin:0; font-size:.74rem; color:{MUTED}; line-height:1.35; }}
.flow .arr {{ flex:0 0 18px; align-self:center; height:2px; background:linear-gradient(90deg,{CYAN},{TEAL}); position:relative; margin:0 2px; }}
.flow .arr::after {{ content:""; position:absolute; right:-1px; top:-4px; border-left:7px solid {TEAL}; border-top:5px solid transparent; border-bottom:5px solid transparent; }}
[class*="st-key-bc_"] button {{ min-height:2.1rem; border-radius:999px; font-size:.8rem; padding:0 10px; }}
[class*="st-key-ui_lang_sel"] div[data-baseweb="select"] > div {{ min-height:2.3rem; border-radius:999px !important; }}
/* ---------- segmented section tabs v2 (scoped to the tab bar only) ---------- */
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .seg-marker) [data-testid="stRadioGroup"], div[data-testid="stVerticalBlock"]:has(> .element-container .seg-marker) [role="radiogroup"] {{ display:flex !important; flex-wrap:wrap; gap:4px !important; background:{CARD}; border:1px solid {LINE}; border-radius:16px; padding:5px; width:fit-content; max-width:100%; box-shadow:0 3px 12px rgba(66,91,154,.07); margin:14px 0 18px 0; }}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .seg-marker) [data-testid="stRadioOption"], div[data-testid="stVerticalBlock"]:has(> .element-container .seg-marker) [role="radiogroup"] > label {{ margin:0 !important; padding:8px 18px !important; border-radius:12px; cursor:pointer; }}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .seg-marker) [data-testid="stRadioOption"] > div > div:first-child:not([data-testid="stMarkdownContainer"]), div[data-testid="stVerticalBlock"]:has(> .element-container .seg-marker) [role="radiogroup"] > label > div:first-of-type {{ display:none !important; }}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .seg-marker) [data-testid="stRadioOption"] p, div[data-testid="stVerticalBlock"]:has(> .element-container .seg-marker) [role="radiogroup"] > label p {{ font-size:.88rem; font-weight:600; color:{INK}; margin:0; white-space:nowrap; }}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .seg-marker) [data-testid="stRadioOption"]:hover {{ background:rgba(66,91,154,.08); }}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .seg-marker) [data-testid="stRadioOption"][data-selected="true"], div[data-testid="stVerticalBlock"]:has(> .element-container .seg-marker) [role="radiogroup"] > label:has(input:checked) {{ background:linear-gradient(90deg,{BLUE},{CYAN}); box-shadow:0 4px 12px rgba(66,91,154,.28); }}
div[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .seg-marker) [data-testid="stRadioOption"][data-selected="true"] p, div[data-testid="stVerticalBlock"]:has(> .element-container .seg-marker) [role="radiogroup"] > label:has(input:checked) p {{ color:#FBFBFF; }}
[class*="st-key-bc_"] button p {{ white-space:nowrap; }}
/* tip, insights, footer */
.tip {{ background:linear-gradient(135deg,rgba(66,91,154,.10),rgba(118,192,236,.16)); border:1px solid {LINE}; border-radius:18px; padding:12px 16px; box-shadow:0 4px 16px rgba(66,91,154,.07); }}
.tip .tt {{ font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.66rem; letter-spacing:.2em; text-transform:uppercase; color:{BLUE}; font-weight:600; margin-bottom:8px; }}
.tip .steps {{ display:flex; gap:10px; flex-wrap:wrap; }}
.tip .step {{ display:flex; align-items:center; gap:9px; background:rgba(255,255,255,.75); border:1px solid {LINE}; border-radius:12px; padding:8px 12px; font-size:.84rem; color:{INK}; flex:1 1 220px; }}
.tip .step i {{ font-style:normal; width:22px; height:22px; border-radius:50%; background:linear-gradient(135deg,{BLUE},{CYAN}); color:#fff; font-size:.74rem; font-weight:600; display:flex; align-items:center; justify-content:center; flex:none; }}
[class*="st-key-tip_ok"] button {{ border-radius:12px; min-height:2.4rem; font-size:.82rem; }}
.ins-title {{ font-family:'IBM Plex Mono',ui-monospace,Consolas,monospace; font-size:.68rem; letter-spacing:.2em; text-transform:uppercase; color:{BLUE}; font-weight:600; margin:6px 2px 2px 2px; }}
.skm-foot {{ display:flex; flex-wrap:wrap; gap:8px 22px; justify-content:space-between; align-items:center; margin:30px 4px 6px 4px; padding:14px 18px; border-top:1px solid {LINE}; font-size:.78rem; color:{MUTED}; }}
.skm-foot b {{ color:{NAVY}; font-weight:600; }}
.skm-foot a {{ color:{BLUE}; text-decoration:none; font-weight:600; }}
.skm-foot a:hover {{ text-decoration:underline; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
try:
    import base64 as _b64
    _ico = _b64.b64encode((ASSETS / "logo_icon.png").read_bytes()).decode()
    st.markdown(f'<div class="skm-splash"><img src="data:image/png;base64,{_ico}"><div class="w">Skill<b>Map</b></div><div class="t">Labour market intelligence</div><div class="ld"><i></i></div></div>', unsafe_allow_html=True)
except Exception:
    pass


# ----------------------------------------------------------------------------------------------- helpers
def fmt(n, sign=False):
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
    return ("-" if neg else ("+" if sign and n > 0 else "")) + s


def rgba(hexc, a):
    h = hexc.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def kpi(label, value, sub="", c1=BLUE, c2=CYAN):
    return f'<div class="kpi" style="--c1:{c1};--c2:{c2}"><div class="l">{tr(label)}</div><div class="v">{value}</div><div class="s">{tr(sub)}</div><div class="bar"></div></div>'


def kpi_row(items):
    cols = st.columns(len(items))
    for c, it in zip(cols, items):
        c.markdown(kpi(*it), unsafe_allow_html=True)


def hero(title, text, tags=(), eyebrow="SkillMap | Labour market intelligence"):
    t = "".join(f'<span class="tag">{tr(x)}</span>' for x in tags)
    st.markdown(f'<div class="hero"><div class="l"><div class="eyebrow">{tr(eyebrow)}</div><h1>{tr(title)}</h1><p>{tr(text)}</p></div><div class="tags">{t}</div></div>', unsafe_allow_html=True)


def panel_card(title, sub=""):
    """Stand-alone heading card (use when the next element is not a single chart)."""
    st.markdown(f'<div class="panel"><h4>{tr(title)}</h4><div class="sub">{tr(sub)}</div></div>', unsafe_allow_html=True)


def panel_open(title, sub=""):
    st.markdown(f'<div class="panel hd"><h4>{tr(title)}</h4><div class="sub">{tr(sub)}</div></div>', unsafe_allow_html=True)


def pill(risk):
    return f'<span class="pill" style="background:{RISK_C.get(risk, BLUE)}">{risk}</span>'



# ----------------------------------------------------------------------------------------------- direct .pkl mode (no API / MySQL needed)
def find_bundle():
    for p in (HERE / "lmis_bundle.pkl", HERE / "notebook" / "lmis_bundle.pkl"):
        if p.exists():
            return p
    hits = sorted(list(HERE.glob("lmis_bundle*.pkl")) + list((HERE / "notebook").glob("lmis_bundle*.pkl")), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


@st.cache_resource(show_spinner=False)
def load_bundle():
    import pickle
    p = find_bundle()
    if p is None:
        return None
    with open(p, "rb") as fh:          # created by the notebook of this project
        return pickle.load(fh)


def local_get(path, params=None):
    B = load_bundle()
    S = pd.DataFrame(B["series"])
    p = params or {}
    if path == "/data":
        return S.to_dict("records")
    if path == "/meta":
        return {k: B[k] for k in ("meta", "policy", "risk_order", "brief_fields", "final_schema", "scenario_risk", "realloc") if k in B}
    if path == "/evaluation":
        keep = ("demand_rolling", "supply_rolling", "demand_fixed", "supply_fixed", "classifier", "classifier_confusion", "demand_audit", "supply_audit")
        return json.loads(json.dumps({k: v for k, v in B["evaluation"].items() if k in keep}, default=str))
    if path == "/kb":
        sys.path.insert(0, str(HERE))
        from load_db import build_kb
        return build_kb(B).to_dict("records")
    if path == "/explain":
        r = S[(S["district"] == p["district"]) & (S["skill"] == p["skill"]) & ((S["state"] == p["state"]) if p.get("state") else True)].iloc[0]
        return {"state": r["state"], "district": r["district"], "skill": r["skill"], "supply_drivers_12m": r["supply_drivers_12m"], "risk_drivers": r["risk_drivers"],
                "ml_risk": r["ml_risk"], "ml_probabilities": {k: float(r[f"ml_prob_{k.lower()}"]) for k in RISKS}}
    if path == "/history":
        H = B["history"]
        ids = list(H["series_ids"])
        sel = S.copy()
        if p.get("state"):
            sel = sel[sel["state"] == p["state"]]
        if p.get("skill"):
            sel = sel[sel["skill"] == p["skill"]]
        pos = [i for i, x in enumerate(ids) if x in set(sel["series_id"])]
        dem = np.asarray(H["demand_const"], float)[:, pos].sum(1)
        sup = np.asarray(H["supply_certified"], float)[:, pos].sum(1)
        enr = np.asarray(H["supply_enrolled"], float)[:, pos].sum(1)
        months = [str(m)[:7] for m in H["dates"]]
        y, m0 = int(months[-1][:4]), int(months[-1][5:7])
        fx = []
        for h in (3, 6, 12):
            m = m0 - 1 + h
            fx.append(f"{y + m // 12}-{m % 12 + 1:02d}")
        return {"months": months, "demand": [round(float(v), 1) for v in dem], "supply": [round(float(v), 1) for v in sup], "enrolled": [round(float(v), 1) for v in enr],
                "forecast_months": fx, "forecast_demand": [float(sel[f"forecast_{h}m"].sum()) for h in (3, 6, 12)], "forecast_supply": [float(sel[f"supply_{h}m"].sum()) for h in (3, 6, 12)]}
    raise KeyError(path)


@st.cache_resource(show_spinner=False)
def local_api_module():
    """Assistant without MySQL: build a SQLite copy from the .pkl and run api.py's own answer engine on it."""
    import importlib
    import local_db
    p = find_bundle()
    dbp = HERE / "skillmap_local.db"
    if (not dbp.exists()) or dbp.stat().st_mtime < p.stat().st_mtime:
        local_db.build(p, dbp)
    os.environ["DATABASE_URL"] = f"sqlite:///{dbp.as_posix()}"
    sys.path.insert(0, str(HERE))
    return importlib.import_module("api")


def local_chat(body):
    try:
        m = local_api_module()
        sid = body.get("session_id") or "local"
        out = m.answer(body["message"], sid, use_llm=True)
        out["session_id"] = sid
        out["sources"] = [x for x in out.get("sources", [])] + ["local SQLite built from lmis_bundle.pkl"]
        return out
    except Exception as e:
        return {"answer": f"The local assistant could not start: {e}", "sources": []}


def local_whatif(b):
    B = load_bundle()
    S = pd.DataFrame(B["series"])
    pol = B["policy"]
    r = S[(S["district"] == b["district"]) & (S["skill"] == b["skill"]) & ((S["state"] == b["state"]) if b.get("state") else True)].iloc[0]
    comp = max(float(r["completion_rate"]), 0.3)
    D0, S0 = float(r["forecast_12m"]), float(r["supply_12m"])
    D1, S1 = D0 * (1 + b["demand_shock_pct"] / 100.0), S0 + b["seats_delta"] * comp * pol["target_utilisation"] / 12.0

    def risk(D, S_):
        gi = (D - S_) / max(D, S_, 1.0)
        bands = pol["shortage_bands"] if gi > 0 else pol["oversupply_bands"]
        a = abs(gi)
        lvl = 3 if a >= bands[2] else 2 if a >= bands[1] else 1 if a >= bands[0] else 0
        if max(D, S_) < pol["low_volume"]:
            lvl = min(lvl, 1)
        return gi, RISKS[lvl]
    g0, r0 = risk(D0, S0); g1, r1 = risk(D1, S1)
    return {"before": {"demand": D0, "supply": S0, "gap": D0 - S0, "gap_index": g0, "risk": r0}, "after": {"demand": D1, "supply": S1, "gap": D1 - S1, "gap_index": g1, "risk": r1}}

# ----------------------------------------------------------------------------------------------- backend access
def _api_up():
    try:
        return requests.get(f"{API}/health", timeout=1.5).status_code == 200
    except Exception:
        return False


def ensure_backend():
    """True when the FastAPI backend (with MySQL) answers. Without it the app reads lmis_bundle.pkl directly.
    Set SKILLMAP_AUTOSTART=1 to let app.py launch the local backend itself."""
    if _api_up():
        return True
    if os.environ.get("SKILLMAP_AUTOSTART") == "1" and API.startswith(("http://127.0.0.1", "http://localhost")) and not st.session_state.get("_spawned"):
        st.session_state["_spawned"] = True
        port = API.rsplit(":", 1)[-1].split("/")[0]
        subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", port], cwd=str(HERE),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(20):
            time.sleep(0.5)
            if _api_up():
                return True
    return False


@st.cache_resource(show_spinner=False)
def http():
    """One keep-alive connection pool for all API calls."""
    s = requests.Session()
    s.mount("http://", HTTPAdapter(pool_connections=4, pool_maxsize=8))
    return s


@st.cache_data(ttl=900, show_spinner=False)
def api_get(path, params=None):
    if st.session_state.get("mode") == "pkl":
        return local_get(path, {k: v for k, v in (params or {}).items() if v not in (None, "", "All")})
    r = http().get(f"{API}{path}", params={k: v for k, v in (params or {}).items() if v not in (None, "", "All")}, timeout=60)
    r.raise_for_status()
    return r.json()


def api_post(path, body):
    if st.session_state.get("mode") == "pkl":
        if path == "/whatif":
            return local_whatif(body)
        return local_chat(body)
    r = http().post(f"{API}{path}", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=120, show_spinner=False)
def load_all():
    df = pd.DataFrame(api_get("/data"))
    meta = api_get("/meta")
    return df, meta


# ----------------------------------------------------------------------------------------------- engine (scenario view)
def classify(gi, D, S, pol):
    gi = np.asarray(gi, float)
    sh, ov = np.array(pol["shortage_bands"]), np.array(pol["oversupply_bands"])
    a = np.abs(gi)
    b = np.where(gi[:, None] > 0, sh[None, :], ov[None, :])
    lvl = (a[:, None] >= b).sum(1)
    lvl = np.where(np.maximum(D, S) < pol["low_volume"], np.minimum(lvl, 1), lvl)
    return np.array(RISKS)[lvl]


def scen_factor(meta, scen):
    m = meta["meta"]
    g = m["scenarios"][scen.lower()]
    return (1 + g) / (1 + m["g_base"])


def view(df, meta, h, scen):
    """Return a copy with D, S, gap, gi, risk for horizon h under a demand scenario."""
    d = df.copy()
    f = scen_factor(meta, scen)
    d["D"] = d[f"forecast_{h}m"] * f
    d["S"] = d[f"supply_{h}m"].astype(float)
    d["gap"] = d["D"] - d["S"]
    d["gi"] = d["gap"] / np.maximum(np.maximum(d["D"], d["S"]), 1.0)
    d["risk_v"] = classify(d["gi"].values, d["D"].values, d["S"].values, meta["policy"])
    return d


def agg_gap(d):
    D, S = d["D"].sum(), d["S"].sum()
    return D, S, D - S


# ----------------------------------------------------------------------------------------------- charts (area style: gradient fill, smooth line, dashed projection tail, ring markers)
LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="IBM Plex Sans, Inter, Segoe UI, sans-serif", color=INK, size=12),
              margin=dict(l=8, r=8, t=30, b=8), hoverlabel=dict(bgcolor=NAVY, font=dict(color="#EAF3FF", size=12), bordercolor=BLUE),
              legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"))


def style(fig, h=380, **kw):
    base = dict(LAYOUT)
    base.update(kw)
    fig.update_layout(height=h, **base)
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, griddash="dot", zeroline=False, linecolor="rgba(0,0,0,0)")
    return fig


def grad(color, top=0.26):
    return dict(type="vertical", colorscale=[[0, rgba(color, 0.0)], [1, rgba(color, top)]])


def area_chart(months, series, fc_months=None, h=400, title=None, band=None):
    """series: list of (name, values, colour, forecast_values or None). Solid line for history, dashed line + ring markers for the forecast."""
    x = pd.to_datetime([m + "-01" for m in months])
    fx = pd.to_datetime([m + "-01" for m in fc_months]) if fc_months else None
    fig = go.Figure()
    for name, y, col, fy in series:
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=col, width=2.6, shape="spline", smoothing=0.9),
                                 fill="tozeroy", fillgradient=grad(col), hovertemplate="%{y:,.0f}<extra>" + name + "</extra>"))
        if fy is not None and fx is not None:
            xx = [x[-1]] + list(fx)
            yy = [y[-1]] + list(fy)
            fig.add_trace(go.Scatter(x=xx, y=yy, name=name + " forecast", mode="lines+markers", line=dict(color=col, width=2.6, dash="dash", shape="spline", smoothing=0.7),
                                     marker=dict(size=[0] + [10] * len(fy), color=CARD, line=dict(color=col, width=2.5)), showlegend=False,
                                     hovertemplate="%{y:,.0f}<extra>" + name + " forecast</extra>"))
    if fx is not None:
        fig.add_vrect(x0=x[-1], x1=fx[-1], fillcolor=rgba(BLUE, 0.05), line_width=0, annotation_text="Forecast", annotation_position="top left",
                      annotation_font=dict(color=MUTED, size=11))
    if band:
        fig.add_hrect(y0=band[0], y1=band[1], fillcolor=rgba(TEAL, 0.10), line=dict(color=TEAL, width=1, dash="dash"))
    style(fig, h, hovermode="x unified", title=dict(text=title or "", font=dict(size=14)))
    return fig


def donut(counts, h=300):
    fig = go.Figure(go.Pie(labels=list(counts), values=list(counts.values()), hole=0.66, sort=False, marker=dict(colors=[RISK_C[k] for k in counts], line=dict(color=CARD, width=3)),
                           textinfo="percent", hovertemplate="%{label}: %{value} series<extra></extra>"))
    fig.add_annotation(text=f"<b>{sum(counts.values())}</b><br><span style='font-size:11px;color:{MUTED}'>series</span>", showarrow=False, font=dict(size=24, color=NAVY))
    return style(fig, h, showlegend=True, legend=dict(orientation="h", y=-0.05))


def risk_stack(d, by, h=420, top=None):
    t = d.groupby([by, "risk_v"]).size().unstack(fill_value=0).reindex(columns=RISKS, fill_value=0)
    t["_s"] = t["High"] + t["Critical"] * 2
    t = t.sort_values("_s").drop(columns="_s")
    if top:
        t = t.tail(top)
    fig = go.Figure()
    for r in RISKS:
        fig.add_trace(go.Bar(y=t.index, x=t[r], name=r, orientation="h", marker=dict(color=RISK_C[r], line=dict(color=CARD, width=1.5)), hovertemplate="%{y}: %{x} " + r + "<extra></extra>"))
    return style(fig, h, barmode="stack", bargap=0.28)


# ----------------------------------------------------------------------------------------------- shared filter bar
def lsel(container, label, options, key, **kw):
    """Selectbox whose widget key carries the interface language, so the shown option text refreshes on a language switch; the chosen value is remembered."""
    ss = st.session_state
    lang = ss.get("ui_lang", "en")
    mem = ss.get(f"{key}__v")
    idx = options.index(mem) if mem in options else kw.pop("index", 0)
    kw.pop("index", None)
    v = container.selectbox(label, options, index=idx, key=f"{key}@{lang}", **kw)
    ss[f"{key}__v"] = v
    return v


def filter_bar(df, meta, key, with_skill=True, with_scen=True):
    box = st.container(border=True)
    with box:
        st.markdown(f'<span class="filter-marker"></span><div class="f-title">{tr("Filters")}</div>', unsafe_allow_html=True)
        c = st.columns(5 if with_skill else 3, gap="medium")
    states = ["All"] + sorted(df["state"].unique())
    st_sel = lsel(c[0], tr("State"), states, f"{key}_st", format_func=tr)
    d = df if st_sel == "All" else df[df["state"] == st_sel]
    i = 1
    sk = "All"
    if with_skill:
        sectors = ["All"] + sorted(df["sector"].unique())
        sec = lsel(c[1], tr("Sector"), sectors, f"{key}_sec", format_func=tr)
        if sec != "All":
            d = d[d["sector"] == sec]
        skills = ["All"] + sorted(d["skill"].unique())
        sk = lsel(c[2], tr("Skill"), skills, f"{key}_sk", format_func=tr)
        if sk != "All":
            d = d[d["skill"] == sk]
        i = 3
    hz = lsel(c[i], tr("Horizon"), [3, 6, 12], f"{key}_h", index=2, format_func=lambda x: f"{x} {tr('months')}, {meta['meta']['target_months'][str(x)]}")
    scen = "Base"
    if with_scen:
        scen = lsel(c[i + 1], tr("Demand scenario"), ["Low", "Base", "High"], f"{key}_sc", index=1, format_func=tr) if with_skill else "Base"
    return d, st_sel, sk, hz, scen


# ===============================================================================================================================
#  PAGES
# ===============================================================================================================================
def sim_note(meta):
    if meta["meta"].get("supply_is_simulated", True):
        st.markdown('<div class="note"><b>Data note.</b> No training-registry file was supplied, so the supply side is simulated from real PMKVY district volumes and documented assumptions. '
                    'Supply, gap, risk and seat figures demonstrate the method; they are not findings about real districts. A real registry file can be plugged into the notebook without code changes.</div>',
                    unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------- interface language (English, Hindi, Marathi)
UI_LANGS = {"English": "en", "हिन्दी": "hi", "मराठी": "mr", "தமிழ்": "ta", "తెలుగు": "te", "বাংলা": "bn", "ગુજરાતી": "gu"}
_TR = {
    # navigation and common
    "Dashboard": ("डैशबोर्ड", "डॅशबोर्ड"),
    "Recommendation Engine": ("सिफारिश इंजन", "शिफारस इंजिन"),
    "Assistant": ("सहायक", "सहाय्यक"),
    "Accessibility": ("सुगम्यता", "सुलभता"),
    "Language": ("भाषा", "भाषा"),
    "Text size": ("अक्षर का आकार", "अक्षरांचा आकार"),
    "Normal": ("सामान्य", "सामान्य"),
    "Large": ("बड़ा", "मोठा"),
    "Extra large": ("बहुत बड़ा", "खूप मोठा"),
    "High contrast": ("उच्च कंट्रास्ट", "उच्च कॉन्ट्रास्ट"),
    "Reduce motion": ("एनिमेशन कम करें", "अॅनिमेशन कमी करा"),
    "Filters": ("फ़िल्टर", "फिल्टर"),
    "State": ("राज्य", "राज्य"),
    "Sector": ("क्षेत्र", "क्षेत्र"),
    "Skill": ("कौशल", "कौशल्य"),
    "Horizon": ("अवधि", "कालावधी"),
    "Demand scenario": ("मांग परिदृश्य", "मागणी परिस्थिती"),
    "Low": ("कम", "कमी"),
    "Base": ("आधार", "मूळ"),
    "High": ("अधिक", "जास्त"),
    "Medium": ("मध्यम", "मध्यम"),
    "Critical": ("गंभीर", "गंभीर"),
    "months": ("महीने", "महिने"),
    "series": ("श्रृंखला", "मालिका"),
    "states": ("राज्य", "राज्ये"),
    "districts": ("ज़िले", "जिल्हे"),
    "skills": ("कौशल", "कौशल्ये"),
    "Data to": ("डेटा", "डेटा"),
    # dashboard
    "Labour market dashboard": ("श्रम बाज़ार डैशबोर्ड", "श्रम बाजार डॅशबोर्ड"),
    "Demand and supply forecasts, skill gap and risk for every district and skill.": ("हर ज़िले और कौशल के लिए मांग-आपूर्ति पूर्वानुमान, कौशल अंतर और जोखिम।", "प्रत्येक जिल्हा आणि कौशल्यासाठी मागणी-पुरवठा अंदाज, कौशल्य तफावत आणि जोखीम."),
    "Overview": ("अवलोकन", "आढावा"),
    "Map and drill-down": ("मानचित्र और ड्रिल-डाउन", "नकाशा आणि ड्रिल-डाउन"),
    "Demand and supply": ("मांग और आपूर्ति", "मागणी आणि पुरवठा"),
    "Gap and risk": ("अंतर और जोखिम", "तफावत आणि जोखीम"),
    "Explainability": ("व्याख्यात्मकता", "स्पष्टीकरण"),
    "Accuracy": ("सटीकता", "अचूकता"),
    "Methodology": ("कार्यप्रणाली", "कार्यपद्धती"),
    "Forecast demand": ("पूर्वानुमानित मांग", "अंदाजित मागणी"),
    "Forecast supply": ("पूर्वानुमानित आपूर्ति", "अंदाजित पुरवठा"),
    "Net gap": ("शुद्ध अंतर", "निव्वळ तफावत"),
    "High and critical": ("उच्च और गंभीर", "जास्त आणि गंभीर"),
    "Training capacity": ("प्रशिक्षण क्षमता", "प्रशिक्षण क्षमता"),
    "shortage": ("कमी", "तुटवडा"),
    "certified per month": ("प्रति माह प्रमाणित", "दरमहा प्रमाणित"),
    "per month today": ("आज प्रति माह", "आज दरमहा"),
    "Demand and supply trajectory": ("मांग और आपूर्ति की दिशा", "मागणी आणि पुरवठ्याची दिशा"),
    "Monthly history (solid) and forecast at 3, 6 and 12 months (dashed, ring markers). Demand is in constant latest-year units.": ("मासिक इतिहास (ठोस रेखा) और 3, 6, 12 माह का पूर्वानुमान (बिंदीदार)। मांग नवीनतम वर्ष की स्थिर इकाइयों में है।", "मासिक इतिहास (घन रेषा) आणि 3, 6, 12 महिन्यांचा अंदाज (ठिपक्यांची). मागणी अलीकडील वर्षाच्या स्थिर एककांत आहे."),
    "Risk mix": ("जोखिम का मिश्रण", "जोखमीचे मिश्रण"),
    "Where the pressure is": ("दबाव कहाँ है", "दबाव कुठे आहे"),
    "States ranked by High and Critical series": ("उच्च और गंभीर श्रृंखला के आधार पर राज्यों की रैंकिंग", "जास्त आणि गंभीर मालिकांनुसार राज्यांची क्रमवारी"),
    "Risk today to action": ("आज का जोखिम से कार्रवाई तक", "आजची जोखीम ते कृती"),
    "Seat decision chosen for each risk class (base plan, number of series)": ("हर जोखिम वर्ग के लिए चुना गया सीट निर्णय (आधार योजना, श्रृंखला की संख्या)", "प्रत्येक जोखीम वर्गासाठी निवडलेला जागा निर्णय (मूळ योजना, मालिकांची संख्या)"),
    "Demand versus supply by skill": ("कौशल के अनुसार मांग बनाम आपूर्ति", "कौशल्यानुसार मागणी विरुद्ध पुरवठा"),
    "Demand map by sector and skill": ("क्षेत्र और कौशल के अनुसार मांग मानचित्र", "क्षेत्र आणि कौशल्यानुसार मागणी नकाशा"),
    "Demand growth outlook": ("मांग वृद्धि का अनुमान", "मागणी वाढीचा अंदाज"),
    "Gap index heatmap": ("अंतर सूचकांक हीटमैप", "तफावत निर्देशांक हीटमॅप"),
    "Risk across horizons": ("विभिन्न अवधियों में जोखिम", "विविध कालावधींतील जोखीम"),
    "Largest shortages": ("सबसे बड़ी कमी", "सर्वात मोठा तुटवडा"),
    "Largest oversupply": ("सबसे अधिक अधिशेष", "सर्वाधिक अतिरिक्त पुरवठा"),
    "Early-warning probabilities": ("पूर्व-चेतावनी संभावनाएँ", "पूर्वसूचना संभाव्यता"),
    "Forecast with uncertainty": ("अनिश्चितता सहित पूर्वानुमान", "अनिश्चिततेसह अंदाज"),
    "What drives the supply forecast": ("आपूर्ति पूर्वानुमान को क्या प्रभावित करता है", "पुरवठा अंदाजावर काय परिणाम करते"),
    "What drives the risk class": ("जोखिम वर्ग को क्या प्रभावित करता है", "जोखीम वर्गावर काय परिणाम करते"),
    # map tab
    "India": ("भारत", "भारत"),
    "Gap index": ("अंतर सूचकांक", "तफावत निर्देशांक"),
    "High and critical share": ("उच्च और गंभीर हिस्सा", "जास्त आणि गंभीर वाटा"),
    "Colour tiles by": ("टाइल का रंग", "टाइलचा रंग"),
    "Pilot state": ("पायलट राज्य", "पायलट राज्य"),
    "Not in pilot": ("पायलट में नहीं", "पायलटमध्ये नाही"),
    "Click a state tile to drill down. Red means shortage, blue means oversupply.": ("ड्रिल-डाउन के लिए राज्य की टाइल पर क्लिक करें। लाल का अर्थ कमी, नीला का अर्थ अधिशेष।", "ड्रिल-डाउनसाठी राज्याच्या टाइलवर क्लिक करा. लाल म्हणजे तुटवडा, निळा म्हणजे अतिरिक्त पुरवठा."),
    "Districts": ("ज़िले", "जिल्हे"),
    "Skills in the selected district": ("चयनित ज़िले के कौशल", "निवडलेल्या जिल्ह्यातील कौशल्ये"),
    "National view": ("राष्ट्रीय दृश्य", "राष्ट्रीय दृश्य"),
    "State view": ("राज्य दृश्य", "राज्य दृश्य"),
    "District view": ("ज़िला दृश्य", "जिल्हा दृश्य"),
    "Select a district": ("ज़िला चुनें", "जिल्हा निवडा"),
    # accuracy
    "Forecast accuracy": ("पूर्वानुमान की सटीकता", "अंदाजाची अचूकता"),
    "Rolling-origin back-test": ("रोलिंग-ओरिजिन बैक-टेस्ट", "रोलिंग-ओरिजिन बॅक-टेस्ट"),
    "Error versus simple baselines": ("सरल आधार रेखाओं की तुलना में त्रुटि", "साध्या आधाररेषांच्या तुलनेत त्रुटी"),
    "Early-warning classifier": ("पूर्व-चेतावनी वर्गीकरणकर्ता", "पूर्वसूचना वर्गीकरणकर्ता"),
    "Data quality audit": ("डेटा गुणवत्ता ऑडिट", "डेटा गुणवत्ता तपासणी"),
    # methodology
    "How the demand index is built": ("मांग सूचकांक कैसे बनता है", "मागणी निर्देशांक कसा तयार होतो"),
    "Data sources": ("डेटा स्रोत", "डेटा स्रोत"),
    "Combine your own data": ("अपना डेटा जोड़ें", "तुमचा स्वतःचा डेटा जोडा"),
    # reco
    "Seat planning": ("सीट योजना", "जागा नियोजन"),
    "Action plan": ("कार्य योजना", "कृती आराखडा"),
    "District planner": ("ज़िला योजनाकार", "जिल्हा नियोजक"),
    "What-if simulator": ("क्या-होगा सिम्युलेटर", "काय-झाले-तर सिम्युलेटर"),
    "Budget optimiser": ("बजट ऑप्टिमाइज़र", "बजेट ऑप्टिमायझर"),
    "Export": ("निर्यात", "निर्यात"),
    "Net seat change by state": ("राज्यवार शुद्ध सीट परिवर्तन", "राज्यनिहाय निव्वळ जागा बदल"),
    "Actions by priority": ("प्राथमिकता के अनुसार कार्रवाई", "प्राधान्यानुसार कृती"),
    "Ranked action plan": ("क्रमबद्ध कार्य योजना", "क्रमवार कृती आराखडा"),
    "Reallocation within the district": ("ज़िले के भीतर पुनर्आवंटन", "जिल्ह्यातील पुनर्वाटप"),
    "Download CSV": ("CSV डाउनलोड करें", "CSV डाउनलोड करा"),
    "Download Excel": ("Excel डाउनलोड करें", "Excel डाउनलोड करा"),
    "Seat targets for the next training cycle": ("अगले प्रशिक्षण चक्र के लिए सीट लक्ष्य", "पुढील प्रशिक्षण चक्रासाठी जागा लक्ष्य"),
    "Planner brief": ("योजनाकार संक्षेप", "नियोजक संक्षेप"),
    "Download PDF brief": ("PDF संक्षेप डाउनलोड करें", "PDF संक्षेप डाउनलोड करा"),
    "Download target workbook": ("लक्ष्य वर्कबुक डाउनलोड करें", "लक्ष्य वर्कबुक डाउनलोड करा"),
    "API for scheme workflows": ("योजना कार्यप्रवाह के लिए API", "योजना कार्यप्रवाहासाठी API"),
    # assistant
    "New chat": ("नई चैट", "नवीन चॅट"),
    "Search chats": ("चैट खोजें", "चॅट शोधा"),
    "Chat history": ("चैट इतिहास", "चॅट इतिहास"),
    "Today": ("आज", "आज"),
    "Yesterday": ("कल", "काल"),
    "Earlier": ("पहले", "आधी"),
    "Clear all history": ("पूरा इतिहास मिटाएँ", "संपूर्ण इतिहास पुसा"),
    "Voice settings": ("आवाज़ सेटिंग", "आवाज सेटिंग"),
    "Speech language": ("बोलने की भाषा", "बोलण्याची भाषा"),
    "Read replies aloud": ("जवाब बोलकर सुनाएँ", "उत्तरे वाचून दाखवा"),
    "AI Assistant": ("एआई सहायक", "एआय सहाय्यक"),
    "What can I help you with?": ("मैं आपकी क्या मदद कर सकता हूँ?", "मी तुम्हाला कशी मदत करू?"),
    "Your labour market analyst": ("आपका श्रम बाज़ार विश्लेषक", "तुमचा श्रम बाजार विश्लेषक"),
    "Ask about demand, supply, skill gap, risk or seat plans. Type a message or use the microphone.": ("मांग, आपूर्ति, कौशल अंतर, जोखिम या सीट योजना के बारे में पूछें। संदेश लिखें या माइक्रोफ़ोन का उपयोग करें।", "मागणी, पुरवठा, कौशल्य तफावत, जोखीम किंवा जागा नियोजनाबद्दल विचारा. संदेश लिहा किंवा मायक्रोफोन वापरा."),
    "Critical shortages": ("गंभीर कमी", "गंभीर तुटवडा"),
    "Where the gap is largest in Bihar": ("बिहार में सबसे बड़ा अंतर कहाँ है", "बिहारमध्ये सर्वात मोठी तफावत कुठे आहे"),
    "District snapshot": ("ज़िले की झलक", "जिल्ह्याचा आढावा"),
    "Demand, supply and risk for Pune": ("पुणे की मांग, आपूर्ति और जोखिम", "पुण्याची मागणी, पुरवठा आणि जोखीम"),
    "Skill demand": ("कौशल की मांग", "कौशल्य मागणी"),
    "Electrician demand in Patna": ("पटना में इलेक्ट्रीशियन की मांग", "पाटण्यात इलेक्ट्रिशियनची मागणी"),
    "Compare states": ("राज्यों की तुलना", "राज्यांची तुलना"),
    "Maharashtra versus Gujarat": ("महाराष्ट्र बनाम गुजरात", "महाराष्ट्र विरुद्ध गुजरात"),
    "Hindi / Hinglish": ("हिंदी / हिंग्लिश", "हिंदी / हिंग्लिश"),
    "Pune me welder ki demand kitni hai": ("पुणे में वेल्डर की मांग कितनी है", "पुण्यात वेल्डरची मागणी किती आहे"),
    "Method": ("पद्धति", "पद्धत"),
    "How is risk calculated": ("जोखिम की गणना कैसे होती है", "जोखमीची गणना कशी होते"),
    "Ask about any state, district or skill. Type, or tap the microphone and speak.": ("किसी भी राज्य, ज़िले या कौशल के बारे में पूछें। लिखें, या माइक्रोफ़ोन दबाकर बोलें।", "कोणत्याही राज्य, जिल्हा किंवा कौशल्याबद्दल विचारा. लिहा, किंवा मायक्रोफोन दाबून बोला."),
    "Conversation": ("बातचीत", "संभाषण"),
    "Message SkillMap AI  (English or Hinglish)": ("SkillMap AI को संदेश लिखें (हिंदी, अंग्रेज़ी या हिंग्लिश)", "SkillMap AI ला संदेश लिहा (मराठी, हिंदी, इंग्रजी)"),
    "Ready when you are.": ("आप तैयार हों तो मैं तैयार हूँ।", "तुम्ही तयार असाल तर मी तयार आहे."),
    "Enter to send": ("भेजने के लिए Enter दबाएँ", "पाठवण्यासाठी Enter दाबा"),
    "Suggested follow-ups": ("सुझाए गए अगले प्रश्न", "सुचवलेले पुढील प्रश्न"),
    "All": ("सभी", "सर्व"),
    "Seats to add": ("जोड़ने योग्य सीटें", "वाढवायच्या जागा"),
    "Seats to release": ("मुक्त करने योग्य सीटें", "मोकळ्या करायच्या जागा"),
    "annual seats": ("वार्षिक सीटें", "वार्षिक जागा"),
    "per month": ("प्रति माह", "दरमहा"),
    "Demand 12M": ("मांग 12 माह", "मागणी 12 महिने"),
    "Supply 12M": ("आपूर्ति 12 माह", "पुरवठा 12 महिने"),
    "Risk (rule-based)": ("जोखिम (नियम-आधारित)", "जोखीम (नियम-आधारित)"),
    "Seat decisions for every district and skill: increase, fill existing seats first, reduce or maintain.": ("हर ज़िले और कौशल के लिए सीट निर्णय: बढ़ाएँ, पहले मौजूदा सीटें भरें, घटाएँ या बनाए रखें।", "प्रत्येक जिल्हा आणि कौशल्यासाठी जागा निर्णय: वाढवा, आधी विद्यमान जागा भरा, कमी करा किंवा कायम ठेवा."),
    "Priority ranked": ("प्राथमिकता क्रम", "प्राधान्यक्रम"),
    "Same-district reallocation": ("उसी ज़िले में पुनर्आवंटन", "त्याच जिल्ह्यात पुनर्वाटप"),
    "States": ("राज्य", "राज्ये"),
    "Click a bar to open the state.": ("राज्य खोलने के लिए बार पर क्लिक करें।", "राज्य उघडण्यासाठी बारवर क्लिक करा."),
    "Click a bar to open the district.": ("ज़िला खोलने के लिए बार पर क्लिक करें।", "जिल्हा उघडण्यासाठी बारवर क्लिक करा."),
    "people per month": ("लोग प्रति माह", "लोक दरमहा"),
    "oversupply": ("अधिशेष", "अतिरिक्त पुरवठा"),
}


_TR_X = {
    "ta": {
        'Dashboard': 'டாஷ்போர்டு',
        'Recommendation Engine': 'பரிந்துரை இயந்திரம்',
        'Assistant': 'உதவியாளர்',
        'Accessibility': 'அணுகல்தன்மை',
        'Language': 'மொழி',
        'Text size': 'எழுத்து அளவு',
        'Normal': 'இயல்பு',
        'Large': 'பெரியது',
        'Extra large': 'மிகப் பெரியது',
        'High contrast': 'அதிக மாறுபாடு',
        'Reduce motion': 'அசைவைக் குறை',
        'Filters': 'வடிகட்டிகள்',
        'State': 'மாநிலம்',
        'Sector': 'துறை',
        'Skill': 'திறன்',
        'Horizon': 'காலம்',
        'Demand scenario': 'தேவை சூழ்நிலை',
        'Low': 'குறைவு',
        'Base': 'அடிப்படை',
        'High': 'அதிகம்',
        'Medium': 'நடுத்தரம்',
        'Critical': 'அவசரம்',
        'months': 'மாதங்கள்',
        'series': 'தொடர்கள்',
        'states': 'மாநிலங்கள்',
        'districts': 'மாவட்டங்கள்',
        'skills': 'திறன்கள்',
        'Data to': 'தரவு வரை',
        'Labour market dashboard': 'தொழிலாளர் சந்தை டாஷ்போர்டு',
        'Demand and supply forecasts, skill gap and risk for every district and skill.': 'ஒவ்வொரு மாவட்டம் மற்றும் திறனுக்கும் தேவை-வழங்கல் முன்கணிப்பு, திறன் இடைவெளி மற்றும் அபாயம்.',
        'Overview': 'மேலோட்டம்',
        'Map and drill-down': 'வரைபடம் மற்றும் ஆழ்ந்து பார்த்தல்',
        'Demand and supply': 'தேவை மற்றும் வழங்கல்',
        'Gap and risk': 'இடைவெளி மற்றும் அபாயம்',
        'Explainability': 'விளக்கத்தன்மை',
        'Accuracy': 'துல்லியம்',
        'Methodology': 'செயல்முறை',
        'Forecast demand': 'முன்கணிப்பு தேவை',
        'Forecast supply': 'முன்கணிப்பு வழங்கல்',
        'Net gap': 'நிகர இடைவெளி',
        'High and critical': 'அதிகம் மற்றும் அவசரம்',
        'Training capacity': 'பயிற்சி திறன்',
        'shortage': 'பற்றாக்குறை',
        'certified per month': 'மாதம் தோறும் சான்றிதழ் பெற்றவர்கள்',
        'per month today': 'இன்று மாதம் தோறும்',
        'Demand and supply trajectory': 'தேவை மற்றும் வழங்கலின் போக்கு',
        'Monthly history (solid) and forecast at 3, 6 and 12 months (dashed, ring markers). Demand is in constant latest-year units.': 'மாதாந்திர வரலாறு (திட்டக்கோடு) மற்றும் 3, 6, 12 மாத முன்கணிப்பு (புள்ளிக்கோடு). தேவை சமீபத்திய ஆண்டின் நிலையான அலகுகளில் உள்ளது.',
        'Risk mix': 'அபாயக் கலவை',
        'Where the pressure is': 'அழுத்தம் எங்கே உள்ளது',
        'States ranked by High and Critical series': 'அதிக மற்றும் அவசரத் தொடர்களின்படி மாநிலங்களின் வரிசை',
        'Risk today to action': 'இன்றைய அபாயம் முதல் நடவடிக்கை வரை',
        'Seat decision chosen for each risk class (base plan, number of series)': 'ஒவ்வொரு அபாய வகைக்கும் தேர்ந்தெடுக்கப்பட்ட இட முடிவு (அடிப்படைத் திட்டம், தொடர்களின் எண்ணிக்கை)',
        'Demand versus supply by skill': 'திறன் வாரியாக தேவை எதிர் வழங்கல்',
        'Demand map by sector and skill': 'துறை மற்றும் திறன் வாரியாக தேவை வரைபடம்',
        'Demand growth outlook': 'தேவை வளர்ச்சி கண்ணோட்டம்',
        'Gap index heatmap': 'இடைவெளிக் குறியீட்டு வெப்ப வரைபடம்',
        'Risk across horizons': 'பல காலங்களில் அபாயம்',
        'Largest shortages': 'மிகப் பெரிய பற்றாக்குறைகள்',
        'Largest oversupply': 'மிகப் பெரிய உபரி',
        'Early-warning probabilities': 'முன்னெச்சரிக்கை நிகழ்தகவுகள்',
        'Forecast with uncertainty': 'நிச்சயமின்மையுடன் முன்கணிப்பு',
        'What drives the supply forecast': 'வழங்கல் முன்கணிப்பை எது தீர்மானிக்கிறது',
        'What drives the risk class': 'அபாய வகையை எது தீர்மானிக்கிறது',
        'India': 'இந்தியா',
        'Gap index': 'இடைவெளிக் குறியீடு',
        'High and critical share': 'அதிக மற்றும் அவசரப் பங்கு',
        'Colour tiles by': 'ஓடுகளின் நிறம்',
        'Pilot state': 'முன்னோடி மாநிலம்',
        'Not in pilot': 'முன்னோடியில் இல்லை',
        'Click a state tile to drill down. Red means shortage, blue means oversupply.': 'ஆழ்ந்து பார்க்க மாநில ஓட்டில் கிளிக் செய்யவும். சிவப்பு என்றால் பற்றாக்குறை, நீலம் என்றால் உபரி.',
        'Districts': 'மாவட்டங்கள்',
        'Skills in the selected district': 'தேர்ந்தெடுத்த மாவட்டத்தின் திறன்கள்',
        'National view': 'தேசியப் பார்வை',
        'State view': 'மாநிலப் பார்வை',
        'District view': 'மாவட்டப் பார்வை',
        'Select a district': 'மாவட்டத்தைத் தேர்ந்தெடுக்கவும்',
        'Forecast accuracy': 'முன்கணிப்பு துல்லியம்',
        'Rolling-origin back-test': 'ரோலிங்-ஆரிஜின் பின்சோதனை',
        'Error versus simple baselines': 'எளிய அடிப்படைகளுடன் பிழை ஒப்பீடு',
        'Early-warning classifier': 'முன்னெச்சரிக்கை வகைப்படுத்தி',
        'Data quality audit': 'தரவு தரத் தணிக்கை',
        'How the demand index is built': 'தேவைக் குறியீடு எவ்வாறு உருவாகிறது',
        'Data sources': 'தரவு மூலங்கள்',
        'Combine your own data': 'உங்கள் சொந்த தரவை இணைக்கவும்',
        'Seat planning': 'இட திட்டமிடல்',
        'Action plan': 'செயல் திட்டம்',
        'District planner': 'மாவட்ட திட்டமிடுபவர்',
        'What-if simulator': 'என்ன-ஆகும் உருவகப்படுத்தி',
        'Budget optimiser': 'பட்ஜெட் மேம்படுத்தி',
        'Export': 'ஏற்றுமதி',
        'Net seat change by state': 'மாநில வாரியாக நிகர இட மாற்றம்',
        'Actions by priority': 'முன்னுரிமை வாரியாக நடவடிக்கைகள்',
        'Ranked action plan': 'வரிசைப்படுத்தப்பட்ட செயல் திட்டம்',
        'Reallocation within the district': 'மாவட்டத்திற்குள் மறுஒதுக்கீடு',
        'Download CSV': 'CSV பதிவிறக்கு',
        'Download Excel': 'Excel பதிவிறக்கு',
        'Seat targets for the next training cycle': 'அடுத்த பயிற்சிச் சுழற்சிக்கான இட இலக்குகள்',
        'Planner brief': 'திட்டமிடுபவர் சுருக்கம்',
        'Download PDF brief': 'PDF சுருக்கத்தைப் பதிவிறக்கு',
        'Download target workbook': 'இலக்கு பணிப்புத்தகத்தைப் பதிவிறக்கு',
        'API for scheme workflows': 'திட்டப் பணிப்பாய்வுகளுக்கான API',
        'New chat': 'புதிய உரையாடல்',
        'Search chats': 'உரையாடல்களைத் தேடு',
        'Chat history': 'உரையாடல் வரலாறு',
        'Today': 'இன்று',
        'Yesterday': 'நேற்று',
        'Earlier': 'முன்பு',
        'Clear all history': 'அனைத்து வரலாற்றையும் அழி',
        'Voice settings': 'குரல் அமைப்புகள்',
        'Speech language': 'பேச்சு மொழி',
        'Read replies aloud': 'பதில்களை உரக்கப் படி',
        'AI Assistant': 'AI உதவியாளர்',
        'What can I help you with?': 'நான் உங்களுக்கு எப்படி உதவலாம்?',
        'Your labour market analyst': 'உங்கள் தொழிலாளர் சந்தை ஆய்வாளர்',
        'Ask about demand, supply, skill gap, risk or seat plans. Type a message or use the microphone.': 'தேவை, வழங்கல், திறன் இடைவெளி, அபாயம் அல்லது இடத் திட்டங்கள் பற்றிக் கேளுங்கள். செய்தியை எழுதவும் அல்லது மைக்ரோஃபோனைப் பயன்படுத்தவும்.',
        'Critical shortages': 'அவசரப் பற்றாக்குறைகள்',
        'Where the gap is largest in Bihar': 'பீகாரில் இடைவெளி எங்கே அதிகம்',
        'District snapshot': 'மாவட்டச் சுருக்கம்',
        'Demand, supply and risk for Pune': 'புனேயின் தேவை, வழங்கல் மற்றும் அபாயம்',
        'Skill demand': 'திறன் தேவை',
        'Electrician demand in Patna': 'பாட்னாவில் எலக்ட்ரீஷியன் தேவை',
        'Compare states': 'மாநிலங்களை ஒப்பிடு',
        'Maharashtra versus Gujarat': 'மகாராஷ்டிரா எதிர் குஜராத்',
        'Hindi / Hinglish': 'இந்தி / ஹிங்கிலிஷ்',
        'Pune me welder ki demand kitni hai': 'புனேயில் வெல்டர் தேவை எவ்வளவு',
        'Method': 'முறை',
        'How is risk calculated': 'அபாயம் எவ்வாறு கணக்கிடப்படுகிறது',
        'Ask about any state, district or skill. Type, or tap the microphone and speak.': 'எந்த மாநிலம், மாவட்டம் அல்லது திறன் பற்றியும் கேளுங்கள். எழுதுங்கள் அல்லது மைக்ரோஃபோனைத் தட்டிப் பேசுங்கள்.',
        'Conversation': 'உரையாடல்',
        'Message SkillMap AI  (English or Hinglish)': 'SkillMap AI-க்கு செய்தி (ஆங்கிலம் அல்லது ஹிங்கிலிஷ்)',
        'Ready when you are.': 'நீங்கள் தயாரானதும் நானும் தயார்.',
        'Enter to send': 'அனுப்ப Enter அழுத்தவும்',
        'Suggested follow-ups': 'பரிந்துரைக்கப்பட்ட அடுத்த கேள்விகள்',
        'All': 'அனைத்தும்',
        'Seats to add': 'சேர்க்க வேண்டிய இடங்கள்',
        'Seats to release': 'விடுவிக்க வேண்டிய இடங்கள்',
        'annual seats': 'ஆண்டு இடங்கள்',
        'per month': 'மாதம் தோறும்',
        'Demand 12M': 'தேவை 12 மாதம்',
        'Supply 12M': 'வழங்கல் 12 மாதம்',
        'Risk (rule-based)': 'அபாயம் (விதி அடிப்படை)',
        'Seat decisions for every district and skill: increase, fill existing seats first, reduce or maintain.': 'ஒவ்வொரு மாவட்டம் மற்றும் திறனுக்கான இட முடிவுகள்: அதிகரி, முதலில் உள்ள இடங்களை நிரப்பு, குறை அல்லது தக்கவை.',
        'Priority ranked': 'முன்னுரிமை வரிசை',
        'Same-district reallocation': 'அதே மாவட்டத்தில் மறுஒதுக்கீடு',
        'States': 'மாநிலங்கள்',
        'Click a bar to open the state.': 'மாநிலத்தைத் திறக்க பட்டையில் கிளிக் செய்யவும்.',
        'Click a bar to open the district.': 'மாவட்டத்தைத் திறக்க பட்டையில் கிளிக் செய்யவும்.',
        'people per month': 'மாதம் தோறும் மக்கள்',
        'oversupply': 'உபரி',
        'Key insights': 'முக்கிய நுண்ணறிவுகள்',
        'Biggest shortage': 'மிகப் பெரிய பற்றாக்குறை',
        'Biggest oversupply': 'மிகப் பெரிய உபரி',
        'Most pressured state': 'அதிக அழுத்தம் உள்ள மாநிலம்',
        'Seats to move': 'மாற்ற வேண்டிய இடங்கள்',
        'Quick start': 'விரைவு தொடக்கம்',
        'Open Map and drill-down and click a state tile.': 'வரைபடம் தாவலைத் திறந்து ஒரு மாநில ஓட்டில் கிளிக் செய்யவும்.',
        'Check the gap and risk, then open the district.': 'இடைவெளி மற்றும் அபாயத்தைப் பார்த்து, மாவட்டத்தைத் திறக்கவும்.',
        'Export seat targets from the Recommendation Engine.': 'பரிந்துரை இயந்திரத்திலிருந்து இட இலக்குகளை ஏற்றுமதி செய்யவும்.',
        'Got it': 'புரிந்தது',
        'Open API documentation': 'API ஆவணத்தைத் திற',
        'Supply data is simulated': 'வழங்கல் தரவு உருவகப்படுத்தப்பட்டது',
    },
    "te": {
        'Dashboard': 'డాష్\u200cబోర్డ్',
        'Recommendation Engine': 'సిఫారసు ఇంజిన్',
        'Assistant': 'సహాయకుడు',
        'Accessibility': 'అందుబాటు',
        'Language': 'భాష',
        'Text size': 'అక్షర పరిమాణం',
        'Normal': 'సాధారణం',
        'Large': 'పెద్దది',
        'Extra large': 'చాలా పెద్దది',
        'High contrast': 'అధిక కాంట్రాస్ట్',
        'Reduce motion': 'కదలికలు తగ్గించు',
        'Filters': 'ఫిల్టర్లు',
        'State': 'రాష్ట్రం',
        'Sector': 'రంగం',
        'Skill': 'నైపుణ్యం',
        'Horizon': 'కాలవ్యవధి',
        'Demand scenario': 'డిమాండ్ దృశ్యం',
        'Low': 'తక్కువ',
        'Base': 'ప్రాథమిక',
        'High': 'ఎక్కువ',
        'Medium': 'మధ్యస్థం',
        'Critical': 'తీవ్రం',
        'months': 'నెలలు',
        'series': 'శ్రేణులు',
        'states': 'రాష్ట్రాలు',
        'districts': 'జిల్లాలు',
        'skills': 'నైపుణ్యాలు',
        'Data to': 'డేటా వరకు',
        'Labour market dashboard': 'శ్రామిక మార్కెట్ డాష్\u200cబోర్డ్',
        'Demand and supply forecasts, skill gap and risk for every district and skill.': 'ప్రతి జిల్లా మరియు నైపుణ్యానికి డిమాండ్-సరఫరా అంచనాలు, నైపుణ్య అంతరం మరియు ప్రమాదం.',
        'Overview': 'అవలోకనం',
        'Map and drill-down': 'మ్యాప్ మరియు లోతైన పరిశీలన',
        'Demand and supply': 'డిమాండ్ మరియు సరఫరా',
        'Gap and risk': 'అంతరం మరియు ప్రమాదం',
        'Explainability': 'వివరణాత్మకత',
        'Accuracy': 'కచ్చితత్వం',
        'Methodology': 'పద్ధతి',
        'Forecast demand': 'అంచనా డిమాండ్',
        'Forecast supply': 'అంచనా సరఫరా',
        'Net gap': 'నికర అంతరం',
        'High and critical': 'ఎక్కువ మరియు తీవ్రం',
        'Training capacity': 'శిక్షణ సామర్థ్యం',
        'shortage': 'కొరత',
        'certified per month': 'నెలకు ధృవీకరించబడినవారు',
        'per month today': 'ఈరోజు నెలకు',
        'Demand and supply trajectory': 'డిమాండ్ మరియు సరఫరా గమనం',
        'Monthly history (solid) and forecast at 3, 6 and 12 months (dashed, ring markers). Demand is in constant latest-year units.': 'నెలవారీ చరిత్ర (ఘన రేఖ) మరియు 3, 6, 12 నెలల అంచనా (చుక్కల రేఖ). డిమాండ్ తాజా సంవత్సరపు స్థిర యూనిట్లలో ఉంది.',
        'Risk mix': 'ప్రమాద మిశ్రమం',
        'Where the pressure is': 'ఒత్తిడి ఎక్కడ ఉంది',
        'States ranked by High and Critical series': 'ఎక్కువ మరియు తీవ్ర శ్రేణుల ఆధారంగా రాష్ట్రాల ర్యాంకింగ్',
        'Risk today to action': 'ఈనాటి ప్రమాదం నుండి చర్య వరకు',
        'Seat decision chosen for each risk class (base plan, number of series)': 'ప్రతి ప్రమాద తరగతికి ఎంచుకున్న సీట్ల నిర్ణయం (ప్రాథమిక ప్రణాళిక, శ్రేణుల సంఖ్య)',
        'Demand versus supply by skill': 'నైపుణ్యం వారీగా డిమాండ్ వర్సెస్ సరఫరా',
        'Demand map by sector and skill': 'రంగం మరియు నైపుణ్యం వారీగా డిమాండ్ మ్యాప్',
        'Demand growth outlook': 'డిమాండ్ వృద్ధి అంచనా',
        'Gap index heatmap': 'అంతరం సూచిక హీట్\u200cమ్యాప్',
        'Risk across horizons': 'వివిధ కాలాల్లో ప్రమాదం',
        'Largest shortages': 'అతిపెద్ద కొరతలు',
        'Largest oversupply': 'అతిపెద్ద మిగులు',
        'Early-warning probabilities': 'ముందస్తు హెచ్చరిక సంభావ్యతలు',
        'Forecast with uncertainty': 'అనిశ్చితితో అంచనా',
        'What drives the supply forecast': 'సరఫరా అంచనాను ఏది ప్రభావితం చేస్తుంది',
        'What drives the risk class': 'ప్రమాద తరగతిని ఏది ప్రభావితం చేస్తుంది',
        'India': 'భారతదేశం',
        'Gap index': 'అంతరం సూచిక',
        'High and critical share': 'ఎక్కువ మరియు తీవ్ర వాటా',
        'Colour tiles by': 'టైల్స్ రంగు',
        'Pilot state': 'పైలట్ రాష్ట్రం',
        'Not in pilot': 'పైలట్\u200cలో లేదు',
        'Click a state tile to drill down. Red means shortage, blue means oversupply.': 'లోతుగా చూడటానికి రాష్ట్ర టైల్\u200cపై క్లిక్ చేయండి. ఎరుపు అంటే కొరత, నీలం అంటే మిగులు.',
        'Districts': 'జిల్లాలు',
        'Skills in the selected district': 'ఎంచుకున్న జిల్లాలోని నైపుణ్యాలు',
        'National view': 'జాతీయ వీక్షణ',
        'State view': 'రాష్ట్ర వీక్షణ',
        'District view': 'జిల్లా వీక్షణ',
        'Select a district': 'జిల్లాను ఎంచుకోండి',
        'Forecast accuracy': 'అంచనా కచ్చితత్వం',
        'Rolling-origin back-test': 'రోలింగ్-ఆరిజిన్ బ్యాక్-టెస్ట్',
        'Error versus simple baselines': 'సరళమైన ప్రాతిపదికలతో పోలిస్తే లోపం',
        'Early-warning classifier': 'ముందస్తు హెచ్చరిక వర్గీకరణ',
        'Data quality audit': 'డేటా నాణ్యత ఆడిట్',
        'How the demand index is built': 'డిమాండ్ సూచిక ఎలా తయారవుతుంది',
        'Data sources': 'డేటా మూలాలు',
        'Combine your own data': 'మీ స్వంత డేటాను కలపండి',
        'Seat planning': 'సీట్ల ప్రణాళిక',
        'Action plan': 'కార్యాచరణ ప్రణాళిక',
        'District planner': 'జిల్లా ప్లానర్',
        'What-if simulator': 'ఏమైతే సిమ్యులేటర్',
        'Budget optimiser': 'బడ్జెట్ ఆప్టిమైజర్',
        'Export': 'ఎగుమతి',
        'Net seat change by state': 'రాష్ట్రాల వారీగా నికర సీట్ల మార్పు',
        'Actions by priority': 'ప్రాధాన్యత వారీగా చర్యలు',
        'Ranked action plan': 'క్రమబద్ధీకరించిన కార్యాచరణ ప్రణాళిక',
        'Reallocation within the district': 'జిల్లా లోపల తిరిగి కేటాయింపు',
        'Download CSV': 'CSV డౌన్\u200cలోడ్ చేయండి',
        'Download Excel': 'Excel డౌన్\u200cలోడ్ చేయండి',
        'Seat targets for the next training cycle': 'తదుపరి శిక్షణ చక్రానికి సీట్ల లక్ష్యాలు',
        'Planner brief': 'ప్లానర్ సారాంశం',
        'Download PDF brief': 'PDF సారాంశం డౌన్\u200cలోడ్ చేయండి',
        'Download target workbook': 'లక్ష్య వర్క్\u200cబుక్ డౌన్\u200cలోడ్ చేయండి',
        'API for scheme workflows': 'పథక వర్క్\u200cఫ్లోల కోసం API',
        'New chat': 'కొత్త చాట్',
        'Search chats': 'చాట్\u200cలను వెతకండి',
        'Chat history': 'చాట్ చరిత్ర',
        'Today': 'ఈరోజు',
        'Yesterday': 'నిన్న',
        'Earlier': 'ఇంతకుముందు',
        'Clear all history': 'మొత్తం చరిత్రను తొలగించు',
        'Voice settings': 'వాయిస్ సెట్టింగ్\u200cలు',
        'Speech language': 'మాట్లాడే భాష',
        'Read replies aloud': 'సమాధానాలను బిగ్గరగా చదువు',
        'AI Assistant': 'AI సహాయకుడు',
        'What can I help you with?': 'నేను మీకు ఎలా సహాయపడగలను?',
        'Your labour market analyst': 'మీ శ్రామిక మార్కెట్ విశ్లేషకుడు',
        'Ask about demand, supply, skill gap, risk or seat plans. Type a message or use the microphone.': 'డిమాండ్, సరఫరా, నైపుణ్య అంతరం, ప్రమాదం లేదా సీట్ల ప్రణాళికల గురించి అడగండి. సందేశం టైప్ చేయండి లేదా మైక్రోఫోన్ ఉపయోగించండి.',
        'Critical shortages': 'తీవ్ర కొరతలు',
        'Where the gap is largest in Bihar': 'బీహార్\u200cలో అంతరం ఎక్కడ ఎక్కువ',
        'District snapshot': 'జిల్లా స్నాప్\u200cషాట్',
        'Demand, supply and risk for Pune': 'పూణే డిమాండ్, సరఫరా మరియు ప్రమాదం',
        'Skill demand': 'నైపుణ్య డిమాండ్',
        'Electrician demand in Patna': 'పాట్నాలో ఎలక్ట్రీషియన్ డిమాండ్',
        'Compare states': 'రాష్ట్రాల పోలిక',
        'Maharashtra versus Gujarat': 'మహారాష్ట్ర వర్సెస్ గుజరాత్',
        'Hindi / Hinglish': 'హిందీ / హింగ్లీష్',
        'Pune me welder ki demand kitni hai': 'పూణేలో వెల్డర్ డిమాండ్ ఎంత',
        'Method': 'పద్ధతి',
        'How is risk calculated': 'ప్రమాదం ఎలా లెక్కిస్తారు',
        'Ask about any state, district or skill. Type, or tap the microphone and speak.': 'ఏ రాష్ట్రం, జిల్లా లేదా నైపుణ్యం గురించైనా అడగండి. టైప్ చేయండి లేదా మైక్రోఫోన్ నొక్కి మాట్లాడండి.',
        'Conversation': 'సంభాషణ',
        'Message SkillMap AI  (English or Hinglish)': 'SkillMap AIకి సందేశం (ఇంగ్లీష్ లేదా హింగ్లీష్)',
        'Ready when you are.': 'మీరు సిద్ధమైతే నేనూ సిద్ధం.',
        'Enter to send': 'పంపడానికి Enter నొక్కండి',
        'Suggested follow-ups': 'సూచించిన తదుపరి ప్రశ్నలు',
        'All': 'అన్నీ',
        'Seats to add': 'జోడించాల్సిన సీట్లు',
        'Seats to release': 'విడుదల చేయాల్సిన సీట్లు',
        'annual seats': 'వార్షిక సీట్లు',
        'per month': 'నెలకు',
        'Demand 12M': 'డిమాండ్ 12 నెలలు',
        'Supply 12M': 'సరఫరా 12 నెలలు',
        'Risk (rule-based)': 'ప్రమాదం (నియమ ఆధారిత)',
        'Seat decisions for every district and skill: increase, fill existing seats first, reduce or maintain.': 'ప్రతి జిల్లా మరియు నైపుణ్యానికి సీట్ల నిర్ణయాలు: పెంచు, ముందు ఉన్న సీట్లు నింపు, తగ్గించు లేదా కొనసాగించు.',
        'Priority ranked': 'ప్రాధాన్యత క్రమం',
        'Same-district reallocation': 'అదే జిల్లాలో తిరిగి కేటాయింపు',
        'States': 'రాష్ట్రాలు',
        'Click a bar to open the state.': 'రాష్ట్రాన్ని తెరవడానికి బార్\u200cపై క్లిక్ చేయండి.',
        'Click a bar to open the district.': 'జిల్లాను తెరవడానికి బార్\u200cపై క్లిక్ చేయండి.',
        'people per month': 'నెలకు ప్రజలు',
        'oversupply': 'మిగులు',
        'Key insights': 'ముఖ్య అంతర్దృష్టులు',
        'Biggest shortage': 'అతిపెద్ద కొరత',
        'Biggest oversupply': 'అతిపెద్ద మిగులు',
        'Most pressured state': 'ఎక్కువ ఒత్తిడి ఉన్న రాష్ట్రం',
        'Seats to move': 'మార్చవలసిన సీట్లు',
        'Quick start': 'త్వరిత ప్రారంభం',
        'Open Map and drill-down and click a state tile.': 'మ్యాప్ ట్యాబ్ తెరిచి ఒక రాష్ట్ర టైల్\u200cపై క్లిక్ చేయండి.',
        'Check the gap and risk, then open the district.': 'అంతరం మరియు ప్రమాదం చూసి, జిల్లాను తెరవండి.',
        'Export seat targets from the Recommendation Engine.': 'సిఫారసు ఇంజిన్ నుండి సీట్ల లక్ష్యాలను ఎగుమతి చేయండి.',
        'Got it': 'అర్థమైంది',
        'Open API documentation': 'API డాక్యుమెంటేషన్ తెరవండి',
        'Supply data is simulated': 'సరఫరా డేటా అనుకరణ చేయబడింది',
    },
    "bn": {
        'Dashboard': 'ড্যাশবোর্ড',
        'Recommendation Engine': 'সুপারিশ ইঞ্জিন',
        'Assistant': 'সহায়ক',
        'Accessibility': 'অ্যাক্সেসিবিলিটি',
        'Language': 'ভাষা',
        'Text size': 'লেখার আকার',
        'Normal': 'স্বাভাবিক',
        'Large': 'বড়',
        'Extra large': 'অনেক বড়',
        'High contrast': 'উচ্চ কনট্রাস্ট',
        'Reduce motion': 'অ্যানিমেশন কমান',
        'Filters': 'ফিল্টার',
        'State': 'রাজ্য',
        'Sector': 'ক্ষেত্র',
        'Skill': 'দক্ষতা',
        'Horizon': 'সময়কাল',
        'Demand scenario': 'চাহিদার পরিস্থিতি',
        'Low': 'কম',
        'Base': 'মূল',
        'High': 'বেশি',
        'Medium': 'মাঝারি',
        'Critical': 'সংকটজনক',
        'months': 'মাস',
        'series': 'ধারা',
        'states': 'রাজ্য',
        'districts': 'জেলা',
        'skills': 'দক্ষতা',
        'Data to': 'ডেটা পর্যন্ত',
        'Labour market dashboard': 'শ্রমবাজার ড্যাশবোর্ড',
        'Demand and supply forecasts, skill gap and risk for every district and skill.': 'প্রতিটি জেলা ও দক্ষতার জন্য চাহিদা-জোগান পূর্বাভাস, দক্ষতার ঘাটতি ও ঝুঁকি।',
        'Overview': 'সারসংক্ষেপ',
        'Map and drill-down': 'মানচিত্র ও বিশদ বিশ্লেষণ',
        'Demand and supply': 'চাহিদা ও জোগান',
        'Gap and risk': 'ঘাটতি ও ঝুঁকি',
        'Explainability': 'ব্যাখ্যাযোগ্যতা',
        'Accuracy': 'নির্ভুলতা',
        'Methodology': 'পদ্ধতি',
        'Forecast demand': 'পূর্বাভাসিত চাহিদা',
        'Forecast supply': 'পূর্বাভাসিত জোগান',
        'Net gap': 'নিট ঘাটতি',
        'High and critical': 'উচ্চ ও সংকটজনক',
        'Training capacity': 'প্রশিক্ষণ ক্ষমতা',
        'shortage': 'ঘাটতি',
        'certified per month': 'প্রতি মাসে প্রত্যয়িত',
        'per month today': 'আজ প্রতি মাসে',
        'Demand and supply trajectory': 'চাহিদা ও জোগানের গতিপথ',
        'Monthly history (solid) and forecast at 3, 6 and 12 months (dashed, ring markers). Demand is in constant latest-year units.': 'মাসিক ইতিহাস (অখণ্ড রেখা) এবং ৩, ৬, ১২ মাসের পূর্বাভাস (বিন্দুযুক্ত রেখা)। চাহিদা সাম্প্রতিক বছরের স্থির এককে দেওয়া।',
        'Risk mix': 'ঝুঁকির মিশ্রণ',
        'Where the pressure is': 'চাপ কোথায়',
        'States ranked by High and Critical series': 'উচ্চ ও সংকটজনক ধারা অনুযায়ী রাজ্যের ক্রম',
        'Risk today to action': 'আজকের ঝুঁকি থেকে পদক্ষেপ',
        'Seat decision chosen for each risk class (base plan, number of series)': 'প্রতিটি ঝুঁকি শ্রেণির জন্য নির্বাচিত আসনের সিদ্ধান্ত (মূল পরিকল্পনা, ধারার সংখ্যা)',
        'Demand versus supply by skill': 'দক্ষতা অনুযায়ী চাহিদা বনাম জোগান',
        'Demand map by sector and skill': 'ক্ষেত্র ও দক্ষতা অনুযায়ী চাহিদার মানচিত্র',
        'Demand growth outlook': 'চাহিদা বৃদ্ধির পূর্বাভাস',
        'Gap index heatmap': 'ঘাটতি সূচকের হিটম্যাপ',
        'Risk across horizons': 'বিভিন্ন সময়কালে ঝুঁকি',
        'Largest shortages': 'সবচেয়ে বড় ঘাটতি',
        'Largest oversupply': 'সবচেয়ে বড় উদ্বৃত্ত',
        'Early-warning probabilities': 'পূর্ব-সতর্কতার সম্ভাবনা',
        'Forecast with uncertainty': 'অনিশ্চয়তাসহ পূর্বাভাস',
        'What drives the supply forecast': 'জোগানের পূর্বাভাস কীসের উপর নির্ভর করে',
        'What drives the risk class': 'ঝুঁকির শ্রেণি কীসের উপর নির্ভর করে',
        'India': 'ভারত',
        'Gap index': 'ঘাটতি সূচক',
        'High and critical share': 'উচ্চ ও সংকটজনক অংশ',
        'Colour tiles by': 'টাইলের রং',
        'Pilot state': 'পাইলট রাজ্য',
        'Not in pilot': 'পাইলটে নেই',
        'Click a state tile to drill down. Red means shortage, blue means oversupply.': 'বিশদ দেখতে রাজ্যের টাইলে ক্লিক করুন। লাল মানে ঘাটতি, নীল মানে উদ্বৃত্ত।',
        'Districts': 'জেলা',
        'Skills in the selected district': 'নির্বাচিত জেলার দক্ষতা',
        'National view': 'জাতীয় দৃশ্য',
        'State view': 'রাজ্য দৃশ্য',
        'District view': 'জেলা দৃশ্য',
        'Select a district': 'একটি জেলা বেছে নিন',
        'Forecast accuracy': 'পূর্বাভাসের নির্ভুলতা',
        'Rolling-origin back-test': 'রোলিং-অরিজিন ব্যাক-টেস্ট',
        'Error versus simple baselines': 'সরল ভিত্তির সঙ্গে ত্রুটির তুলনা',
        'Early-warning classifier': 'পূর্ব-সতর্কতা শ্রেণিবিন্যাসক',
        'Data quality audit': 'ডেটার মান নিরীক্ষা',
        'How the demand index is built': 'চাহিদা সূচক কীভাবে তৈরি হয়',
        'Data sources': 'ডেটার উৎস',
        'Combine your own data': 'নিজের ডেটা যোগ করুন',
        'Seat planning': 'আসন পরিকল্পনা',
        'Action plan': 'কর্মপরিকল্পনা',
        'District planner': 'জেলা পরিকল্পক',
        'What-if simulator': 'কী-হলে সিমুলেটর',
        'Budget optimiser': 'বাজেট অপটিমাইজার',
        'Export': 'রপ্তানি',
        'Net seat change by state': 'রাজ্য অনুযায়ী নিট আসন পরিবর্তন',
        'Actions by priority': 'অগ্রাধিকার অনুযায়ী পদক্ষেপ',
        'Ranked action plan': 'ক্রমানুসারে কর্মপরিকল্পনা',
        'Reallocation within the district': 'জেলার ভিতরে পুনর্বণ্টন',
        'Download CSV': 'CSV ডাউনলোড করুন',
        'Download Excel': 'Excel ডাউনলোড করুন',
        'Seat targets for the next training cycle': 'পরবর্তী প্রশিক্ষণ চক্রের আসন লক্ষ্য',
        'Planner brief': 'পরিকল্পকের সারসংক্ষেপ',
        'Download PDF brief': 'PDF সারসংক্ষেপ ডাউনলোড করুন',
        'Download target workbook': 'লক্ষ্য ওয়ার্কবুক ডাউনলোড করুন',
        'API for scheme workflows': 'প্রকল্পের কর্মপ্রবাহের জন্য API',
        'New chat': 'নতুন চ্যাট',
        'Search chats': 'চ্যাট খুঁজুন',
        'Chat history': 'চ্যাটের ইতিহাস',
        'Today': 'আজ',
        'Yesterday': 'গতকাল',
        'Earlier': 'আগে',
        'Clear all history': 'সব ইতিহাস মুছুন',
        'Voice settings': 'ভয়েস সেটিংস',
        'Speech language': 'কথা বলার ভাষা',
        'Read replies aloud': 'উত্তর জোরে পড়ে শোনান',
        'AI Assistant': 'এআই সহায়ক',
        'What can I help you with?': 'আমি আপনাকে কীভাবে সাহায্য করতে পারি?',
        'Your labour market analyst': 'আপনার শ্রমবাজার বিশ্লেষক',
        'Ask about demand, supply, skill gap, risk or seat plans. Type a message or use the microphone.': 'চাহিদা, জোগান, দক্ষতার ঘাটতি, ঝুঁকি বা আসন পরিকল্পনা সম্পর্কে জিজ্ঞাসা করুন। বার্তা লিখুন বা মাইক্রোফোন ব্যবহার করুন।',
        'Critical shortages': 'সংকটজনক ঘাটতি',
        'Where the gap is largest in Bihar': 'বিহারে ঘাটতি কোথায় সবচেয়ে বেশি',
        'District snapshot': 'জেলার ঝলক',
        'Demand, supply and risk for Pune': 'পুনের চাহিদা, জোগান ও ঝুঁকি',
        'Skill demand': 'দক্ষতার চাহিদা',
        'Electrician demand in Patna': 'পাটনায় ইলেকট্রিশিয়ানের চাহিদা',
        'Compare states': 'রাজ্যের তুলনা',
        'Maharashtra versus Gujarat': 'মহারাষ্ট্র বনাম গুজরাট',
        'Hindi / Hinglish': 'হিন্দি / হিংলিশ',
        'Pune me welder ki demand kitni hai': 'পুনেতে ওয়েল্ডারের চাহিদা কত',
        'Method': 'পদ্ধতি',
        'How is risk calculated': 'ঝুঁকি কীভাবে হিসাব করা হয়',
        'Ask about any state, district or skill. Type, or tap the microphone and speak.': 'যেকোনো রাজ্য, জেলা বা দক্ষতা সম্পর্কে জিজ্ঞাসা করুন। লিখুন, অথবা মাইক্রোফোন চেপে বলুন।',
        'Conversation': 'কথোপকথন',
        'Message SkillMap AI  (English or Hinglish)': 'SkillMap AI-কে বার্তা (ইংরেজি বা হিংলিশ)',
        'Ready when you are.': 'আপনি প্রস্তুত হলেই আমি প্রস্তুত।',
        'Enter to send': 'পাঠাতে Enter চাপুন',
        'Suggested follow-ups': 'প্রস্তাবিত পরবর্তী প্রশ্ন',
        'All': 'সব',
        'Seats to add': 'যোগ করার আসন',
        'Seats to release': 'ছাড়ার আসন',
        'annual seats': 'বার্ষিক আসন',
        'per month': 'প্রতি মাসে',
        'Demand 12M': 'চাহিদা ১২ মাস',
        'Supply 12M': 'জোগান ১২ মাস',
        'Risk (rule-based)': 'ঝুঁকি (নিয়মভিত্তিক)',
        'Seat decisions for every district and skill: increase, fill existing seats first, reduce or maintain.': 'প্রতিটি জেলা ও দক্ষতার জন্য আসনের সিদ্ধান্ত: বাড়ান, আগে বিদ্যমান আসন পূরণ করুন, কমান বা বজায় রাখুন।',
        'Priority ranked': 'অগ্রাধিকারক্রম',
        'Same-district reallocation': 'একই জেলার মধ্যে পুনর্বণ্টন',
        'States': 'রাজ্য',
        'Click a bar to open the state.': 'রাজ্য খুলতে বারে ক্লিক করুন।',
        'Click a bar to open the district.': 'জেলা খুলতে বারে ক্লিক করুন।',
        'people per month': 'প্রতি মাসে মানুষ',
        'oversupply': 'উদ্বৃত্ত',
        'Key insights': 'মূল অন্তর্দৃষ্টি',
        'Biggest shortage': 'সবচেয়ে বড় ঘাটতি',
        'Biggest oversupply': 'সবচেয়ে বড় উদ্বৃত্ত',
        'Most pressured state': 'সবচেয়ে চাপে থাকা রাজ্য',
        'Seats to move': 'স্থানান্তরযোগ্য আসন',
        'Quick start': 'দ্রুত শুরু',
        'Open Map and drill-down and click a state tile.': 'মানচিত্র ট্যাব খুলে একটি রাজ্যের টাইলে ক্লিক করুন।',
        'Check the gap and risk, then open the district.': 'ঘাটতি ও ঝুঁকি দেখে জেলা খুলুন।',
        'Export seat targets from the Recommendation Engine.': 'সুপারিশ ইঞ্জিন থেকে আসন লক্ষ্য রপ্তানি করুন।',
        'Got it': 'বুঝেছি',
        'Open API documentation': 'API ডকুমেন্টেশন খুলুন',
        'Supply data is simulated': 'জোগানের ডেটা সিমুলেটেড',
    },
    "gu": {
        'Dashboard': 'ડેશબોર્ડ',
        'Recommendation Engine': 'ભલામણ એન્જિન',
        'Assistant': 'સહાયક',
        'Accessibility': 'સુલભતા',
        'Language': 'ભાષા',
        'Text size': 'અક્ષરનું કદ',
        'Normal': 'સામાન્ય',
        'Large': 'મોટું',
        'Extra large': 'ખૂબ મોટું',
        'High contrast': 'ઉચ્ચ કોન્ટ્રાસ્ટ',
        'Reduce motion': 'એનિમેશન ઘટાડો',
        'Filters': 'ફિલ્ટર',
        'State': 'રાજ્ય',
        'Sector': 'ક્ષેત્ર',
        'Skill': 'કૌશલ્ય',
        'Horizon': 'સમયગાળો',
        'Demand scenario': 'માંગ પરિદૃશ્ય',
        'Low': 'ઓછું',
        'Base': 'મૂળ',
        'High': 'વધુ',
        'Medium': 'મધ્યમ',
        'Critical': 'ગંભીર',
        'months': 'મહિના',
        'series': 'શ્રેણી',
        'states': 'રાજ્યો',
        'districts': 'જિલ્લા',
        'skills': 'કૌશલ્યો',
        'Data to': 'ડેટા સુધી',
        'Labour market dashboard': 'શ્રમ બજાર ડેશબોર્ડ',
        'Demand and supply forecasts, skill gap and risk for every district and skill.': 'દરેક જિલ્લા અને કૌશલ્ય માટે માંગ-પુરવઠાનું અનુમાન, કૌશલ્ય ખાઈ અને જોખમ.',
        'Overview': 'અવલોકન',
        'Map and drill-down': 'નકશો અને વિગતવાર દર્શન',
        'Demand and supply': 'માંગ અને પુરવઠો',
        'Gap and risk': 'ખાઈ અને જોખમ',
        'Explainability': 'સમજાવટક્ષમતા',
        'Accuracy': 'ચોકસાઈ',
        'Methodology': 'પદ્ધતિ',
        'Forecast demand': 'અનુમાનિત માંગ',
        'Forecast supply': 'અનુમાનિત પુરવઠો',
        'Net gap': 'ચોખ્ખી ખાઈ',
        'High and critical': 'વધુ અને ગંભીર',
        'Training capacity': 'તાલીમ ક્ષમતા',
        'shortage': 'અછત',
        'certified per month': 'દર મહિને પ્રમાણિત',
        'per month today': 'આજે દર મહિને',
        'Demand and supply trajectory': 'માંગ અને પુરવઠાની દિશા',
        'Monthly history (solid) and forecast at 3, 6 and 12 months (dashed, ring markers). Demand is in constant latest-year units.': 'માસિક ઇતિહાસ (સળંગ રેખા) અને 3, 6, 12 મહિનાનું અનુમાન (ટપકાંવાળી રેખા). માંગ તાજેતરના વર્ષના સ્થિર એકમોમાં છે.',
        'Risk mix': 'જોખમનું મિશ્રણ',
        'Where the pressure is': 'દબાણ ક્યાં છે',
        'States ranked by High and Critical series': 'વધુ અને ગંભીર શ્રેણી મુજબ રાજ્યોનો ક્રમ',
        'Risk today to action': 'આજના જોખમથી કાર્યવાહી સુધી',
        'Seat decision chosen for each risk class (base plan, number of series)': 'દરેક જોખમ વર્ગ માટે પસંદ કરેલો બેઠકોનો નિર્ણય (મૂળ યોજના, શ્રેણીની સંખ્યા)',
        'Demand versus supply by skill': 'કૌશલ્ય મુજબ માંગ વિરુદ્ધ પુરવઠો',
        'Demand map by sector and skill': 'ક્ષેત્ર અને કૌશલ્ય મુજબ માંગનો નકશો',
        'Demand growth outlook': 'માંગ વૃદ્ધિનો અંદાજ',
        'Gap index heatmap': 'ખાઈ સૂચકાંક હીટમેપ',
        'Risk across horizons': 'જુદા જુદા સમયગાળામાં જોખમ',
        'Largest shortages': 'સૌથી મોટી અછત',
        'Largest oversupply': 'સૌથી મોટો વધારો',
        'Early-warning probabilities': 'પૂર્વ-ચેતવણી સંભાવનાઓ',
        'Forecast with uncertainty': 'અનિશ્ચિતતા સાથે અનુમાન',
        'What drives the supply forecast': 'પુરવઠાના અનુમાનને શું અસર કરે છે',
        'What drives the risk class': 'જોખમ વર્ગને શું અસર કરે છે',
        'India': 'ભારત',
        'Gap index': 'ખાઈ સૂચકાંક',
        'High and critical share': 'વધુ અને ગંભીર હિસ્સો',
        'Colour tiles by': 'ટાઇલનો રંગ',
        'Pilot state': 'પાયલોટ રાજ્ય',
        'Not in pilot': 'પાયલોટમાં નથી',
        'Click a state tile to drill down. Red means shortage, blue means oversupply.': 'વિગતવાર જોવા માટે રાજ્યની ટાઇલ પર ક્લિક કરો. લાલ એટલે અછત, વાદળી એટલે વધારો.',
        'Districts': 'જિલ્લા',
        'Skills in the selected district': 'પસંદ કરેલા જિલ્લાના કૌશલ્યો',
        'National view': 'રાષ્ટ્રીય દૃશ્ય',
        'State view': 'રાજ્ય દૃશ્ય',
        'District view': 'જિલ્લા દૃશ્ય',
        'Select a district': 'જિલ્લો પસંદ કરો',
        'Forecast accuracy': 'અનુમાનની ચોકસાઈ',
        'Rolling-origin back-test': 'રોલિંગ-ઓરિજિન બેક-ટેસ્ટ',
        'Error versus simple baselines': 'સરળ આધારરેખાઓ સામે ભૂલ',
        'Early-warning classifier': 'પૂર્વ-ચેતવણી વર્ગીકરણકર્તા',
        'Data quality audit': 'ડેટા ગુણવત્તા ઓડિટ',
        'How the demand index is built': 'માંગ સૂચકાંક કેવી રીતે બને છે',
        'Data sources': 'ડેટા સ્ત્રોતો',
        'Combine your own data': 'તમારો પોતાનો ડેટા ઉમેરો',
        'Seat planning': 'બેઠક આયોજન',
        'Action plan': 'કાર્ય યોજના',
        'District planner': 'જિલ્લા આયોજક',
        'What-if simulator': 'શું-થશે સિમ્યુલેટર',
        'Budget optimiser': 'બજેટ ઓપ્ટિમાઇઝર',
        'Export': 'નિકાસ',
        'Net seat change by state': 'રાજ્ય મુજબ ચોખ્ખો બેઠક ફેરફાર',
        'Actions by priority': 'અગ્રતા મુજબ કાર્યવાહી',
        'Ranked action plan': 'ક્રમબદ્ધ કાર્ય યોજના',
        'Reallocation within the district': 'જિલ્લાની અંદર પુનઃફાળવણી',
        'Download CSV': 'CSV ડાઉનલોડ કરો',
        'Download Excel': 'Excel ડાઉનલોડ કરો',
        'Seat targets for the next training cycle': 'આગામી તાલીમ ચક્ર માટે બેઠક લક્ષ્યો',
        'Planner brief': 'આયોજક સારાંશ',
        'Download PDF brief': 'PDF સારાંશ ડાઉનલોડ કરો',
        'Download target workbook': 'લક્ષ્ય વર્કબુક ડાઉનલોડ કરો',
        'API for scheme workflows': 'યોજના કાર્યપ્રવાહ માટે API',
        'New chat': 'નવી ચેટ',
        'Search chats': 'ચેટ શોધો',
        'Chat history': 'ચેટ ઇતિહાસ',
        'Today': 'આજે',
        'Yesterday': 'ગઈકાલે',
        'Earlier': 'અગાઉ',
        'Clear all history': 'બધો ઇતિહાસ સાફ કરો',
        'Voice settings': 'અવાજ સેટિંગ્સ',
        'Speech language': 'બોલવાની ભાષા',
        'Read replies aloud': 'જવાબ મોટેથી વાંચો',
        'AI Assistant': 'AI સહાયક',
        'What can I help you with?': 'હું તમને કેવી રીતે મદદ કરી શકું?',
        'Your labour market analyst': 'તમારા શ્રમ બજાર વિશ્લેષક',
        'Ask about demand, supply, skill gap, risk or seat plans. Type a message or use the microphone.': 'માંગ, પુરવઠો, કૌશલ્ય ખાઈ, જોખમ અથવા બેઠક યોજના વિશે પૂછો. સંદેશ લખો અથવા માઇક્રોફોનનો ઉપયોગ કરો.',
        'Critical shortages': 'ગંભીર અછત',
        'Where the gap is largest in Bihar': 'બિહારમાં ખાઈ ક્યાં સૌથી વધુ છે',
        'District snapshot': 'જિલ્લાની ઝલક',
        'Demand, supply and risk for Pune': 'પુણેની માંગ, પુરવઠો અને જોખમ',
        'Skill demand': 'કૌશલ્યની માંગ',
        'Electrician demand in Patna': 'પટનામાં ઇલેક્ટ્રિશિયનની માંગ',
        'Compare states': 'રાજ્યોની સરખામણી',
        'Maharashtra versus Gujarat': 'મહારાષ્ટ્ર વિરુદ્ધ ગુજરાત',
        'Hindi / Hinglish': 'હિન્દી / હિંગ્લિશ',
        'Pune me welder ki demand kitni hai': 'પુણેમાં વેલ્ડરની માંગ કેટલી છે',
        'Method': 'પદ્ધતિ',
        'How is risk calculated': 'જોખમની ગણતરી કેવી રીતે થાય છે',
        'Ask about any state, district or skill. Type, or tap the microphone and speak.': 'કોઈપણ રાજ્ય, જિલ્લા કે કૌશલ્ય વિશે પૂછો. લખો, અથવા માઇક્રોફોન દબાવીને બોલો.',
        'Conversation': 'વાતચીત',
        'Message SkillMap AI  (English or Hinglish)': 'SkillMap AI ને સંદેશ (અંગ્રેજી અથવા હિંગ્લિશ)',
        'Ready when you are.': 'તમે તૈયાર હો ત્યારે હું તૈયાર છું.',
        'Enter to send': 'મોકલવા માટે Enter દબાવો',
        'Suggested follow-ups': 'સૂચવેલા આગળના પ્રશ્નો',
        'All': 'બધા',
        'Seats to add': 'ઉમેરવાની બેઠકો',
        'Seats to release': 'મુક્ત કરવાની બેઠકો',
        'annual seats': 'વાર્ષિક બેઠકો',
        'per month': 'દર મહિને',
        'Demand 12M': 'માંગ 12 મહિના',
        'Supply 12M': 'પુરવઠો 12 મહિના',
        'Risk (rule-based)': 'જોખમ (નિયમ આધારિત)',
        'Seat decisions for every district and skill: increase, fill existing seats first, reduce or maintain.': 'દરેક જિલ્લા અને કૌશલ્ય માટે બેઠકના નિર્ણયો: વધારો, પહેલા હાલની બેઠકો ભરો, ઘટાડો અથવા જાળવો.',
        'Priority ranked': 'અગ્રતા ક્રમ',
        'Same-district reallocation': 'એ જ જિલ્લામાં પુનઃફાળવણી',
        'States': 'રાજ્યો',
        'Click a bar to open the state.': 'રાજ્ય ખોલવા બાર પર ક્લિક કરો.',
        'Click a bar to open the district.': 'જિલ્લો ખોલવા બાર પર ક્લિક કરો.',
        'people per month': 'દર મહિને લોકો',
        'oversupply': 'વધારો',
        'Key insights': 'મુખ્ય તારણો',
        'Biggest shortage': 'સૌથી મોટી અછત',
        'Biggest oversupply': 'સૌથી મોટો વધારો',
        'Most pressured state': 'સૌથી વધુ દબાણવાળું રાજ્ય',
        'Seats to move': 'ખસેડવાની બેઠકો',
        'Quick start': 'ઝડપી શરૂઆત',
        'Open Map and drill-down and click a state tile.': 'નકશો ટેબ ખોલી રાજ્યની ટાઇલ પર ક્લિક કરો.',
        'Check the gap and risk, then open the district.': 'ખાઈ અને જોખમ જુઓ, પછી જિલ્લો ખોલો.',
        'Export seat targets from the Recommendation Engine.': 'ભલામણ એન્જિનમાંથી બેઠક લક્ષ્યો નિકાસ કરો.',
        'Got it': 'સમજાયું',
        'Open API documentation': 'API દસ્તાવેજ ખોલો',
        'Supply data is simulated': 'પુરવઠા ડેટા સિમ્યુલેટેડ છે',
    },
}
_TR.update({k: (v[0], v[1]) for k, v in {'Key insights': ('मुख्य निष्कर्ष', 'मुख्य निष्कर्ष'), 'Biggest shortage': ('सबसे बड़ी कमी', 'सर्वात मोठा तुटवडा'), 'Biggest oversupply': ('सबसे बड़ा अधिशेष', 'सर्वात मोठा अतिरिक्त पुरवठा'), 'Most pressured state': ('सबसे अधिक दबाव वाला राज्य', 'सर्वाधिक दबाव असलेले राज्य'), 'Seats to move': ('स्थानांतरित करने योग्य सीटें', 'हलवायच्या जागा'), 'Quick start': ('शुरुआत कैसे करें', 'सुरुवात कशी करावी'), 'Open Map and drill-down and click a state tile.': ('मानचित्र और ड्रिल-डाउन खोलें और किसी राज्य की टाइल पर क्लिक करें।', 'नकाशा आणि ड्रिल-डाउन उघडा आणि राज्याच्या टाइलवर क्लिक करा.'), 'Check the gap and risk, then open the district.': ('अंतर और जोखिम देखें, फिर ज़िला खोलें।', 'तफावत आणि जोखीम पाहा, मग जिल्हा उघडा.'), 'Export seat targets from the Recommendation Engine.': ('सिफारिश इंजन से सीट लक्ष्य निर्यात करें।', 'शिफारस इंजिनमधून जागांचे लक्ष्य निर्यात करा.'), 'Got it': ('समझ गया', 'समजले'), 'Open API documentation': ('API दस्तावेज़ खोलें', 'API दस्तऐवज उघडा'), 'Supply data is simulated': ('आपूर्ति डेटा सिम्युलेटेड है', 'पुरवठा डेटा सिम्युलेटेड आहे')}.items()})
_TR_L = {k.lower(): v for k, v in _TR.items()}
_TR_XL = {c: {k.lower(): v for k, v in m.items()} for c, m in _TR_X.items()}


def tr(s):
    """Translate a visible UI string into the chosen interface language; unknown strings stay in English."""
    lang = st.session_state.get("ui_lang", "en")
    if lang == "en" or not isinstance(s, str):
        return s
    if lang in _TR_X:
        hit = _TR_X[lang].get(s) or _TR_XL[lang].get(s.lower())
        if hit:
            return hit
    t = _TR.get(s) or _TR_L.get(s.lower())
    if not t:
        m = re.match(r"^of (\d+) series$", s)
        if m:
            n = m.group(1)
            return {"hi": f"{n} श्रृंखला में से", "mr": f"{n} मालिकांपैकी", "ta": f"{n} தொடர்களில்", "te": f"{n} శ్రేణులలో", "bn": f"{n}টি ধারার মধ্যে", "gu": f"{n} શ્રેણીમાંથી"}.get(lang, s)
        m = re.match(r"^people per month, (.+)$", s)
        if m:
            w = m.group(1)
            return {"hi": f"लोग प्रति माह, {w}", "mr": f"लोक दरमहा, {w}", "ta": f"மாதம் தோறும் மக்கள், {w}", "te": f"నెలకు ప్రజలు, {w}", "bn": f"প্রতি মাসে মানুষ, {w}", "gu": f"દર મહિને લોકો, {w}"}.get(lang, s)
        return s
    if lang in ("hi", "mr"):
        return t[0] if lang == "hi" else t[1]
    return s


# ----------------------------------------------------------------------------------------------- national to district map, accuracy, methodology, exports
# Tile-grid map of India (one tile per state or UT, roughly geographic). Pilot states are coloured, the rest stay grey.
TILES = {
    "Jammu and Kashmir": (2, 0, "JK"), "Ladakh": (3, 0, "LA"),
    "Punjab": (1, 1, "PB"), "Himachal Pradesh": (2, 1, "HP"), "Uttarakhand": (3, 1, "UK"),
    "Rajasthan": (0, 2, "RJ"), "Haryana": (1, 2, "HR"), "Delhi": (2, 2, "DL"), "Uttar Pradesh": (3, 2, "UP"), "Bihar": (4, 2, "BR"), "Sikkim": (5, 2, "SK"), "Arunachal Pradesh": (7, 2, "AR"),
    "Gujarat": (0, 3, "GJ"), "Madhya Pradesh": (2, 3, "MP"), "Chhattisgarh": (3, 3, "CG"), "Jharkhand": (4, 3, "JH"), "West Bengal": (5, 3, "WB"), "Assam": (6, 3, "AS"), "Nagaland": (7, 3, "NL"),
    "Maharashtra": (1, 4, "MH"), "Telangana": (3, 4, "TS"), "Odisha": (4, 4, "OD"), "Meghalaya": (6, 4, "ML"), "Manipur": (7, 4, "MN"),
    "Goa": (1, 5, "GA"), "Karnataka": (2, 5, "KA"), "Andhra Pradesh": (3, 5, "AP"), "Tripura": (6, 5, "TR"), "Mizoram": (7, 5, "MZ"),
    "Kerala": (2, 6, "KL"), "Tamil Nadu": (3, 6, "TN"),
}


def _agg(d, by):
    g = d.groupby(by).agg(D=("D", "sum"), S=("S", "sum"), n=("D", "size"), hc=("risk_v", lambda x: int(x.isin(["High", "Critical"]).sum())))
    g["gap"] = g["D"] - g["S"]
    g["gi"] = g["gap"] / np.maximum(np.maximum(g["D"], g["S"]), 1.0)
    g["share"] = g["hc"] / g["n"]
    return g


def _picked(ev):
    """Return the customdata of the first selected point of a plotly selection event, or None."""
    try:
        pts = (ev or {}).get("selection", {}).get("points", [])
    except Exception:
        return None, None
    if not pts:
        return None, None
    c = pts[0].get("customdata")
    if isinstance(c, (list, tuple)):
        c = c[0] if c else None
    return c, str(pts[0].get("point_index")) + str(pts[0].get("curve_number"))


def _plotly_pick(fig, key):
    try:
        return st.plotly_chart(fig, width="stretch", key=key, on_select="rerun", selection_mode="points")
    except TypeError:
        st.plotly_chart(fig, width="stretch", key=key)
        return None


def _bars(g, col, labels, key, h=380, fmt_val=None):
    g = g.sort_values(col)
    cols = [("#D6455D" if v > 0 else CYAN) for v in g[col]]
    fig = go.Figure(go.Bar(y=g.index, x=g[col], orientation="h", marker=dict(color=cols), customdata=list(g.index),
                           hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
    style(fig, h, margin=dict(l=8, r=8, t=10, b=8), showlegend=False)
    fig.update_xaxes(title=labels)
    fig.update_yaxes(automargin=True)
    return _plotly_pick(fig, key)


def dash_map(d):
    ss = st.session_state
    ss.setdefault("map_state", None)
    ss.setdefault("map_district", None)
    g = _agg(d, "state")
    if ss["map_state"] not in g.index:
        ss["map_state"], ss["map_district"] = None, None
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        metric = st.radio(tr("Colour tiles by"), ["Gap index", "High and critical share", "Net gap"], horizontal=True, format_func=tr, key="map_metric")
        panel_open(tr("National view"), tr("Click a state tile to drill down. Red means shortage, blue means oversupply."))
        col = {"Gap index": "gi", "High and critical share": "share", "Net gap": "gap"}[metric]
        vals = g[col]
        lim = max(float(vals.abs().max()), 0.05 if col == "gi" else 1.0)
        if col == "share":
            scale, cmin, cmax = [[0, "#F4F2FB"], [1, "#D6455D"]], 0.0, max(float(vals.max()), 0.1)
        else:
            scale, cmin, cmax = [[0, CYAN], [0.5, "#F4F2FB"], [1, "#D6455D"]], -lim, lim
        xs, ys, txt, cd, v, tc, hv = [], [], [], [], [], [], []
        gx, gy, gt = [], [], []
        for name, (cx, ry, ab) in TILES.items():
            if name in g.index:
                r = g.loc[name]
                xs.append(cx); ys.append(-ry); txt.append(ab); cd.append(name); v.append(float(r[col]))
                norm = (float(r[col]) - cmin) / (cmax - cmin) if cmax != cmin else 0.5
                tc.append("#FFFFFF" if (norm > 0.72 or (col != "share" and norm < 0.12)) else NAVY)
                hv.append(f"<b>{name}</b><br>Demand {fmt(r['D'])} | Supply {fmt(r['S'])}<br>Net gap {fmt(r['gap'], True)} | Gap index {r['gi']:+.2f}<br>High and critical: {int(r['hc'])} of {int(r['n'])} series")
            else:
                gx.append(cx); gy.append(-ry); gt.append(ab)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gx, y=gy, mode="markers+text", text=gt, textfont=dict(color="#9AA3C4", size=11), hoverinfo="skip", showlegend=False,
                                 marker=dict(symbol="square", size=56, color="#E8EAF5", line=dict(color="#FFFFFF", width=2))))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=txt, textfont=dict(color=tc, size=13, family="IBM Plex Mono, monospace"), customdata=cd, hovertext=hv, hoverinfo="text", showlegend=False,
                                 marker=dict(symbol="square", size=56, color=v, colorscale=scale, cmin=cmin, cmax=cmax, line=dict(color="#FFFFFF", width=2),
                                             colorbar=dict(thickness=9, len=0.55, outlinewidth=0, tickfont=dict(size=10)))))
        if ss["map_state"]:
            cx, ry, _ = TILES[ss["map_state"]]
            fig.add_trace(go.Scatter(x=[cx], y=[-ry], mode="markers", hoverinfo="skip", showlegend=False, marker=dict(symbol="square-open", size=64, line=dict(color=NAVY, width=3))))
        style(fig, 470, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
        fig.update_xaxes(visible=False, range=[-0.7, 7.8])
        fig.update_yaxes(visible=False, range=[-6.8, 0.8], scaleanchor="x", scaleratio=1)
        ev = _plotly_pick(fig, "map_tiles")
        picked, sig = _picked(ev)
        if picked in g.index and sig != ss.get("map_sig"):
            ss["map_sig"] = sig
            ss["map_state"], ss["map_district"] = picked, None
        st.caption(tr("Pilot state") + ": " + ", ".join(sorted(g.index)) + "  |  " + tr("Not in pilot") + ": " + str(len(TILES) - len(g)))
    with right:
        # breadcrumb
        bc = st.columns([1.1, 2.4, 2.4, 0.3])
        if bc[0].button(tr("India"), key="bc_in", width="stretch"):
            ss["map_state"], ss["map_district"], ss["map_sig"] = None, None, None
            st.rerun()
        if ss["map_state"] and bc[1].button("> " + ss["map_state"], key="bc_st", width="stretch"):
            ss["map_district"] = None
            st.rerun()
        if ss["map_district"]:
            bc[2].button("> " + ss["map_district"], key="bc_di", width="stretch", disabled=True)
        stt, dis = ss["map_state"], ss["map_district"]
        if not stt:
            sub, level = d, "national"
        elif not dis:
            sub = d[d["state"] == stt]
            level = "state"
        else:
            sub = d[(d["state"] == stt) & (d["district"] == dis)]
            level = "district"
        D_, S_ = sub["D"].sum(), sub["S"].sum()
        hc_ = int(sub["risk_v"].isin(["High", "Critical"]).sum())
        k_a = [(tr("Forecast demand"), fmt(D_), "people per month"), (tr("Forecast supply"), fmt(S_), "certified per month")]
        k_b = [(tr("Net gap"), fmt(D_ - S_, True), "shortage" if D_ > S_ else "oversupply", "#D6455D" if D_ > S_ else CYAN, "#FF95A5"),
               (tr("High and critical"), fmt(hc_), f"of {len(sub)} " + tr("series"), "#D6455D", "#FF95A5")]
        kpi_row(k_a)
        kpi_row(k_b)
        if level == "national":
            panel_open(tr("States"), tr("Click a bar to open the state."))
            ev2 = _bars(g, "gap", "Net gap (people per month)", "map_bars_states", 330)
            p2, s2 = _picked(ev2)
            if p2 in g.index and s2 != ss.get("map_sig_b"):
                ss["map_sig_b"] = s2
                ss["map_state"], ss["map_district"] = p2, None
                st.rerun()
        elif level == "state":
            gd = _agg(sub, "district")
            panel_open(tr("Districts") + " | " + stt, tr("Click a bar to open the district."))
            ev2 = _bars(gd, "gap", "Net gap (people per month)", f"map_bars_{stt}", 330)
            p2, s2 = _picked(ev2)
            if p2 in gd.index and s2 != ss.get("map_sig_d"):
                ss["map_sig_d"] = s2
                ss["map_district"] = p2
                st.rerun()
            pick = st.selectbox(tr("Select a district"), [""] + sorted(gd.index), key=f"map_pick_{stt}")
            if pick:
                ss["map_district"] = pick
                st.rerun()
        else:
            panel_open(tr("Skills in the selected district"), f"{dis}, {stt}")
            t = sub.sort_values("gap", ascending=False)[["skill", "sector", "D", "S", "gap", "risk_v", "recommendation", "seat_change", "priority"]].rename(columns={
                "skill": "Skill", "sector": "Sector", "D": "Demand", "S": "Supply", "gap": "Gap", "risk_v": "Risk", "recommendation": "Recommendation", "seat_change": "Seat change", "priority": "Priority"})
            st.dataframe(t.round({"Demand": 0, "Supply": 0, "Gap": 0}), width="stretch", hide_index=True, height=330)
            st.download_button("CSV", t.to_csv(index=False).encode("utf-8"), f"skillmap_{dis}.csv", "text/csv")


def _eval_data():
    try:
        return api_get("/evaluation")
    except Exception:
        B = load_bundle()
        if B is None:
            return None
        keep = ("demand_rolling", "supply_rolling", "demand_fixed", "supply_fixed", "classifier", "classifier_confusion", "demand_audit", "supply_audit")
        return json.loads(json.dumps({k: v for k, v in B["evaluation"].items() if k in keep}, default=str))


def dash_accuracy(meta):
    E = _eval_data()
    if not E:
        st.info("Evaluation results are not available. Run load_db.py again or keep lmis_bundle.pkl next to app.py.")
        return
    M = meta["meta"]
    names = M["model_names"]
    inv = {v: k for k, v in names.items()}
    cs = ["#425B9A", "#76C0EC", "#9AA3C4", "#FF95A5", "#D6455D", "#F2C879", "#2B3A6B"]

    def block(key, label, selkey, note):
        T = pd.DataFrame(E[key])
        items = []
        for h in (3, 6, 12):
            code = inv.get(M[selkey][str(h)], "ENS")
            r = T[(T["Horizon"] == h) & (T["Model"] == code)].iloc[0]
            b = T[(T["Horizon"] == h) & (T["Model"] == "NAIVE3")].iloc[0]
            items.append((f"{label} {h}M MAPE", f"{r['MAPE']:.1f}%", f"naive {b['MAPE']:.1f}% | {100 * (1 - r['MAPE'] / b['MAPE']):.0f}% lower error", BLUE, CYAN))
        kpi_row(items)
        metric = st.radio("Metric", ["MAPE", "WAPE", "MAE", "RMSE"], horizontal=True, key=f"acc_m_{key}")
        panel_open(f"{label} {tr('Error versus simple baselines')}", "Lower is better. Grey and blue bars are naive baselines; the selected model is the ensemble or LightGBM")
        fig = go.Figure()
        order = ["NAIVE3", "SNAIVE", "LR", "XGB", "LGBM", "ENS"]
        for i, code in enumerate(order):
            s = T[T["Model"] == code].sort_values("Horizon")
            if s.empty:
                continue
            fig.add_trace(go.Bar(x=[f"{h}M" for h in s["Horizon"]], y=s[metric], name=names.get(code, code), marker=dict(color=cs[i % len(cs)], line=dict(color=CARD, width=1)),
                                 hovertemplate="%{x} " + names.get(code, code) + ": %{y:.2f}<extra></extra>"))
        style(fig, 330, barmode="group")
        fig.update_yaxes(title=metric)
        st.plotly_chart(fig, width="stretch", key=f"acc_{key}")
        st.caption(note)

    panel_card(tr("Forecast accuracy"), tr("Rolling-origin back-test") + ": every month of the last year used as a forecast origin, all 756 series (9,072 forecasts per horizon)")
    block("demand_rolling", "Demand", "demand_selected", "Lower is better. The selected model beats the simple baselines at every horizon.")
    st.write("")
    panel_card("Supply forecast accuracy", "Measured on the simulated registry, so it is optimistic compared with what real data would show")
    block("supply_rolling", "Supply", "supply_selected", "Supply is simulated in this build: the numbers show the method, not real district performance.")
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    C = pd.DataFrame(E["classifier"])
    with c1:
        panel_open(tr("Early-warning classifier"), "Predicts the risk class 12 months ahead (hold-out year, 9,072 cases)")
        fig = go.Figure()
        for i, mt in enumerate(["Accuracy", "Macro F1", "Within one class"]):
            fig.add_trace(go.Bar(x=C["model"], y=C[mt], name=mt, marker=dict(color=[CYAN, "#9AA3C4", "#425B9A", "#FF95A5"][i % 4]), hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
        style(fig, 320, barmode="group")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, width="stretch", key="acc_clf")
    with c2:
        panel_open("Confusion matrix", "Rows = actual risk class, columns = predicted (XGBoost classifier)")
        cm = np.array(E["classifier_confusion"], dtype=float)
        lab = ["Low", "Medium", "High", "Critical"][: cm.shape[0]]
        fig = go.Figure(go.Heatmap(z=cm, x=lab, y=lab, colorscale=[[0, "#F4F2FB"], [1, BLUE]], text=cm.astype(int), texttemplate="%{text}", xgap=3, ygap=3, showscale=False))
        style(fig, 320, margin=dict(l=8, r=8, t=10, b=8))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch", key="acc_cm")
    st.write("")
    panel_card(tr("Data quality audit"), "Checks run on the raw demand and supply files before modelling, and what was done about each")
    a1, a2 = st.columns(2, gap="large")
    a1.markdown("**Demand data (NCS based)**")
    a1.dataframe(pd.DataFrame(E["demand_audit"]), width="stretch", hide_index=True, height=300)
    a2.markdown("**Supply data (simulated registry)**")
    a2.dataframe(pd.DataFrame(E["supply_audit"]), width="stretch", hide_index=True, height=300)


SOURCES = [
    ("National Career Service (NCS) vacancies", "Vacancies and hiring intensity", "District x skill x month", "Monthly", "NCO coded", 0.35, "Live in this build"),
    ("Job portals", "Postings, salary offered", "District x skill x month", "Daily to monthly", "Title mapped to NCO", 0.25, "Adapter ready (upload below)"),
    ("Industry and Sector Skill Council hiring signals", "Planned hiring, vacancy surveys", "State x sector", "Quarterly", "NSQF sector", 0.15, "Adapter ready (upload below)"),
    ("e-Shram registrations", "Workers by occupation and district", "District x occupation", "Monthly", "NCO coded", 0.15, "Adapter ready (upload below)"),
    ("PLFS (periodic labour force survey)", "Employment and wage structure", "State x NCO division", "Quarterly", "NCO coded", 0.10, "Adapter ready (upload below)"),
]


def dash_method_intro(df):
    panel_card(tr("How the demand index is built"), "Heterogeneous sources are cleaned, coded to NCO and NSQF, put on one scale and blended into a single demand index per district and trade")
    steps = [("1", "Ingest", "NCS, job portals, SSC hiring, e-Shram, PLFS"), ("2", "Clean and code", "Canonical names, NCO and NSQF mapping, duplicates, gaps"),
             ("3", "Normalise", "Constant-price units, z-score within series"), ("4", "Blend", "Weighted demand index (below)"),
             ("5", "Forecast", "3, 6, 12 months, 80 percent interval"), ("6", "Gap and risk", "Against training capacity and seats"), ("7", "Act", "Targets, API and export")]
    flow = '<div class="flow">' + '<span class="arr"></span>'.join(f'<div class="st"><b>{n}</b><h5>{t}</h5><p>{s}</p></div>' for n, t, s in steps) + "</div>"
    st.markdown(flow, unsafe_allow_html=True)
    st.markdown('<div class="note" style="margin-top:10px"><b>Demand index.</b> DI(district, trade, month) = sum over sources k of w<sub>k</sub> x z<sub>k</sub>(district, trade, month) / sum of w<sub>k</sub> for the sources present, where z<sub>k</sub> is the source signal standardised within its own series. Missing sources are skipped and the remaining weights are re-scaled, so the index stays comparable as new sources are added.</div>', unsafe_allow_html=True)
    st.write("")
    panel_card(tr("Data sources"), "What each source adds and how it enters the index. Weights are design defaults that a planner can change")
    S = pd.DataFrame(SOURCES, columns=["Source", "Signal", "Granularity", "Refresh", "Coding", "Default weight", "Status"])
    st.dataframe(S, width="stretch", hide_index=True, height=235)
    st.write("")
    panel_card(tr("Combine your own data"), "Upload a CSV with columns state, district, skill, month, signal, source. It is mapped to the canonical names, standardised and blended into a demand index preview")
    tmpl = pd.DataFrame({"state": ["Maharashtra"] * 4, "district": ["Pune"] * 4, "skill": ["Welder"] * 4, "month": ["2026-01", "2026-02", "2026-01", "2026-02"],
                         "signal": [120, 135, 80, 95], "source": ["Job portal", "Job portal", "e-Shram", "e-Shram"]})
    c1, c2 = st.columns([3, 1])
    up = c1.file_uploader("CSV file", type="csv", key="dm_up", label_visibility="collapsed")
    c2.download_button("Template CSV", tmpl.to_csv(index=False).encode("utf-8"), "skillmap_signal_template.csv", "text/csv", width="stretch")
    c2.caption("Example rows only")
    if up is None:
        return
    import difflib
    try:
        u = pd.read_csv(up)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return
    need = {"state", "district", "skill", "month", "signal", "source"}
    if not need.issubset(u.columns):
        st.error("Missing columns: " + ", ".join(sorted(need - set(u.columns))))
        return

    def canon(series, ref):
        ref = list(ref)
        low = {r.lower(): r for r in ref}
        out = []
        for x in series.astype(str):
            k = x.strip().lower()
            out.append(low.get(k) or (low[difflib.get_close_matches(k, list(low), 1, 0.78)[0]] if difflib.get_close_matches(k, list(low), 1, 0.78) else None))
        return out
    n0 = len(u)
    u["state"] = canon(u["state"], df["state"].unique())
    u["district"] = canon(u["district"], df["district"].unique())
    u["skill"] = canon(u["skill"], df["skill"].unique())
    u["month"] = pd.to_datetime(u["month"], errors="coerce").dt.to_period("M").astype(str)
    u["signal"] = pd.to_numeric(u["signal"], errors="coerce")
    u = u.dropna(subset=["state", "district", "skill", "signal"]).query("month != 'NaT'")
    st.caption(f"{len(u)} of {n0} rows matched the canonical states, districts and skills; the rest were dropped.")
    if u.empty:
        return
    srcs = sorted(u["source"].astype(str).unique())
    wcols = st.columns(len(srcs))
    w = {s_: wcols[i].slider(s_, 0.0, 1.0, 1.0 / len(srcs), 0.05, key=f"dm_w_{s_}") for i, s_ in enumerate(srcs)}
    u["z"] = u.groupby(["district", "skill", "source"])["signal"].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 0 else 1.0))
    u["w"] = u["source"].map(w)
    idx = u.groupby(["state", "district", "skill", "month"]).apply(lambda g_: float((g_["z"] * g_["w"]).sum() / max(g_["w"].sum(), 1e-9)), include_groups=False).rename("Demand index").reset_index()
    st.dataframe(idx.round({"Demand index": 3}), width="stretch", hide_index=True, height=240)
    one = idx.assign(series=idx["district"] + " | " + idx["skill"])
    fig = go.Figure()
    for sname, g_ in one.groupby("series"):
        fig.add_trace(go.Scatter(x=g_["month"], y=g_["Demand index"], name=sname, mode="lines+markers", line=dict(width=3, shape="spline")))
    style(fig, 300)
    st.plotly_chart(fig, width="stretch", key="dm_idx")
    st.caption("Preview of the blend. To change the forecasts themselves, add the file to the notebook's demand step and re-run it; no code changes are needed.")


# ---- seat target workbook and planner brief
def targets_table(df, state=None):
    t = df if not state else df[df["state"] == state]
    t = t.sort_values("priority_rank").copy()
    t["target_seats"] = (t["annual_seats"].fillna(0) + t["seat_change"].fillna(0)).round().astype(int)
    cols = ["priority_rank", "priority", "state", "district", "skill", "sector", "risk", "skill_gap", "annual_seats", "seat_change", "target_seats", "recommendation", "recommendation_detail"]
    return t[cols].rename(columns={"priority_rank": "Rank", "priority": "Priority", "state": "State", "district": "District", "skill": "Trade", "sector": "Sector", "risk": "Risk", "skill_gap": "Gap (people/month, 12M)",
                                   "annual_seats": "Current annual seats", "seat_change": "Seat change", "target_seats": "Target seats", "recommendation": "Recommendation", "recommendation_detail": "Reason"})


def targets_workbook(df, meta, state=None):
    t = targets_table(df, state)
    s = t.groupby("State").agg(**{"Current annual seats": ("Current annual seats", "sum"), "Seat increase": ("Seat change", lambda x: int(x[x > 0].sum())), "Seat reduction": ("Seat change", lambda x: int(-x[x < 0].sum())),
                                   "Target seats": ("Target seats", "sum"), "Critical or High series": ("Risk", lambda x: int(x.isin(["Critical", "High"]).sum()))}).reset_index()
    note = pd.DataFrame({"Read me": [
        "SkillMap seat targets for the next training cycle.",
        "Target seats = current annual seats + recommended seat change (12-month gap, policy bands in the Recommendation Engine).",
        "Priority P1 = top 10 percent by priority score (risk weight x absolute gap), P2 = next 20 percent.",
        "Supply data in this build is simulated from PMKVY district volumes; use as a method demonstration, not as a finding about real districts.",
        f"Generated {time.strftime('%d %b %Y %H:%M')}. Data to {meta['meta']['last_data_month']}."]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        note.to_excel(xw, sheet_name="Read me", index=False)
        t.to_excel(xw, sheet_name="District targets", index=False)
        s.to_excel(xw, sheet_name="State summary", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(48, max(10, max(len(str(c.value or "")) for c in col[:60]) + 2))
            ws.freeze_panes = "A2"
    return buf.getvalue()


def planner_brief_pdf(df, meta, state=None):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    d = df if not state else df[df["state"] == state]
    navy, blue = colors.HexColor("#2B3A6B"), colors.HexColor("#425B9A")
    ss_ = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=ss_["Title"], textColor=navy, fontSize=20, alignment=0, spaceAfter=2)
    H2 = ParagraphStyle("h2", parent=ss_["Heading2"], textColor=blue, fontSize=12, spaceBefore=10, spaceAfter=4)
    P = ParagraphStyle("p", parent=ss_["BodyText"], fontSize=8.5, leading=11)

    def table(rows, widths):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 7.5), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F3FB")]),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D3D8EE")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        return t
    short = d[d["skill_gap"] > 0].sort_values("priority_rank").head(10)
    over = d[d["seat_change"] < 0].sort_values("seat_change").head(10)
    inc, cut = int(d[d["seat_change"] > 0]["seat_change"].sum()), int(-d[d["seat_change"] < 0]["seat_change"].sum())
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm, title="SkillMap planner brief")
    el = [Paragraph("SkillMap planner brief", H1),
          Paragraph(f"Scope: {state or 'All pilot states'} | Data to {meta['meta']['last_data_month']} | Generated {time.strftime('%d %b %Y')}", P), Spacer(1, 4)]
    kp = [["Series", "Shortage series", "Oversupply series", "Critical or High", "Seats to add", "Seats to release"],
          [len(d), int((d["skill_gap"] > 0).sum()), int((d["skill_gap"] < 0).sum()), int(d["risk"].isin(["Critical", "High"]).sum()), f"{inc:,}", f"{cut:,}"]]
    el += [table(kp, [30 * mm] * 6), Paragraph("Largest shortages (add or fill seats first)", H2)]
    rows = [["Rank", "State", "District", "Trade", "Risk", "Gap/month", "Seat change", "Action"]] + [[int(r.priority_rank), r.state, r.district, r.skill, r.risk, f"{int(r.skill_gap):+d}", f"{int(r.seat_change):+d}", r.recommendation] for r in short.itertuples()]
    el.append(table(rows, [11 * mm, 24 * mm, 26 * mm, 40 * mm, 15 * mm, 18 * mm, 20 * mm, 28 * mm]))
    el.append(Paragraph("Largest seat reductions (oversupplied trades)", H2))
    rows = [["Rank", "State", "District", "Trade", "Risk", "Gap/month", "Seat change", "Action"]] + [[int(r.priority_rank), r.state, r.district, r.skill, r.risk, f"{int(r.skill_gap):+d}", f"{int(r.seat_change):+d}", r.recommendation] for r in over.itertuples()]
    el.append(table(rows, [11 * mm, 24 * mm, 26 * mm, 40 * mm, 15 * mm, 18 * mm, 20 * mm, 28 * mm]))
    el.append(Paragraph("State seat shifts", H2))
    sg = d.groupby("state").agg(inc=("seat_change", lambda x: int(x[x > 0].sum())), cut=("seat_change", lambda x: int(-x[x < 0].sum())), crit=("risk", lambda x: int(x.isin(["Critical", "High"]).sum()))).reset_index()
    rows = [["State", "Seats to add", "Seats to release", "Critical or High series"]] + [[r.state, f"{r.inc:,}", f"{r.cut:,}", r.crit] for r in sg.itertuples()]
    el.append(table(rows, [45 * mm, 35 * mm, 38 * mm, 50 * mm]))
    el += [Spacer(1, 8), Paragraph("Method: risk uses the gap index (demand minus supply over the larger of the two) with policy bands; seats = extra certified people per month x 12 / (completion rate x 90 percent utilisation). "
                                   "<b>Data note:</b> supply is simulated from PMKVY district volumes in this build, so the figures demonstrate the method and are not findings about real districts.", P)]
    doc.build(el)
    return buf.getvalue()


def lazy_tabs(names, key):
    """Segmented tab bar that renders only the selected section (st.tabs would build every chart of every tab on each rerun)."""
    with st.container():
        st.markdown('<span class="seg-marker"></span>', unsafe_allow_html=True)
        return st.radio("Section", names, horizontal=True, key=key, label_visibility="collapsed", format_func=tr)


def quick_start():
    """Dismissible three-step guide for first-time visitors."""
    if st.session_state.get("tip_off"):
        return
    c1, c2 = st.columns([9, 1.3], gap="small")
    steps = ["Open Map and drill-down and click a state tile.", "Check the gap and risk, then open the district.", "Export seat targets from the Recommendation Engine."]
    chips = "".join(f'<div class="step"><i>{i + 1}</i><span>{tr(t)}</span></div>' for i, t in enumerate(steps))
    c1.markdown(f'<div class="tip"><div class="tt">{tr("Quick start")}</div><div class="steps">{chips}</div></div>', unsafe_allow_html=True)
    with c2:
        st.write("")
        if st.button(tr("Got it"), key="tip_ok", width="stretch"):
            st.session_state["tip_off"] = True
            st.rerun()


def insight_cards(d, d0):
    """Plain-language headline findings for planners: biggest shortage, biggest oversupply, most pressured state, seats to move."""
    top = d.loc[d["gap"].idxmax()]
    low = d.loc[d["gap"].idxmin()]
    hot = d[d["risk_v"].isin(["High", "Critical"])].groupby("state").size().sort_values(ascending=False)
    add = int(d0.loc[d0["seat_change"] > 0, "seat_change"].sum())
    rel = int(-d0.loc[d0["seat_change"] < 0, "seat_change"].sum())

    def card(label, val, sub, c1, c2):
        return (f'<div class="kpi" style="--c1:{c1};--c2:{c2}"><div class="l">{tr(label)}</div>'
                f'<div class="v" style="font-size:1.05rem;line-height:1.35;min-height:2.85rem">{val}</div><div class="s">{sub}</div><div class="bar"></div></div>')
    items = [card("Biggest shortage", f"{top['skill']}, {top['district']}", f"{fmt(top['gap'], True)} {tr('people per month')}", "#FF95A5", "#D6455D"),
             card("Biggest oversupply", f"{low['skill']}, {low['district']}", f"{fmt(low['gap'], True)} {tr('people per month')}", CYAN, BLUE),
             card("Most pressured state", f"{hot.index[0]}" if len(hot) else "-", f"{int(hot.iloc[0])} {tr('High and critical')}" if len(hot) else "", "#F2C879", "#D6455D"),
             card("Seats to move", f"+{fmt(add)}  /  -{fmt(rel)}", f"{tr('Seats to add')} / {tr('Seats to release')}", BLUE, CYAN)]
    st.markdown(f'<div class="ins-title">{tr("Key insights")}</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for c, h in zip(cols, items):
        c.markdown(h, unsafe_allow_html=True)


def page_dashboard():
    df, meta = load_all()
    M = meta["meta"]
    hero("Dashboard", "Demand and supply forecasts, skill gap and risk for every district and skill.", eyebrow="Labour market dashboard", tags=[f"{M['n_series']} {tr('series')}", f"{M['n_states']} {tr('states')}", f"{M['n_districts']} {tr('districts')}", f"{M['n_skills']} {tr('skills')}", f"{tr('Data to')} {M['last_data_month']}"])
    quick_start()
    d0, st_sel, sk_sel, hz, scen = filter_bar(df, meta, "db")
    if d0.empty:
        st.warning("No series match this filter.")
        return
    d = view(d0, meta, hz, scen)
    TN = ["Overview", "Map and drill-down", "Demand and supply", "Gap and risk", "Explainability", "Accuracy", "Methodology"]
    sel_tab = lazy_tabs(TN, "db_tab")

    if sel_tab == "Map and drill-down":
        dash_map(d)
    if sel_tab == "Accuracy":
        dash_accuracy(meta)

    # ---- Overview
    if sel_tab == "Overview":
        D, S, G = agg_gap(d)
        hi = int(d["risk_v"].isin(["High", "Critical"]).sum())
        kpi_row([("Forecast demand", fmt(D), f"people per month, {M['target_months'][str(hz)]}", BLUE, CYAN),
                 ("Forecast supply", fmt(S), "certified per month", TEAL, GREEN),
                 ("Net gap", fmt(G, True), "shortage" if G > 0 else "oversupply", "#D6455D" if G > 0 else TEAL, "#FF95A5" if G > 0 else GREEN),
                 ("High and Critical", fmt(hi), f"of {len(d)} series", "#FF95A5", "#D6455D"),
                 ("Training capacity", fmt(d0["training_capacity"].sum()), "per month today", NAVY, BLUE)])
        st.write("")
        insight_cards(d, d0)
        st.write("")
        c1, c2 = st.columns([2.1, 1])
        with c1:
            h = api_get("/history", {"state": None if st_sel == "All" else st_sel, "skill": None if sk_sel == "All" else sk_sel})
            f = scen_factor(meta, scen)
            panel_open("Demand and supply trajectory", "Monthly history (solid) and forecast at 3, 6 and 12 months (dashed, ring markers). Demand is in constant latest-year units.")
            fig = area_chart(h["months"], [("Demand", h["demand"], BLUE, [v * f for v in h["forecast_demand"]]), ("Supply", h["supply"], TEAL, h["forecast_supply"])], h["forecast_months"], 330)
            st.plotly_chart(fig, width="stretch")
        with c2:
            panel_open("Risk mix", f"Series by risk class at {hz} months")
            cnt = {r: int((d["risk_v"] == r).sum()) for r in RISKS}
            st.plotly_chart(donut(cnt, 300), width="stretch")
        c3, c4 = st.columns([1.3, 1])
        with c3:
            panel_open("Where the pressure is", "States ranked by High and Critical series")
            st.plotly_chart(risk_stack(d, "state", 340), width="stretch")
        with c4:
            panel_open("Risk today to action", "Seat decision chosen for each risk class (base plan, number of series)")
            ct = d0.groupby(["risk", "recommendation"]).size().unstack(fill_value=0).reindex(RISKS)
            fig = go.Figure()
            for act in ACT_C:
                if act in ct.columns:
                    fig.add_trace(go.Bar(y=[f"{r} risk" for r in ct.index], x=ct[act], name=act, orientation="h", marker=dict(color=ACT_C[act], line=dict(color=CARD, width=1.5)),
                                         hovertemplate="%{y}: %{x} series<extra>" + act + "</extra>"))
            st.plotly_chart(style(fig, 340, barmode="stack", bargap=0.3), width="stretch")

    # ---- Demand and supply
    if sel_tab == "Demand and supply":
        c1, c2 = st.columns([1.3, 1])
        with c1:
            panel_open("Demand versus supply by skill", f"{hz}-month forecast, {scen.lower()} scenario")
            t = d.groupby("skill")[["D", "S"]].sum().sort_values("D")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=t.index, x=t["D"], name="Demand", orientation="h", marker=dict(color=BLUE), hovertemplate="%{y}: %{x:,.0f}<extra>Demand</extra>"))
            fig.add_trace(go.Bar(y=t.index, x=t["S"], name="Supply", orientation="h", marker=dict(color=TEAL), hovertemplate="%{y}: %{x:,.0f}<extra>Supply</extra>"))
            st.plotly_chart(style(fig, 370, barmode="group", bargap=0.25), width="stretch")
        with c2:
            panel_open("Demand map by sector and skill", "Area = forecast demand, colour = gap index (blue oversupply, red shortage)")
            tm = d.groupby(["sector", "skill"]).agg(D=("D", "sum"), S=("S", "sum")).reset_index()
            tm["gi"] = (tm["D"] - tm["S"]) / np.maximum(np.maximum(tm["D"], tm["S"]), 1)
            fig = go.Figure(go.Treemap(labels=tm["skill"], parents=tm["sector"], values=tm["D"], marker=dict(colors=tm["gi"], colorscale=[[0, CYAN], [0.5, MIST2], [1, "#D6455D"]], cmid=0, line=dict(width=2, color=CARD)),
                                       texttemplate="%{label}", hovertemplate="%{label}<br>demand %{value:,.0f}<extra></extra>"))
            st.plotly_chart(style(fig, 370), width="stretch")
        panel_open("Demand growth outlook", "Forecast change versus current level, by skill (12 months)")
        gdf = d0.groupby("skill").agg(c=("current_demand", "sum"), f=("forecast_12m", "sum")).reset_index()
        gdf["g"] = (gdf["f"] / gdf["c"] - 1) * 100
        gdf = gdf.sort_values("g")
        fig = go.Figure(go.Bar(x=gdf["skill"], y=gdf["g"], marker=dict(color=np.where(gdf["g"] >= 0, BLUE, "#FF95A5")), hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
        st.plotly_chart(style(fig, 230), width="stretch")

    # ---- Gap and risk
    if sel_tab == "Gap and risk":
        hc1, hc2 = st.columns([2.2, 1])
        with hc1:
            panel_open("Gap index heatmap", "State by skill, positive (red) means shortage, negative (blue) means oversupply")
            hm = d.groupby(["state", "skill"]).agg(D=("D", "sum"), S=("S", "sum")).reset_index()
            hm["gi"] = (hm["D"] - hm["S"]) / np.maximum(np.maximum(hm["D"], hm["S"]), 1)
            pv = hm.pivot(index="state", columns="skill", values="gi")
            fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, zmid=0, colorscale=[[0, CYAN], [0.5, MIST], [1, "#D6455D"]], xgap=3, ygap=3,
                                       colorbar=dict(thickness=10, title="gap index"), hovertemplate="%{y} / %{x}<br>gap index %{z:.2f}<extra></extra>"))
            fig.update_xaxes(tickangle=-35)
            st.plotly_chart(style(fig, 400), width="stretch")
        with hc2:
            panel_open("Risk across horizons", "Series by risk class at 3, 6 and 12 months (selected scenario)")
            fig = go.Figure()
            for r in RISKS:
                fig.add_trace(go.Bar(x=[f"{x} months" for x in (3, 6, 12)], y=[int((view(d0, meta, x, scen)["risk_v"] == r).sum()) for x in (3, 6, 12)], name=r, marker=dict(color=RISK_C[r])))
            st.plotly_chart(style(fig, 400, barmode="stack", bargap=0.45), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            panel_open("Largest shortages", "Top 12 district and skill combinations by gap")
            top = d.sort_values("gap", ascending=False).head(12).iloc[::-1]
            lab = top["skill"] + ", " + top["district"]
            fig = go.Figure(go.Bar(y=lab, x=top["gap"], orientation="h", marker=dict(color=[RISK_C[r] for r in top["risk_v"]]), hovertemplate="%{y}: %{x:,.0f} per month<extra></extra>"))
            st.plotly_chart(style(fig, 320), width="stretch")
        with c2:
            panel_open("Largest oversupply", "Top 12 where supply exceeds demand")
            bot = d.sort_values("gap").head(12).iloc[::-1]
            lab = bot["skill"] + ", " + bot["district"]
            fig = go.Figure(go.Bar(y=lab, x=bot["gap"], orientation="h", marker=dict(color=[RISK_C[r] for r in bot["risk_v"]]), hovertemplate="%{y}: %{x:,.0f} per month<extra></extra>"))
            st.plotly_chart(style(fig, 320), width="stretch")

    # ---- Explainability
    if sel_tab == "Explainability":
        c = st.columns(3)
        s_st = c[0].selectbox("State", sorted(df["state"].unique()), key="ex_st")
        s_di = c[1].selectbox("District", sorted(df[df["state"] == s_st]["district"].unique()), key="ex_di")
        s_sk = c[2].selectbox("Skill", sorted(df[(df["state"] == s_st) & (df["district"] == s_di)]["skill"].unique()), key="ex_sk")
        r = df[(df["state"] == s_st) & (df["district"] == s_di) & (df["skill"] == s_sk)].iloc[0]
        ex = api_get("/explain", {"district": s_di, "skill": s_sk, "state": s_st})
        kpi_row([("Risk (rule-based)", r["risk"], f"gap index {r['gap_index']:+.2f}", RISK_C[r["risk"]], BLUE),
                 ("Early-warning class", ex["ml_risk"], "XGBoost, 12 months ahead", BLUE, CYAN),
                 ("Demand 12M", fmt(r["forecast_12m"]), f"80% interval {fmt(r['forecast_12m_low'])} to {fmt(r['forecast_12m_high'])}", BLUE, CYAN),
                 ("Supply 12M", fmt(r["supply_12m"]), f"interval {fmt(r['supply_12m_low'])} to {fmt(r['supply_12m_high'])}", TEAL, GREEN)])
        st.write("")
        c1, c2 = st.columns([1, 1.2])
        with c1:
            panel_open("Early-warning probabilities", "Chance of each risk class 12 months ahead")
            p = ex["ml_probabilities"]
            fig = go.Figure(go.Bar(x=list(p), y=[p[k] for k in p], marker=dict(color=[RISK_C[k] for k in p]), text=[f"{p[k]:.0%}" for k in p], textposition="outside"))
            fig.update_yaxes(range=[0, 1.1], tickformat=".0%")
            st.plotly_chart(style(fig, 270), width="stretch")
        with c2:
            panel_open("Forecast with uncertainty", "Demand and supply at 3, 6 and 12 months with the 80 percent interval")
            hs = ["3M", "6M", "12M"]
            fig = go.Figure()
            for nm, col, pre in (("Demand", BLUE, "forecast"), ("Supply", TEAL, "supply")):
                key = (lambda h: f"forecast_{h}m") if pre == "forecast" else (lambda h: f"supply_{h}m")
                y = [r[key(h)] for h in (3, 6, 12)]
                hi = [r[key(h) + "_high"] for h in (3, 6, 12)]
                lo = [r[key(h) + "_low"] for h in (3, 6, 12)]
                fig.add_trace(go.Scatter(x=hs + hs[::-1], y=hi + lo[::-1], fill="toself", fillcolor=rgba(col, 0.14), line=dict(width=0), hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=hs, y=y, name=nm, mode="lines+markers", line=dict(color=col, width=3, shape="spline"), marker=dict(size=10, color=CARD, line=dict(color=col, width=2.5))))
            st.plotly_chart(style(fig, 270, hovermode="x unified"), width="stretch")
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="panel"><h4>{tr("What drives the supply forecast")}</h4><div class="sub">Main SHAP contributions</div><p>{ex["supply_drivers_12m"] or "Not available"}</p></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="panel"><h4>{tr("What drives the risk class")}</h4><div class="sub">Main SHAP contributions</div><p>{ex["risk_drivers"] or "Not available"}</p></div>', unsafe_allow_html=True)

    # ---- Data and method
    if sel_tab == "Methodology":
        dash_method_intro(df)
        sim_note(meta)
        st.caption("Data source: " + ("lmis_bundle.pkl read directly" if st.session_state.get("mode") == "pkl" else "MySQL database through the FastAPI backend"))
        kb = api_get("/kb")
        cats = {}
        for k in kb:
            cats.setdefault(k["category"], []).append(k)
        names = {"overview": "Overview", "data": "Data", "method": "Method", "accuracy": "Accuracy", "limits": "Limitations", "usage": "Usage", "glossary": "Glossary"}
        for cat, docs in cats.items():
            st.subheader(names.get(cat, cat.title()))
            for dd in docs:
                with st.expander(dd["title"]):
                    st.write(dd["content"])
        st.subheader("Series table")
        show = d[["state", "district", "skill", "D", "S", "gap", "gi", "risk_v", "recommendation", "priority"]].rename(columns={"state": "State", "district": "District", "skill": "Skill", "D": "Demand", "S": "Supply", "gap": "Gap", "gi": "Gap index", "risk_v": "Risk", "recommendation": "Recommendation", "priority": "Priority"})
        st.dataframe(show.round({"Demand": 0, "Supply": 0, "Gap": 0, "Gap index": 3}), width="stretch", hide_index=True, height=380)


def page_reco():
    df, meta = load_all()
    hero("Recommendation Engine", "Seat decisions for every district and skill: increase, fill existing seats first, reduce or maintain.",
         ["Priority ranked", "Same-district reallocation", "What-if simulator", "Budget optimiser"], eyebrow="Seat planning")
    TN = ["Action plan", "District planner", "What-if simulator", "Budget optimiser", "Export"]
    sel_tab = lazy_tabs(TN, "rc_tab")

    if sel_tab == TN[0]:
        c = st.columns(4)
        st_sel = c[0].selectbox("State", ["All"] + sorted(df["state"].unique()), key="rc_st")
        act = c[1].selectbox("Action", ["All"] + list(ACT_C), key="rc_act")
        pr = c[2].selectbox("Priority", ["All", "P1", "P2", "P3"], key="rc_pr")
        rk = c[3].multiselect("Risk", RISKS, default=RISKS, key="rc_rk")
        d = df.copy()
        if st_sel != "All":
            d = d[d["state"] == st_sel]
        if act != "All":
            d = d[d["recommendation"] == act]
        if pr != "All":
            d = d[d["priority"].str.startswith(pr)]
        d = d[d["risk"].isin(rk)]
        add, cut = int(d[d["seat_change"] > 0]["seat_change"].sum()), int(-d[d["seat_change"] < 0]["seat_change"].sum())
        kpi_row([("Seats to add", fmt(add), "annual seats", "#D6455D", "#FF95A5"), ("Seats to release", fmt(cut), "annual seats", BLUE, CYAN),
                 ("Net change", fmt(add - cut, True), "annual seats", NAVY, BLUE), ("P1 actions", fmt(d["priority"].str.startswith("P1").sum()), "act now", "#D6455D", "#F2C879"),
                 ("Series in view", fmt(len(d)), f"of {len(df)}", TEAL, GREEN)])
        st.write("")
        c1, c2 = st.columns([1.1, 1])
        with c1:
            panel_open("Net seat change by state", "Waterfall of annual seats: increases to the right, reductions to the left")
            w = d.groupby("state")["seat_change"].sum().sort_values()
            fig = go.Figure(go.Waterfall(x=list(w.index) + ["Total"], y=list(w.values) + [0], measure=["relative"] * len(w) + ["total"],
                                         increasing=dict(marker=dict(color="#D6455D")), decreasing=dict(marker=dict(color=BLUE)), totals=dict(marker=dict(color=NAVY)), connector=dict(line=dict(color=LINE))))
            st.plotly_chart(style(fig, 330), width="stretch")
        with c2:
            panel_open("Actions by priority", "Number of series per action and priority")
            t = d.assign(p=d["priority"].str[:2]).groupby(["recommendation", "p"]).size().unstack(fill_value=0)
            fig = go.Figure()
            for p_, col in (("P1", "#D6455D"), ("P2", "#F2C879"), ("P3", TEAL)):
                if p_ in t:
                    fig.add_trace(go.Bar(y=t.index, x=t[p_], name=p_, orientation="h", marker=dict(color=col, line=dict(color=CARD, width=1.5))))
            st.plotly_chart(style(fig, 330, barmode="stack", bargap=0.35), width="stretch")
        panel_open("Ranked action plan", "Sorted by priority score (risk weight times absolute gap)")
        t = d.sort_values("priority_rank")[["priority_rank", "priority", "state", "district", "skill", "risk", "skill_gap", "recommendation", "seat_change", "recommendation_detail"]]
        t.columns = ["Rank", "Priority", "State", "District", "Skill", "Risk", "Gap 12M", "Recommendation", "Seat change", "Reason"]
        st.dataframe(t, width="stretch", hide_index=True, height=420, column_config={"Seat change": st.column_config.NumberColumn(format="%+d"), "Gap 12M": st.column_config.NumberColumn(format="%+d")})

    if sel_tab == TN[1]:
        c = st.columns(2)
        s = c[0].selectbox("State", sorted(df["state"].unique()), key="dp_st")
        di = c[1].selectbox("District", sorted(df[df["state"] == s]["district"].unique()), key="dp_di")
        d = df[(df["state"] == s) & (df["district"] == di)].sort_values("skill_gap")
        short, over = d[d["skill_gap"] > 0], d[d["seat_change"] < 0]
        kpi_row([("Demand 12M", fmt(d["forecast_12m"].sum()), "per month", BLUE, CYAN), ("Supply 12M", fmt(d["supply_12m"].sum()), "per month", TEAL, GREEN),
                 ("Skills in shortage", fmt(len(short)), f"of {len(d)}", "#D6455D", "#FF95A5"), ("Seats to release", fmt(-over["seat_change"].sum()), "from oversupplied skills", BLUE, CYAN)])
        st.write("")
        c1, c2 = st.columns([1.2, 1])
        with c1:
            panel_open(f"Skill gap in {di}", "People per month at 12 months; bar colour is the risk class")
            fig = go.Figure(go.Bar(y=d["skill"], x=d["skill_gap"], orientation="h", marker=dict(color=[RISK_C[r] for r in d["risk"]]), hovertemplate="%{y}: %{x:+,.0f}<extra></extra>"))
            st.plotly_chart(style(fig, 350), width="stretch")
        with c2:
            panel_open("Reallocation within the district", "Seats released by oversupplied skills can be offered to the shortage skills below")
            if len(over) and len(short):
                pool = int(-over["seat_change"].sum())
                need = short[short["seat_change"] > 0].sort_values("priority_rank")
                rows, left = [], pool
                for _, r in need.iterrows():
                    give = min(left, int(r["seat_change"]))
                    if give > 0:
                        rows.append({"Move seats to": r["skill"], "Seats": give, "Of needed": int(r["seat_change"]), "Risk": r["risk"]})
                        left -= give
                if rows:
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    st.caption("Source skills: " + ", ".join(f"{r.skill} (-{fmt(-r.seat_change)})" for r in over.itertuples()))
                else:
                    st.info("No shortage skill in this district needs additional seats.")
            else:
                st.info("This district has no oversupplied skill to release seats, or no shortage that needs them.")
        t = d.sort_values("priority_rank")[["priority", "skill", "risk", "forecast_12m", "supply_12m", "skill_gap", "recommendation", "seat_change", "recommendation_detail"]]
        t.columns = ["Priority", "Skill", "Risk", "Demand 12M", "Supply 12M", "Gap", "Recommendation", "Seat change", "Reason"]
        st.dataframe(t, width="stretch", hide_index=True)

    if sel_tab == TN[2]:
        c = st.columns(3)
        s = c[0].selectbox("State", sorted(df["state"].unique()), key="wi_st")
        di = c[1].selectbox("District", sorted(df[df["state"] == s]["district"].unique()), key="wi_di")
        sk = c[2].selectbox("Skill", sorted(df[(df["state"] == s) & (df["district"] == di)]["skill"].unique()), key="wi_sk")
        c = st.columns(2)
        sd = c[0].slider("Change in annual seats", -3000, 3000, 0, 50, key="wi_sd")
        sh = c[1].slider("Demand shock (percent)", -30, 40, 0, 1, key="wi_sh")
        try:
            r = api_post("/whatif", {"state": s, "district": di, "skill": sk, "seats_delta": sd, "demand_shock_pct": float(sh)})
            b, a = r["before"], r["after"]
            kpi_row([("Gap before", fmt(b["gap"], True), f"risk {b['risk']}", RISK_C[b["risk"]], BLUE), ("Gap after", fmt(a["gap"], True), f"risk {a['risk']}", RISK_C[a["risk"]], BLUE),
                     ("Demand after", fmt(a["demand"]), f"{sh:+d}% shock", BLUE, CYAN), ("Supply after", fmt(a["supply"]), f"{sd:+d} annual seats", TEAL, GREEN)])
            st.write("")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Before", "After"], y=[b["demand"], a["demand"]], name="Demand", marker=dict(color=BLUE)))
            fig.add_trace(go.Bar(x=["Before", "After"], y=[b["supply"], a["supply"]], name="Supply", marker=dict(color=TEAL)))
            st.plotly_chart(style(fig, 280, barmode="group", bargap=0.4), width="stretch")
        except Exception as e:
            st.error(f"The what-if service did not respond: {e}")

    if sel_tab == TN[3]:
        panel_open("Budget optimiser", "Allocate a limited number of new seats to the highest-priority shortages first")
        inc = df[df["seat_change"] > 0].sort_values("priority_rank").copy()
        tot = int(inc["seat_change"].sum())
        budget = st.slider("Seats available to add (annual)", 0, max(tot, 100), min(5000, tot), 100, key="bo_b")
        inc["cum"] = inc["seat_change"].cumsum()
        inc["alloc"] = np.clip(budget - (inc["cum"] - inc["seat_change"]), 0, inc["seat_change"])
        inc["share"] = inc["alloc"] / inc["seat_change"]
        covered = float((inc["skill_gap"].clip(lower=0) * inc["share"]).sum())
        total_short = float(inc["skill_gap"].clip(lower=0).sum())
        kpi_row([("Seats allocated", fmt(inc["alloc"].sum()), f"of {fmt(tot)} requested", BLUE, CYAN), ("Series funded", fmt((inc["alloc"] > 0).sum()), f"of {len(inc)}", TEAL, GREEN),
                 ("Shortage addressed", f"{covered / max(total_short, 1):.0%}", f"{fmt(covered)} of {fmt(total_short)} per month", "#D6455D", "#FF95A5")])
        st.write("")
        cs = inc["seat_change"].cumsum().values
        cov = (inc["skill_gap"].clip(lower=0) * 1.0).cumsum().values / max(total_short, 1) * 100
        fig = go.Figure(go.Scatter(x=cs, y=cov, mode="lines", line=dict(color=BLUE, width=3, shape="spline"), fill="tozeroy", fillgradient=grad(BLUE), hovertemplate="%{x:,.0f} seats: %{y:.0f}% of shortage<extra></extra>"))
        fig.add_vline(x=budget, line=dict(color="#D6455D", dash="dash"), annotation_text="Budget", annotation_font=dict(color="#D6455D"))
        fig.update_xaxes(title="Cumulative seats (in priority order)"); fig.update_yaxes(title="Cumulative shortage covered (percent)")
        st.plotly_chart(style(fig, 300), width="stretch")
        t = inc[inc["alloc"] > 0][["priority_rank", "state", "district", "skill", "risk", "seat_change", "alloc"]]
        t.columns = ["Rank", "State", "District", "Skill", "Risk", "Requested", "Allocated"]
        st.dataframe(t, width="stretch", hide_index=True, height=300)

    if sel_tab == TN[4]:
        sim_note(meta)
        panel_open("Seat targets for the next training cycle", "District x trade targets (current annual seats + recommended change), ready for scheme target-setting")
        scope = st.selectbox(tr("State"), ["All"] + sorted(df["state"].unique()), key="ex_state")
        stt = None if scope == "All" else scope
        t = targets_table(df, stt)
        k1, k2, k3 = st.columns(3)
        with k1:
            st.download_button("CSV", t.to_csv(index=False).encode("utf-8"), "skillmap_targets.csv", "text/csv", width="stretch")
        with k2:
            try:
                st.download_button(tr("Download target workbook"), targets_workbook(df, meta, stt), "skillmap_targets.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")
            except Exception as e:
                st.caption(f"Excel export needs openpyxl ({e}).")
        with k3:
            try:
                st.download_button(tr("Download PDF brief"), planner_brief_pdf(df, meta, stt), "skillmap_planner_brief.pdf", "application/pdf", width="stretch")
            except Exception as e:
                st.caption(f"PDF brief needs reportlab ({e}).")
        st.dataframe(t, width="stretch", hide_index=True, height=340)
        st.write("")
        panel_open("API for scheme workflows", "The same targets are served by the FastAPI backend, so scheme systems can pull them directly")
        base = API if st.session_state.get("mode") == "api" else "http://localhost:8000"
        st.code(f"GET {base}/targets?state=Bihar&risk=Critical&fmt=json\nGET {base}/targets?fmt=csv\nGET {base}/recommendations\nGET {base}/high-risk\nGET {base}/docs   # interactive Swagger", language="text")
        if st.session_state.get("mode") != "api":
            st.caption("The backend is not running right now; start it with: uvicorn api:app --port 8000")


def render_chat_payload(p, key, stream=False):
    """Chat-style reply: the text (typed out when new), then compact KPI cards, an area chart and an optional table."""
    text = p["answer"]
    if stream and hasattr(st, "write_stream"):
        import re as _re

        def _gen():
            toks = _re.split(r"(\s+)", text)
            dt = min(0.004, 0.25 / max(len(toks), 1))
            for w in toks:
                yield w
                time.sleep(dt)
        st.write_stream(_gen())
    else:
        st.markdown(text)
    if p.get("kpis"):
        kpi_row([(k["label"], k["value"], k.get("sub", "")) for k in p["kpis"]])
        st.write("")
    ch = p.get("chart")
    if ch and ch.get("months"):
        fig = area_chart(ch["months"], [("Demand", ch["demand"], BLUE, ch.get("forecast_demand")), ("Supply", ch["supply"], TEAL, ch.get("forecast_supply"))], ch.get("forecast_months"), 300)
        st.plotly_chart(fig, width="stretch", key=f"ch_{key}")
    if p.get("table"):
        with st.expander("View data table"):
            st.dataframe(pd.DataFrame(p["table"]), width="stretch", hide_index=True)


@st.cache_resource(show_spinner=False)
def chat_avatars():
    """Avatars drawn in the SkillMap palette (no emoji): user = person glyph on indigo, assistant = logo mark on cream."""
    from PIL import Image, ImageDraw
    S = 160
    u = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(u)
    d.rounded_rectangle((0, 0, S - 1, S - 1), radius=44, fill="#425B9A")
    d.ellipse((S * 0.34, S * 0.20, S * 0.66, S * 0.52), fill="#FFF6DC")
    d.pieslice((S * 0.20, S * 0.52, S * 0.80, S * 1.10), 180, 360, fill="#76C0EC")
    a = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(a)
    d.rounded_rectangle((0, 0, S - 1, S - 1), radius=44, fill="#FFF6DC", outline="#FF95A5", width=5)
    try:
        ic = Image.open(ASSETS / "logo_icon.png").convert("RGBA")
        ic.thumbnail((int(S * 0.62), int(S * 0.62)))
        a.paste(ic, ((S - ic.width) // 2, (S - ic.height) // 2), ic)
    except Exception:
        pass
    return u, a


WELCOME = ("Hello, I am the SkillMap Assistant. Ask me about the demand, supply, skill gap, risk or seat recommendation of any state, district or skill. "
           "You can write in English or Hinglish, for example *Pune me welder ki demand kitni hai* or *Bihar me sabse zyada shortage kaha hai*.")
CHIPS = ["Details of Pune", "Top critical shortages in Bihar", "Electrician demand in Patna", "Pune me welder ki demand kitni hai?", "Compare Maharashtra and Gujarat", "How is risk calculated?"]


def _ask(q):
    try:
        return api_post("/chat", {"message": q, "session_id": st.session_state.sid})
    except Exception as e:
        return {"answer": f"The assistant could not answer right now: {e}"}


@st.cache_resource(show_spinner=False)
def _logo_b64():
    import base64
    try:
        return base64.b64encode((ASSETS / "logo_icon.png").read_bytes()).decode()
    except Exception:
        return ""


VOICE_JS = r"""(function () {
  if (window.__skmLoaded) return;
  window.__skmLoaded = true;
  var D = document, W = window;
  var SR = W.SpeechRecognition || W.webkitSpeechRecognition;
  var synth = W.speechSynthesis;
  var S = { lang: 'hi-IN', speak: true, lastId: null, rec: null, on: false, stream: null, ctx: null, raf: 0 };
  var MIC = '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/></svg>';
  var MICOFF = '<svg viewBox="0 0 24 24" width="34" height="34" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3M4 4l16 16"/></svg>';
  var MICON = MICOFF.replace('M4 4l16 16', '');
  var css = '' +
  '#skm-voice{position:fixed;inset:0;z-index:100000;display:none;flex-direction:column;color:#F7FAFE;font-family:"IBM Plex Sans",system-ui,sans-serif;background:radial-gradient(900px 520px at 50% 28%,rgba(118,192,236,.34),transparent 62%),radial-gradient(700px 420px at 85% 95%,rgba(255,149,165,.22),transparent 60%),linear-gradient(160deg,#25325E,#35508F 60%,#2B3A6B);}' +
  '#skm-voice.open{display:flex;animation:skmv-in .35s ease both}' +
  '@keyframes skmv-in{from{opacity:0;transform:scale(1.02)}to{opacity:1;transform:none}}' +
  '#skm-voice .vh{display:flex;align-items:center;gap:14px;padding:16px 28px;border-bottom:1px solid rgba(255,255,255,.12);background:rgba(255,255,255,.04)}' +
  '#skm-voice .vh .ic{width:42px;height:42px;border-radius:12px;background:#FFF6DC;display:flex;align-items:center;justify-content:center}' +
  '#skm-voice .vh .ic img{width:30px;height:30px}' +
  '#skm-voice .vh .e{font-family:"IBM Plex Mono",monospace;font-size:.66rem;letter-spacing:.22em;text-transform:uppercase;color:#76C0EC}' +
  '#skm-voice .vh .t{font-size:1.05rem;font-weight:600}' +
  '#skm-voice .vh .sp{flex:1}' +
  '#skm-voice select{background:rgba(255,255,255,.10);color:#F7FAFE;border:1px solid rgba(255,255,255,.22);border-radius:10px;padding:7px 10px;font-size:.82rem}' +
  '#skm-voice select option{color:#2B3A6B}' +
  '#skm-voice .x{width:40px;height:40px;border-radius:12px;border:1px solid rgba(255,255,255,.25);background:rgba(255,255,255,.08);color:#fff;font-size:1.1rem;cursor:pointer}' +
  '#skm-voice .x:hover{background:rgba(255,255,255,.18)}' +
  '#skm-voice .stage{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;padding:10px 20px}' +
  '#skm-voice .orb{position:relative;width:240px;height:240px;display:flex;align-items:center;justify-content:center;--lvl:0}' +
  '#skm-voice .orb i{position:absolute;border-radius:50%;border:1px solid rgba(118,192,236,.40);inset:0;transition:transform .12s linear,opacity .2s}' +
  '#skm-voice .orb i:nth-child(1){transform:scale(calc(.62 + var(--lvl)*.10));opacity:.9}' +
  '#skm-voice .orb i:nth-child(2){transform:scale(calc(.78 + var(--lvl)*.22));opacity:.55}' +
  '#skm-voice .orb i:nth-child(3){transform:scale(calc(.94 + var(--lvl)*.36));opacity:.30}' +
  '#skm-voice.live .orb i{background:rgba(118,192,236,.07);animation:skmv-pulse 2.2s ease-in-out infinite}' +
  '#skm-voice.live .orb i:nth-child(2){animation-delay:.35s}#skm-voice.live .orb i:nth-child(3){animation-delay:.7s}' +
  '@keyframes skmv-pulse{0%,100%{box-shadow:0 0 0 0 rgba(118,192,236,.25)}50%{box-shadow:0 0 38px 8px rgba(118,192,236,.22)}}' +
  '#skm-voice .mic{position:relative;z-index:2;width:84px;height:84px;border-radius:50%;border:none;cursor:pointer;background:#FFF6DC;color:#D6455D;display:flex;align-items:center;justify-content:center;box-shadow:0 12px 34px rgba(0,0,0,.30);transition:transform .2s,background .2s,color .2s}' +
  '#skm-voice .mic:hover{transform:scale(1.06)}' +
  '#skm-voice.live .mic{background:linear-gradient(135deg,#76C0EC,#FF95A5);color:#fff}' +
  '#skm-voice .h{font-size:1.15rem;font-weight:600;margin-top:6px}' +
  '#skm-voice .s{font-size:.86rem;color:#C9D6F2;margin-bottom:8px}' +
  '#skm-voice .box{position:relative;width:min(620px,92vw)}' +
  '#skm-voice textarea{width:100%;min-height:104px;resize:none;border-radius:16px;border:1px solid rgba(255,255,255,.22);background:rgba(255,255,255,.09);color:#fff;padding:14px 150px 14px 16px;font:inherit;font-size:.96rem;outline:none;box-sizing:border-box}' +
  '#skm-voice textarea::placeholder{color:#AFC0E6}' +
  '#skm-voice textarea:focus{border-color:#76C0EC;box-shadow:0 0 0 3px rgba(118,192,236,.25)}' +
  '#skm-voice .send{position:absolute;right:12px;bottom:12px;border:none;border-radius:12px;padding:10px 16px;font-weight:600;font-size:.85rem;cursor:pointer;background:linear-gradient(90deg,#76C0EC,#FF95A5);color:#2B3A6B;opacity:.45;pointer-events:none;transition:opacity .2s,transform .2s}' +
  '#skm-voice .send.on{opacity:1;pointer-events:auto}#skm-voice .send.on:hover{transform:translateY(-1px)}' +
  '.skm-mic{position:absolute;right:58px;top:50%;transform:translateY(-50%);width:36px;height:36px;border-radius:12px;border:1px solid #D3D8EE;background:#F1F3FC;color:#425B9A;cursor:pointer;display:flex;align-items:center;justify-content:center;z-index:5;transition:background .2s,transform .2s}' +
  '.skm-mic:hover{background:#E2E8F8;transform:translateY(-50%) scale(1.06)}' +
  '[data-testid="stChatInput"] textarea{padding-right:110px !important}' +
  '#skm-speaking{position:fixed;right:22px;bottom:22px;z-index:99990;display:none;align-items:center;gap:10px;padding:9px 12px 9px 14px;border-radius:999px;background:#2B3A6B;color:#F7FAFE;font:600 .8rem "IBM Plex Sans",sans-serif;box-shadow:0 10px 26px rgba(43,58,107,.35)}' +
  '#skm-speaking.on{display:flex;animation:skmv-in .3s ease both}' +
  '#skm-speaking .bars{display:flex;gap:3px;align-items:center;height:16px}' +
  '#skm-speaking .bars b{width:3px;border-radius:2px;background:#76C0EC;animation:skmv-bar 1s ease-in-out infinite}' +
  '#skm-speaking .bars b:nth-child(1){height:6px}#skm-speaking .bars b:nth-child(2){height:14px;animation-delay:.15s}#skm-speaking .bars b:nth-child(3){height:9px;animation-delay:.3s}#skm-speaking .bars b:nth-child(4){height:13px;animation-delay:.45s}' +
  '@keyframes skmv-bar{0%,100%{transform:scaleY(.4)}50%{transform:scaleY(1.25)}}' +
  '#skm-speaking button{border:none;border-radius:999px;background:rgba(255,255,255,.16);color:#fff;padding:5px 11px;font-size:.74rem;cursor:pointer}';
  var st = D.createElement('style'); st.textContent = css; D.head.appendChild(st);

  var icon = (W.__skmIcon || '');
  var ov = D.createElement('div'); ov.id = 'skm-voice';
  ov.innerHTML = '<div class="vh"><div class="ic">' + (icon ? '<img src="' + icon + '">' : '') + '</div><div><div class="e">SkillMap Voice</div><div class="t">Talk to your assistant</div></div><div class="sp"></div>' +
    '<select id="skmv-lang"><option value="hi-IN">Hindi / Hinglish</option><option value="mr-IN">Marathi</option><option value="ta-IN">Tamil</option><option value="te-IN">Telugu</option><option value="bn-IN">Bengali</option><option value="gu-IN">Gujarati</option><option value="en-IN">English (India)</option><option value="en-US">English (US)</option></select><button class="x" title="Close">&#10005;</button></div>' +
    '<div class="stage"><div class="orb"><i></i><i></i><i></i><button class="mic" title="Start or stop listening"></button></div>' +
    '<div class="h">Tap the microphone when you are ready</div><div class="s">Your words will appear below. Review them before sending.</div>' +
    '<div class="box"><textarea placeholder="Try: Electrician demand in Patna, or Pune me welder ki demand kitni hai"></textarea><button class="send">Send request</button></div></div>';
  D.body.appendChild(ov);
  var sp = D.createElement('div'); sp.id = 'skm-speaking';
  sp.innerHTML = '<div class="bars"><b></b><b></b><b></b><b></b></div><span>Speaking</span><button>Stop</button>';
  D.body.appendChild(sp);

  var ta = ov.querySelector('textarea'), send = ov.querySelector('.send'), mic = ov.querySelector('.mic'),
      head = ov.querySelector('.h'), sub = ov.querySelector('.s'), orb = ov.querySelector('.orb'), sel = ov.querySelector('#skmv-lang');
  mic.innerHTML = MICOFF;
  function upd() { if (ta.value.trim()) send.classList.add('on'); else send.classList.remove('on'); }
  ta.addEventListener('input', upd);
  sel.addEventListener('change', function () { S.lang = sel.value; if (S.on) { stop(); setTimeout(start, 250); } });

  function levels() {
    try {
      navigator.mediaDevices.getUserMedia({ audio: true }).then(function (stm) {
        if (!S.on) { stm.getTracks().forEach(function (t) { t.stop(); }); return; }
        S.stream = stm;
        var AC = W.AudioContext || W.webkitAudioContext; S.ctx = new AC();
        var an = S.ctx.createAnalyser(); an.fftSize = 256; S.ctx.createMediaStreamSource(stm).connect(an);
        var buf = new Uint8Array(an.frequencyBinCount);
        (function loop() {
          if (!S.on) return;
          an.getByteFrequencyData(buf); var s = 0; for (var i = 0; i < buf.length; i++) s += buf[i];
          orb.style.setProperty('--lvl', Math.min(1, (s / buf.length) / 70).toFixed(2));
          S.raf = requestAnimationFrame(loop);
        })();
      }).catch(function () { });
    } catch (e) { }
  }
  function setLive(v) {
    S.on = v; ov.classList.toggle('live', v); mic.innerHTML = v ? MICON : MICOFF;
    head.textContent = v ? 'Listening' : 'Tap the microphone when you are ready';
    sub.textContent = v ? 'Speak now. Tap the microphone again to stop.' : 'Your words will appear below. Review them before sending.';
    if (!v) { orb.style.setProperty('--lvl', 0); try { cancelAnimationFrame(S.raf); } catch (e) { } if (S.stream) { S.stream.getTracks().forEach(function (t) { t.stop(); }); S.stream = null; } if (S.ctx) { try { S.ctx.close(); } catch (e) { } S.ctx = null; } }
  }
  function start() {
    if (!SR) { head.textContent = 'Speech input is not supported in this browser'; sub.textContent = 'Please use Chrome or Edge, or type your question below.'; return; }
    try { synth && synth.cancel(); } catch (e) { }
    var r = new SR(); S.rec = r; r.lang = S.lang; r.interimResults = true; r.continuous = true;
    var base = ta.value ? ta.value.trim() + ' ' : '';
    r.onresult = function (e) {
      var fin = '', it = '';
      for (var i = 0; i < e.results.length; i++) { var x = e.results[i]; if (x.isFinal) fin += x[0].transcript; else it += x[0].transcript; }
      ta.value = (base + fin + ' ' + it).trim(); upd();
    };
    r.onerror = function (e) {
      setLive(false);
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') { head.textContent = 'Microphone access is blocked'; sub.textContent = 'Allow the microphone for this site in the browser address bar, then try again.'; }
      else if (e.error === 'no-speech') { head.textContent = 'No speech detected'; sub.textContent = 'Tap the microphone and try again.'; }
    };
    r.onend = function () { if (S.on) setLive(false); };
    try { r.start(); setLive(true); levels(); } catch (e) { setLive(false); }
  }
  function stop() { try { S.rec && S.rec.stop(); } catch (e) { } setLive(false); }
  mic.addEventListener('click', function () { S.on ? stop() : start(); });
  function openV() { sel.value = S.lang; ov.classList.add('open'); ta.value = ''; upd(); setLive(false); setTimeout(function () { ta.focus(); }, 60); }
  function closeV() { stop(); ov.classList.remove('open'); }
  ov.querySelector('.x').addEventListener('click', closeV);
  D.addEventListener('keydown', function (e) { if (e.key === 'Escape' && ov.classList.contains('open')) closeV(); });

  function submit(text) {
    var t = D.querySelector('[data-testid="stChatInputTextArea"]') || D.querySelector('[data-testid="stChatInput"] textarea');
    if (!t) return false;
    var setter = Object.getOwnPropertyDescriptor(W.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(t, text); t.dispatchEvent(new Event('input', { bubbles: true }));
    setTimeout(function () {
      var b = D.querySelector('[data-testid="stChatInputSubmitButton"]') || D.querySelector('[data-testid="stChatInput"] button');
      if (b) b.click(); else t.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', keyCode: 13, bubbles: true }));
    }, 90);
    return true;
  }
  send.addEventListener('click', function () {
    var v = ta.value.trim(); if (!v) return;
    try { sessionStorage.setItem('skmv', '1'); } catch (e) { }
    closeV(); submit(v);
  });
  ta.addEventListener('keydown', function (e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send.click(); } });

  function clean(t) {
    return t.replace(/\|[^\n]*\|/g, ' ').replace(/[*_`#>~]/g, '').replace(/\[(.*?)\]\(.*?\)/g, '$1').replace(/^\s*[-\d]+[.)]?\s+/gm, '').replace(/\s*\n+\s*/g, '. ').replace(/\s+/g, ' ').replace(/\.\s*\./g, '.').trim().slice(0, 700);
  }
  function pickVoice(lang) {
    var vs = synth ? synth.getVoices() : []; var p = lang.slice(0, 2);
    return vs.filter(function (v) { return v.lang === lang; })[0] || vs.filter(function (v) { return v.lang.slice(0, 2) === p; })[0] || null;
  }
  function stopSpeak() { try { synth.cancel(); } catch (e) { } sp.classList.remove('on'); }
  sp.querySelector('button').addEventListener('click', stopSpeak);
  function say(text) {
    if (!synth) return;
    var t = clean(text); if (!t) return;
    try { synth.cancel(); } catch (e) { }
    var parts = t.match(/[^.!?]+[.!?]*/g) || [t], chunks = [], cur = '';
    parts.forEach(function (p) { if ((cur + p).length > 180) { if (cur) chunks.push(cur); cur = p; } else cur += p; });
    if (cur) chunks.push(cur);
    chunks.forEach(function (c, i) {
      var u = new SpeechSynthesisUtterance(c.trim()); u.lang = S.lang; u.rate = 0.97;
      var v = pickVoice(S.lang); if (v) u.voice = v;
      if (i === 0) u.onstart = function () { sp.classList.add('on'); };
      if (i === chunks.length - 1) { u.onend = function () { sp.classList.remove('on'); }; u.onerror = u.onend; }
      synth.speak(u);
    });
  }
  if (synth && synth.onvoiceschanged !== undefined) synth.onvoiceschanged = function () { };

  function ensure() {
    var wrap = D.querySelector('[data-testid="stChatInput"]');
    if (!wrap || wrap.querySelector('.skm-mic')) return;
    var b = D.createElement('button'); b.className = 'skm-mic'; b.type = 'button'; b.title = 'Speak your question'; b.innerHTML = MIC;
    b.addEventListener('click', function (e) { e.preventDefault(); e.stopPropagation(); openV(); });
    wrap.style.position = 'relative'; wrap.appendChild(b);
  }
  setInterval(ensure, 600); ensure();

  W.__skmUpdate = function (cfg) {
    if (cfg.icon) { W.__skmIcon = cfg.icon; var im = ov.querySelector('.ic'); if (im && !im.firstChild) im.innerHTML = '<img src="' + cfg.icon + '">'; }
    S.lang = cfg.lang || S.lang; S.speak = !!cfg.speak;
    if (cfg.id && cfg.text && cfg.id !== S.lastId) {
      S.lastId = cfg.id; var f = false; try { f = sessionStorage.getItem('skmv') === '1'; sessionStorage.removeItem('skmv'); } catch (e) { }
      if (cfg.speak || f) say(cfg.text);
    }
  };
})();
"""

SPEECH_LANGS = {"Hindi / Hinglish": "hi-IN", "Marathi": "mr-IN", "Tamil": "ta-IN", "Telugu": "te-IN", "Bengali": "bn-IN", "Gujarati": "gu-IN", "English (India)": "en-IN", "English (US)": "en-US"}
STARTERS = [
    ("Critical shortages", "Where the gap is largest in Bihar", "Top critical shortages in Bihar"),
    ("District snapshot", "Demand, supply and risk for Pune", "Details of Pune"),
    ("Skill demand", "Electrician demand in Patna", "Electrician demand in Patna"),
    ("Compare states", "Maharashtra versus Gujarat", "Compare Maharashtra and Gujarat"),
    ("Hindi / Hinglish", "Pune me welder ki demand kitni hai", "Pune me welder ki demand kitni hai?"),
    ("Method", "How is risk calculated", "How is risk calculated?"),
]


def voice_bridge(lang, speak, fresh):
    """Injects the voice layer into the page: mic button in the chat box, full-screen voice mode, speech to text and text to speech."""
    import streamlit.components.v1 as components
    cfg = {"lang": lang, "speak": bool(speak), "icon": "data:image/png;base64," + _logo_b64()}
    if fresh:
        cfg.update(fresh)
    html = ("<script>(function(){var P=window.parent,D=P.document;"
            "if(!P.__skmLoaded){var s=D.createElement('script');s.textContent=" + json.dumps(VOICE_JS) + ";D.head.appendChild(s);}"
            "var n=0;(function go(){if(P.__skmUpdate){P.__skmUpdate(" + json.dumps(cfg) + ");}else if(n++<40){setTimeout(go,100);}})();})();</script>")
    components.html(html, height=0)


CHAT_FILE = HERE / "skillmap_chats.json"      # fallback only, used if the SQL database cannot be reached


def _conv_call(op, cid=None, body=None):
    """Chat history lives in the SQL database (table chat_conversation): MySQL through the API, or the local SQLite copy."""
    if st.session_state.get("mode") == "api":
        if op == "list":
            return http().get(f"{API}/conversations", timeout=15).json()
        if op == "save":
            http().put(f"{API}/conversations/{cid}", json=body, timeout=15).raise_for_status()
        elif op == "delete":
            http().delete(f"{API}/conversations/{cid}", timeout=15).raise_for_status()
        elif op == "clear":
            http().delete(f"{API}/conversations", timeout=15).raise_for_status()
        return None
    m = local_api_module()
    if op == "list":
        return m.conv_list()
    if op == "save":
        return m.conv_save(cid, m.ConvIn(**body))
    if op == "delete":
        return m.conv_delete(cid)
    if op == "clear":
        return m.conv_clear()


def _chats_load():
    try:
        rows = _conv_call("list")
        if isinstance(rows, list):
            return rows
    except Exception:
        pass
    try:
        return json.loads(CHAT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _chats_save(op="save", conv=None):
    """Write one change to SQL; if SQL is unavailable keep a JSON copy so nothing is lost."""
    try:
        if op == "save":
            _conv_call("save", conv["id"], {"title": conv["title"], "ts": conv["ts"], "d": conv.get("d", ""), "msgs": conv["msgs"]})
        elif op == "delete":
            _conv_call("delete", conv)
        else:
            _conv_call("clear")
        return
    except Exception:
        pass
    try:
        CHAT_FILE.write_text(json.dumps(st.session_state.chats[:60], ensure_ascii=False, default=str), encoding="utf-8")
    except Exception:
        pass


def _chat_store(first_q=None):
    """Create or update the current conversation in the history list and save it to the SQL database."""
    ss = st.session_state
    cid = ss.get("chat_id")
    stamp = time.strftime("%d %b, %H:%M")
    conv = next((c for c in ss.chats if c["id"] == cid), None) if cid else None
    if conv:
        conv["msgs"], conv["ts"] = ss.msgs, stamp
        ss.chats.remove(conv)
    else:
        cid = uuid.uuid4().hex[:10]
        ss.chat_id = cid
        conv = {"id": cid, "title": (first_q or "New chat").strip()[:60], "ts": stamp, "d": time.strftime("%Y-%m-%d"), "msgs": ss.msgs}
    ss.chats.insert(0, conv)
    _chats_save("save", conv)


def page_assistant():
    ss = st.session_state
    if "sid" not in ss:
        ss.sid = uuid.uuid4().hex[:12]
    if "chats" not in ss:
        ss.chats = _chats_load()
    if "msgs" not in ss or (ss.msgs and ss.msgs[0].get("welcome")):
        ss.msgs = []
    ss.setdefault("chat_id", None)
    ss.setdefault("hist_open", True)
    ua, aa = chat_avatars()
    mode = "SkillMap database (MySQL)" if ss.get("mode") == "api" else "SkillMap local database"
    if ss.hist_open:
        side, main = st.columns([1.05, 3.7], gap="medium")
    else:
        side, main = None, st.container()
    lang_name = ss.get("_lang", "Hindi / Hinglish")
    speak = ss.get("_speak", True)
    if side is not None:
        with side:
            try:
                rail = st.container(border=True)
            except TypeError:
                rail = st.container()
            with rail:
                st.markdown(f'''<span class="rail-marker"></span><div class="as-brand"><img src="data:image/png;base64,{_logo_b64()}"><div><b>SkillMap AI</b><span>Labour market assistant</span></div></div>''', unsafe_allow_html=True)
                if st.button("+  " + tr("New chat"), key="newc", width="stretch"):
                    ss.msgs = []
                    ss.chat_id = None
                    ss.sid = uuid.uuid4().hex[:12]
                    st.rerun()
                srch = st.text_input(tr("Search chats"), placeholder=tr("Search chats"), key="hist_q", label_visibility="collapsed")
                st.markdown('<div class="hist-title"><span>'+tr("Chat history")+'</span><span>%d</span></div>' % len(ss.chats), unsafe_allow_html=True)
                try:
                    hbox = st.container(height=300, border=False)
                except TypeError:
                    hbox = st.container()
                with hbox:
                    items = [c for c in ss.chats if not srch or srch.lower() in c["title"].lower()]
                    if not items:
                        st.markdown('<div class="hist-empty">%s</div>' % ("No chats match your search." if srch else "No conversations yet. Ask your first question and it will appear here."), unsafe_allow_html=True)
                    today = time.strftime("%Y-%m-%d")
                    yest = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
                    last_grp = None
                    for c in items[:50]:
                        d = c.get("d", "")
                        grp = "Today" if d == today else "Yesterday" if d == yest else "Earlier"
                        if grp != last_grp:
                            st.markdown(f'<div class="hist-grp">{tr(grp)}</div>', unsafe_allow_html=True)
                            last_grp = grp
                        r1, r2 = st.columns([6, 1], gap="small")
                        if r1.button(c["title"], key=f"hs_{c['id']}", help=c["ts"], type="primary" if c["id"] == ss.chat_id else "secondary", width="stretch"):
                            ss.chat_id = c["id"]
                            ss.msgs = c["msgs"]
                            ss.sid = uuid.uuid4().hex[:12]
                            st.rerun()
                        if r2.button("\u00d7", key=f"hd_{c['id']}", help="Delete this chat"):
                            ss.chats = [x for x in ss.chats if x["id"] != c["id"]]
                            if ss.chat_id == c["id"]:
                                ss.chat_id, ss.msgs = None, []
                            _chats_save("delete", c["id"])
                            st.rerun()
                if ss.chats and st.button(tr("Clear all history"), key="clrall", width="stretch"):
                    ss.chats, ss.chat_id, ss.msgs = [], None, []
                    _chats_save("clear")
                    st.rerun()
                with st.expander(tr("Voice settings")):
                    lang_name = st.selectbox(tr("Speech language"), list(SPEECH_LANGS), index=list(SPEECH_LANGS).index(lang_name), key="as_lang")
                    speak = st.toggle(tr("Read replies aloud"), value=speak, key="as_speak")
                ss["_lang"], ss["_speak"] = lang_name, speak
                st.markdown(f'<div class="rail-foot"><i></i><span>Chats saved in {mode}</span></div>', unsafe_allow_html=True)
    with main:
        try:
            box = st.container(border=True)
        except TypeError:
            box = st.container()
        with box:
            t1, t2 = st.columns([0.5, 8], gap="small")
            if t1.button("\u2630", key="tog_hist", help="Hide history" if ss.hist_open else "Show history", width="stretch"):
                ss.hist_open = not ss.hist_open
                st.rerun()
            cur = next((c["title"] for c in ss.chats if c["id"] == ss.chat_id), None)
            t2.markdown(f'<span class="as-marker"></span><div class="as-top" style="border:none;padding:0 4px"><div><div class="as-title">{cur or tr("AI Assistant")}</div><div class="as-sub">{tr("Conversation") if cur else tr("Ask about any state, district or skill. Type, or tap the microphone and speak.")}</div></div><div class="as-pill"><i></i>SkillMap engine</div></div>', unsafe_allow_html=True)
            st.markdown('<hr class="chat-line">', unsafe_allow_html=True)
            try:
                area = st.container(height=520, border=False)
            except TypeError:
                try:
                    area = st.container(height=500)
                except TypeError:
                    area = st.container()
            typed = st.chat_input(tr("Message SkillMap AI  (English or Hinglish)"))
            st.markdown('<div class="as-hint"><span>'+tr("Ready when you are.")+'</span><span>'+tr("Enter to send")+'</span></div><div class="as-disc">The assistant can make mistakes. Check important numbers in the dashboard. Voice recognition and playback depend on your browser and available voices.</div>', unsafe_allow_html=True)
    pending = None
    q = None
    with area:
        if not ss.msgs:
            st.markdown(f'''<div class="as-hero"><div class="ic"><img src="data:image/png;base64,{_logo_b64()}"></div><div class="e">{tr("Your labour market analyst")}</div><h2>{tr("What can I help you with?")}</h2><p>{tr("Ask about demand, supply, skill gap, risk or seat plans. Type a message or use the microphone.")}</p></div>''', unsafe_allow_html=True)
            with st.container():
                st.markdown('<span class="as-cards"></span>', unsafe_allow_html=True)
                cc = st.columns(2, gap="small")
                for i, (t_, s_, qq) in enumerate(STARTERS):
                    if cc[i % 2].button(f"**{tr(t_)}**  \n{tr(s_)}", key=f"st{i}", width="stretch"):
                        pending = qq
        else:
            for i, m in enumerate(ss.msgs):
                with st.chat_message(m["role"], avatar=ua if m["role"] == "user" else aa):
                    if m["role"] == "user":
                        st.markdown('<span class="u-msg"></span>', unsafe_allow_html=True)
                        st.write(m["content"])
                    else:
                        st.markdown('<span class="a-msg"></span>', unsafe_allow_html=True)
                        render_chat_payload(m["payload"], f"{ss.chat_id}_{i}")
                    st.markdown(f'<div class="msg-time">{m.get("ts", "")}</div>', unsafe_allow_html=True)
            fu = ss.msgs[-1].get("payload", {}).get("follow_ups") or []
            if fu:
                st.caption(tr("Suggested follow-ups"))
                cc = st.columns(min(len(fu), 3))
                for j, f_ in enumerate(fu[:3]):
                    if cc[j].button(f_, key=f"fu{len(ss.msgs)}_{j}", width="stretch"):
                        pending = f_
        q = typed or pending
        if q:
            now = time.strftime("%H:%M")
            ss.msgs.append({"role": "user", "content": q, "ts": now})
            with st.chat_message("user", avatar=ua):
                st.markdown('<span class="u-msg"></span>', unsafe_allow_html=True)
                st.write(q)
                st.markdown(f'<div class="msg-time">{now}</div>', unsafe_allow_html=True)
            with st.chat_message("assistant", avatar=aa):
                st.markdown('<span class="a-msg"></span>', unsafe_allow_html=True)
                typing = st.empty()
                typing.markdown('<div class="typing"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
                p = _ask(q)
                typing.empty()
                render_chat_payload(p, f"new_{len(ss.msgs)}", stream=True)
            ss.msgs.append({"role": "assistant", "payload": p, "ts": time.strftime("%H:%M")})
            _chat_store(q)
            ss["speak_now"] = {"id": uuid.uuid4().hex[:8], "text": p.get("answer", "")}
    fresh = ss.pop("speak_now", None) if not q else None
    voice_bridge(SPEECH_LANGS[lang_name], speak, fresh)
    if q:
        st.rerun()


# ----------------------------------------------------------------------------------------------- shell (top navigation)
PAGES = {"Dashboard": page_dashboard, "Recommendation Engine": page_reco, "Assistant": page_assistant}

_chk = st.session_state.get("_be")
if _chk is None or time.time() - _chk[0] > 60:
    _chk = (time.time(), ensure_backend())
    st.session_state["_be"] = _chk
if _chk[1]:
    st.session_state["mode"] = "api"
elif load_bundle() is not None:
    st.session_state["mode"] = "pkl"
else:
    st.error("No data source found. Put lmis_bundle.pkl next to app.py (the notebook writes it), or start the backend: uvicorn api:app --port 8000.")
    st.stop()

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Dashboard"
cols = st.columns([1.15, 0.95, 1.7, 0.95, 1.55])
with cols[0]:
    st.markdown('<span class="nav-marker"></span>', unsafe_allow_html=True)
    st.image(str(ASSETS / "logo_lockup.png"), width=118)
for i, name in enumerate(PAGES):
    with cols[1 + i]:
        if st.button(tr(name), key=f"nav_{name}", type="primary" if st.session_state["nav_page"] == name else "secondary", width="stretch"):
            st.session_state["nav_page"] = name
            st.rerun()
with cols[4]:
    lc1, lc2 = st.columns([1.5, 1], gap="small")
    names_ = list(UI_LANGS)
    cur_ = next(k for k, v in UI_LANGS.items() if v == st.session_state.get("ui_lang", "en"))
    pick_ = lc1.selectbox("Language", names_, index=names_.index(cur_), key="ui_lang_sel", label_visibility="collapsed")
    if UI_LANGS[pick_] != st.session_state.get("ui_lang", "en"):
        st.session_state["ui_lang"] = UI_LANGS[pick_]
        st.session_state["_lang"] = {"hi": "Hindi / Hinglish", "mr": "Marathi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali", "gu": "Gujarati", "en": "English (India)"}[UI_LANGS[pick_]]
        st.session_state.pop("as_lang", None)
        st.rerun()
    try:
        pop_ = lc2.popover("Aa", help=tr("Accessibility"))
    except TypeError:
        pop_ = lc2.popover("Aa")
    with pop_:
        st.markdown(f"**{tr('Accessibility')}**")
        size_ = st.select_slider(tr("Text size"), ["Normal", "Large", "Extra large"], key="a11y_size", format_func=tr)
        hc_ = st.toggle(tr("High contrast"), key="a11y_hc")
        rm_ = st.toggle(tr("Reduce motion"), key="a11y_rm")
_dev = ""
if st.session_state.get("ui_lang", "en") != "en":
    _dev = (".hero h1, .hero p, .eyebrow, .tip .tt, .ins-title, .skm-foot, .tip .step span, .kpi .l, .kpi .v, .kpi .s, .f-title, .as-cap, .hist-title, .hist-grp, .tag, .as-pill, .as-title, .as-sub, .panel h4, .panel .sub, button p, label p, .stMarkdown p "
            "{ letter-spacing:0 !important; text-transform:none !important; font-family:'Noto Sans Devanagari','Noto Sans Tamil','Noto Sans Telugu','Noto Sans Bengali','Noto Sans Gujarati','Nirmala UI','Mangal','IBM Plex Sans',sans-serif !important; line-height:1.55 !important; }")
_zoom = {"Normal": 1.0, "Large": 1.12, "Extra large": 1.25}[st.session_state.get("a11y_size", "Normal")]
_a11y = _dev + (f".block-container {{ zoom:{_zoom}; }}" if _zoom != 1.0 else "")
if st.session_state.get("a11y_hc"):
    _a11y += (".stApp { background:#FFFFFF !important; } .kpi, .panel, .panel.hd, div[data-testid=\"stVerticalBlockBorderWrapper\"], div[data-baseweb=\"select\"] > div { border:2px solid #1B2548 !important; background:#FFFFFF !important; } "
              ".kpi .l, .kpi .s, .panel .sub, .as-sub, .hist-empty, .msg-time, [data-testid=\"stCaptionContainer\"] { color:#1B2548 !important; } "
              "p, span, label, div, h1, h2, h3, h4 { color:#0B1230; } .hero * { color:#FFFFFF !important; } button:focus-visible, input:focus-visible { outline:3px solid #D6455D !important; outline-offset:2px; }")
if st.session_state.get("a11y_rm"):
    _a11y += " *, *::before, *::after { animation:none !important; transition:none !important; } .skm-splash { display:none !important; }"
if _a11y:
    st.markdown(f"<style>{_a11y}</style>", unsafe_allow_html=True)
PAGES[st.session_state["nav_page"]]()
_docs = f'<a href="{API}/docs" target="_blank">{tr("Open API documentation")}</a>' if st.session_state.get("mode") == "api" else ""
st.markdown(f'<div class="skm-foot"><span><b>SkillMap</b> &nbsp;|&nbsp; AI-enabled labour market intelligence for skilling</span><span>{tr("Supply data is simulated")}</span>{_docs}</div>', unsafe_allow_html=True)