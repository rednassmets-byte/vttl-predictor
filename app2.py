"""
VTTL Ranking Predictor — Streamlit App
Run with:  streamlit run app.py
Place ranking_model.pkl in the same folder.
"""

import pickle
import numpy as np
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="VTTL Ranking Predictor", page_icon="🏓", layout="centered")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stAppViewContainer"] { background: #0d0f14; color: #e8eaf0; }
[data-testid="stHeader"] { background: transparent; }

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.title-block { text-align: center; padding: 2rem 0 1rem; }
.title-block h1 {
    font-size: 2.8rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(135deg, #f0f4ff 0%, #7eb8f7 60%, #3d7fcc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
}
.title-block p { color: #6b7280; font-size: 1rem; margin-top: 0.4rem; font-weight: 300; }

.card {
    background: #181b24; border: 1px solid #2a2d3a;
    border-radius: 16px; padding: 2rem; margin: 1rem 0;
}

.rank-badge {
    display: inline-block; font-family: 'Syne', sans-serif;
    font-size: 2.4rem; font-weight: 800; padding: 0.3rem 1.2rem;
    border-radius: 12px; letter-spacing: 2px;
}
.rank-up   { background: #0f2318; color: #4ade80; border: 2px solid #166534; }
.rank-down { background: #2a0f0f; color: #f87171; border: 2px solid #7f1d1d; }
.rank-same { background: #1e2535; color: #7eb8f7; border: 2px solid #3d6fa8; }

.result-label {
    color: #6b7280; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem;
}
.result-name {
    font-family: 'Syne', sans-serif; font-size: 1.5rem;
    font-weight: 800; color: #e8eaf0;
}
.chip {
    display: inline-block; background: #1e2535; color: #9ca3af;
    border-radius: 20px; padding: 0.2rem 0.75rem; font-size: 0.78rem; margin: 0.15rem;
}
.prob-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.5rem; }
.prob-rank { font-family: 'Syne', sans-serif; font-weight: 700; width: 2.5rem; font-size: 0.9rem; }
.prob-bar-bg { flex: 1; background: #1e2535; border-radius: 6px; height: 9px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 6px; }
.prob-pct { color: #9ca3af; font-size: 0.82rem; width: 3.5rem; text-align: right; }

div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background: #1e2535 !important; border: 1px solid #2a2d3a !important;
    border-radius: 10px !important; color: #e8eaf0 !important;
}
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label {
    color: #9ca3af !important; font-size: 0.82rem !important;
    text-transform: uppercase; letter-spacing: 0.8px;
}
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #3b82f6) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.65rem 2rem !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RANK_ORDER = ['A','B0','B2','B4','B6','C0','C2','C4','C6',
              'D0','D2','D4','D6','E0','E2','E4','E6','NG']
RANK_TO_IDX = {r: i for i, r in enumerate(RANK_ORDER)}
# Find ranking_model.pkl — checks script folder first, then cwd
def _find_model():
    candidates = [
        Path(__file__).resolve().parent / "ranking_model.pkl",
        Path.cwd() / "ranking_model.pkl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # return first so error message shows the right path

MODEL_PATH = _find_model()


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def build_kaart(member):
    kaart = {r: [0, 0] for r in RANK_ORDER}
    entries = getattr(member, "ResultEntries", None)
    if not entries:
        return kaart
    for entry in entries:
        opp_rank = None
        for field in ("OpponentRanking", "OpponentClassement", "OpponentRankingIndex", "Ranking"):
            val = getattr(entry, field, None)
            if val:
                opp_rank = str(val).strip().upper()
                break
        if not opp_rank or opp_rank not in RANK_TO_IDX:
            continue
        outcome = str(getattr(entry, "Result", "") or "").strip().upper()
        if outcome in ("V", "W", "WIN", "1"):
            kaart[opp_rank][0] += 1
        elif outcome in ("D", "L", "LOSS", "DEFEAT", "0"):
            kaart[opp_rank][1] += 1
    return kaart


def build_features(current_rank, kaart, category, province, arts):
    cur_idx = RANK_TO_IDX.get(current_rank, 8)
    feature_cols = arts["feature_cols"]
    age_group_map = arts.get("age_group_map", {})
    feats = {col: 0 for col in feature_cols}
    tw = tl = wa = la = ws = ls = wb = lb = 0

    for rank, (wins, losses) in kaart.items():
        ri = RANK_TO_IDX.get(rank, 8)
        tw += wins
        tl += losses
        if ri < cur_idx:
            wa += wins
            la += losses
        elif ri == cur_idx:
            ws += wins
            ls += losses
        else:
            wb += wins
            lb += losses
        if f"w_{rank}" in feats:
            feats[f"w_{rank}"] = wins
        if f"l_{rank}" in feats:
            feats[f"l_{rank}"] = losses

    tg = tw + tl
    feats["total_wins"]     = tw
    feats["total_losses"]   = tl
    feats["total_games"]    = tg
    feats["win_rate"]       = tw / tg if tg else 0
    feats["wins_above"]     = wa
    feats["losses_above"]   = la
    feats["wins_same"]      = ws
    feats["losses_same"]    = ls
    feats["wins_below"]     = wb
    feats["losses_below"]   = lb
    feats["above_win_rate"] = wa / (wa + la) if (wa + la) else 0
    feats["same_win_rate"]  = ws / (ws + ls) if (ws + ls) else 0
    feats["below_win_rate"] = wb / (wb + lb) if (wb + lb) else 0
    feats["current_rank_idx"] = cur_idx

    # Age group features — model understands youth/junior/senior/veteran
    age_val = age_group_map.get(category.upper(), 6)
    if "age_group" in feats:
        feats["age_group"]  = age_val
    if "is_youth" in feats:
        feats["is_youth"]   = 1 if age_val < 4 else 0
    if "is_junior" in feats:
        feats["is_junior"]  = 1 if age_val == 4 else 0
    if "is_veteran" in feats:
        feats["is_veteran"] = 1 if age_val > 6 else 0

    return np.array([feats[c] for c in feature_cols]).reshape(1, -1)


def get_direction(cur, pred):
    ci = RANK_TO_IDX.get(cur, 8)
    pi = RANK_TO_IDX.get(pred, 8)
    if pi < ci:
        return "up", "⬆ PROMOTION", "rank-up"
    if pi > ci:
        return "down", "⬇ DROP", "rank-down"
    return "same", "➡ SAME RANK", "rank-same"


def rank_color(rank):
    idx = RANK_TO_IDX.get(rank, 8)
    if idx <= 3:
        return "#fbbf24"
    if idx <= 7:
        return "#60a5fa"
    if idx <= 11:
        return "#a78bfa"
    return "#9ca3af"


# Youth categories — bonus is baked into the ML model's training targets
# CAD/MIN/BEN/PRE: model trained with +1 rank on next_rank targets
# JUN/J19/J21: model trained on real data (they grade easier naturally)
YOUTH_CATS  = {"CAD", "MIN", "BEN", "PRE"}
JUNIOR_CATS = {"JUN", "J19", "J21"}

def apply_youth_bonus(pred_rank, category):
    """Youth bonus is already in the model — just flag for display."""
    cat = category.strip().upper()
    if cat in YOUTH_CATS:
        return pred_rank, 1   # bonus already applied by model
    if cat in JUNIOR_CATS:
        return pred_rank, -1  # flag as junior (grades easier, no rank shift)
    return pred_rank, 0


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
  <h1>🏓 VTTL Predictor</h1>
  <p>Predict a player's next ranking from their match history</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"ranking_model.pkl not found. Place it in: {MODEL_PATH.parent}")
    st.stop()

arts = load_model()

# Input form
col1, col2 = st.columns([3, 1])
with col1:
    name_input = st.text_input("Player name", placeholder="e.g. Frank Hollak")
with col2:
    season_input = st.number_input("Season", min_value=15, max_value=26, value=26, step=1)

username = ""
password = ""

predict_btn = st.button("Predict next ranking")

# ── Phase 1: fetch members on predict button click ────────────────────────────
if predict_btn:
    if not name_input.strip():
        st.warning("Please enter a player name.")
        st.stop()

    with st.spinner("Fetching from VTTL API…"):
        try:
            from pyvttl import VttlApi
            api = VttlApi()

            # Step 1: search by name to find UniqueIndex
            search_result = api.getMembers(
                name_search=name_input.strip(),
                season=int(season_input),
            )
            found = list(getattr(search_result, "MemberEntries", None) or [])
            if not found:
                st.warning(f"No player found for '{name_input}' in season {int(season_input)}.")
                st.stop()

            # Step 2: fetch full results for each found player
            members = []
            for m in found:
                uid = getattr(m, "UniqueIndex", None)
                if uid is None:
                    continue
                detail = api.getMembers(
                    unique_index=int(uid),
                    season=int(season_input),
                    with_results=True,
                    with_opponent_ranking_evaluation=True,
                )
                detail_entries = list(getattr(detail, "MemberEntries", None) or [])
                if detail_entries:
                    members.append(detail_entries[0])

        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    if not members:
        st.warning(f"Could not load match results for '{name_input}' in season {int(season_input)}.")
        st.stop()

    # Store fetched members and season in session state
    st.session_state["_members"] = members
    st.session_state["_season"] = int(season_input)
    st.session_state["_member"] = members[0] if len(members) == 1 else None

# ── Phase 2: player selection (if multiple found) ─────────────────────────────
if st.session_state.get("_members") and st.session_state["_member"] is None:
    members = st.session_state["_members"]
    options = {
        f"{getattr(m,'FirstName','')} {getattr(m,'LastName','')} "
        f"— {getattr(m,'Club','?')} (#{getattr(m,'UniqueIndex','?')})": m
        for m in members
    }
    chosen_key = st.selectbox(f"Found {len(members)} players — select one:", list(options.keys()))
    if st.button("Confirm selection"):
        st.session_state["_member"] = options[chosen_key]
        st.rerun()
    st.stop()

# ── Phase 3: run prediction ───────────────────────────────────────────────────
if "_member" not in st.session_state:
    st.stop()

member = st.session_state["_member"]
season_input = st.session_state.get("_season", int(season_input))

# Re-init API for category lookup
try:
    from pyvttl import VttlApi as _VttlApi
    api = _VttlApi()
except Exception:
    api = None

# Extract player info
full_name    = f"{getattr(member,'FirstName','')} {getattr(member,'LastName','')}".strip()
unique_index = getattr(member, "UniqueIndex", "?")
current_rank = str(getattr(member, "Ranking", "NG") or "NG").strip().upper()

# Step 3: get real category via getPlayerCategories (works without credentials)
# Priority order — pick the highest age group if multiple categories returned
CAT_PRIORITY = {
    'PRE': 0, 'BEN': 1, 'MIN': 2, 'CAD': 3,
    'JUN': 4, 'J19': 4, 'J21': 5,
    'SEN': 6,
    'V40': 7, 'V50': 8, 'V60': 9, 'V65': 10,
    'V70': 11, 'V75': 12, 'V80': 13, 'V85': 14,
}
category = "SEN"
try:
    cat_result = api.getPlayerCategories(unique_index=int(unique_index), season=int(season_input))
    # Collect ALL short names returned
    found_cats = []
    cat_entries = getattr(cat_result, "CategoryEntries", None) or []
    if not isinstance(cat_entries, list):
        cat_entries = [cat_entries]
    for ce in cat_entries:
        short = str(getattr(ce, "ShortName", "") or "").strip().upper()
        if short and short not in ("NONE", ""):
            found_cats.append(short)
    # Also try direct ShortName on root object
    short = str(getattr(cat_result, "ShortName", "") or "").strip().upper()
    if short and short not in ("NONE", ""):
        found_cats.append(short)
    # Pick the category with highest priority (highest age group = most specific)
    if found_cats:
        category = max(found_cats, key=lambda c: CAT_PRIORITY.get(c, 6))
except Exception:
    pass

# If API gave nothing, infer from match history as fallback
if category == "SEN":
    entries = getattr(member, "ResultEntries", None) or []
    series = " ".join(str(getattr(e, "TournamentSerieName", "") or "").lower() for e in entries)
    match_ids = " ".join(str(getattr(e, "MatchId", "") or "") for e in entries)
    if any(x in series for x in ("cadet", "kadett")):
        category = "CAD"
    elif any(x in series for x in ("miniem", "minime")):
        category = "MIN"
    elif any(x in series for x in ("benjamin",)):
        category = "BEN"
    elif any(x in series for x in ("junior",)):
        category = "JUN"
    elif "jeugd" in series or "youth" in series:
        category = "CAD"
    elif "J" in match_ids[:30]:  # PANTJ prefix = jeugd
        category = "CAD" 
club         = str(getattr(member, "Club", "") or "").strip()
province     = str(
    getattr(member, "Province", "")
    or getattr(member, "ClubCategory", "")
    or "Antwerpen"
).strip()

kaart = build_kaart(member)
total_games = sum(w + l for w, l in kaart.values())

if current_rank not in RANK_TO_IDX:
    st.error(f"Unknown ranking '{current_rank}' — cannot predict.")
    st.stop()

MIN_GAMES = 15
if total_games < MIN_GAMES:
    st.markdown(f"""
    <div style="background:#1e1a0f;border:1px solid #92400e;border-radius:14px;
                padding:1.5rem;margin-top:1rem;text-align:center">
        <div style="font-size:2rem;margin-bottom:0.5rem">⚠️</div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.2rem;
                    color:#fbbf24;margin-bottom:0.4rem">Not enough data</div>
        <div style="color:#9ca3af;font-size:0.9rem">
            <b style="color:#e8eaf0">{full_name}</b> has only played
            <b style="color:#fbbf24">{total_games} match{'es' if total_games != 1 else ''}</b>
            this season.<br>At least <b style="color:#e8eaf0">{MIN_GAMES} matches</b>
            are needed for a reliable prediction.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Still show the match history if there's anything
    played_rows = []
    for r in RANK_ORDER:
        w, l = kaart.get(r, [0, 0])
        if w + l > 0:
            played_rows.append({"Opponent rank": r, "Wins": w, "Losses": l,
                                 "Games": w + l, "Win %": f"{w/(w+l)*100:.0f}%"})
    if played_rows:
        import pandas as pd
        with st.expander("Match history so far"):
            st.dataframe(pd.DataFrame(played_rows), hide_index=True, use_container_width=True)
    st.stop()

# Predict
X              = build_features(current_rank, kaart, category, province, arts)
pred_idx       = arts["model"].predict(X)[0]
pred_rank_base = arts["le_target"].inverse_transform([pred_idx])[0]
proba          = arts["model"].predict_proba(X)[0]
confidence     = float(proba[pred_idx])
top5 = [
    (arts["le_target"].inverse_transform([i])[0], float(proba[i]))
    for i in np.argsort(proba)[::-1][:5]
]

# Apply youth category bonus
pred_rank, youth_bonus = apply_youth_bonus(pred_rank_base, category)
dir_key, dir_label, dir_css = get_direction(current_rank, pred_rank)

# Render result card

st.markdown(f"""
<div style="margin-bottom:1.5rem">
    <div class="result-name">{full_name}</div>
    <div style="margin-top:0.4rem">
        <span class="chip">#{unique_index}</span>
        <span class="chip">{club}</span>
        <span class="chip">{category}</span>
        <span class="chip">Season {int(season_input)}</span>
        <span class="chip">{total_games} games</span>
    </div>
</div>
""", unsafe_allow_html=True)

cur_col = rank_color(current_rank)
st.markdown(f"""
<div style="display:flex;align-items:center;gap:2rem;margin-bottom:1.5rem;flex-wrap:wrap">
    <div style="text-align:center">
        <div class="result-label">Current rank</div>
        <span class="rank-badge"
              style="color:{cur_col};background:#1e2535;border:2px solid {cur_col}44">
            {current_rank}
        </span>
    </div>
    <div style="font-size:2rem;color:#374151">&#8594;</div>
    <div style="text-align:center">
        <div class="result-label">Predicted next</div>
        <span class="rank-badge {dir_css}">{pred_rank}</span>
    </div>
    <div style="text-align:center">
        <div class="result-label">Verdict</div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;
                    font-size:1.05rem;color:#e8eaf0">{dir_label}</div>
        <div style="color:#6b7280;font-size:0.85rem;margin-top:0.2rem">
            {confidence*100:.0f}% confidence
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Show youth info badge
if youth_bonus == 1:
    st.markdown('''
    <div style="background:#1a2e1a;border:1px solid #166534;border-radius:10px;
                padding:0.6rem 1rem;margin-bottom:0.5rem;display:flex;
                align-items:center;gap:0.6rem">
        <span style="font-size:1.2rem">🟢</span>
        <div>
            <span style="color:#4ade80;font-family:Syne,sans-serif;font-weight:700;
                         font-size:0.95rem">Youth bonus applied by model</span>
            <div style="color:#6b7280;font-size:0.78rem;margin-top:0.1rem">
                CAD / MIN / BEN / PRE players are predicted +1 rank higher
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
elif youth_bonus == -1:
    st.markdown('''
    <div style="background:#1a2218;border:1px solid #3d6fa8;border-radius:10px;
                padding:0.6rem 1rem;margin-bottom:0.5rem;display:flex;
                align-items:center;gap:0.6rem">
        <span style="font-size:1.2rem">🔵</span>
        <div>
            <span style="color:#7eb8f7;font-family:Syne,sans-serif;font-weight:700;
                         font-size:0.95rem">Junior grading advantage</span>
            <div style="color:#6b7280;font-size:0.78rem;margin-top:0.1rem">
                JUN / J19 / J21 players grade easier — reflected in the model
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


st.markdown('<div class="result-label" style="margin-bottom:0.6rem">Top predictions</div>',
            unsafe_allow_html=True)

for rank, prob in top5:
    is_top  = rank == pred_rank
    bar_w   = int(prob * 100)
    bar_col = "linear-gradient(90deg,#1d4ed8,#60a5fa)" if is_top else "linear-gradient(90deg,#374151,#4b5563)"
    txt_col = "#e8eaf0" if is_top else "#6b7280"
    st.markdown(f"""
    <div class="prob-row">
        <div class="prob-rank" style="color:{txt_col}">{rank}</div>
        <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{bar_w}%;background:{bar_col}"></div>
        </div>
        <div class="prob-pct">{prob*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Match history
played_rows = []
for r in RANK_ORDER:
    w, l = kaart.get(r, [0, 0])
    if w + l > 0:
        played_rows.append({
            "Opponent rank": r,
            "Wins": w,
            "Losses": l,
            "Games": w + l,
            "Win %": f"{w/(w+l)*100:.0f}%",
        })

if played_rows:
    import pandas as pd
    with st.expander("Match history (kaart)"):
        st.dataframe(pd.DataFrame(played_rows), hide_index=True, use_container_width=True)
else:
    st.info("No match results found for this player/season.")