# -------------------- app.py --------------------
import random
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.components.v1 import iframe  # only used in admin view

# ── CONFIG ────────────────────────────────────────────────────────────
DB_FILE      = "votes.db"
SLIDES_FILE  = "slides.csv"      # columns: slide_id,model_id,gdrive_file_id
GDRIVE_RAW   = "https://drive.google.com/uc?export=download&id={}"
REQ_RANKS    = {1, 2, 3, 4}

# ── GLOBAL STYLE (one string, fixes the CSS NameError) ────────────────
CSS = """
<style>
/* compact page */
main .block-container{padding-top:1rem;padding-bottom:1rem;max-width:100%;}
header, #MainMenu, footer{visibility:hidden;}
/* radios side-by-side + highlight selected */
.stRadio>div{flex-direction:row;justify-content:center;}
label[data-baseweb="radio"]{margin-right:.8rem;}
label[data-selected="true"] span{background:#ff4b4b;color:#fff;padding:2px 6px;border-radius:3px;}
/* 2-col grid tight spacing */
[data-testid="column"]{width:calc(50% - .5rem)!important;flex:1 1 calc(50% - .5rem)!important;}
[data-testid="column"]>div{background:#fff;border-radius:6px;padding:.5rem;margin:.25rem;
                            box-shadow:0 1px 2px rgba(0,0,0,.12);}
.stButton button{font-weight:bold;font-size:12px;padding:2px 4px;}
</style>
"""

# ── DB HELPERS ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_conn(db_path: str = DB_FILE) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS votes(
              vote_id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id TEXT, slide_id INT, model_id TEXT,
              rank INT, points INT,
              voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_slide ON votes(user_id,slide_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_model       ON votes(model_id)")
    return conn  # DO NOT close – kept alive by Streamlit

def record_votes(user_id: str, slide_id: int, ranks: dict[int, int]) -> bool:
    """Validate ranks {model:rank} and write to DB."""
    if set(ranks.values()) != REQ_RANKS:
        st.error("⚠️ Please assign each rank (1-4) exactly once.")
        return False
    rows = [
        (user_id, slide_id, m, r, 5 - r, datetime.utcnow())
        for m, r in ranks.items()
    ]
    conn = get_conn()
    conn.executemany(
        "INSERT INTO votes(user_id,slide_id,model_id,rank,points,voted_at) "
        "VALUES(?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return True

def fetch_scoreboard() -> pd.DataFrame:
    q = """SELECT model_id,
                  SUM(points)  AS total_points,
                  COUNT(*)     AS vote_count,
                  ROUND(AVG(points),2) AS avg_points
           FROM votes
           GROUP BY model_id
           ORDER BY total_points DESC"""
    return pd.read_sql_query(q, get_conn())

# ── DATA LOADING ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_slides(path: str = SLIDES_FILE) -> pd.DataFrame:
    if not Path(path).is_file():
        st.error(f"❌ '{path}' not found.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    needed = {"slide_id", "model_id", "gdrive_file_id"}
    if not needed.issubset(df.columns):
        st.error(f"CSV must have columns: {', '.join(needed)}")
        return pd.DataFrame()
    df["url"] = df["gdrive_file_id"].apply(lambda fid: GDRIVE_RAW.format(fid))
    return df

# ── UTILITIES ─────────────────────────────────────────────────────────
def extract_id(val: str) -> str | None:
    """Return raw Drive file-id from either a bare id or any share URL."""
    if "/" not in val and len(val) > 20:
        return val
    try:
        return next(p for p in val.split("/") if len(p) > 20 and "." not in p)
    except StopIteration:
        return None

def blind_order(slide_id: int, models: list[str]) -> list[str]:
    key = f"order_{slide_id}"
    if key not in st.session_state:
        st.session_state[key] = random.sample(models, k=len(models))
    return st.session_state[key]

# ── VIDEO WIDGET ──────────────────────────────────────────────────────
def video_block(slide_id: int, model: str, label: str, gdrive_id: str, current_rank: int | None, disabled: bool):
    file_id = extract_id(gdrive_id)
    if not file_id:
        st.error("Bad Drive link/ID")
        return None
    # 16:9 responsive iframe
    st.markdown(f"""
    <div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;">
      <iframe src="https://drive.google.com/file/d/{file_id}/preview"
              style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"
              allowfullscreen></iframe>
    </div>""", unsafe_allow_html=True)

    if disabled:
        st.markdown(f"**Rank: {current_rank}**")
        return None

    # radio buttons (horizontal)
    return st.radio(
        f"{label} – choose rank",
        options=[1, 2, 3, 4],
        index=current_rank - 1 if current_rank else None,
        horizontal=True,
        key=f"radio_{slide_id}_{model}",
    )

# ── MAIN APP ──────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config("Video Ranking Tool", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    # Admin view toggle via URL "…?results=true"
    is_admin = st.query_params.get("results", ["false"])[0].lower() == "true"
    df_slides = load_slides()
    if df_slides.empty:
        st.stop()

    # Initialise session vars
    st.session_state.setdefault("uid", str(uuid.uuid4()))
    st.session_state.setdefault("voted", set())        # slides already saved
    st.session_state.setdefault("rankings", {})        # slide_id → {model:rank}

    if is_admin:
        show_admin()
        return

    st.title("🎬 Video Ranking Tool")

    for sid in sorted(df_slides["slide_id"].unique()):
        st.divider()
        voted = sid in st.session_state.voted
        st.header(f"Slide {sid} {'✅' if voted else ''}", anchor=False)

        subset = df_slides.query("slide_id == @sid")
        models = subset["model_id"].tolist()

        # ensure 4 videos
        if len(models) != 4:
            st.error("This slide does not have exactly 4 videos; skipping.")
            continue

        # ensure rankings dict exists
        st.session_state.rankings.setdefault(sid, {})

        # 2×2 grid
        cols = st.columns(2) + st.columns(2)
        changed = False
        for col, model, idx in zip(cols, blind_order(sid, models), range(4)):
            with col:
                gid = subset.loc[subset.model_id == model, "gdrive_file_id"].iloc[0]
                new_rank = video_block(
                    slide_id=sid,
                    model=model,
                    label=f"Video {idx+1}",
                    gdrive_id=gid,
                    current_rank=st.session_state.rankings[sid].get(model),
                    disabled=voted,
                )
                if new_rank and new_rank != st.session_state.rankings[sid].get(model):
                    # remove this rank from whoever had it
                    for m, r in st.session_state.rankings[sid].items():
                        if r == new_rank:
                            st.session_state.rankings[sid][m] = None
                    st.session_state.rankings[sid][model] = new_rank
                    changed = True

        if changed:
            st.rerun()

        # progress + submit
        if not voted:
            assigned = [r for r in st.session_state.rankings[sid].values() if r]
            st.progress(len(assigned) / 4)
            if len(assigned) == 4:
                if st.button(f"Submit Slide {sid}", key=f"submit_{sid}", type="primary"):
                    if record_votes(st.session_state.uid, sid, st.session_state.rankings[sid]):
                        st.session_state.voted.add(sid)
                        # Clean up radio keys for this slide
                        for k in list(st.session_state.keys()):
                            if k.startswith(f'radio_{sid}_'):
                                del st.session_state[k]
                        st.success("Saved! 🎉")
                        st.experimental_rerun()
            else:
                st.info("Rank all four videos to enable Submit.")

    st.divider()
    st.write("✅ **Thank you for participating!**")

# ── ADMIN / RESULTS VIEW ──────────────────────────────────────────────
def show_admin():
    st.title("📊 Live Results")
    df = fetch_scoreboard()
    if df.empty:
        st.info("No votes yet.")
        return
    st.bar_chart(df.set_index("model_id")["total_points"])
    st.dataframe(df.rename(columns={
        "model_id": "Model",
        "total_points": "Total Pts",
        "vote_count": "Votes",
        "avg_points": "Avg Pts/Vote",
    }))
    # optional raw table download
    csv = df.to_csv(index=False).encode()
    st.download_button("Download CSV", csv, "scoreboard.csv", "text/csv")

# ── RUN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
