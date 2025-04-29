import streamlit as st
import pandas as pd
import sqlite3
import uuid
import random
from pathlib import Path
from datetime import datetime
import streamlit.components.v1 as components

# --- Configuration ---
DB_FILE = "votes.db"
SLIDES_FILE = "slides.csv"
GDRIVE_BASE_URL = "https://drive.google.com/uc?export=download&id={}"
REQUIRED_RANKS = {1, 2, 3, 4}

# --- Database Setup ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False) # Allow access from multiple threads if Streamlit uses them
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    return conn

def init_db():
    """Initializes the database table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            vote_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            slide_id INTEGER NOT NULL,
            model_id TEXT NOT NULL,
            rank INTEGER NOT NULL,
            points INTEGER NOT NULL, -- Calculated in Python before insert
            voted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    # Optional: Add indexes for faster querying if data grows
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_slide ON votes (user_id, slide_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON votes (model_id);")
    conn.commit()
    conn.close()

# --- Data Loading ---
@st.cache_data
def load_slides(csv_path):
    """Loads slide data from CSV and constructs video URLs."""
    try:
        df = pd.read_csv(csv_path)
        # Validate columns
        if not {'slide_id', 'model_id', 'gdrive_file_id'}.issubset(df.columns):
            st.error(f"Error: {csv_path} must contain 'slide_id', 'model_id', and 'gdrive_file_id' columns.")
            return pd.DataFrame() # Return empty dataframe on error

        # Construct Google Drive URL - Use .loc to avoid SettingWithCopyWarning
        df['url'] = df['gdrive_file_id'].apply(lambda file_id: GDRIVE_BASE_URL.format(file_id))
        return df
    except FileNotFoundError:
        st.error(f"Error: Slides file '{csv_path}' not found. Please create it.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading slides: {e}")
        return pd.DataFrame()

# --- Application Logic ---
def get_blind_order(slide_id, models_list):
    """Gets or generates a shuffled order for models for a given slide and session."""
    session_key = f"order_{slide_id}"
    if session_key not in st.session_state.order:
        shuffled_list = models_list[:]  # Create a copy
        random.shuffle(shuffled_list)
        st.session_state.order[session_key] = shuffled_list
        # st.write(f"DEBUG: Shuffled order for slide {slide_id}: {shuffled_list}") # Uncomment for debugging
    return st.session_state.order[session_key]

def record_votes(user_id, slide_id, ranks_dict):
    """Validates and records votes into the database."""
    conn = get_db_connection()
    try:
        rows_to_insert = []
        submitted_ranks = set(ranks_dict.values())

        # Validation 1: Check if all ranks 1-4 are present exactly once
        if submitted_ranks != REQUIRED_RANKS:
            st.error(f"⚠️ Please assign each rank (1, 2, 3, 4) exactly once for Slide {slide_id}.")
            return False

        # Validation 2: Check if this user already voted for this slide in DB (optional but good)
        # For simplicity in this version, we rely on the session state `voted_slides` check primarily.
        # A DB check could prevent re-voting if the user refreshes after voting but before the session state is fully set.
        # cursor = conn.cursor()
        # cursor.execute("SELECT 1 FROM votes WHERE user_id = ? AND slide_id = ?", (user_id, slide_id))
        # if cursor.fetchone():
        #     st.warning(f"Votes already recorded for Slide {slide_id} by this user ID.")
        #     return False # Already voted

        timestamp = datetime.now()
        for model_id, rank in ranks_dict.items():
            points = 5 - rank
            rows_to_insert.append({
                "user_id": user_id,
                "slide_id": slide_id,
                "model_id": str(model_id), # Ensure model_id is string
                "rank": rank,
                "points": points,
                "voted_at": timestamp
            })

        df_votes = pd.DataFrame(rows_to_insert)
        df_votes.to_sql("votes", conn, if_exists="append", index=False)
        conn.commit()
        st.session_state.voted_slides.add(slide_id) # Mark as voted in session
        return True
    except sqlite3.Error as e:
        st.error(f"Database error while saving votes: {e}")
        conn.rollback() # Rollback changes on error
        return False
    finally:
        conn.close()

# --- Scoreboard ---
# @st.cache_data(ttl=30) # Cache scoreboard for 30 seconds for 'live' feel without constant query
def get_scoreboard():
    """Queries the database and returns the aggregated scores."""
    conn = get_db_connection()
    try:
        query = """
            SELECT
                model_id,
                SUM(points) AS total_points
            FROM votes
            GROUP BY model_id
            ORDER BY total_points DESC;
        """
        # Use Pandas to read SQL query directly into a DataFrame
        score_df = pd.read_sql_query(query, conn)
        return score_df
    except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
        # Handle case where table might not exist yet or other DB errors
        st.warning(f"Could not retrieve scoreboard yet or DB error: {e}")
        return pd.DataFrame(columns=['model_id', 'total_points']) # Return empty df
    finally:
        conn.close()

# --- Main App ---
st.set_page_config(page_title="Blind Video Test", layout="wide")

# Initialize Database
init_db()

# Load Slide Data
slides_df = load_slides(SLIDES_FILE)

# Initialize Session State if not already done
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    st.session_state.order = {} # Stores { "order_slide_id": [shuffled_models] }
    st.session_state.voted_slides = set() # Stores {slide_id} that have been voted on
    st.session_state.slide_ranks = {} # Stores { slide_id: {model_id: rank} } before submission

st.sidebar.title("📊 Live Scoreboard")
st.sidebar.write("Points: Rank 1 = 4pts, Rank 2 = 3pts, Rank 3 = 2pts, Rank 4 = 1pt")
scoreboard_df = get_scoreboard()
if not scoreboard_df.empty:
    # Ensure model_id is suitable for index/charting (string)
    scoreboard_df['model_id'] = scoreboard_df['model_id'].astype(str)
    # Use model_id directly if it's unique and suitable as identifier
    st.sidebar.bar_chart(scoreboard_df.set_index('model_id')['total_points'])
else:
    st.sidebar.info("No votes recorded yet.")

st.sidebar.markdown("---")
st.sidebar.info(f"Your User ID (for this session): `{st.session_state.user_id}`")


st.title("🎬 Blind Video Comparison Tool")
st.markdown("Please watch the 4 videos for each slide below. The order is randomized for each user. Assign a unique rank (1 = Best, 4 = Worst) to each video.")

if slides_df.empty:
    st.warning("No slide data loaded. Cannot display tests.")
else:
    unique_slide_ids = slides_df['slide_id'].unique()
    unique_slide_ids.sort() # Process slides in order

    for slide_id in unique_slide_ids:
        st.divider() # Visually separate slides
        slide_voted = slide_id in st.session_state.voted_slides
        header_text = f"Slide {slide_id}"
        if slide_voted:
            header_text += " (✅ Voted)"
        st.header(header_text)

        # Get models specific to this slide
        slide_models_df = slides_df[slides_df['slide_id'] == slide_id]
        models_for_slide = slide_models_df['model_id'].tolist()

        # Ensure 4 models per slide as per design
        if len(models_for_slide) != 4:
             st.error(f"Configuration Error: Slide {slide_id} does not have exactly 4 models defined in {SLIDES_FILE}. Skipping this slide.")
             continue

        # Get the randomized order for display
        shuffled_model_ids = get_blind_order(slide_id, models_for_slide)

        # Initialize ranks for this slide if not already done
        if slide_id not in st.session_state.slide_ranks:
             st.session_state.slide_ranks[slide_id] = {model_id: 0 for model_id in models_for_slide} # 0 = unranked

        # Display videos and ranking widgets side-by-side
        cols = st.columns(4)
        current_ranks = st.session_state.slide_ranks[slide_id] # Get current ranks for this slide

        for i, model_id in enumerate(shuffled_model_ids):
            with cols[i]:
                # Get data for the current model
                row = slide_models_df.loc[slide_models_df['model_id'] == model_id].iloc[0]
                gdrive_link_or_id = row['gdrive_file_id'] # This might be a full URL or just an ID

                # Extract the actual file ID
                try:
                    # Assume it might be a full URL like /file/d/ID/view... or just the ID
                    parts = gdrive_link_or_id.split('/')
                    file_id = next(p for p in reversed(parts) if p and '?' not in p and '=' not in p and len(p) > 20) # Find the likely ID
                    if not file_id:
                        raise ValueError("Could not extract file ID")
                except Exception:
                     st.error(f"Could not extract GDrive File ID from: {gdrive_link_or_id}")
                     file_id = None # Prevent further errors

                # Using a placeholder text that doesn't reveal the model_id
                st.subheader(f"Video Option {i+1}")
                if file_id:
                    try:
                        # Use iframe with preview URL
                        preview_url = f"https://drive.google.com/file/d/{file_id}/preview"
                        components.iframe(preview_url, height=315) # Adjust height if needed
                    except Exception as e:
                        st.error(f"Could not load video {i+1}. GDrive ID: {file_id}. Error: {e}")
                else:
                    st.warning(f"Skipping video {i+1} due to GDrive ID extraction error.")

                # Use st.selectbox for ranking to easily see current selection
                rank_key = f"rank_{slide_id}_{model_id}"
                selected_rank = st.selectbox(
                    f"Rank Video Option {i+1}",
                    options=[0, 1, 2, 3, 4], # Include 0 for 'unranked'
                    format_func=lambda x: f"Rank {x}" if x > 0 else "Select Rank",
                    index=current_ranks[model_id], # Set initial value from session state
                    key=rank_key,
                    disabled=slide_voted # Disable if already voted
                )
                # Update session state immediately on change (selectbox triggers rerun)
                st.session_state.slide_ranks[slide_id][model_id] = selected_rank


        # Submit Button Area for the current slide
        st.markdown("---") # Small separator before button
        submit_key = f"submit_{slide_id}"
        if not slide_voted:
            if st.button("Submit Ranks for Slide " + str(slide_id), key=submit_key):
                # Retrieve the latest ranks directly from session state just before submitting
                ranks_to_submit = st.session_state.slide_ranks[slide_id]

                # Check if any rank is still 0 (unselected)
                if 0 in ranks_to_submit.values():
                    st.error(f"⚠️ Please assign a rank (1-4) to all 4 video options for Slide {slide_id}.")
                else:
                    # Attempt to record votes
                    success = record_votes(st.session_state.user_id, slide_id, ranks_to_submit)
                    if success:
                        st.success(f"✅ Votes for Slide {slide_id} submitted successfully!")
                        # Optional: Clear ranks for the slide from session state after successful submission
                        # st.session_state.slide_ranks[slide_id] = {m: 0 for m in models_for_slide}
                        st.rerun() # Rerun to update scoreboard and disable widgets
                    # Error messages (validation, DB error) are handled within record_votes

        elif slide_voted:
             st.info(f"You have already submitted your rankings for Slide {slide_id}.")

    st.divider()
    st.header("✅ Thank You!")
    st.markdown("All slides processed. You can see the current overall scores in the sidebar.") 