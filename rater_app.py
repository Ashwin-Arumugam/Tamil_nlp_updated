import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import time

# =========================================================
# 1. CONFIGURATION
# =========================================================

st.set_page_config(page_title="Model Eval Tool", layout="wide")

MASTER_SHEET_GID = 1905633307
USER_CORRECTION_GID = 677241304

MODEL_SHEET_GIDS = {
    "A": 364113859,
    "B": 952136825,
    "C": 656105801,
    "D": 1630302691,
    "E": 803791042,
    "F": 141437423,
}

MODEL_TAB_NAMES = {
    "A": "qwen",
    "B": "nemotron",
    "C": "ministral",
    "D": "kimik2",
    "E": "gpt",
    "F": "gemma"
}

RATING_COLS = ["submission_id", "user", "rating"]
CORRECTION_COLS = ["submission_id", "user", "user_corrected"]

# =========================================================
# 2. STATE MANAGEMENT & LOADING
# =========================================================

if "username" not in st.session_state:
    st.title("Login")
    with st.form("login"):
        user = st.text_input("Username")
        if st.form_submit_button("Start") and user:
            st.session_state.username = user.strip()
            st.session_state.conn = st.connection("gsheets", type=GSheetsConnection)
            st.cache_data.clear()
            if "local_dfs" in st.session_state:
                del st.session_state.local_dfs
            st.rerun()
    st.stop()

if "local_dfs" not in st.session_state:
    st.session_state.local_dfs = {}

@st.cache_data(show_spinner=False, ttl=600)
def load_master_data(_conn):
    df = _conn.read(worksheet_id=MASTER_SHEET_GID)
    if df is None or df.empty:
        st.error("Master sheet is empty.")
        st.stop()
    df = df.dropna(how="all")
    unique_sentences = df["incorrect"].unique().tolist()
    return df, unique_sentences

def clean_rating_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=RATING_COLS)
    for col in RATING_COLS:
        if col not in df.columns:
            df[col] = None
    df["submission_id"] = df["submission_id"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors='coerce').fillna(0).astype(int)
    return df[RATING_COLS]

def load_all_tabs_into_variables(_conn):
    for m_id, gid in MODEL_SHEET_GIDS.items():
        try:
            df = _conn.read(worksheet_id=gid, ttl=0)
            if df is None or df.empty:
                st.session_state.local_dfs[m_id] = pd.DataFrame(columns=RATING_COLS)
            else:
                st.session_state.local_dfs[m_id] = clean_rating_df(df)
        except Exception:
            st.session_state.local_dfs[m_id] = pd.DataFrame(columns=RATING_COLS)

    try:
        df = _conn.read(worksheet_id=USER_CORRECTION_GID, ttl=0)
        if df is None or df.empty:
            st.session_state.local_dfs["corrections"] = pd.DataFrame(columns=CORRECTION_COLS)
        else:
            if "submission_id" in df.columns:
                df["submission_id"] = df["submission_id"].astype(str)
            valid_cols = [c for c in CORRECTION_COLS if c in df.columns]
            st.session_state.local_dfs["corrections"] = df[valid_cols]
    except:
        st.session_state.local_dfs["corrections"] = pd.DataFrame(columns=CORRECTION_COLS)

if not st.session_state.local_dfs:
    with st.spinner("Initializing variables from cloud..."):
        load_master_data(st.session_state.conn)
        load_all_tabs_into_variables(st.session_state.conn)

master_df, unique_list = load_master_data(st.session_state.conn)

# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def get_model_specific_row_id(master_df, current_incorrect, model_id):
    mask = (master_df["incorrect"] == current_incorrect) & (master_df["id"] == model_id)
    subset = master_df[mask]
    if not subset.empty:
        return str(subset.index[0] + 2) 
    return None

def get_existing_rating(m_id, sub_id):
    df = st.session_state.local_dfs.get(m_id)
    if df is not None and not df.empty:
        mask = (df["submission_id"] == str(sub_id)) & (df["user"] == st.session_state.username)
        match = df[mask]
        if not match.empty:
            try:
                val = int(match.iloc[0]["rating"])
                return val if val > 0 else None
            except:
                return None
    return None

def update_local_variable(key, new_row_df, sub_id):
    current_df = st.session_state.local_dfs.get(key)
    if current_df is not None and not current_df.empty:
        mask = (current_df["submission_id"] == str(sub_id)) & \
               (current_df["user"] == st.session_state.username)
        current_df = current_df[~mask]
    else:
        if key == "corrections":
            current_df = pd.DataFrame(columns=CORRECTION_COLS)
        else:
            current_df = pd.DataFrame(columns=RATING_COLS)

    updated_df = pd.concat([new_row_df, current_df], ignore_index=True)
    st.session_state.local_dfs[key] = updated_df

def save_to_local_memory(current_incorrect, versions):
    """Pulls current widget states and saves to memory ONLY."""
    model_ids = sorted(MODEL_TAB_NAMES.keys())
    for m_id in model_ids:
        val = st.session_state.get(f"pills_{m_id}_{st.session_state.u_index}")
        if val is not None:
            current_model_row_id = get_model_specific_row_id(master_df, current_incorrect, m_id)
            if current_model_row_id:
                new_row = pd.DataFrame([{
                    "submission_id": str(current_model_row_id),
                    "user": str(st.session_state.username),
                    "rating": int(val)
                }])
                update_local_variable(m_id, new_row, current_model_row_id)
            
    manual_fix_val = st.session_state.get(f"fix_{st.session_state.u_index}")
    if manual_fix_val:
        if not versions.empty:
            general_sub_id = str(versions.index[0] + 2)
            user_row = pd.DataFrame([{
                "submission_id": str(general_sub_id),
                "user": str(st.session_state.username),
                "user_corrected": str(manual_fix_val)
            }])
            update_local_variable("corrections", user_row, general_sub_id)

def sync_to_cloud():
    """Writes all local_dfs to Google Sheets."""
    save_bar = st.sidebar.progress(0, text="Syncing to Cloud...")
    model_ids = sorted(MODEL_TAB_NAMES.keys())
    total_tabs = len(model_ids) + 1
    current_tab = 0
    
    for key, df in st.session_state.local_dfs.items():
        if not df.empty:
            if key in MODEL_TAB_NAMES:
                tab_name = MODEL_TAB_NAMES[key]
            elif key == "corrections":
                tab_name = "user_corrections"
            else:
                continue 
            
            save_bar.progress((current_tab / total_tabs), text=f"Writing {tab_name}...")
            try:
                st.session_state.conn.update(worksheet=tab_name, data=df)
                time.sleep(1.0) 
            except Exception as e:
                st.sidebar.error(f"Failed to write {tab_name}: {e}")
            
            current_tab += 1

    save_bar.progress(1.0, text="Sync Complete!")
    time.sleep(1.0)
    save_bar.empty()

if "u_index" not in st.session_state:
    st.session_state.u_index = 0

# =========================================================
# 4. SIDEBAR LOGIC
# =========================================================

with st.sidebar:
    st.markdown(f"Logged in as: **{st.session_state.username}**")
    if st.button("Logout & Save", type="primary", use_container_width=True):
        # 1. Catch the current screen's unsaved ratings before logging out
        if st.session_state.u_index < len(unique_list):
            current_incorrect = unique_list[st.session_state.u_index]
            versions = master_df[master_df["incorrect"] == current_incorrect]
            save_to_local_memory(current_incorrect, versions)
        
        # 2. Push local variables to Google Sheets
        sync_to_cloud()
        
        # 3. Clear session and return to login
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

# =========================================================
# 5. UI
# =========================================================

# Handle End of List
if st.session_state.u_index >= len(unique_list):
    st.success("🎉 You've reached the end! Please click **Logout & Save** in the sidebar to upload your evaluations.")
    st.stop()

current_incorrect = unique_list[st.session_state.u_index]
versions = master_df[master_df["incorrect"] == current_incorrect]

# --- Navigation (Top) ---
c1, c2, c3 = st.columns([1, 6, 1])

if c1.button("⬅️ Prev") and st.session_state.u_index > 0:
    save_to_local_memory(current_incorrect, versions)
    st.session_state.u_index -= 1
    st.rerun()
    
c2.markdown(f"<center><b>Sentence {st.session_state.u_index + 1} of {len(unique_list)}</b></center>", unsafe_allow_html=True)

if c3.button("Next ➡️"):
    save_to_local_memory(current_incorrect, versions)
    st.session_state.u_index += 1
    st.rerun()

st.info(f"**Original:** {current_incorrect}")
st.divider()

# --- Ratings ---
model_ids = sorted(MODEL_TAB_NAMES.keys())
rating_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

rows = [model_ids[:3], model_ids[3:]]
for row_ids in rows:
    cols = st.columns(3)
    for i, m_id in enumerate(row_ids):
        with cols[i]:
            m_row = versions[versions["id"] == m_id]
            if not m_row.empty:
                specific_sub_id = str(m_row.index[0] + 2)
                
                st.markdown(f"**{MODEL_TAB_NAMES[m_id].capitalize()}**")
                st.success(m_row.iloc[0]["corrected"])
                
                key = f"pills_{m_id}_{st.session_state.u_index}"
                if key not in st.session_state:
                    saved_val = get_existing_rating(m_id, specific_sub_id)
                    if saved_val:
                        st.session_state[key] = saved_val
                
                st.pills("Rate", rating_options, key=key, label_visibility="collapsed")
            else:
                st.warning(f"No data for {MODEL_TAB_NAMES[m_id]}")

st.divider()
st.text_area("Correction (Optional):", key=f"fix_{st.session_state.u_index}")
