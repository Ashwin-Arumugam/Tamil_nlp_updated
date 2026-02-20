import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
import time

# =========================================================
# 1. CONFIGURATION & CSS
# =========================================================

st.set_page_config(page_title="Model Eval Tool", layout="wide")

# Aggressive CSS to force pills onto a single line
st.markdown(
    """
    <style>
    /* Force the flex container to never wrap */
    div[data-testid="stPills"] > div {
        flex-wrap: nowrap !important;
        gap: 2px !important; 
        overflow-x: auto !important; 
        padding-bottom: 4px; 
    }
    
    /* Shrink the individual pill buttons */
    div[data-testid="stPills"] button {
        padding: 2px 6px !important; 
        min-width: 30px !important; 
        min-height: 32px !important;
        font-size: 14px !important;
    }
    
    /* Remove any extra internal padding Streamlit adds to the text */
    div[data-testid="stPills"] button p {
        margin: 0px !important;
        padding: 0px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MASTER_SHEET_GID = 1905633307 

MODEL_TAB_NAMES = {
    "A": "qwen",
    "B": "nemotron",
    "C": "ministral",
    "D": "kimik2",
    "E": "gpt",
    "F": "gemma"
}

# Added 'reason' to the rating columns schema
RATING_COLS = ["submission_id", "user", "rating", "reason"]
CORRECTION_COLS = ["submission_id", "user", "user_corrected"]

REASON_OPTIONS = ["spelling error", "not fluent enough", "grammar error"]

# =========================================================
# 2. STATE MANAGEMENT & LOADING
# =========================================================

if "username" not in st.session_state:
    st.title("Tamil NLP grammar correction")
    with st.form("entry_form"):
        user = st.text_input("Name")
        if st.form_submit_button("Continue") and user:
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
    df = _conn.read(worksheet=0, ttl=0) 
    if df is None or df.empty:
        st.error("Master sheet is empty or could not be loaded.")
        st.stop()
    df = df.dropna(how="all")
    unique_sentences = df["incorrect"].unique().tolist()
    return df, unique_sentences

def clean_rating_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=RATING_COLS)
    
    if "submission_id" in df.columns:
        df = df.dropna(subset=["submission_id"])
        
    for col in RATING_COLS:
        if col not in df.columns:
            df[col] = None

    df["submission_id"] = df["submission_id"].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df = df[~df["submission_id"].isin(["", "None", "nan"])]
    
    df["user"] = df["user"].astype(str).str.strip()
    
    df["rating"] = pd.to_numeric(df["rating"], errors='coerce')
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    
    # Clean the reason column
    df["reason"] = df["reason"].fillna("").astype(str)
    df["reason"] = df["reason"].replace(["nan", "None"], "")
    
    return df[RATING_COLS]

def clean_correction_df(df):
    if df is None or df.empty:
        return pd.DataFrame(columns=CORRECTION_COLS)
        
    if "submission_id" in df.columns:
        df = df.dropna(subset=["submission_id"])
        
    for col in CORRECTION_COLS:
        if col not in df.columns:
            df[col] = None
            
    df["submission_id"] = df["submission_id"].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df = df[~df["submission_id"].isin(["", "None", "nan"])]
    df["user"] = df["user"].astype(str).str.strip()
    
    return df[CORRECTION_COLS]

def load_all_tabs_into_variables(_conn):
    for m_id, tab_name in MODEL_TAB_NAMES.items():
        try:
            df = _conn.read(worksheet=tab_name, ttl=0)
            st.session_state.local_dfs[m_id] = clean_rating_df(df)
        except Exception as e:
            st.error(f"Error loading tab '{tab_name}': {e}")
            st.session_state.local_dfs[m_id] = pd.DataFrame(columns=RATING_COLS)

    try:
        df = _conn.read(worksheet="user_corrections", ttl=0)
        st.session_state.local_dfs["corrections"] = clean_correction_df(df)
    except Exception as e:
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

def get_existing_reason(m_id, sub_id):
    df = st.session_state.local_dfs.get(m_id)
    if df is not None and not df.empty:
        mask = (df["submission_id"] == str(sub_id)) & (df["user"] == st.session_state.username)
        match = df[mask]
        if not match.empty:
            reason_str = str(match.iloc[0]["reason"])
            if reason_str and reason_str not in ["nan", "None", ""]:
                # Convert comma-separated string back to a list
                return [r.strip() for r in reason_str.split(",") if r.strip() in REASON_OPTIONS]
    return []

def get_existing_correction(sub_id):
    df = st.session_state.local_dfs.get("corrections")
    if df is not None and not df.empty:
        mask = (df["submission_id"] == str(sub_id)) & (df["user"] == st.session_state.username)
        match = df[mask]
        if not match.empty:
            return str(match.iloc[0]["user_corrected"])
    return ""

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
    model_ids = sorted(MODEL_TAB_NAMES.keys())
    for m_id in model_ids:
        val = st.session_state.get(f"pills_{m_id}_{st.session_state.u_index}")
        if val is not None:
            current_model_row_id = get_model_specific_row_id(master_df, current_incorrect, m_id)
            if current_model_row_id:
                # Retrieve the selected reasons if rating <= 7
                reason_key = f"reason_{m_id}_{st.session_state.u_index}"
                reason_list = st.session_state.get(reason_key, [])
                
                # If rating > 7, we don't save a reason even if one was previously selected
                reason_str = ", ".join(reason_list) if (val <= 7 and reason_list) else ""
                
                new_row = pd.DataFrame([{
                    "submission_id": str(current_model_row_id),
                    "user": str(st.session_state.username),
                    "rating": int(val),
                    "reason": reason_str
                }])
                update_local_variable(m_id, new_row, current_model_row_id)
            
    manual_fix_val = st.session_state.get(f"fix_{st.session_state.u_index}")
    if manual_fix_val and manual_fix_val.strip() != "":
        if not versions.empty:
            general_sub_id = str(versions.index[0] + 2)
            user_row = pd.DataFrame([{
                "submission_id": str(general_sub_id),
                "user": str(st.session_state.username),
                "user_corrected": str(manual_fix_val)
            }])
            update_local_variable("corrections", user_row, general_sub_id)

def sync_to_cloud():
    save_bar = st.progress(0, text="Syncing to Cloud...")
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
                st.error(f"Failed to write {tab_name}: {e}")
            
            current_tab += 1

    save_bar.progress(1.0, text="Sync Complete!")
    time.sleep(1.0)
    save_bar.empty()

def get_first_unrated_index(unique_list, master_df):
    for idx, sentence in enumerate(unique_list):
        versions = master_df[master_df["incorrect"] == sentence]
        for m_id in MODEL_TAB_NAMES.keys():
            m_row = versions[versions["id"] == m_id]
            if not m_row.empty:
                specific_sub_id = str(m_row.index[0] + 2)
                if get_existing_rating(m_id, specific_sub_id) is None:
                    return idx
    return len(unique_list)

if "u_index" not in st.session_state:
    st.session_state.u_index = get_first_unrated_index(unique_list, master_df)

# =========================================================
# 4. MAIN UI & LOGOUT HEADER
# =========================================================

top_c1, top_c2 = st.columns([8, 2])
with top_c1:
    st.markdown(f"👤 Name: **{st.session_state.username}**")
with top_c2:
    if st.button("Save & Exit", type="primary", use_container_width=True):
        if st.session_state.u_index < len(unique_list):
            current_incorrect = unique_list[st.session_state.u_index]
            versions = master_df[master_df["incorrect"] == current_incorrect]
            save_to_local_memory(current_incorrect, versions)
        
        sync_to_cloud()
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

st.divider()

if st.session_state.u_index >= len(unique_list):
    st.success("🎉 You've reached the end! Please click **Save & Exit** above to upload your evaluations.")
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
    all_rated = True
    for m_id in MODEL_TAB_NAMES.keys():
        m_row = versions[versions["id"] == m_id]
        if not m_row.empty:
            val = st.session_state.get(f"pills_{m_id}_{st.session_state.u_index}")
            if val is None:
                all_rated = False
                break
                
    if all_rated:
        save_to_local_memory(current_incorrect, versions)
        st.session_state.u_index += 1
        st.rerun()
    else:
        st.toast("⚠️ Please rate all displayed models before proceeding to the next sentence.", icon="🚨")

st.info(f"**Original:** {current_incorrect}")
st.divider()

# --- Ratings ---
model_ids = sorted(MODEL_TAB_NAMES.keys())
rating_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

rows = [model_ids[:3], model_ids[3:]]
for row_idx, row_ids in enumerate(rows):
    cols = st.columns(3)
    for i, m_id in enumerate(row_ids):
        with cols[i]:
            m_row = versions[versions["id"] == m_id]
            if not m_row.empty:
                specific_sub_id = str(m_row.index[0] + 2)
                
                st.markdown(f"**{MODEL_TAB_NAMES[m_id].capitalize()}** <span style='color:red'>*</span>", unsafe_allow_html=True)
                st.success(m_row.iloc[0]["corrected"])
                
                key = f"pills_{m_id}_{st.session_state.u_index}"
                if key not in st.session_state:
                    saved_val = get_existing_rating(m_id, specific_sub_id)
                    if saved_val:
                        st.session_state[key] = saved_val
                
                # We save the selected pill to a variable to conditionally show the dropdown
                selected_rating = st.pills("Rate", rating_options, key=key, label_visibility="collapsed")
                
                # Conditional Reason Dropdown for ratings <= 7
                if selected_rating is not None and selected_rating <= 7:
                    reason_key = f"reason_{m_id}_{st.session_state.u_index}"
                    
                    if reason_key not in st.session_state:
                        saved_reason = get_existing_reason(m_id, specific_sub_id)
                        st.session_state[reason_key] = saved_reason
                        
                    st.multiselect(
                        "Why this rating?", 
                        options=REASON_OPTIONS,
                        key=reason_key
                    )
            else:
                st.warning(f"No data for {MODEL_TAB_NAMES[m_id]}")
                
    # Buffer space between the two rows so layout stays clean
    if row_idx < len(rows) - 1:
        st.markdown("<br><br>", unsafe_allow_html=True)

st.divider()

# Pre-fill correction box if data exists
general_sub_id = str(versions.index[0] + 2) if not versions.empty else None
existing_correction = get_existing_correction(general_sub_id) if general_sub_id else ""
st.text_area("Correction (Optional):", value=existing_correction, key=f"fix_{st.session_state.u_index}")
