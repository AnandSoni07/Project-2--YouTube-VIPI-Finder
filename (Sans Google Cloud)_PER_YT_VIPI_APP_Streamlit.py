import streamlit as st
import os
import json
import configparser
import logging
import pandas as pd
import time
from datetime import datetime
import isodate
import altair as alt
import io
import zipfile

try:
    import openpyxl  # noqa
except ImportError:
    st.error("The 'openpyxl' package is not installed. Please install it via 'pip install openpyxl' and restart.")
    st.stop()

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# -------------------------------------------------------------------------
# GLOBALS/CONFIG
# -------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USAGE_FILE = os.path.join(SCRIPT_DIR, "usage.json")
DAILY_LIMIT = 3000
ALLOWED_TARGETS = [10, 50, 100, 200, 400, 600, 800, 1000]
MAX_KEYWORDS = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -------------------------------------------------------------------------
# 1) USAGE PERSISTENCE
# -------------------------------------------------------------------------
def load_daily_usage():
    """Load daily usage from usage.json if it exists and matches today's date."""
    if not os.path.exists(USAGE_FILE):
        return 0, None
    try:
        with open(USAGE_FILE, 'r') as f:
            data = json.load(f)
        usage_date = data.get('usage_date')
        if usage_date == str(datetime.now().date()):
            return data.get('daily_usage', 0), usage_date
        else:
            return 0, None
    except:
        return 0, None

def save_daily_usage(usage):
    """Save the daily usage + today's date to usage.json."""
    data = {
        'daily_usage': usage,
        'usage_date': str(datetime.now().date())
    }
    with open(USAGE_FILE, 'w') as f:
        json.dump(data, f)

# -------------------------------------------------------------------------
# 2) CREDENTIAL LOADING
# -------------------------------------------------------------------------
def load_credentials():
    """Load 'youtube_api_key' from secrets or local config."""
    # Try Streamlit secrets first
    try:
        creds = st.secrets["credentials"]
        api_key = creds["youtube_api_key"]
        logging.info("Loaded YouTube API key from Streamlit Secrets.")
        return api_key
    except KeyError:
        logging.warning("Secrets found but no 'youtube_api_key'. Checking local config.")
    except Exception as e:
        logging.warning(f"No valid secrets found ({e}). Checking local config.")

    # Fallback: parameters.config
    CONFIG_PATH = os.path.join(SCRIPT_DIR, "parameters.config")
    if not os.path.exists(CONFIG_PATH):
        logging.error(f"No parameters.config at {CONFIG_PATH}")
        return None

    config = configparser.ConfigParser()
    read_files = config.read(CONFIG_PATH)
    if not read_files:
        logging.error(f"Could not read parameters.config at {CONFIG_PATH}")
        return None

    try:
        api_key = config.get("credentials", "youtube_api_key")
        if not api_key:
            logging.error("parameters.config found but 'youtube_api_key' is blank.")
            return None
        logging.info("Loaded YouTube API key from local parameters.config.")
        return api_key
    except Exception as e:
        logging.error(f"Error reading 'youtube_api_key': {e}")
        return None

# -------------------------------------------------------------------------
# 3) DONUT CHART
# -------------------------------------------------------------------------
def draw_credit_donut(used, total=3000):
    used = max(0, min(used, total))
    remaining = total - used

    data = pd.DataFrame({
        'Category': ['Used', 'Remaining'],
        'Value': [used, remaining]
    })

    chart = (
        alt.Chart(data)
        .mark_arc(outerRadius=60, innerRadius=40)
        .encode(
            theta='Value:Q',
            color=alt.Color(
                'Category:N',
                scale=alt.Scale(domain=['Used','Remaining'], range=['#E4572E','#76B041']),
                legend=None
            ),
            tooltip=['Category:N', 'Value:Q']
        )
        .properties(width=200, height=200)
    )
    return chart

# -------------------------------------------------------------------------
# 4) YOUTUBE API ROUTINES
# -------------------------------------------------------------------------
def fetch_videos_for_keyword(youtube, keyword, target_count, collected_urls, all_results):
    logging.info(f"Collecting {target_count} videos for keyword: {keyword}")
    next_page_token = None
    results_collected = 0

    while results_collected < target_count:
        search_response = youtube.search().list(
            q=keyword,
            type='video',
            part='id,snippet',
            maxResults=50,
            order='viewCount',
            pageToken=next_page_token
        ).execute()

        for item in search_response['items']:
            video_id = item['id']['videoId']
            video_url = f'https://www.youtube.com/watch?v={video_id}'
            if video_url in collected_urls:
                continue

            video_title = item['snippet']['title']
            channel_id = item['snippet']['channelId']
            channel_title = item['snippet']['channelTitle']
            channel_url = f'https://www.youtube.com/channel/{channel_id}'
            published_at = item['snippet']['publishedAt']

            collected_urls.add(video_url)
            all_results.append({
                'Video Title': video_title,
                'Video URL': video_url,
                'Channel Name': channel_title,
                'Channel URL': channel_url,
                'Published Date': published_at,
                'Keyword': keyword
            })
            results_collected += 1
            if results_collected >= target_count:
                break

        next_page_token = search_response.get('nextPageToken')
        if not next_page_token or results_collected >= target_count:
            break
        time.sleep(0.5)

def fetch_additional_details(youtube, results, progress_bar=None, current_progress=0.0):
    if not results:
        return

    video_ids = [r['Video URL'].split('=')[-1] for r in results]
    channel_ids = list(set(r['Channel URL'].split('/')[-1] for r in results))

    # Video stats
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        resp = youtube.videos().list(
            part='statistics,contentDetails',
            id=','.join(batch)
        ).execute()

        for item in resp.get('items', []):
            vid_id = item['id']
            view_count = item['statistics'].get('viewCount', 'N/A')
            duration = format_duration(item['contentDetails'].get('duration', 'PT0S'))

            for r in results:
                if r['Video URL'].endswith(vid_id):
                    r['View Count'] = view_count
                    r['Duration'] = duration

        if progress_bar:
            current_progress += 0.05
            progress_bar.progress(min(current_progress, 1.0))

    # Channel stats
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i+50]
        resp = youtube.channels().list(
            part='snippet,statistics',
            id=','.join(batch)
        ).execute()

        for citem in resp.get('items', []):
            cid = citem['id']
            subs = citem['statistics'].get('subscriberCount', 'N/A')
            country = citem['snippet'].get('country', 'N/A')

            for r in results:
                if r['Channel URL'].endswith(cid):
                    r['Number of Subscribers'] = subs
                    r['Country of Origin'] = country

        if progress_bar:
            current_progress += 0.05
            progress_bar.progress(min(current_progress, 1.0))

def format_duration(duration_iso):
    parsed = isodate.parse_duration(duration_iso)
    total_seconds = int(parsed.total_seconds())
    h, remainder = divmod(total_seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02}:{m:02}:{s:02}"

def run_youtube_search(api_key, keyword_dict):
    youtube = build('youtube', 'v3', developerKey=api_key, cache_discovery=False)
    all_results = []
    collected_urls = set()
    total_keywords = len(keyword_dict)

    progress_bar = st.progress(0.0)
    step = 0.0

    # Basic collection
    for idx, (kw, tcount) in enumerate(keyword_dict.items(), start=1):
        st.write(f"**Fetching videos for:** {kw}")
        fetch_videos_for_keyword(youtube, kw, tcount, collected_urls, all_results)
        step = float(idx) / float(total_keywords + 2)
        progress_bar.progress(min(step, 1.0))

    # Additional details
    fetch_additional_details(youtube, all_results, progress_bar, step)
    progress_bar.progress(1.0)

    return pd.DataFrame(all_results)

# -------------------------------------------------------------------------
# 5) EXCEL PROCESSING & ZIP HELPERS
# -------------------------------------------------------------------------
def process_data(df: pd.DataFrame):
    required_cols = {"Channel Name", "View Count"}
    if not required_cols.issubset(df.columns):
        st.error("Required columns 'Channel Name' and 'View Count' are missing. Cannot process.")
        return None, None

    channel_counts = df['Channel Name'].value_counts()
    df_all_videos = df.copy()
    df_all_videos['Number of Video Published'] = df_all_videos['Channel Name'].map(channel_counts)

    df_unique_channels = df_all_videos.loc[
        df_all_videos.groupby('Channel Name')['View Count'].idxmax()
    ]
    return df_all_videos, df_unique_channels

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.read()

def create_zip_file(file_dict):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for fname, fbytes in file_dict.items():
            zf.writestr(fname, fbytes)
    zip_buffer.seek(0)
    return zip_buffer.read()

# -------------------------------------------------------------------------
# 6) STREAMLIT APP (UI/UX IMPROVED)
# -------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="PER - YouTube VIPI Discovery Tool", 
        layout="wide"
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-size: 16px;
        }
        h1 {
            font-size: 40px !important;
            font-weight: 700;
        }
        h2, h3, h4 {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        div.stButton > button:first-child {
            background-color: #4CAF50 !important;
            color: white;
            font-size: 1rem;
            padding: 0.6em 1.2em;
            border-radius: 8px;
            margin-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Load daily usage
    used, usage_date = load_daily_usage()
    leftover = DAILY_LIMIT - used

    # Sidebar: Usage & Credits
    with st.sidebar:
        st.title("Usage & Credits")

        donut_chart = draw_credit_donut(used, DAILY_LIMIT)
        st.altair_chart(donut_chart, use_container_width=True)

        st.write(f"**Daily Limit:** {DAILY_LIMIT}")
        st.write(f"**Used:** {used}")
        st.write(f"**Remaining:** {leftover}")
        st.caption("_1 credit = 1 video_")

        # Load/Check API Key
        api_key = load_credentials()
        if not api_key:
            st.warning("No valid YouTube API key found. Please update credentials to proceed.")

    st.title("PER - YouTube VIPI Discovery Tool")
    st.write("Use this app to search for YouTube videos by keywords, fetch top-viewed videos, and export the results to Excel.")
    st.divider()

    # Step 1: Configure Search
    st.header("Step 1: Configure Your Search")

    # -- Let user pick how many keywords (outside a form, so immediate re-run)
    st.subheader("Number of Keywords")
    selected_keyword_count = st.selectbox(
        "Select how many keywords you want to search:",
        list(range(1, MAX_KEYWORDS + 1)),
        index=3
    )

    # Prepare session state for all possible keywords
    if "keyword_entries" not in st.session_state:
        st.session_state["keyword_entries"] = [
            {"keyword": "", "count": 10} for _ in range(MAX_KEYWORDS)
        ]

    # -- Distribute Credits
    with st.expander("Distribute Credits Evenly (Optional)"):
        st.write("Enter a total number of credits below; the app will split them across your selected keywords.")
        total_distribution = st.number_input(
            "Total Credits to Distribute", 
            min_value=0, 
            max_value=DAILY_LIMIT, 
            value=0, 
            step=50
        )
        if st.button("Apply Distribution"):
            if total_distribution == 0:
                st.warning("Distribution total is zero; nothing to apply.")
            else:
                def pick_closest(value, allowed=ALLOWED_TARGETS):
                    return min(allowed, key=lambda x: abs(x - value))

                portion = total_distribution // selected_keyword_count
                for i in range(selected_keyword_count):
                    st.session_state["keyword_entries"][i]["count"] = pick_closest(portion)
                st.success(f"Distributed ~{portion} credits per keyword (rounded to nearest allowed).")

    # -- Enter Keywords & Target Counts
    st.subheader("Enter Keywords & Target Video Counts")
    for i in range(selected_keyword_count):
        col1, col2 = st.columns([2,1], gap="small")
        with col1:
            st.session_state["keyword_entries"][i]["keyword"] = st.text_input(
                f"Keyword {i+1}",
                value=st.session_state["keyword_entries"][i]["keyword"],
                key=f"kw_{i}"
            )
        with col2:
            current_count_val = st.session_state["keyword_entries"][i]["count"]
            if current_count_val not in ALLOWED_TARGETS:
                current_count_val = 10
            st.session_state["keyword_entries"][i]["count"] = st.selectbox(
                "Target Count",
                ALLOWED_TARGETS,
                index=ALLOWED_TARGETS.index(current_count_val),
                key=f"count_{i}"
            )

    # Base Name for output
    st.subheader("Base Name for Output Files")
    base_name = st.text_input(
        "Enter a base name for your processed files:",
        value="MySearch"
    )

    # Check total requested credits
    total_requested = 0
    for i in range(selected_keyword_count):
        kw = st.session_state["keyword_entries"][i]["keyword"].strip()
        if kw:
            total_requested += st.session_state["keyword_entries"][i]["count"]

    if total_requested > leftover:
        st.error(
            f"You've requested {total_requested} credits, but only {leftover} remain. "
            "Please reduce your target counts."
        )

    # Run Search Button
    run_button = st.button("Run YouTube Search & Process to Excel")
    if run_button:
        if not api_key:
            st.error("No valid API key. Please fix your credentials in the sidebar before proceeding.")
            return

        if total_requested > leftover:
            st.error("You are exceeding your daily credit limit. Please reduce target counts.")
            return

        if total_requested <= 0:
            st.error("No valid (non-empty) keywords or zero total requested. Nothing to search.")
            return

        # Build dict {keyword: count} for the selected number
        keyword_dict = {}
        for i in range(selected_keyword_count):
            kw = st.session_state["keyword_entries"][i]["keyword"].strip()
            cnt = st.session_state["keyword_entries"][i]["count"]
            if kw:
                keyword_dict[kw] = cnt

        try:
            # Fetch from YouTube
            df_results = run_youtube_search(api_key, keyword_dict)
            new_used = used + total_requested
            save_daily_usage(new_used)

            st.success(f"Search complete! Found {len(df_results)} unique videos.")
            st.info(f"Used {new_used} of {DAILY_LIMIT} credits today.")

            if not df_results.empty:
                df_all, df_unique = process_data(df_results)
                if df_all is not None and df_unique is not None:
                    st.subheader("Summary of Results")

                    # Show stats
                    total_channels = df_unique['Channel Name'].nunique()
                    st.write(f"**Total Videos Fetched:** {len(df_all)}")
                    st.write(f"**Total Unique Channels:** {total_channels}")

                    # Optional: Quick bar chart of top channels
                    st.write("**Top Channels by View Count (from unique channels)**")
                    df_unique['View Count'] = pd.to_numeric(df_unique['View Count'], errors='coerce').fillna(0)
                    top_channels = df_unique.nlargest(10, 'View Count')
                    chart = (
                        alt.Chart(top_channels)
                        .mark_bar()
                        .encode(
                            x=alt.X('View Count:Q', title='View Count'),
                            y=alt.Y('Channel Name:N', sort='-x', title='Channel Name'),
                            tooltip=['Channel Name', 'View Count']
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(chart, use_container_width=True)

                    st.divider()

                    st.subheader("Download Your Results")
                    all_videos_bytes = df_to_excel_bytes(df_all)
                    unique_channels_bytes = df_to_excel_bytes(df_unique)

                    # Construct final filenames
                    all_videos_filename = f"{base_name} All Videos.xlsx"
                    unique_filename = f"{base_name} Unique Channels.xlsx"

                    file_dict = {
                        all_videos_filename: all_videos_bytes,
                        unique_filename: unique_channels_bytes
                    }
                    zipped_files = create_zip_file(file_dict)

                    st.download_button(
                        label="Download Processed Files (ZIP)",
                        data=zipped_files,
                        file_name=f"{base_name}_processed.zip",
                        mime="application/zip"
                    )
                else:
                    st.warning("Data processing returned None. Check your data or columns.")
            else:
                st.warning("No results returned for these keywords.")

        except HttpError as e:
            st.error(f"YouTube API error: {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

if __name__ == "__main__":
    main()
