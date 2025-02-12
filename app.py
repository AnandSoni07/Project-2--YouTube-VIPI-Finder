import streamlit as st
import os
import json
import configparser
import logging
import pandas as pd
import time
from datetime import datetime, timedelta
import isodate
import altair as alt
import io
import zipfile

try:
    import openpyxl  # noqa
except ImportError:
    st.error("The 'openpyxl' package is not installed. Please install it via 'pip install openpyxl' and restart.")
    st.stop()

# Google API libraries
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.cloud import storage
from google.oauth2 import service_account

import pytz  # For handling timezones

##############################################################################
# GLOBALS/CONFIG
##############################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

USAGE_FILE_BLOB_NAME = "usage.json"
DAILY_LIMIT = 3000

# We removed '10', so the minimum is 50
ALLOWED_TARGETS = [50, 100, 200, 400, 600, 800, 1000]
MAX_KEYWORDS = 10

BUCKET_NAME = "youtube-vipi-tool-usage-bucket"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Define the Pacific timezone
PACIFIC_TZ = pytz.timezone("US/Pacific")

##############################################################################
# 1) LOAD / SAVE USAGE FROM GCS
##############################################################################
def get_gcs_credentials():
    """Load GCS credentials from st.secrets or local config."""
    try:
        gcp_sa = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(dict(gcp_sa))
        logging.info("Loaded GCS credentials from Streamlit Secrets (gcp_service_account).")
        return credentials
    except Exception:
        pass

    config_path = os.path.join(SCRIPT_DIR, "parameters.config")
    if not os.path.exists(config_path):
        st.error(f"parameters.config not found at {config_path}; cannot load GCS credentials.")
        return None

    config = configparser.ConfigParser()
    read_files = config.read(config_path)
    if not read_files:
        st.error(f"Could not read parameters.config at {config_path}; cannot load GCS credentials.")
        return None

    if not config.has_section("gcp_service_account"):
        st.error("No [gcp_service_account] section found in parameters.config.")
        return None

    try:
        gcp_sa_info = {
            "type": config.get("gcp_service_account", "type"),
            "project_id": config.get("gcp_service_account", "project_id"),
            "private_key_id": config.get("gcp_service_account", "private_key_id"),
            "private_key": config.get("gcp_service_account", "private_key").replace("\\n", "\n"),
            "client_email": config.get("gcp_service_account", "client_email"),
            "client_id": config.get("gcp_service_account", "client_id"),
            "auth_uri": config.get("gcp_service_account", "auth_uri"),
            "token_uri": config.get("gcp_service_account", "token_uri"),
            "auth_provider_x509_cert_url": config.get("gcp_service_account", "auth_provider_x509_cert_url"),
            "client_x509_cert_url": config.get("gcp_service_account", "client_x509_cert_url"),
        }
        credentials = service_account.Credentials.from_service_account_info(gcp_sa_info)
        logging.info("Loaded GCS credentials from local parameters.config [gcp_service_account].")
        return credentials
    except Exception as e:
        st.error(f"Error parsing GCP service account from parameters.config: {e}")
        return None

def get_storage_client():
    creds = get_gcs_credentials()
    if not creds:
        return None
    return storage.Client(credentials=creds, project=creds.project_id)

def load_daily_usage():
    """Load usage from GCS if usage.json is for today's (Pacific) date, else reset to 0."""
    try:
        client = get_storage_client()
        if not client:
            return 0, None
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(USAGE_FILE_BLOB_NAME)

        if not blob.exists():
            logging.warning("usage.json does not exist in GCS. Starting usage at 0.")
            return 0, None

        data_str = blob.download_as_text()
        data = json.loads(data_str)
        usage_date = data.get("usage_date")

        # Compare with today's date in Pacific
        today_pacific = datetime.now(PACIFIC_TZ).date()
        if usage_date == str(today_pacific):
            return data.get("daily_usage", 0), usage_date
        else:
            return 0, None
    except Exception as e:
        st.error(f"Could not load usage.json from GCS: {e}\nWill start usage at 0 for now.")
        logging.error(f"GCS error in load_daily_usage: {e}")
        return 0, None

def save_daily_usage(usage):
    """Save daily usage to GCS as usage.json, keyed by today's (Pacific) date."""
    today_pacific = datetime.now(PACIFIC_TZ).date()
    data = {
        "daily_usage": usage,
        "usage_date": str(today_pacific)
    }
    try:
        client = get_storage_client()
        if not client:
            return
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(USAGE_FILE_BLOB_NAME)
        blob.upload_from_string(json.dumps(data), content_type="application/json")
        logging.info("Saved usage.json to GCS successfully.")
    except Exception as e:
        st.error(f"Could not save usage.json to GCS: {e}")
        logging.error(f"GCS error in save_daily_usage: {e}")

##############################################################################
# 2) CREDENTIAL LOADING (YouTube API Key)
##############################################################################
def load_credentials():
    """Load 'youtube_api_key' from st.secrets or local config."""
    try:
        api_key = st.secrets["credentials"]["youtube_api_key"]
        logging.info("Loaded YouTube API key from Streamlit Secrets.")
        return api_key
    except Exception:
        pass

    config_path = os.path.join(SCRIPT_DIR, "parameters.config")
    if not os.path.exists(config_path):
        st.error(f"No parameters.config found at {config_path}")
        return None

    config = configparser.ConfigParser()
    read_files = config.read(config_path)
    if not read_files:
        st.error(f"Could not read parameters.config at {config_path}")
        return None

    try:
        api_key = config.get("credentials", "youtube_api_key")
        if not api_key:
            st.error("parameters.config found but 'youtube_api_key' is blank.")
            return None
        logging.info("Loaded YouTube API key from local parameters.config.")
        return api_key
    except Exception as e:
        st.error(f"Error reading 'youtube_api_key' from parameters.config: {e}")
        return None

##############################################################################
# 3) DONUT CHART
##############################################################################
def draw_credit_donut(used, total=3000):
    used = max(0, min(used, total))
    remaining = total - used
    df = pd.DataFrame({
        'Category': ['Used', 'Remaining'],
        'Value': [used, remaining]
    })

    chart = (
        alt.Chart(df)
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

##############################################################################
# 4) YOUTUBE DATA FETCHING / FILTERING
##############################################################################
def format_duration(duration_iso):
    if not duration_iso:
        return "00:00:00"
    parsed = isodate.parse_duration(duration_iso)
    total_seconds = int(parsed.total_seconds())
    h, remainder = divmod(total_seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02}:{m:02}:{s:02}"

def passes_subscriber_filter(sub_count, below20k, below50k, above50k):
    """
    Return True if sub_count meets ANY of the checked booleans.
    If none are True => "No filter."
    """
    if not any([below20k, below50k, above50k]):
        return True
    conditions = []
    if below20k:
        conditions.append(sub_count < 20000)
    if below50k:
        conditions.append(sub_count < 50000)
    if above50k:
        conditions.append(sub_count >= 50000)
    return any(conditions)

from googleapiclient.errors import HttpError

def fetch_videos_for_keyword_until_filtered(
    youtube, keyword, target_count, published_after,
    below20k, below50k, above50k,
    daily_usage
):
    """
    Loop through search pages, up to 50 per page, gather raw,
    fetch channel/video stats, apply sub filter => filtered
    Stop if enough filtered or no more pages
    """
    raw_results = []
    filtered_results = []
    collected_urls_raw = set()
    collected_urls_filtered = set()
    next_page_token = None

    while len(filtered_results) < target_count:
        search_req = youtube.search().list(
            q=keyword,
            type='video',
            part='id,snippet',
            maxResults=50,
            order='viewCount',
            pageToken=next_page_token,
            publishedAfter=published_after
        )
        try:
            resp = search_req.execute()
        except HttpError as e:
            st.error(f"YouTube API error in search request: {e}")
            break

        items = resp.get('items', [])
        daily_usage += len(items)

        if not items:
            break

        page_raw = []
        video_ids_for_stats = []
        channel_ids_for_stats = []

        for item in items:
            vid_id = item['id'].get('videoId')
            snippet = item['snippet']
            if not vid_id or not snippet:
                continue

            video_url = f"https://www.youtube.com/watch?v={vid_id}"
            if video_url in collected_urls_raw:
                continue

            channel_id = snippet['channelId']
            channel_title = snippet['channelTitle']
            row = {
                "Video Title": snippet['title'],
                "Video URL": video_url,
                "Channel Name": channel_title,
                "Channel ID": channel_id,
                "Published Date": snippet['publishedAt'],
                "Keyword": keyword
            }
            page_raw.append(row)
            video_ids_for_stats.append(vid_id)
            channel_ids_for_stats.append(channel_id)

        if not page_raw:
            next_page_token = resp.get('nextPageToken')
            if not next_page_token:
                break
            continue

        for r in page_raw:
            raw_results.append(r)
            collected_urls_raw.add(r["Video URL"])

        # fetch channel stats
        channel_stats_map = {}
        unique_channels = list(set(channel_ids_for_stats))
        for i in range(0, len(unique_channels), 50):
            batch = unique_channels[i:i+50]
            try:
                c_resp = youtube.channels().list(
                    part="snippet,statistics",
                    id=",".join(batch)
                ).execute()
            except HttpError as e:
                st.error(f"YouTube API error (channels) for {batch}: {e}")
                continue
            for citem in c_resp.get("items", []):
                cid = citem["id"]
                subs_value = citem["statistics"].get("subscriberCount", None)
                if subs_value is None:
                    subs = "HIDDEN"
                else:
                    subs = subs_value
                country = citem["snippet"].get("country", "N/A")
                channel_stats_map[cid] = (subs, country)

        # fetch video stats
        video_stats_map = {}
        for i in range(0, len(video_ids_for_stats), 50):
            batch = video_ids_for_stats[i:i+50]
            try:
                v_resp = youtube.videos().list(
                    part="statistics,contentDetails",
                    id=",".join(batch)
                ).execute()
            except HttpError as e:
                st.error(f"YouTube API error (videos) for {batch}: {e}")
                continue
            for vitem in v_resp.get("items", []):
                vid = vitem["id"]
                stats = vitem.get("statistics", {})
                content_details = vitem.get("contentDetails", {})
                view_count = stats.get("viewCount", "0")
                duration = format_duration(content_details.get("duration"))
                video_stats_map[vid] = (view_count, duration)

        # apply sub filter
        for r in page_raw:
            cid = r["Channel ID"]
            (subs_raw, country) = channel_stats_map.get(cid, ("HIDDEN", "N/A"))

            # interpret HIDDEN as 0
            if isinstance(subs_raw, str) and subs_raw == "HIDDEN":
                sub_count_num = 0
            else:
                sub_count_num = int(subs_raw)

            r["Number of Subscribers"] = subs_raw
            r["Country of Origin"] = country

            vid_id = r["Video URL"].split("=")[-1]
            vstat = video_stats_map.get(vid_id, ("0", "00:00:00"))
            r["View Count"] = vstat[0]
            r["Duration"] = vstat[1]

            if passes_subscriber_filter(sub_count_num, below20k, below50k, above50k):
                if r["Video URL"] not in collected_urls_filtered:
                    filtered_results.append(r)
                    collected_urls_filtered.add(r["Video URL"])
                    if len(filtered_results) >= target_count:
                        break

        next_page_token = resp.get('nextPageToken')
        if not next_page_token:
            break
        time.sleep(0.3)

    return raw_results, filtered_results, daily_usage

##############################################################################
# 5) EXCEL PROCESSING & ZIP HELPERS
##############################################################################
def process_data(df: pd.DataFrame):
    """Create 'All Videos' + 'Unique Channels' from the filtered set."""
    required_cols = {"Channel Name", "View Count"}
    if not required_cols.issubset(df.columns):
        st.error("Required columns 'Channel Name' and 'View Count' are missing. Cannot process.")
        return None, None

    df["View Count"] = pd.to_numeric(df["View Count"], errors="coerce").fillna(0)
    channel_counts = df["Channel Name"].value_counts()

    df_all_videos = df.copy()
    df_all_videos["Number of Video Published"] = df_all_videos["Channel Name"].map(channel_counts)

    # For each channel, pick the row with the highest view count
    df_unique_channels = df_all_videos.loc[
        df_all_videos.groupby("Channel Name")["View Count"].idxmax()
    ]
    return df_all_videos, df_unique_channels

def df_to_excel_bytes_with_info(main_df: pd.DataFrame, info_df: pd.DataFrame,
                                data_sheet_name="Data", info_sheet_name="Filters") -> bytes:
    """
    Write main_df to one sheet, info_df to a second sheet in the same Excel file.
    Return the resulting bytes.
    """
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        main_df.to_excel(writer, index=False, sheet_name=data_sheet_name)
        info_df.to_excel(writer, index=False, sheet_name=info_sheet_name)
    buffer.seek(0)
    return buffer.read()

def create_zip_file(file_dict):
    """
    file_dict: {filename: file_bytes}
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for fname, fbytes in file_dict.items():
            zf.writestr(fname, fbytes)
    zip_buffer.seek(0)
    return zip_buffer.read()

##############################################################################
# 6) STREAMLIT APP
##############################################################################
def main():
    st.set_page_config(page_title="PER - YouTube VIPI Discovery Tool", layout="wide")

    st.write("**Initializing...**")

    used, usage_date = load_daily_usage()
    leftover = DAILY_LIMIT - used
    if leftover < 0:
        leftover = 0

    # Sidebar
    with st.sidebar:
        st.title("Usage & Credits")
        donut_chart = draw_credit_donut(used, DAILY_LIMIT)
        st.altair_chart(donut_chart, use_container_width=True)

        st.write(f"**Daily Limit:** {DAILY_LIMIT}")
        st.write(f"**Used:** {used}")
        st.write(f"**Remaining:** {leftover}")

        # Show time until 12:00 PM Pacific
        now_pacific = datetime.now(PACIFIC_TZ)
        next_noon = now_pacific.replace(hour=12, minute=0, second=0, microsecond=0)
        if now_pacific >= next_noon:
            next_noon += timedelta(days=1)
        time_to_reset = next_noon - now_pacific
        hours = time_to_reset.seconds // 3600
        minutes = (time_to_reset.seconds // 60) % 60
        st.write(f"Credits reset in {hours}h {minutes}m (12:00 PM Pacific Time).")

        api_key = load_credentials()
        if not api_key:
            st.warning("No valid YouTube API key found. Please update credentials to proceed.")

    # Title
    st.title("PER - YouTube VIPI Discovery Tool")
    st.write("Use this app to search for YouTube videos by keywords, fetch top-viewed videos, and export the results to Excel.")
    st.divider()

    # Renamed: “Configure Your Search”
    st.header("Configure Your Search")

    # 1) Number of Keywords
    st.subheader("Number of Keywords")
    selected_keyword_count = st.selectbox(
        "Select how many keywords you want to search:",
        list(range(1, MAX_KEYWORDS + 1)),
        index=3
    )

    if "keyword_entries" not in st.session_state:
        st.session_state["keyword_entries"] = [
            {"keyword": "", "count": 50} for _ in range(MAX_KEYWORDS)
        ]

    # Optional distribution
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

    # 2) Enter Keywords & Target Video Counts
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
                current_count_val = 50
            st.session_state["keyword_entries"][i]["count"] = st.selectbox(
                "Target Count",
                ALLOWED_TARGETS,
                index=ALLOWED_TARGETS.index(current_count_val),
                key=f"count_{i}"
            )

    # 3) Upload Date Filter (radio)
    st.subheader("Upload Date Filter")
    date_filter_choice = st.radio(
        "Select an upload date filter:",
        ["No filter", "Last 3 Months", "Last 6 Months", "Last 1 Year"],
        index=0
    )
    published_after_value = None
    if date_filter_choice != "No filter":
        now_utc = datetime.utcnow()
        if date_filter_choice == "Last 3 Months":
            days_ago = 90
        elif date_filter_choice == "Last 6 Months":
            days_ago = 180
        else:  # "Last 1 Year"
            days_ago = 365
        cutoff = now_utc - timedelta(days=days_ago)
        published_after_value = cutoff.isoformat("T") + "Z"

    # 4) Subscriber Count Filter (radio)
    st.subheader("Subscriber Count Filter")
    sub_filter_choice = st.radio(
        "Pick one subscriber range:",
        ["No filter", "Up to 20K", "Up to 50K", "Above 50K"],
        index=0
    )

    below20k = False
    below50k = False
    above50k = False
    if sub_filter_choice == "Up to 20K":
        below20k = True
    elif sub_filter_choice == "Up to 50K":
        below50k = True
    elif sub_filter_choice == "Above 50K":
        above50k = True

    # 5) Give Your Output File a Name
    st.subheader("Give Your Output File a Name")
    base_name = st.text_input("Type your output file base name here:", value="MySearch")

    # Now the main button
    run_button = st.button("Run YouTube Search & Process Excel File")
    if run_button:
        if not api_key:
            st.error("No valid API key. Please fix your credentials in the sidebar.")
            return

        # Compute total_requested
        total_requested = 0
        for i in range(selected_keyword_count):
            kw = st.session_state["keyword_entries"][i]["keyword"].strip()
            if kw:
                total_requested += st.session_state["keyword_entries"][i]["count"]

        if total_requested > leftover:
            st.error(f"You've requested {total_requested} credits, but only {leftover} remain.")
            return

        if total_requested <= 0:
            st.error("No valid (non-empty) keywords or zero total requested. Nothing to search.")
            return

        youtube = build('youtube', 'v3', developerKey=api_key, cache_discovery=False)
        new_used = used

        global_raw_list = []
        global_filtered_list = []

        progress_bar = st.progress(0.0)
        valid_keywords = [e for e in st.session_state["keyword_entries"] if e["keyword"].strip()]
        total_kw = len(valid_keywords)
        done_kw = 0

        try:
            for entry in valid_keywords:
                kw = entry["keyword"].strip()
                tcount = entry["count"]

                st.write(f"**Fetching up to {tcount} filtered videos for:** {kw} ...")
                raw_res, filtered_res, new_used = fetch_videos_for_keyword_until_filtered(
                    youtube=youtube,
                    keyword=kw,
                    target_count=tcount,
                    published_after=published_after_value,
                    below20k=below20k,
                    below50k=below50k,
                    above50k=above50k,
                    daily_usage=new_used
                )

                global_raw_list.extend(raw_res)
                global_filtered_list.extend(filtered_res)

                done_kw += 1
                progress_bar.progress(done_kw / total_kw)

            save_daily_usage(new_used)
            st.success(f"Search complete! Found {len(global_filtered_list)} filtered videos total.")
            st.info(f"Used {new_used} of {DAILY_LIMIT} credits today.")

            # Convert to DataFrame
            df_raw = pd.DataFrame(global_raw_list)
            df_filtered = pd.DataFrame(global_filtered_list)

            if not df_raw.empty:
                df_raw.drop_duplicates(subset=["Video URL"], inplace=True)
            if not df_filtered.empty:
                df_filtered.drop_duplicates(subset=["Video URL"], inplace=True)

            # Process to get All Videos Filtered + Unique Channels
            if df_filtered.empty:
                st.warning("No results passed the sub-count filter. The raw file may still have items though.")
                df_all_filtered = df_filtered
                df_unique_filtered = pd.DataFrame()
            else:
                df_all_filtered, df_unique_filtered = process_data(df_filtered)

            # Build the filter info in a DataFrame
            info_rows = []
            info_rows.append({"Parameter": "Upload Date Filter", "Value": date_filter_choice})
            info_rows.append({"Parameter": "Subscriber Filter", "Value": sub_filter_choice})

            # List keywords & counts
            for idx, ent in enumerate(valid_keywords, start=1):
                info_rows.append({
                    "Parameter": f"Keyword {idx}",
                    "Value": f"{ent['keyword']} (count={ent['count']})"
                })

            info_df = pd.DataFrame(info_rows)

            # 1) Filtered Data
            filtered_bytes = b""
            if not df_all_filtered.empty:
                filtered_bytes = df_to_excel_bytes_with_info(
                    df_all_filtered, info_df,
                    data_sheet_name="All Videos Filtered",
                    info_sheet_name="Filters"
                )

            # 2) Unique Data
            unique_bytes = b""
            if not df_unique_filtered.empty:
                unique_bytes = df_to_excel_bytes_with_info(
                    df_unique_filtered, info_df,
                    data_sheet_name="Unique Channels Filtered",
                    info_sheet_name="Filters"
                )

            # 3) Raw Data (EXCLUDING whatever is in the filtered set)
            raw_bytes = b""
            if not df_raw.empty:
                if not df_all_filtered.empty:
                    # Exclude any video that's already in the "All Videos Filtered"
                    df_raw_excluding = df_raw[~df_raw["Video URL"].isin(df_all_filtered["Video URL"])]
                else:
                    # If there's nothing in df_all_filtered, just keep df_raw as is
                    df_raw_excluding = df_raw

                if not df_raw_excluding.empty:
                    raw_bytes = df_to_excel_bytes_with_info(
                        df_raw_excluding, info_df,
                        data_sheet_name="Raw Data",
                        info_sheet_name="Filters"
                    )

            # Build final ZIP
            file_dict = {}
            if filtered_bytes:
                file_dict[f"{base_name} All Videos Filtered.xlsx"] = filtered_bytes
            if unique_bytes:
                file_dict[f"{base_name} Unique Channels Filtered.xlsx"] = unique_bytes
            if raw_bytes:
                file_dict[f"{base_name} Raw Videos Unfiltered (Excluding Filtered).xlsx"] = raw_bytes

            zipped_files = create_zip_file(file_dict)

            st.subheader("Download Your Results")
            st.download_button(
                label="Download Processed Files (ZIP)",
                data=zipped_files,
                file_name=f"{base_name}_processed.zip",
                mime="application/zip"
            )

        except HttpError as e:
            st.error(f"YouTube API error: {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")


if __name__ == "__main__":
    main()
