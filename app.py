#!/usr/bin/env python3
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

# -------------- Google Cloud / Usage-limiting Libraries --------------
from google.cloud import storage
from google.oauth2 import service_account

# -------------- YouTube API --------------
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# -------------- Zendesk Sell (BaseCRM) --------------
from basecrm import Client as BaseClient
import basecrm.errors

# -------------- Timezone --------------
import pytz

# -------------- Country Mapping --------------
import pycountry  # pip install pycountry

# ---------------------------------------------------------------------
# GLOBAL CONFIG
# ---------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

DAILY_LIMIT = 3000
MAX_KEYWORDS = 3
ALLOWED_TARGETS = [50, 100, 200, 400, 600, 800, 1000]

BUCKET_NAME = "youtube-vipi-tool-usage-bucket"
USAGE_FILE_BLOB_NAME = "usage.json"
PACIFIC_TZ = pytz.timezone("US/Pacific")

# ---------------------------------------------------------------------
# Owners Map
# ---------------------------------------------------------------------
owners_map = {
    "1": (6445741, "Nicolas Beaumont"),
    "2": (6445920, "Masaya SATO"),
    "3": (6551966, "Stefan Bakir"),
    "4": (6558051, "Julien Cinquin"),
    "5": (6558054, "Michael Wayne Plant"),
    "6": (6585866, "Yangkun SHI"),
    "7": (6818690, "Anand SONI"),
    "8": (8441298, "Marie-Dominique BONARDI"),
    "9": (8593282, "Killian DARE"),
    "10": (8722384, "Marina Andrianoelison"),
    "11": (8809361, "Maryjo FERNANDES"),
}

# ---------------------------------------------------------------------
# Helper: safe_api_call
# ---------------------------------------------------------------------
def safe_api_call(func, *args, **kwargs):
    for attempt in range(5):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"[CRM] API call error: {e}. Retrying ({attempt+1}/5)...")
            time.sleep(0.2)
    logging.error("Failed after 5 retries.")
    return None

# ---------------------------------------------------------------------
# 1) GCS USAGE-LIMITING
# ---------------------------------------------------------------------
def get_gcs_credentials():
    try:
        gcp_sa = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(dict(gcp_sa))
        logging.info("Loaded GCS credentials from Streamlit Secrets.")
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
        logging.info("Loaded GCS credentials from local parameters.config.")
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
        today_pacific = datetime.now(PACIFIC_TZ).date()
        if usage_date == str(today_pacific):
            return data.get("daily_usage", 0), usage_date
        else:
            return 0, None
    except Exception as e:
        st.error(f"Could not load usage.json from GCS: {e}")
        logging.error(f"GCS error in load_daily_usage: {e}")
        return 0, None

def save_daily_usage(usage):
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

# ---------------------------------------------------------------------
# 2) Load YouTube API Key
# ---------------------------------------------------------------------
def load_youtube_api_key():
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

# ---------------------------------------------------------------------
# 3) Credit Donut
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# 4) YouTube Data Fetching (unchanged except for sort order)
# ---------------------------------------------------------------------
def format_duration(duration_iso):
    if not duration_iso:
        return "00:00:00"
    parsed = isodate.parse_duration(duration_iso)
    total_seconds = int(parsed.total_seconds())
    h, remainder = divmod(total_seconds, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02}:{m:02}:{s:02}"

def passes_subscriber_filter(sub_count, below20k, below50k, above50k):
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

def map_country_code_to_fullname(country_code):
    if country_code == "N/A":
        return "N/A"
    try:
        country = pycountry.countries.get(alpha_2=country_code)
        if country:
            return country.name
    except Exception:
        pass
    return country_code

def is_youtube_short(video_url, duration_str):
    """
    Determines if a video is likely a YouTube Short
    Args:
        video_url: The full YouTube video URL
        duration_str: Duration in format "HH:MM:SS"
    
    Returns:
        Boolean indicating if it's likely a Short
    """
    # Check URL pattern (if it contains /shorts/)
    is_shorts_url = "/shorts/" in video_url
    
    # Check for common shorts hashtags in the URL
    has_shorts_hashtag = "#shorts" in video_url.lower() or "#short" in video_url.lower()
    
    # Check duration (Shorts are typically ≤ 60 seconds)
    try:
        h, m, s = map(int, duration_str.split(":"))
        duration_seconds = h * 3600 + m * 60 + s
        is_shorts_duration = duration_seconds <= 60
    except:
        is_shorts_duration = False
    
    # If URL is explicitly a Shorts URL, has shorts hashtag, or duration is under 60s
    return is_shorts_url or has_shorts_hashtag or is_shorts_duration

def get_standard_country_name(input_country):
    if not input_country or input_country.upper() == "N/A":
        return "N/A"
    try:
        if len(input_country) == 2:
            country = pycountry.countries.get(alpha_2=input_country.upper())
            if country:
                return country.name
        elif len(input_country) == 3:
            country = pycountry.countries.get(alpha_3=input_country.upper())
            if country:
                return country.name
    except Exception:
        pass
    custom_mapping = {
        "USA": "United States",
        "US": "United States",
        "UK": "United Kingdom",
    }
    return custom_mapping.get(input_country.upper(), input_country)

def build_channel_url(citem):
    snippet = citem.get("snippet", {})
    brand   = citem.get("brandingSettings", {}).get("channel", {})
    possible_custom = snippet.get("customUrl", "").strip()
    possible_vanity = brand.get("vanityUrl", "").strip()
    if possible_custom.startswith("@"):
        return f"https://www.youtube.com/{possible_custom}"
    elif possible_vanity.startswith("@"):
        return f"https://www.youtube.com/{possible_vanity}"
    else:
        cid = citem["id"]
        return f"https://www.youtube.com/channel/{cid}"

def fetch_videos_for_keyword_until_filtered(
    youtube, keyword, target_count, published_after,
    below20k, below50k, above50k,
    daily_usage, order_filter, shorts_filter="Include all videos"
):
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
            order=order_filter,
            pageToken=next_page_token,
            publishedAfter=published_after
        )
        
        # Log the search parameters to verify order_filter is correctly passed
        logging.info(f"YouTube API search with: keyword='{keyword}', order='{order_filter}', filter={shorts_filter}")
        
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
                "Channel ID": channel_id,
                "Channel Name": channel_title,
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

        from googleapiclient.errors import HttpError
        channel_stats_map = {}
        unique_channels = list(set(channel_ids_for_stats))
        for i in range(0, len(unique_channels), 50):
            batch = unique_channels[i:i+50]
            try:
                c_resp = youtube.channels().list(
                    part="snippet,statistics,brandingSettings",
                    id=",".join(batch)
                ).execute()
            except HttpError as e:
                st.error(f"YouTube API error (channels) for {batch}: {e}")
                continue
            for citem in c_resp.get("items", []):
                cid = citem["id"]
                subs_value = citem["statistics"].get("subscriberCount", None)
                subs = subs_value if subs_value else "HIDDEN"
                country_code = citem["snippet"].get("country", "N/A")
                final_url = build_channel_url(citem)
                channel_stats_map[cid] = {
                    "subs": subs,
                    "country_code": country_code,
                    "final_url": final_url
                }

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

        for r in page_raw:
            cid = r["Channel ID"]
            c_info = channel_stats_map.get(cid, None)
            if not c_info:
                sub_count_num = 0
                final_url = f"https://www.youtube.com/channel/{cid}"
                cc = "N/A"
            else:
                sub_count_str = c_info["subs"]
                cc = c_info["country_code"]
                final_url = c_info["final_url"]
                sub_count_num = 0 if sub_count_str == "HIDDEN" else int(sub_count_str)
            r["Country of Origin"] = map_country_code_to_fullname(cc)
            r["Channel URL"] = final_url
            vid_id = r["Video URL"].split("=")[-1]
            vstat = video_stats_map.get(vid_id, ("0", "00:00:00"))
            r["View Count"] = vstat[0]
            r["Duration"] = vstat[1]
            r["Number of Subscribers"] = sub_count_num
            
            # Detect if this is a YouTube Short
            r["Is Short"] = is_youtube_short(r["Video URL"], vstat[1])
            
            # Apply Shorts filter if needed
            if (shorts_filter == "Exclude Shorts" and r["Is Short"]) or \
               (shorts_filter == "Only show Shorts" and not r["Is Short"]):
                continue  # Skip this video based on Shorts filter
                
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

# ---------------------------------------------------------------------
# Aggregation and DataFrame Building Helpers
# ---------------------------------------------------------------------
def aggregate_video_info(video_df: pd.DataFrame) -> dict:
    video_info = {}
    if video_df.empty:
        return video_info
    video_df["View Count Numeric"] = pd.to_numeric(video_df["View Count"], errors="coerce").fillna(0)
    grouped = video_df.groupby("Channel ID")
    for channel_id, group in grouped:
        sorted_group = group.sort_values(by="View Count Numeric", ascending=False)
        lines = []
        count = 1
        for _, row in sorted_group.head(3).iterrows():
            lines.append(f"{count}. {row['Video Title']} ({row['Video URL']})")
            count += 1
        video_info[channel_id] = "\n".join(lines)
    return video_info

def build_unique_channels_df(video_df: pd.DataFrame, is_filtered: bool, video_info: dict = None) -> pd.DataFrame:
    if video_df.empty:
        return pd.DataFrame()
    video_df["Number of Subscribers"] = pd.to_numeric(video_df["Number of Subscribers"], errors="coerce").fillna(0)
    grouping = video_df.groupby("Channel ID", as_index=False).agg({
        "Channel Name": "first",
        "Country of Origin": "first",
        "Number of Subscribers": "max",
        "Channel URL": "first"
    })
    grouping["IsFiltered"] = is_filtered
    grouping["Video Info"] = grouping["Channel ID"].map(video_info) if video_info else ""
    return grouping

def analyze_search_results_by_order(df_videos, sort_order):
    """
    Analyzes video results to validate if the sort order appears to be applied correctly.
    Returns stats and a boolean indicating if the sort order appears valid.
    """
    if df_videos.empty:
        return {}, False
    
    # Add numeric view count field
    df_videos["View Count Numeric"] = pd.to_numeric(df_videos["View Count"], errors="coerce").fillna(0)
    
    # Get basic stats
    total_videos = len(df_videos)
    total_views = df_videos["View Count Numeric"].sum()
    avg_views = df_videos["View Count Numeric"].mean()
    median_views = df_videos["View Count Numeric"].median()
    
    # Check if sort order appears valid
    if sort_order == "Most Watched":
        # For viewCount, we expect the first videos to have more views than later ones
        # Check if the first quarter has significantly more views than the last quarter
        if total_videos >= 4:
            quarter_size = total_videos // 4
            first_quarter = df_videos.iloc[:quarter_size]["View Count Numeric"].mean()
            last_quarter = df_videos.iloc[-quarter_size:]["View Count Numeric"].mean()
            order_seems_valid = first_quarter > (last_quarter * 1.5)  # 50% more views in first quarter
        else:
            # Not enough videos for comparison
            order_seems_valid = True
    else:
        # For relevance, we can't easily verify through statistics
        # We'll check if there's keyword match in titles - this is imperfect but helpful
        keywords = df_videos["Keyword"].unique()
        keyword_match_count = 0
        for idx, row in df_videos.iterrows():
            title = row["Video Title"].lower()
            keyword = row["Keyword"].lower()
            if keyword in title:
                keyword_match_count += 1
                
        keyword_match_percent = (keyword_match_count / total_videos) * 100 if total_videos > 0 else 0
        order_seems_valid = keyword_match_percent >= 40  # At least 40% should have keyword in title
    
    return {
        "total_videos": total_videos,
        "total_views": total_views,
        "avg_views": avg_views,
        "median_views": median_views,
        "order_seems_valid": order_seems_valid
    }, order_seems_valid

# ---------------------------------------------------------------------
# Excel/ZIP Helpers
# ---------------------------------------------------------------------
def df_to_excel_bytes_with_info(main_df: pd.DataFrame, info_df: pd.DataFrame,
                                data_sheet_name="Data", info_sheet_name="Filters") -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        main_df.to_excel(writer, index=False, sheet_name=data_sheet_name)
        info_df.to_excel(writer, index=False, sheet_name=info_sheet_name)
    buffer.seek(0)
    return buffer.read()

def create_zip_file(file_dict):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for fname, fbytes in file_dict.items():
            zf.writestr(fname, fbytes)
    zip_buffer.seek(0)
    return zip_buffer.read()

# ---------------------------------------------------------------------
# Zendesk Sell / BaseCRM Integration
# ---------------------------------------------------------------------
def load_zendesk_sell_token():
    try:
        return st.secrets["zendesk_sell"]["api_token"]
    except Exception:
        pass
    config_path = os.path.join(SCRIPT_DIR, "parameters.config")
    if not os.path.exists(config_path):
        st.warning(f"No parameters.config found at {config_path}, cannot load ZSell token.")
        return None
    config = configparser.ConfigParser()
    read_files = config.read(config_path)
    if not read_files or not config.has_section("zendesk_sell"):
        st.warning("Could not load Zendesk Sell token from parameters.config.")
        return None
    try:
        return config.get("zendesk_sell", "api_token")
    except Exception as e:
        st.warning(f"Error reading 'api_token' from parameters.config [zendesk_sell]: {e}")
        return None

def parse_channel_id_from_crm(link: str) -> str:
    link = link.strip().lower()
    if "/@" in link:
        return link
    prefix = "https://www.youtube.com/channel/"
    if link.startswith(prefix):
        return link[len(prefix):]
    return link

# ---------------------------------------------------------------------
# Real Lead Creation Functions (from original script)
# ---------------------------------------------------------------------
def create_lead_and_note(z_client, lead_payload, channel_row):
    new_lead = z_client.leads.create(lead_payload)
    
    # Start with a separator
    lines = ["---\n"]
    
    # Channel Info section
    lines.append("**Channel Info:**\n")
    lines.append(f"Channel Name: {channel_row.get('Channel Name','Unknown')}")
    lines.append(f"Subscribers: {channel_row.get('Number of Subscribers',0)}")
    lines.append(f"Country: {channel_row.get('Country of Origin','N/A')}")
    lines.append(f"Channel URL: {channel_row.get('Channel URL','')}")
    lines.append("")
    
    # Video information section
    video_info = channel_row.get("Video Info", "")
    if video_info:
        lines.append("**Top Videos:**\n")
        lines.append(video_info)
        lines.append("")
    
    # Search filters section
    search_filters = st.session_state.get("search_filters", None)
    if search_filters:
        lines.append("**Search Filters:**\n")
        lines.append(f"Upload Date Filter: {search_filters.get('Upload Date Filter', 'No filter')}")
        lines.append(f"Subscriber Count Filter: {search_filters.get('Subscriber Count Filter', 'No filter')}")
        lines.append(f"Keyword(s): {search_filters.get('Keyword(s)', 'None')}")
        lines.append(f"Sort Order: {search_filters.get('Sort Order', 'Relevance')}")
        lines.append(f"Shorts Filter: {search_filters.get('Shorts Filter', 'Include all videos')}")
    
    note_content = "\n".join(lines)
    z_client.notes.create({
        "resource_type": "lead",
        "resource_id": new_lead["id"],
        "content": note_content
    })
    return new_lead

def partial_import(z_client, df, first_keyword, existing_channels):
    updated = df.copy()
    updated["CRM"] = ""
    updated["Lead ID"] = ""
    summary = {
        "imported_count": 0,
        "duplicate_count": 0,
        "missing_id_count": 0,
        "error_count": 0
    }
    for idx, row in updated.iterrows():
        channel_url = row.get("Channel URL", "").strip()
        if not channel_url:
            summary["missing_id_count"] += 1
            updated.at[idx, "CRM"] = "No Channel URL"
            continue
        dedup_key = parse_channel_id_from_crm(channel_url)
        if dedup_key in existing_channels:
            summary["duplicate_count"] += 1
            updated.at[idx, "CRM"] = "Duplicate in CRM"
            continue
        base_tag = "VIPI-FILTERED" if row.get("IsFiltered") else "VIPI-RAW"
        subs_num = int(row.get("Number of Subscribers", 0))
        payload = {
            "first_name": "TBD",
            "last_name": "TBD",
            "industry": "VIPI (Influencers)",
            "status": "Qualification - New",
            "owner_id": 6818690,  # adjust if needed
            "tags": ["Youtube-VIPI", base_tag, first_keyword.upper()],
            "address": {
                "country": get_standard_country_name(row.get("Country of Origin", "N/A"))
            },
            "custom_fields": {
                "Youtube": channel_url,
                "Youtube Subscribers": subs_num,
                "VIPI Category": "YouTubers",
                "VIPI Priority": "1 - Bronze"
            }
        }
        try:
            new_lead = create_lead_and_note(z_client, payload, row)
            updated.at[idx, "CRM"] = "Imported"
            updated.at[idx, "Lead ID"] = str(new_lead["id"])
            summary["imported_count"] += 1
            existing_channels.add(dedup_key)
        except Exception as e:
            summary["error_count"] += 1
            updated.at[idx, "CRM"] = f"Error: {e}"
    return updated, summary

# ---------------------------------------------------------------------
# Updated CRM Fetching: Internal Filtering by Tag "Youtube-VIPI"
# ---------------------------------------------------------------------
def fetch_existing_youtube_values(z_client):
    """
    For each owner in owners_map, fetch contacts and leads (using owner_id),
    then internally filter records to keep only those that are tagged with "Youtube-VIPI".
    It is assumed that if a record is tagged with "Youtube-VIPI", the "Youtube" custom field is filled.
    Returns:
        all_vals: set of normalized YouTube channel IDs,
        total_leads_filtered: count of leads matching,
        total_contacts_filtered: count of contacts matching.
    """
    all_vals = set()
    total_contacts_filtered = 0
    total_leads_filtered = 0
    per_page = 100

    # Process contacts for each owner.
    for owner_key, (owner_id, owner_name) in owners_map.items():
        page = 1
        contacts = safe_api_call(z_client.contacts.list, page=page, per_page=per_page, owner_id=owner_id)
        while contacts:
            for ct in contacts:
                cfields = ct.get("custom_fields", {})
                youtube_val = cfields.get("Youtube")
                tags = ct.get("tags", [])
                if youtube_val and ("Youtube-VIPI" in tags):
                    cid = parse_channel_id_from_crm(youtube_val)
                    all_vals.add(cid)
                    total_contacts_filtered += 1
            if len(contacts) < per_page:
                break
            page += 1
            contacts = safe_api_call(z_client.contacts.list, page=page, per_page=per_page, owner_id=owner_id)

    # Process leads for each owner.
    for owner_key, (owner_id, owner_name) in owners_map.items():
        page = 1
        leads = safe_api_call(z_client.leads.list, page=page, per_page=per_page, owner_id=owner_id)
        while leads:
            for ld in leads:
                cfields = ld.get("custom_fields", {})
                youtube_val = cfields.get("Youtube")
                tags = ld.get("tags", [])
                if youtube_val and ("Youtube-VIPI" in tags):
                    cid = parse_channel_id_from_crm(youtube_val)
                    all_vals.add(cid)
                    total_leads_filtered += 1
            if len(leads) < per_page:
                break
            page += 1
            leads = safe_api_call(z_client.leads.list, page=page, per_page=per_page, owner_id=owner_id)

    return all_vals, total_leads_filtered, total_contacts_filtered

# ---------------------------------------------------------------------
# 7) MAIN STREAMLIT APP
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="YouTube VIPI Discovery Tool", layout="wide")
    if "search_history" not in st.session_state:
        st.session_state["search_history"] = []
        
    # Initialize session state variables
    if "search_done" not in st.session_state:
        st.session_state["search_done"] = False
    if "duplicates_checked" not in st.session_state:
        st.session_state["duplicates_checked"] = False
        
    if "usage_loaded" not in st.session_state:
        used, usage_date = load_daily_usage()
        st.session_state["usage_loaded"] = True
        st.session_state["used_credits"] = used
        st.session_state["usage_date"] = usage_date
    else:
        used = st.session_state["used_credits"]
        usage_date = st.session_state["usage_date"]

    leftover = DAILY_LIMIT - used
    leftover = max(leftover, 0)

    with st.sidebar:
        st.title("Usage & Credits")
        donut_chart = draw_credit_donut(used, DAILY_LIMIT)
        st.altair_chart(donut_chart, use_container_width=True)
        st.write(f"**Daily Limit:** {DAILY_LIMIT}")
        st.write(f"**Used:** {used}")
        st.write(f"**Remaining:** {leftover}")
        now_pacific = datetime.now(PACIFIC_TZ)
        next_noon = now_pacific.replace(hour=12, minute=0, second=0, microsecond=0)
        if now_pacific >= next_noon:
            next_noon += timedelta(days=1)
        time_to_reset = next_noon - now_pacific
        hours = time_to_reset.seconds // 3600
        minutes = (time_to_reset.seconds // 60) % 60
        st.write(f"Credits reset in {hours}h {minutes}m (12:00 PM Pacific Time).")
        api_key = load_youtube_api_key()
        if not api_key:
            st.warning("No valid YouTube API key found. Please update credentials to proceed.")

    st.title("YouTube VIPI Discovery Tool")
    st.divider()
    st.header("Configure Your Search")
    st.subheader("Number of Keywords (Max 3)")
    selected_keyword_count = st.selectbox("Select how many keywords:", list(range(1, MAX_KEYWORDS+1)), index=0)
    if "keyword_entries" not in st.session_state:
        st.session_state["keyword_entries"] = [{"keyword": "", "count": 50} for _ in range(MAX_KEYWORDS)]
    with st.expander("Distribute Credits Evenly (Optional)"):
        st.write("Enter a total number of credits; they'll be split among your selected keywords.")
        total_distribution = st.number_input("Total Credits to Distribute", min_value=0, max_value=DAILY_LIMIT, value=0, step=50)
        if st.button("Apply Distribution"):
            if total_distribution == 0:
                st.warning("Distribution total is zero; nothing to apply.")
            else:
                def pick_closest(value, allowed=ALLOWED_TARGETS):
                    return min(allowed, key=lambda x: abs(x - value))
                portion = total_distribution // selected_keyword_count
                for i in range(selected_keyword_count):
                    st.session_state["keyword_entries"][i]["count"] = pick_closest(portion)
                st.success(f"Distributed ~{portion} credits per keyword (rounded).")
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
    st.subheader("Upload Date Filter")
    date_filter_choice = st.radio("Select an upload date filter:", ["No filter", "Last 3 Months", "Last 6 Months", "Last 1 Year"], index=0)
    published_after_value = None
    if date_filter_choice != "No filter":
        now_utc = datetime.utcnow()
        days_ago = 90 if date_filter_choice == "Last 3 Months" else 180 if date_filter_choice == "Last 6 Months" else 365
        cutoff = now_utc - timedelta(days=days_ago)
        published_after_value = cutoff.isoformat("T") + "Z"
    st.subheader("Subscriber Count Filter")
    sub_filter_choice = st.radio("Pick one subscriber range:", ["No filter", "Up to 20K", "Up to 50K", "Above 50K"], index=0)
    below20k = (sub_filter_choice == "Up to 20K")
    below50k = (sub_filter_choice == "Up to 50K")
    above50k = (sub_filter_choice == "Above 50K")
    st.subheader("Select Sort Order")
    sort_order = st.radio("Choose sort order:", options=["Relevance", "Most Watched"], index=0)
    # Convert sort order to the corresponding YouTube API parameter
    youtube_order = "relevance" if sort_order == "Relevance" else "viewCount"
    
    st.subheader("YouTube Shorts Filter")
    shorts_filter = st.radio(
        "Include YouTube Shorts?",
        ["Include all videos", "Exclude Shorts", "Only show Shorts"],
        index=0
    )
    
    st.subheader("Give Your Output File a Name")
    base_name = st.text_input("Output file base name:", value="MySearch")
    run_search = st.button("Run YouTube Search & Process Channels")
    if run_search:
        if not api_key:
            st.error("No valid YouTube API key. Please update your credentials in the sidebar.")
            return
        total_requested = 0
        valid_keywords = []
        for i in range(selected_keyword_count):
            kw = st.session_state["keyword_entries"][i]["keyword"].strip()
            cnt = st.session_state["keyword_entries"][i]["count"]
            if kw:
                total_requested += cnt
                valid_keywords.append({"keyword": kw, "count": cnt})
        used = st.session_state["used_credits"]
        leftover = DAILY_LIMIT - used
        if total_requested > leftover:
            st.error(f"You've requested {total_requested} credits, but only {leftover} remain.")
            return
        if total_requested <= 0:
            st.error("No valid (non-empty) keywords or zero total requested. Nothing to search.")
            return
        youtube = build('youtube', 'v3', developerKey=api_key, cache_discovery=False)
        new_used = used
        raw_videos_all = []
        filtered_videos_all = []
        progress_bar = st.progress(0.0)
        done_kw = 0
        total_kw = len(valid_keywords)
        for entry in valid_keywords:
            kw = entry["keyword"]
            tcount = entry["count"]
            st.write(f"**Fetching up to {tcount} filtered videos for:** {kw} ...")
            rv, fv, new_used = fetch_videos_for_keyword_until_filtered(
                youtube=youtube,
                keyword=kw,
                target_count=tcount,
                published_after=published_after_value,
                below20k=below20k,
                below50k=below50k,
                above50k=above50k,
                daily_usage=new_used,
                order_filter=youtube_order,
                shorts_filter=shorts_filter
            )
            raw_videos_all.extend(rv)
            filtered_videos_all.extend(fv)
            done_kw += 1
            progress_bar.progress(done_kw / total_kw)
        save_daily_usage(new_used)
        st.session_state["used_credits"] = new_used
        st.success(f"Search complete! Found {len(filtered_videos_all)} filtered videos total.")
        st.info(f"Used {new_used} of {DAILY_LIMIT} credits so far today.")
        
        # Show applied filters for verification
        st.write("**Applied filters:**")
        st.write(f"• Sort order: **{sort_order}** (YouTube API parameter: '{youtube_order}')")
        st.write(f"• Shorts filter: **{shorts_filter}**")
        st.write(f"• Upload date: **{date_filter_choice}**")
        st.write(f"• Subscriber range: **{sub_filter_choice}**")
        
        # Analyze if sort order seems to be working
        if len(filtered_videos_all) > 0:
            df_filtered_videos = pd.DataFrame(filtered_videos_all)
            
            # Add View Count Numeric for analysis
            df_filtered_videos["View Count Numeric"] = pd.to_numeric(df_filtered_videos["View Count"], errors="coerce").fillna(0)
            
            analysis, seems_valid = analyze_search_results_by_order(df_filtered_videos, sort_order)
            
            with st.expander("Sort Order Analysis (Click to expand)"):
                if sort_order == "Most Watched":
                    st.write("**Most Watched Analysis:**")
                    st.write(f"• Average views: {int(analysis.get('avg_views', 0)):,}")
                    if analysis.get('total_videos', 0) >= 4:
                        quarter_size = analysis.get('total_videos', 0) // 4
                        st.write(f"• First {quarter_size} videos avg views: {int(df_filtered_videos.iloc[:quarter_size]['View Count Numeric'].mean()):,}")
                        st.write(f"• Last {quarter_size} videos avg views: {int(df_filtered_videos.iloc[-quarter_size:]['View Count Numeric'].mean()):,}")
                else:
                    st.write("**Relevance Analysis:**")
                    # Count how many video titles contain the keyword
                    keywords = df_filtered_videos["Keyword"].unique()
                    keyword_match_count = 0
                    for idx, row in df_filtered_videos.iterrows():
                        title = row["Video Title"].lower()
                        keyword = row["Keyword"].lower()
                        if keyword in title:
                            keyword_match_count += 1
                    
                    keyword_match_percent = (keyword_match_count / len(df_filtered_videos)) * 100
                    st.write(f"• Videos with keyword in title: {keyword_match_count} ({keyword_match_percent:.1f}%)")
                
                if seems_valid:
                    st.success(f"The '{sort_order}' sort order appears to be working properly.")
                else:
                    st.warning(f"The '{sort_order}' sort order may not be working as expected. Review the results.")
        
        sum_of_videos = len(raw_videos_all) + len(filtered_videos_all)
        st.session_state["total_videos_discovered"] = sum_of_videos
        df_all_videos = pd.concat([pd.DataFrame(raw_videos_all), pd.DataFrame(filtered_videos_all)], ignore_index=True)
        video_info_dict = aggregate_video_info(df_all_videos)
        df_raw_videos = pd.DataFrame(raw_videos_all)
        df_filtered_videos = pd.DataFrame(filtered_videos_all)
        df_raw_channels = build_unique_channels_df(df_raw_videos, is_filtered=False, video_info=video_info_dict)
        df_filtered_channels = build_unique_channels_df(df_filtered_videos, is_filtered=True, video_info=video_info_dict)
        df_all_channels = pd.concat([df_raw_channels, df_filtered_channels], ignore_index=True)
        df_all_channels.drop_duplicates(subset=["Channel ID"], inplace=True)
        discovered_count = len(df_all_channels)
        
        # Store all relevant DataFrames and result data in session state
        st.session_state["raw_videos_all"] = raw_videos_all
        st.session_state["filtered_videos_all"] = filtered_videos_all
        st.session_state["df_raw_videos"] = df_raw_videos
        st.session_state["df_filtered_videos"] = df_filtered_videos
        st.session_state["df_raw_channels"] = df_raw_channels
        st.session_state["df_filtered_channels"] = df_filtered_channels
        st.session_state["df_all_channels"] = df_all_channels
        st.session_state["video_info_dict"] = video_info_dict
        st.session_state["discovered_count"] = discovered_count
        
        st.session_state["search_done"] = True
        st.session_state["duplicates_checked"] = False
        
        # Save search filter info in session state for use in lead notes.
        st.session_state["search_filters"] = {
            "Upload Date Filter": date_filter_choice,
            "Subscriber Count Filter": sub_filter_choice,
            "Keyword(s)": ", ".join([kv["keyword"] for kv in valid_keywords]),
            "Sort Order": sort_order,
            "Shorts Filter": shorts_filter
        }
        info_rows = [
            {"Parameter": "Upload Date Filter", "Value": date_filter_choice},
            {"Parameter": "Subscriber Filter", "Value": sub_filter_choice},
            {"Parameter": "Sort Order", "Value": sort_order},
            {"Parameter": "Shorts Filter", "Value": shorts_filter}
        ]
        idx = 1
        for kv in valid_keywords:
            info_rows.append({"Parameter": f"Keyword {idx}", "Value": f"{kv['keyword']} (count={kv['count']})"})
            idx += 1
        info_df = pd.DataFrame(info_rows)
        
        # Create Excel files for download
        all_bytes = df_to_excel_bytes_with_info(df_all_channels, info_df, "All Channels", "Filters")
        raw_bytes = df_to_excel_bytes_with_info(df_raw_channels, info_df, "RAW Channels", "Filters")
        filt_bytes = df_to_excel_bytes_with_info(df_filtered_channels, info_df, "FILTERED Channels", "Filters")
        
        # Store these in session state
        st.session_state["all_bytes"] = all_bytes
        st.session_state["raw_bytes"] = raw_bytes
        st.session_state["filt_bytes"] = filt_bytes
        
        # Create zip file
        file_dict = {}
        file_dict[f"{base_name} ALL Channels.xlsx"] = all_bytes
        file_dict[f"{base_name} RAW Channels.xlsx"] = raw_bytes
        file_dict[f"{base_name} FILTERED Channels.xlsx"] = filt_bytes
        final_zip = create_zip_file(file_dict)
        
        # Store zip in session state
        st.session_state["final_zip"] = final_zip
        
        st.subheader("Download Your Results")
        st.download_button("Download Processed Channels (ZIP)",
                           data=final_zip,
                           file_name=f"{base_name}_processed_channels.zip",
                           mime="application/zip")
        
        # Add this search to history
        search_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "keywords": ", ".join([kv["keyword"] for kv in valid_keywords]),
            "sort_order": sort_order,
            "shorts_filter": shorts_filter,
            "date_filter": date_filter_choice,
            "subscriber_filter": sub_filter_choice,
            "result_count": len(filtered_videos_all)
        }
        
        # Calculate average views if results exist
        if len(filtered_videos_all) > 0:
            df_temp = pd.DataFrame(filtered_videos_all)
            df_temp["View Count Numeric"] = pd.to_numeric(df_temp["View Count"], errors="coerce").fillna(0)
            search_entry["avg_views"] = int(df_temp["View Count Numeric"].mean())
        else:
            search_entry["avg_views"] = 0
        
        # Keep only the last 5 searches
        st.session_state["search_history"].append(search_entry)
        if len(st.session_state["search_history"]) > 5:
            st.session_state["search_history"] = st.session_state["search_history"][-5:]
            
        # Display search history comparison if there are multiple entries
        if len(st.session_state["search_history"]) > 1:
            with st.expander("Compare with Previous Searches"):
                history_df = pd.DataFrame(st.session_state["search_history"])
                st.dataframe(history_df, use_container_width=True)
                
                # Only show chart if we have multiple searches with different parameters
                if len(history_df["sort_order"].unique()) > 1:
                    st.write("**Result counts by sort order:**")
                    chart = alt.Chart(history_df).mark_bar().encode(
                        x='sort_order',
                        y='result_count',
                        color='sort_order',
                        tooltip=['keywords', 'sort_order', 'result_count', 'avg_views']
                    ).properties(width=300, height=200)
                    st.altair_chart(chart)
                    
    # Add a check here to reload previous search data if needed
    elif st.session_state.get("search_done") and not st.session_state.get("duplicates_checked"):
        # If we have previous search data but haven't checked duplicates, show the download button again
        if "final_zip" in st.session_state and "base_name" in locals():
            st.subheader("Download Your Previous Results")
            st.download_button("Download Processed Channels (ZIP)",
                               data=st.session_state["final_zip"],
                               file_name=f"{base_name}_processed_channels.zip",
                               mime="application/zip")

    if st.session_state.get("search_done"):
        st.divider()
        st.header("Fetch Existing CRM Data")
        z_token = load_zendesk_sell_token()
        if not z_token:
            st.warning("No Zendesk Sell token found. Please update your config first.")
            return
        z_client = BaseClient(access_token=z_token)
        if st.button("Fetch Existing CRM Data"):
            # Check if necessary dataframes exist in session state
            if "df_raw_channels" not in st.session_state or "df_filtered_channels" not in st.session_state:
                st.error("Search results data is missing. Please run the search again.")
                return
                
            existing_set, lead_count, contact_count = fetch_existing_youtube_values(z_client)
            st.session_state["existing_channels"] = existing_set
            st.session_state["lead_count"] = lead_count
            st.session_state["contact_count"] = contact_count

            df_raw = st.session_state.get("df_raw_channels", pd.DataFrame()).copy()
            df_filt = st.session_state.get("df_filtered_channels", pd.DataFrame()).copy()

            st.session_state["raw_before"] = len(df_raw)
            st.session_state["filt_before"] = len(df_filt)

            new_filt_rows = []
            for idx, row in df_filt.iterrows():
                # Add safety check for row
                if row is None:
                    continue
                link = row.get("Channel URL", "")
                # Add safety check for link
                if link is None:
                    continue
                link = link.strip()
                if link:
                    dedup_key = parse_channel_id_from_crm(link)
                    if dedup_key not in existing_set:
                        new_filt_rows.append(row)
            df_filt = pd.DataFrame(new_filt_rows)

            filtered_keys = set()
            for idx, row in df_filt.iterrows():
                # Add safety check for row
                if row is None:
                    continue
                link = row.get("Channel URL", "")
                # Add safety check for link
                if link is None:
                    continue
                link = link.strip()
                if link:
                    dedup_key = parse_channel_id_from_crm(link)
                    filtered_keys.add(dedup_key)

            new_raw_rows = []
            for idx, row in df_raw.iterrows():
                # Add safety check for row
                if row is None:
                    continue
                link = row.get("Channel URL", "")
                # Add safety check for link
                if link is None:
                    continue
                link = link.strip()
                if link:
                    dedup_key = parse_channel_id_from_crm(link)
                    if (dedup_key not in existing_set) and (dedup_key not in filtered_keys):
                        new_raw_rows.append(row)
            df_raw = pd.DataFrame(new_raw_rows)

            df_filt["Import?"] = True
            df_raw["Import?"] = True

            st.session_state["df_raw_nodup"] = df_raw
            st.session_state["df_filt_nodup"] = df_filt
            st.session_state["duplicates_checked"] = True
        
        # Rest of the CRM processing code...
        if st.session_state.get("duplicates_checked"):
            st.subheader("Summary Statistics")
            total_videos = st.session_state.get("total_videos_discovered", 0)
            st.write(f"- **Total Videos Discovered for this search:** {total_videos}")
            discovered_total = st.session_state.get("discovered_count", 0)
            st.write(f"- **YouTube VIPIs Discovered for this search:** {discovered_total}")
            st.write(f"- **YouTube VIPIs in CRM (Contacts):** {st.session_state['contact_count']}")
            st.write(f"- **YouTube VIPIs in CRM (Leads):** {st.session_state['lead_count']}")
            raw_before = st.session_state.get("raw_before", 0)
            filt_before = st.session_state.get("filt_before", 0)
            df_rn = st.session_state.get("df_raw_nodup", pd.DataFrame())
            df_fn = st.session_state.get("df_filt_nodup", pd.DataFrame())
            
            # Get unique channel IDs from before deduplication
            all_channels_before = set()
            for _, row in st.session_state.get("df_raw_channels", pd.DataFrame()).iterrows():
                if "Channel URL" in row and row.get("Channel URL") is not None and row["Channel URL"].strip():
                    all_channels_before.add(parse_channel_id_from_crm(row["Channel URL"]))
            for _, row in st.session_state.get("df_filtered_channels", pd.DataFrame()).iterrows():
                if "Channel URL" in row and row.get("Channel URL") is not None and row["Channel URL"].strip():
                    all_channels_before.add(parse_channel_id_from_crm(row["Channel URL"]))
                
            # Get unique channel IDs after deduplication
            all_channels_after = set()
            for _, row in df_rn.iterrows():
                if "Channel URL" in row and row.get("Channel URL") is not None and row["Channel URL"].strip():
                    all_channels_after.add(parse_channel_id_from_crm(row["Channel URL"]))
            for _, row in df_fn.iterrows():
                if "Channel URL" in row and row.get("Channel URL") is not None and row["Channel URL"].strip():
                    all_channels_after.add(parse_channel_id_from_crm(row["Channel URL"]))
                
            # True duplicate count
            total_removed = len(all_channels_before) - len(all_channels_after)
            
            st.write(f"- **Duplicates between discovered and CRM:** {total_removed}")
            st.write(f"- **Filtered Channels to Import (currently):** {len(df_fn)}")
            st.write(f"- **RAW (Unfiltered) Channels to Import (currently):** {len(df_rn)}")
            st.divider()
            st.header("Step 2: Select Which FILTERED Channels to Import")
            df_fn = st.session_state.get("df_filt_nodup", pd.DataFrame())
            if df_fn.empty:
                st.info("No new FILTERED channels found (or all were duplicates).")
            else:
                columns_order_f = ["Import?", "Channel ID", "Channel URL", "Channel Name",
                                     "Number of Subscribers", "Country of Origin", "IsFiltered", "Video Info"]
                columns_order_f = [c for c in columns_order_f if c in df_fn.columns]
                df_fn = df_fn[columns_order_f]
                edited_filt = st.data_editor(
                    df_fn,
                    column_config={
                        "Import?": st.column_config.CheckboxColumn(
                            "Import?",
                            help="Uncheck if you do not want to import this channel",
                            default=True
                        )
                    },
                    hide_index=True,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="filtered_editor"
                )
                st.session_state["edited_filt"] = edited_filt
                selected_filtered_count = edited_filt["Import?"].sum()
                st.write(f"**Filtered Channels Currently Selected:** {selected_filtered_count}")
            st.divider()
            st.header("Select Which RAW (Unfiltered) Channels to Import")
            df_rn = st.session_state.get("df_raw_nodup", pd.DataFrame())
            if df_rn.empty:
                st.info("No new RAW channels found (or all were duplicates).")
            else:
                columns_order_r = ["Import?", "Channel ID", "Channel URL", "Channel Name",
                                     "Number of Subscribers", "Country of Origin", "IsFiltered", "Video Info"]
                columns_order_r = [c for c in columns_order_r if c in df_rn.columns]
                df_rn = df_rn[columns_order_r]
                edited_raw = st.data_editor(
                    df_rn,
                    column_config={
                        "Import?": st.column_config.CheckboxColumn(
                            "Import?",
                            help="Uncheck if you do not want to import this channel",
                            default=True
                        )
                    },
                    hide_index=True,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="raw_editor"
                )
                st.session_state["edited_raw"] = edited_raw
                selected_raw_count = edited_raw["Import?"].sum()
                st.write(f"**RAW Channels Currently Selected:** {selected_raw_count}")
            st.divider()
            st.header("Perform CRM Import")
            if st.button("Import Selected Leads into CRM"):
                existing_channels = st.session_state["existing_channels"]
                final_imported_frames = []
                edited_filt = st.session_state.get("edited_filt", pd.DataFrame())
                if not edited_filt.empty:
                    subset_f = edited_filt[edited_filt["Import?"] == True].copy()
                    if not subset_f.empty:
                        st.write(f"Importing {len(subset_f)} FILTERED channels ...")
                        first_kw = "NONE"
                        if "keyword_entries" in st.session_state:
                            all_kws = [k["keyword"] for k in st.session_state["keyword_entries"] if k["keyword"]]
                        if all_kws:
                            first_kw = all_kws[0]
                        updated_f, summary_f = partial_import(z_client, subset_f, first_kw, existing_channels)
                        st.success(
                            f"Filtered import done: Imported={summary_f['imported_count']}, "
                            f"Duplicates={summary_f['duplicate_count']}, No Channel URL={summary_f['missing_id_count']}, "
                            f"Errors={summary_f['error_count']}"
                        )
                        final_imported_frames.append(updated_f)
                    else:
                        st.info("No FILTERED channels selected for import.")
                edited_raw = st.session_state.get("edited_raw", pd.DataFrame())
                if not edited_raw.empty:
                    subset_r = edited_raw[edited_raw["Import?"] == True].copy()
                    if not subset_r.empty:
                        st.write(f"Importing {len(subset_r)} RAW channels ...")
                        first_kw = "NONE"
                        if "keyword_entries" in st.session_state:
                            all_kws = [k["keyword"] for k in st.session_state["keyword_entries"] if k["keyword"]]
                            if all_kws:
                                first_kw = all_kws[0]
                        updated_r, summary_r = partial_import(z_client, subset_r, first_kw, existing_channels)
                        st.success(
                            f"RAW import done: Imported={summary_r['imported_count']}, "
                            f"Duplicates={summary_r['duplicate_count']}, No Channel URL={summary_r['missing_id_count']}, "
                            f"Errors={summary_r['error_count']}"
                        )
                        final_imported_frames.append(updated_r)
                    else:
                        st.info("No RAW channels selected for import.")
                if final_imported_frames:
                    combined_import_df = pd.concat(final_imported_frames, ignore_index=True)
                    
                    filtered_keys = set()
                    df_filt = st.session_state.get("df_filt_nodup", pd.DataFrame())
                    for idx, row in df_filt.iterrows():
                        if row is None:
                            continue
                        link = row.get("Channel URL", "")
                        if link is None:
                            continue
                        link = link.strip()
                        if link:
                            dedup_key = parse_channel_id_from_crm(link)
                            filtered_keys.add(dedup_key)
                    
                    df_duplicates = create_duplicates_df(
                        raw_before=st.session_state.get("raw_before", 0),
                        raw_after=len(df_rn),
                        filt_before=st.session_state.get("filt_before", 0),
                        filt_after=len(df_fn),
                        existing_channels=existing_channels,
                        filtered_keys=filtered_keys
                    )
                    
                    file_dict = {}
                    
                    crm_import_bytes = df_to_excel_bytes_with_info(
                        combined_import_df, pd.DataFrame(),
                        data_sheet_name="CRM Import Results",
                        info_sheet_name="Info"
                    )
                    file_dict["CRM Import Results.xlsx"] = crm_import_bytes
                    
                    file_dict[f"{base_name} ALL Channels.xlsx"] = st.session_state.get("all_bytes")
                    file_dict[f"{base_name} RAW Channels.xlsx"] = st.session_state.get("raw_bytes")
                    file_dict[f"{base_name} FILTERED Channels.xlsx"] = st.session_state.get("filt_bytes")
                    
                    if not df_duplicates.empty:
                        duplicates_bytes = df_to_excel_bytes_with_info(
                            df_duplicates, pd.DataFrame(),
                            data_sheet_name="Duplicates",
                            info_sheet_name="Info"
                        )
                        file_dict[f"{base_name} Duplicates.xlsx"] = duplicates_bytes
                    
                    final_zip_all = create_zip_file(file_dict)
                    
                    st.download_button(
                        "Download CRM Import Results + Processed Data",
                        data=final_zip_all,
                        file_name=f"{base_name}_all_results.zip",
                        mime="application/zip"
                    )
                else:
                    st.info("No leads imported. Possibly all channels unchecked or empty results.")

def create_duplicates_df(raw_before, raw_after, filt_before, filt_after, existing_channels, filtered_keys):
    duplicates = []
    
    # Safety check for raw channels
    df_raw_channels = st.session_state.get("df_raw_channels")
    if df_raw_channels is not None and not df_raw_channels.empty:
        for idx, row in df_raw_channels.iterrows():
            if row is None:
                continue
            link = row.get("Channel URL", "")
            if link is None:
                continue
            link = link.strip()
            if link:
                dedup_key = parse_channel_id_from_crm(link)
                if dedup_key in existing_channels:
                    row_data = row.to_dict()
                    row_data["Duplicate Type"] = "Already exists in CRM"
                    row_data["Channel Type"] = "RAW"
                    duplicates.append(row_data)
                elif dedup_key in filtered_keys:
                    row_data = row.to_dict()
                    row_data["Duplicate Type"] = "Already exists in FILTERED results"
                    row_data["Channel Type"] = "RAW"
                    duplicates.append(row_data)

    # Safety check for filtered channels
    df_filtered_channels = st.session_state.get("df_filtered_channels")
    if df_filtered_channels is not None and not df_filtered_channels.empty:
        for idx, row in df_filtered_channels.iterrows():
            if row is None:
                continue
            link = row.get("Channel URL", "")
            if link is None:
                continue
            link = link.strip()
            if link:
                dedup_key = parse_channel_id_from_crm(link)
                if dedup_key in existing_channels:
                    row_data = row.to_dict()
                    row_data["Duplicate Type"] = "Already exists in CRM"
                    row_data["Channel Type"] = "FILTERED"
                    duplicates.append(row_data)

    if not duplicates:
        return pd.DataFrame()

    df_duplicates = pd.DataFrame(duplicates)
    
    columns_order = ["Channel Type", "Duplicate Type", "Channel ID", "Channel URL", "Channel Name",
                    "Number of Subscribers", "Country of Origin", "Video Info"]
    columns_order = [c for c in columns_order if c in df_duplicates.columns]
    remaining_cols = [c for c in df_duplicates.columns if c not in columns_order]
    df_duplicates = df_duplicates[columns_order + remaining_cols]
    
    return df_duplicates

if __name__ == "__main__":
    main()