from googleapiclient.discovery import build
import pandas as pd
import time
from datetime import datetime
import os
import logging
import isodate

# Replace with your actual YouTube Data API key
API_KEY ="AIzaSyBcwqCHmwALaF4_OY7RLl_hzCpfCLZexWg"

# Build the YouTube service
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Define keywords and target result counts
keywords = {
"nik": 10,

}

def main():
    # Setup logging
    downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(downloads_folder, f'youtube_data_collection_{timestamp}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Bulk keyword search started")

    all_results = []
    collected_urls = set()  # Track URLs to avoid duplicates

    for keyword, target_count in keywords.items():
        logging.info(f"Starting data collection for query: {keyword}, aiming for {target_count} results")
        print(f"\nCollecting data for keyword: {keyword}")
        next_page_token = None
        results_collected = 0
        
        try:
            while results_collected < target_count:
                # Search for videos related to the query, sorted by view count
                search_response = youtube.search().list(
                    q=keyword,
                    type='video',
                    part='id,snippet',
                    maxResults=50,  # API max per page
                    order='viewCount',
                    pageToken=next_page_token
                ).execute()

                # Process each video in the search response
                for item in search_response['items']:
                    video_id = item['id']['videoId']
                    video_url = f'https://www.youtube.com/watch?v={video_id}'

                    # Skip if URL already exists
                    if video_url in collected_urls:
                        continue
                    
                    # Basic video details
                    video_title = item['snippet']['title']
                    channel_id = item['snippet']['channelId']
                    channel_title = item['snippet']['channelTitle']
                    channel_url = f'https://www.youtube.com/channel/{channel_id}'

                    # Published date formatting
                    published_at = item['snippet']['publishedAt']
                    published_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                    formatted_published_date = published_date.strftime("%Y-%m-%d %H:%M:%S")

                    # Add URL to set to avoid future duplicates
                    collected_urls.add(video_url)

                    # Append to all results
                    all_results.append({
                        'Video Title': video_title,
                        'Video URL': video_url,
                        'Channel Name': channel_title,
                        'Channel URL': channel_url,
                        'Published Date': formatted_published_date,
                        'Keyword': keyword  # Track keyword for analysis
                    })
                    results_collected += 1

                    logging.info(f"Collected data for video: {video_title}")

                    # Stop if target count reached for this keyword
                    if results_collected >= target_count:
                        break

                # Check if there is a next page
                next_page_token = search_response.get('nextPageToken')
                if not next_page_token or results_collected >= target_count:
                    break

                time.sleep(1)  # Respect API rate limits

        except Exception as e:
            logging.error(f"Error during data collection for keyword '{keyword}': {e}")
            print(f"An error occurred for '{keyword}': {e}")

    # Fetch view counts, duration, and other details
    print("\nFetching additional details for all videos...")
    logging.info("Fetching additional details for all videos")
    fetch_additional_details(all_results)

    # Save final results to CSV
    save_results_to_csv(all_results, downloads_folder, timestamp)
    logging.info("Bulk keyword search completed successfully")

def fetch_additional_details(results):
    # Collect unique video and channel IDs for batch requests
    video_ids = [result['Video URL'].split('=')[-1] for result in results]
    channel_ids = list(set([result['Channel URL'].split('/')[-1] for result in results]))

    # Fetch video statistics and duration
    for i in range(0, len(video_ids), 50):
        batch_video_ids = video_ids[i:i + 50]
        video_response = youtube.videos().list(
            part='statistics,contentDetails',
            id=','.join(batch_video_ids)
        ).execute()

        for video_item in video_response['items']:
            video_id = video_item['id']
            view_count = video_item['statistics'].get('viewCount', 'N/A')
            duration = format_duration(video_item['contentDetails'].get('duration', 'PT0S'))

            # Update matching results entry
            for result in results:
                if result['Video URL'].endswith(video_id):
                    result['View Count'] = view_count
                    result['Duration'] = duration

    # Fetch channel subscriber count and country
    for i in range(0, len(channel_ids), 50):
        batch_channel_ids = channel_ids[i:i + 50]
        channel_response = youtube.channels().list(
            part='snippet,statistics',
            id=','.join(batch_channel_ids)
        ).execute()

        for channel_item in channel_response['items']:
            channel_id = channel_item['id']
            subscriber_count = channel_item['statistics'].get('subscriberCount', 'N/A')
            country = channel_item['snippet'].get('country', 'N/A')

            # Update matching results entry
            for result in results:
                if result['Channel URL'].endswith(channel_id):
                    result['Number of Subscribers'] = subscriber_count
                    result['Country of Origin'] = country

def format_duration(duration_iso):
    parsed_duration = isodate.parse_duration(duration_iso)
    total_seconds = int(parsed_duration.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def save_results_to_csv(results, downloads_folder, timestamp):
    filename = os.path.join(downloads_folder, f'YouTube_Influencers_bulk_{timestamp}.csv')
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"\nData saved to '{filename}'")
    logging.info(f"Data saved to '{filename}'")

if __name__ == '__main__':
    main()
