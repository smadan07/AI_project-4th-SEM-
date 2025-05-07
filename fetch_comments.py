import re
from googleapiclient.discovery import build

# Replace with your actual API key and YouTube video URL
api_key = "AIzaSyAWg-YWnEKrs3LJGYfPagorNG56b1cxEGU"
video_url = "https://www.youtube.com/watch?v=JiMTOMkuM2Y&ab_channel=MnemonicMaster"

# Function to extract video ID from the YouTube URL
def extract_video_id(url):
    match = re.search(r"v=([^&]+)", url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

video_id = extract_video_id(video_url)

# Build the YouTube service object
youtube = build('youtube', 'v3', developerKey=api_key)

def get_comments(video_id, max_results=100):
    """
    Fetches YouTube comments and stores them in the required dictionary format.

    Returns:
        dict: Dictionary where keys are usernames and values are dictionaries 
              containing comment text and date.
    """
    comments = {}
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        order="time"
    )
    
    while request:
        response = request.execute()
        for item in response.get("items", []):
            comment_data = item["snippet"]["topLevelComment"]["snippet"]
            username = comment_data["authorDisplayName"]
            comment_text = comment_data["textDisplay"]
            comment_date = comment_data["publishedAt"]
            
            # Store the comment using the username as the key
            comments[username] = {
                "comment": comment_text,
                "date": comment_date
            }

        # Handle pagination
        request = youtube.commentThreads().list_next(request, response)
    
    return comments

# Fetch comments
comments = get_comments(video_id)
print(f"Total comments fetched: {len(comments)}")

# Print the dictionary (sample output)
for username, data in list(comments.items())[:5]:  # Show first 5 users
    print(f"{username}: {data}")
