import os
import re
import pickle
from datetime import datetime
import string

# Flask and Google API Client
from flask import Flask, render_template, request, redirect, url_for, flash
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Data Handling and NLP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from emoji import demojize
from nltk.stem import PorterStemmer
import logging

# Plotting
import plotly.express as px
import plotly.io as pio

# --- Configuration ---
app = Flask(__name__)
app.secret_key = 'your_very_secret_key_123!' # Change this to something random and secret
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- YouTube API Setup ---
# IMPORTANT: Replace with your actual API key.
# Consider using environment variables for production for better security.
API_KEY = "AIzaSyAWg-YWnEKrs3LJGYfPagorNG56b1cxEGU" # <--- PASTE YOUR ACTUAL YOUTUBE DATA API v3 KEY HERE

youtube = None
if API_KEY == "YOUR_API_KEY" or not API_KEY:
    logging.warning("!!!! YouTube API Key is missing or placeholder. YouTube functionality will be disabled. !!!!")
else:
    try:
        # Verify the API Key format slightly (basic check)
        if not re.match(r'AIza[0-9A-Za-z_-]{35}', API_KEY):
             logging.warning("API Key format looks potentially incorrect. Proceeding anyway...")
        youtube = build('youtube', 'v3', developerKey=API_KEY)
        logging.info("YouTube service built successfully.")
    except Exception as e:
        logging.error(f"Failed to build YouTube service: {e}")
        youtube = None # Ensure it's None if build fails

# --- Sentiment Analysis Model Loading ---
vectorizer = None
model = None
try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logging.info("Sentiment model and vectorizer loaded successfully.")
except FileNotFoundError:
    logging.error("Error: vectorizer.pkl or sentiment_model.pkl not found in the current directory.")
except Exception as e:
    logging.error(f"Error loading model/vectorizer files: {e}")

# --- Helper Functions ---

def extract_video_id(url):
    """Extracts video ID from various YouTube URL formats."""
    if not url: return None
    patterns = [
        r'(?:v=|\/|vi=)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:youtube\.com\/embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtube\.com\/shorts\/)([0-9A-Za-z_-]{11})',
        r'(?:youtube\.com\/live\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            logging.info(f"Extracted Video ID: {match.group(1)}")
            return match.group(1)
    logging.warning(f"Could not extract Video ID from URL: {url}")
    return None

def get_comments(video_id, max_results_per_page=100, max_total_results=5000):
    """Fetches YouTube comments, handling potential errors."""
    if not youtube:
        raise ConnectionError("YouTube service not initialized (check API Key).")
    if not video_id:
        raise ValueError("Invalid Video ID provided to get_comments.")

    comments_list = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=min(max_results_per_page, max_total_results),
        order="time", # Fetch most recent first (useful for pagination limit)
        textFormat="plainText" # Get plain text, suitable for analysis
    )

    fetched_count = 0
    page_count = 0
    while request and fetched_count < max_total_results:
        page_count += 1
        logging.info(f"Fetching comments page {page_count} (fetched so far: {fetched_count}/{max_total_results})")
        try:
            response = request.execute()
        except HttpError as e:
             error_content = e.content.decode('utf-8') if e.content else "{no content}"
             logging.error(f"YouTube API HTTP Error: Status {e.resp.status}, Reason: {e.reason}, Content: {error_content}")
             if e.resp.status == 403:
                 if 'commentsDisabled' in error_content: raise ValueError("Comments are disabled for this video.")
                 elif 'quotaExceeded' in error_content: raise ConnectionError("YouTube API Quota Exceeded. Please try again later or check your quota.")
                 else: raise ConnectionError(f"YouTube API Forbidden Error (Status 403): {e.reason}. Check API key permissions/restrictions.")
             elif e.resp.status == 404: raise ValueError(f"Video not found (ID: {video_id}) or comments unavailable (Status 404).")
             else: raise ConnectionError(f"An unexpected YouTube API HTTP error occurred: {e}")
        except Exception as e: # Catch other potential errors (network, etc.)
            logging.error(f"Non-HTTP error during comment fetch: {e}", exc_info=True)
            raise ConnectionError(f"An unexpected error occurred while fetching comments: {e}")

        items = response.get("items", [])
        if not items and page_count == 1: # No comments found on the first page
             logging.info("No comments found for this video.")
             # break # Exit loop if no comments on first page

        for item in items:
            if fetched_count >= max_total_results:
                break
            try:
                comment_data = item["snippet"]["topLevelComment"]["snippet"]
                username = comment_data.get("authorDisplayName", "Unknown User")
                comment_text = comment_data.get("textOriginal", "") # Use original text for analysis
                display_text = comment_data.get("textDisplay", comment_text) # Use display text if different, fallback to original

                # Parse date string to datetime object
                comment_date_str = comment_data.get("publishedAt")
                if comment_date_str:
                    try:
                        # Handle standard ISO format with Z or offset
                        comment_date = datetime.fromisoformat(comment_date_str.replace('Z', '+00:00'))
                    except ValueError:
                        logging.warning(f"Could not parse date string '{comment_date_str}' for comment by {username}. Using current UTC time.")
                        comment_date = datetime.utcnow()
                else:
                     logging.warning(f"Missing 'publishedAt' field for comment by {username}. Using current UTC time.")
                     comment_date = datetime.utcnow()

                comments_list.append({
                    "username": username,
                    "comment": comment_text,      # Text used for analysis
                    "display_comment": display_text, # Text potentially for display
                    "date": comment_date          # Datetime object
                })
                fetched_count += 1
            except KeyError as ke:
                 logging.warning(f"Skipping comment due to missing key: {ke} in item snippet: {item.get('snippet', {})}")
            except Exception as inner_e:
                 logging.warning(f"Skipping comment due to unexpected error during processing: {inner_e}", exc_info=True)


        # Handle pagination
        next_page_token = response.get('nextPageToken')
        if next_page_token and fetched_count < max_total_results:
            request = youtube.commentThreads().list_next(previous_request=request, previous_response=response)
        else:
            request = None # Stop fetching

    logging.info(f"Finished fetching. Total comments retrieved: {len(comments_list)}")
    return comments_list

def preprocess_text(text):
    """Preprocesses text for sentiment analysis."""
    if not isinstance(text, str): text = str(text) # Ensure string input
    text = text.lower()                            # Lowercase
    text = re.sub(r'http\S+|www\S+', '', text)     # Remove URLs/links
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = demojize(text, delimiters=(" ", " "))  # Convert emojis to text representations (e.g., :thumbs_up:)
    tokens = text.split()                          # Tokenize by whitespace
    stemmer = PorterStemmer()
    try:
        # Stemming can sometimes fail on weird inputs, though rare
        stemmed_tokens = [stemmer.stem(token) for token in tokens]
    except Exception as stem_e:
        logging.warning(f"Stemming error on token list: {tokens}. Error: {stem_e}. Returning original tokens.")
        stemmed_tokens = tokens
    processed_text = ' '.join(stemmed_tokens)      # Join back into string
    # logging.debug(f"Processed text: '{processed_text[:100]}...'") # Optional debug log
    return processed_text

def predict_sentiment(comments_data):
    """Predicts sentiment for a list of comment dictionaries."""
    if not model or not vectorizer:
        raise RuntimeError("Sentiment model or vectorizer not loaded. Cannot predict.")

    # Extract comment texts (using the 'comment' key which holds 'textOriginal')
    comment_texts = [item.get('comment', '') for item in comments_data]

    if not comment_texts:
        logging.info("No comment texts found to predict sentiment.")
        return comments_data # Return original data if no text

    logging.info(f"Preprocessing {len(comment_texts)} comments for sentiment analysis...")
    cleaned_comments = [preprocess_text(comment) for comment in comment_texts]

    logging.info("Transforming comments with vectorizer...")
    try:
        X_new = vectorizer.transform(cleaned_comments)
    except Exception as e:
        logging.error(f"Error transforming text with vectorizer: {e}", exc_info=True)
        raise RuntimeError(f"Vectorizer transformation failed: {e}")

    logging.info("Predicting sentiment with model...")
    try:
        predictions = model.predict(X_new)
        # probabilities = model.predict_proba(X_new) # Optional: get probabilities
    except Exception as e:
        logging.error(f"Error predicting sentiment with model: {e}", exc_info=True)
        raise RuntimeError(f"Model prediction failed: {e}")

    # Add predictions back to the original data list
    for i, item in enumerate(comments_data):
        # Assuming model predicts 1 for Positive, 0 for Negative
        item['sentiment'] = int(predictions[i]) # Ensure integer type

    logging.info("Sentiment prediction complete.")
    return comments_data # Return list of dicts with added 'sentiment' key


def create_plotly_chart(df, title, xaxis_title):
    """Creates a Plotly line chart HTML snippet for Positive Ratio."""
    div_id = f"plotly-{xaxis_title.lower().replace(' ', '-').replace('(', '').replace(')', '')}" # Create a valid div ID
    chart_wrapper_class = 'plotly-chart-container' # CSS class for the outer div

    if df.empty or 'ratio' not in df.columns or df['ratio'].isnull().all():
        logging.warning(f"No valid data to plot for '{title}'. DataFrame empty or 'ratio' column missing/all NaN.")
        return f"<div class='{chart_wrapper_class}' id='{div_id}-wrapper'><p>No data available to plot for {xaxis_title}.</p></div>"

    # Ensure the index is DatetimeIndex for time series plotting
    df_plot = df.copy()
    if not isinstance(df_plot.index, pd.DatetimeIndex):
        logging.warning(f"Plotting '{title}': Index is not DatetimeIndex. Attempting conversion if 'date' column exists.")
        if 'date' in df_plot.columns: # Check if 'date' column exists from a potential reset_index
             try:
                 df_plot['date'] = pd.to_datetime(df_plot['date'])
                 df_plot = df_plot.set_index('date')
                 logging.info("Successfully converted 'date' column to DatetimeIndex for plotting.")
             except Exception as e:
                 logging.error(f"Failed to convert 'date' column to datetime index for plot '{title}': {e}")
                 return f"<div class='{chart_wrapper_class}' id='{div_id}-wrapper'><p>Plotting error: Invalid date format.</p></div>"
        else: # Fallback if index isn't datetime and no suitable 'date' column
            logging.error(f"Cannot plot '{title}': Index is not DatetimeIndex and no 'date' column found.")
            return f"<div class='{chart_wrapper_class}' id='{div_id}-wrapper'><p>Plotting error: Date information missing or invalid.</p></div>"

    logging.info(f"Generating Plotly chart: '{title}'")
    try:
        fig = px.line(
            df_plot,
            y='ratio',      # Plot the calculated positive ratio
            title=title,
            labels={'index': xaxis_title, 'ratio': 'Positive Sentiment Ratio (%)'}, # Improved labels
            markers=True    # Show markers on data points
        )

        fig.update_layout(
            yaxis_range=[0, 100.5],  # Allow slightly above 100 for visual clarity if needed
            yaxis_ticksuffix="%",
            hovermode="x unified", # Good default for time series hover
            template="plotly_white", # Clean appearance
            margin=dict(l=50, r=30, t=60, b=50), # Adjust margins
            xaxis_title=xaxis_title, # Explicitly set axis titles
            yaxis_title='Positive Ratio (%)'
        )
        # Ensure x-axis date formatting is reasonable (Plotly often handles this well automatically)
        # fig.update_xaxes(dtick="M1", tickformat="%b\n%Y") # Example: Monthly ticks if needed

        # Convert figure to HTML div. include_plotlyjs='cdn' adds the <script> tag.
        plot_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn', div_id=div_id)

        # Return the HTML wrapped in our container div
        return f"<div class='{chart_wrapper_class}' id='{div_id}-wrapper'>{plot_html}</div>"

    except Exception as plot_err:
         logging.error(f"Error generating Plotly chart '{title}': {plot_err}", exc_info=True)
         return f"<div class='{chart_wrapper_class}' id='{div_id}-wrapper'><p>Error creating plot: {plot_err}</p></div>"

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    # Clear any previous results/errors if just loading the page
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles the analysis request."""
    video_url = request.form.get('video_url', '').strip() # Get URL and remove leading/trailing whitespace
    results = None
    error_message = None

    # --- Input Validation and Service Checks ---
    if not video_url:
        flash("Please enter a YouTube video URL.", "error")
        return redirect(url_for('index'))

    if not youtube:
        flash("YouTube service is not available. Please check the API Key in the server configuration.", "error")
        return redirect(url_for('index'))

    if not model or not vectorizer:
         flash("Sentiment analysis model is not available. Please check model files.", "error")
         return redirect(url_for('index'))

    try:
        # --- Core Analysis Steps ---
        video_id = extract_video_id(video_url)
        if not video_id:
            # Use ValueError for bad input format
            raise ValueError("Invalid or unsupported YouTube URL format. Could not find Video ID.")

        # 1. Fetch Comments
        logging.info(f"Attempting to fetch comments for video ID: {video_id}")
        # Limit fetch, e.g., max_total_results=500 for performance
        comments_data = get_comments(video_id, max_total_results=5000)

        if not comments_data:
            # Use info flash if comments simply aren't found vs an error
            flash(f"No comments were found for this video (ID: {video_id}), or comments are disabled.", "info")
            # Render index but show the URL that was tried
            return render_template('index.html', video_url=video_url)

        # 2. Predict Sentiment
        logging.info(f"Attempting sentiment prediction for {len(comments_data)} comments...")
        comments_with_sentiment = predict_sentiment(list(comments_data)) # Use a copy

        # 3. Calculate Statistics
        sentiments = [item.get('sentiment') for item in comments_with_sentiment if item.get('sentiment') is not None]
        total_analyzed = len(sentiments)
        if total_analyzed == 0:
             flash("Sentiment analysis could not be performed on the fetched comments.", "warning")
             return render_template('index.html', video_url=video_url)

        positive_comments_count = sum(s == 1 for s in sentiments)
        negative_comments_count = total_analyzed - positive_comments_count # Assumes binary 0 or 1
        positive_percentage = (positive_comments_count / total_analyzed * 100)
        negative_percentage = (negative_comments_count / total_analyzed * 100)

        # 4. Prepare Data & Generate Plotly Charts
        logging.info("Preparing data and generating plots...")
        df = pd.DataFrame(comments_with_sentiment)
        # Ensure 'date' column is datetime type before proceeding
        if 'date' not in df.columns or df['date'].isnull().all():
             raise ValueError("Date information is missing or invalid in fetched comments.")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']) # Remove rows where date conversion failed
        df = df.sort_values(by='date') # Essential for time series resampling

        # --- Daily Ratio Calculation ---
        df_daily = df.set_index('date').resample('D')['sentiment'].agg(
            positive_count=lambda x: (x == 1).sum(),
            total_count='count'
        ).dropna(subset=['total_count']) # Drop days with no comments after resampling
        df_daily['ratio'] = (df_daily['positive_count'] / df_daily['total_count'] * 100).fillna(0)

        # --- Monthly Ratio Calculation ---
        df_monthly = df.set_index('date').resample('ME')['sentiment'].agg( # 'ME' = Month End
             positive_count=lambda x: (x == 1).sum(),
             total_count='count'
        ).dropna(subset=['total_count'])
        df_monthly['ratio'] = (df_monthly['positive_count'] / df_monthly['total_count'] * 100).fillna(0)

        # Create Plotly HTML snippets using the helper function
        daily_plot_html = create_plotly_chart(df_daily, "Daily Positive Comment Ratio (%)", "Date (Day)")
        monthly_plot_html = create_plotly_chart(df_monthly, "Monthly Positive Comment Ratio (%)", "Date (Month)")

        # 5. Find Top Recent Comments
        logging.info("Finding top recent comments...")
        # df is already sorted by date ascending, so use tail() for recent OR sort descending and use head()
        df_sorted_recent = df.sort_values(by='date', ascending=False)

        top_positive = df_sorted_recent[df_sorted_recent['sentiment'] == 1].head(5).to_dict('records')
        top_negative = df_sorted_recent[df_sorted_recent['sentiment'] == 0].head(5).to_dict('records')

        # Format dates and prepare display text for the template
        for comment_dict in top_positive + top_negative:
            # Format date string
            if isinstance(comment_dict.get('date'), datetime):
                 comment_dict['date_str'] = comment_dict['date'].strftime('%Y-%m-%d %H:%M')
            else:
                 comment_dict['date_str'] = str(comment_dict.get('date', 'N/A')) # Fallback

            # Get display text (prefer 'display_comment' if available, else 'comment')
            comment_dict['display_text'] = comment_dict.get('display_comment', comment_dict.get('comment', 'Error: Text Missing'))


        # Prepare final results dictionary for the template
        results = {
            'total_comments': len(comments_data), # Total fetched
            'analyzed_comments': total_analyzed, # Total successfully analyzed
            'positive_count': positive_comments_count,
            'negative_count': negative_comments_count,
            'positive_percentage': round(positive_percentage, 2),
            'negative_percentage': round(negative_percentage, 2),
            'daily_plot_html': daily_plot_html,     # Plotly HTML string
            'monthly_plot_html': monthly_plot_html, # Plotly HTML string
            'top_positive': top_positive,           # List of dicts
            'top_negative': top_negative,           # List of dicts
            'video_url': video_url,                 # Pass URL back for context
            'video_id': video_id                    # Pass ID back for context
        }

    # --- Error Handling Block ---
    except ValueError as e: # Specific errors related to input or data validity
        logging.warning(f"Value Error during analysis: {e}")
        error_message = str(e)
    except ConnectionError as e: # Specific errors related to external services (YouTube API)
        logging.error(f"Connection Error during analysis: {e}")
        error_message = str(e)
    except RuntimeError as e: # Specific errors related to model/vectorizer execution
         logging.error(f"Runtime Error during analysis: {e}")
         error_message = f"Internal processing error: {e}"
    except Exception as e: # Catch-all for any other unexpected errors
        logging.exception("An unexpected error occurred during analysis.") # Log full traceback
        error_message = f"An unexpected server error occurred. Please try again later." # User-friendly message

    # --- Render Response ---
    if error_message:
        flash(error_message, "error")
        # Render index even on error, pass back the attempted URL
        return render_template('index.html', video_url=request.form.get('video_url'))
    else:
        # Successfully completed, render with results
        logging.info(f"Analysis successful for video ID: {video_id}")
        return render_template('index.html', results=results)


# --- Main Execution ---
if __name__ == '__main__':
    # Ensure necessary directories exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    # Run the Flask app
    # Set debug=False for production environment
    app.run(debug=True) # Listen on all interfaces if needed (e.g., for Docker)
    # app.run(debug=True) # Default: localhost only

# https://www.youtube.com/watch?v=yVTNge3sXpg&ab_channel=BeerBiceps
# https://www.youtube.com/watch?v=DyBi72AIKJQ&ab_channel=BeerBiceps
# https://www.youtube.com/watch?v=LHLaP7g1SaA