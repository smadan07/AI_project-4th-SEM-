import re
import string
from emoji import demojize
from nltk.stem import PorterStemmer
from textblob import TextBlob
import pandas as pd
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Replace common chat shortcuts with full words
    # Repository Link : https://github.com/rishabhverma17/sms_slang_translator/blob/master/slang.txt
    chat_words = {
        "AFAIK": "as far as i know",
        "AFK": "away from keyboard",
        "ASAP": "as soon as possible",
        "ATK": "at the keyboard",
        "ATM": "at the moment",
        "A3": "anytime anywhere anyplace",
        "BAK": "back at keyboard",
        "BBL": "be back later",
        "BBS": "be back soon",
        "BFN": "bye for now",
        "B4N": "bye for now",
        "BRB": "be right back",
        "BRT": "be right there",
        "BTW": "by the way",
        "B4": "before",
        "CU": "see you",
        "CUL8R": "see you later",
        "CYA": "see you",
        "FAQ": "frequently asked questions",
        "FC": "fingers crossed",
        "FWIW": "for what its worth",
        "FYI": "for your information",
        "GAL": "get a life",
        "GG": "good game",
        "GN": "good night",
        "GMTA": "great minds think alike",
        "GR8": "great",
        "G9": "genius",
        "IC": "i see",
        "ICQ": "i seek you",
        "ILU": "i love you",
        "IMHO": "in my honest opinion",
        "IMO": "in my opinion",
        "IOW": "in other words",
        "IRL": "in real life",
        "KISS": "keep it simple stupid",
        "LDR": "long distance relationship",
        "LMAO": "laughing my a off",
        "LOL": "laughing out loud",
        "LTNS": "long time no see",
        "L8R": "later",
        "MTE": "my thoughts exactly",
        "M8": "mate",
        "NRN": "no reply necessary",
        "OIC": "oh i see",
        "PITA": "pain in the a",
        "PRT": "party",
        "PRW": "parents are watching",
        "QPSA?": "que pasa?",
        "ROFL": "rolling on the floor laughing",
        "ROFLOL": "rolling on the floor laughing out loud",
        "ROTFLMAO": "rolling on the floor laughing my a off",
        "SK8": "skate",
        "STATS": "your sex and age",
        "ASL": "age sex location",
        "THX": "thank you",
        "TTFN": "ta-ta for now",
        "TTYL": "talk to you later",
        "U": "you",
        "U2": "you too",
        "U4E": "yours for ever",
        "WB": "welcome back",
        "WTF": "what the f",
        "WTG": "way to go",
        "WUF": "where are you from",
        "W8": "wait",
        "7K": "sick laughing",
        "TFW": "that feeling when",
        "MFW": "my face when",
        "MRW": "my reaction when",
        "IFYP": "i feel your pain",
        "TNTL": "trying not to laugh",
        "JK": "just kidding",
        "IDC": "i don't care",
        "ILY": "i love you",
        "IMU": "i miss you",
        "ADIH": "another day in hell",
        "ZZZ": "sleeping bored tired",
        "WYWH": "wish you were here",
        "TIME": "tears in my eyes",
        "BAE": "before anyone else",
        "FIMH": "forever in my heart",
        "BSAAW": "big smile and a wink",
        "BWL": "bursting with laughter",
        "BFF": "best friends forever",
        "CSL": "can't stop laughing",
        "ur": "your",
        "gr8": "great",
        "b4": "before",
        "l8r": "later",
        "idk": "i don't know",
        "omg": "oh my god",
        "btw": "by the way",
        "thx": "thanks",
        "pls": "please",
        "lol": "laugh out loud"
    }
    
    # Replace chat shortcuts
    words = text.split()
    words = [chat_words.get(word, word) for word in words]
    text = ' '.join(words)
    
    # Convert emoji to text descriptions
    text = demojize(text)
    
    # Correct spelling errors using TextBlob
    text = str(TextBlob(text).correct())
    
    # Split text into tokens using whitespace
    tokens = text.split()
    
    # Initialize the stemmer and apply stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back into a single string
    processed_text = ' '.join(stemmed_tokens)
    return processed_text

# Example usage:
sample_text = "OMG, I loveing this video! Check out http://example.com 😊 lol"


df=pd.read_csv('D:\AI Project\Twitter_sentiment_analysis_dataset.csv')
print(df.head)