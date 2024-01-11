import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

nltk.download('vader_lexicon')

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-us,en;q=0.5'
}

# Initialize analyzers
flipkart_sid = SentimentIntensityAnalyzer()
snapdeal_sid = SentimentIntensityAnalyzer()
nykaa_sid = SentimentIntensityAnalyzer()

def sentiment_Vader(text, sid):
    if text.isdigit():
        rating = int(text)
        if rating >= 4:
            return "positive"
        elif rating >= 2:
            return "neutral"
        else:
            return "negative"
    else:
        over_all_polarity = sid.polarity_scores(text)
        print(f"Sentiment analysis result: {over_all_polarity}")
        if over_all_polarity['compound'] >= 0.05:
            return "positive"
        elif over_all_polarity['compound'] <= -0.05:
            return "negative"
        else:
            return "neutral"

def analyze_flipkart(url_base):
    all_reviews = []

    for i in range(1, 44):
        url = f"{url_base}{i}"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to retrieve page {i}. Status code: {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        review_divs = soup.find_all('div', {'class': 't-ZTKy'})
        reviews = [c.div.div.get_text(strip=True) for c in review_divs]
        all_reviews.extend(reviews)

    # Perform sentiment analysis on the collected reviews
    data = pd.DataFrame({'review': all_reviews})
    data['polarity'] = data['review'].apply(lambda review: sentiment_Vader(review, flipkart_sid))

    return generate_results(data)

def analyze_snapdeal(url_base):
    all_reviews = []

    for i in range(1, 44):
        url = f"{url_base}{i}"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to retrieve page {i}. Status code: {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        review_divs = soup.find_all('div', {'class': 'user-review'})
        reviews = [review_div.find("p").text for review_div in review_divs]
        all_reviews.extend(reviews)

    all_reviews = list(set(all_reviews))

    # Perform sentiment analysis on the collected reviews
    data = pd.DataFrame({'review': all_reviews})
    data['polarity'] = data['review'].apply(lambda review: sentiment_Vader(review, snapdeal_sid))

    return generate_results(data)

def analyze_nykaa(url_base):
    all_reviews = []

    for i in range(1, 44):
        url = f"{url_base}{i}"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Failed to retrieve page {i}. Status code: {response.status_code}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        review_divs = soup.find_all('div', {'class': 'css-ggpltb'})
        reviews = [review_div.find("h4").text for review_div in review_divs]
        all_reviews.extend(reviews)

    all_reviews = list(set(all_reviews))

    # Perform sentiment analysis on the collected reviews
    data = pd.DataFrame({'review': all_reviews})
    data['polarity'] = data['review'].apply(lambda review: sentiment_Vader(review, nykaa_sid))

    return generate_results(data)

def generate_results(data):
    # Display the count of each sentiment category
    sentiment_counts = data['polarity'].value_counts()

    # Ensure all sentiments are included in the pie chart
    all_sentiments = ['positive', 'neutral', 'negative']
    sentiment_counts = sentiment_counts.reindex(all_sentiments, fill_value=0)

    # Create a pie chart
    custom_colors = ['#21de74', '#3053cf', '#f00f1f']  # Green, Blue, Red
    plt.figure(figsize=(10, 10))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140, colors=custom_colors)
    plt.title('Sentiment Analysis Results')

    # Save the plot to a BytesIO object
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    # Encode the plot to base64 for displaying in HTML
    plot_url = base64.b64encode(img.getvalue()).decode()

    all_reviews_text = ' '.join(data['review'])
    wordcloud = WordCloud(width=500, height=300, background_color='white').generate(all_reviews_text)

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save the word cloud to a BytesIO object
    wordcloud_img = BytesIO()
    plt.savefig(wordcloud_img, format='png')
    wordcloud_img.seek(0)

    # Encode the word cloud to base64 for displaying in HTML
    wordcloud_url = base64.b64encode(wordcloud_img.getvalue()).decode()

    return sentiment_counts, plot_url, wordcloud_url

def main():
    st.title("Sentiment Analysis App")

    selected_analyzer = st.radio("Select an analyzer", ('Flipkart', 'Snapdeal', 'Nykaa'))

    url_base = st.text_input("Enter the URL base:")
