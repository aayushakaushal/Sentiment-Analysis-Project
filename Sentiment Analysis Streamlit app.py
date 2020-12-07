import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import streamlit as st

# Scraping mouthshut.com data
@st.cache(allow_output_mutation=True)
def mouthshut():
    URL = ""
    Final = []
    for x in range(0, 8):
        if x == 1:
            URL = "https://www.mouthshut.com/product-reviews/Kotak-811-Mobile-Banking-reviews-925917218"

        else:
            URL = "https://www.mouthshut.com/product-reviews/Kotak-811-Mobile-Banking-reviews-925917218-page-{}".format(
                x)

        r = requests.get(URL)
        soup = BeautifulSoup(r.content, 'html.parser')
        reviews = []  # a list to store reviews

        # Use a CSS selector to extract all the review containers
        review_divs = soup.select('div.col-10.review')
        for element in review_divs:
            review = {'Review_Title': element.a.text, 'URL': element.a['href'],
                      'Review': element.find('div', {'class': ['more', 'reviewdata']}).text.strip(),
                      'Stars': len(element.find('div', "rating").findAll("i", "rated-star"))}
            reviews.append(review)

        Final.extend(reviews)

    df = pd.DataFrame(Final)
    return df

# Scraping bankbazaar.com data
@st.cache(allow_output_mutation=True)
def bankbazaar():
    URL = ""
    Final = []
    for x in range(0, 2):
        if x == 0:
            URL = "https://www.bankbazaar.com/reviews/kotak-mahindra-bank/debit-card.html"

        else:
            URL = "https://www.bankbazaar.com/reviews/kotak-mahindra-bank/debit-card.html?reviewPageNumber=2"

        r = requests.get(URL)
        soup = BeautifulSoup(r.content, 'html.parser')
        reviews = []  # a list to store reviews

        # Use a CSS selector to extract all the review containers
        review_divs = soup.select('li.review-box')
        for element in review_divs:
            review = {'Review_Title': element.a.text, 'URL': element.a['href'],
                      'Review': element.find('div', {'class': ['text_here', 'review-desc-more']}).text.strip(),
                      'Stars': element.input['value']}
            reviews.append(review)

        Final.extend(reviews)

    df = pd.DataFrame(Final)
    return df

# Scraping creditkaro.com data
@st.cache(allow_output_mutation=True)
def creditkaro():
    Final = []
    URL = "https://www.creditkaro.com/savings-account/kotak-811-savings-account"

    r = requests.get(URL)
    soup = BeautifulSoup(r.content, 'html.parser')
    reviews = []  # a list to store reviews

    # Use a CSS selector to extract all the review containers
    review_divs = soup.select('div.col-lg-3.col-md-3.col-sm-12.col-xs-12')
    for element in review_divs:
        review = {'Review_Title': 'Creditkaro review', 'URL': ' ',
                  'Review': element.p.text,
                  'Stars': len(element.find('div', "listing-rating card-popup-rainingvis").findAll("i", "fa fa-star"))}
        reviews.append(review)

    Final.extend(reviews)

    df = pd.DataFrame(Final)
    return df

# Scraping appgrooves.com data
@st.cache(allow_output_mutation=True)
def appgrooves():
    URL = ""
    Final = []
    for x in range(2):
        if x == 0:
            URL = "https://appgrooves.com/app/kotak-811-and-mobile-banking-by-kotak-mahindra-bank-ltd/positive"
            ref = '[id^="positive_mosthelpful_reviews"]'
        else:
            URL = "https://appgrooves.com/app/kotak-811-and-mobile-banking-by-kotak-mahindra-bank-ltd/negative"
            ref = '[id^="negative_mosthelpful_reviews"]'

        soup = BeautifulSoup(requests.get(URL).content, 'html.parser')

        all_data = []
        while True:
            for r in soup.select(ref):
                rating = r.select_one('.rating-stars')['class'][-1].split('-')[-1]
                txt = r.select_one('[id^="review-item-body"]').get_text(strip=True)
                title = r.select_one('.title').get_text(strip=True)
                all_data.append({'Review_Title': title, 'URL': ' ',
                                 'Review': txt, 'Stars': rating})

            btn = soup.select_one('button:contains("Show More Positive Reviews")')
            if not btn:
                break

            data = requests.get('https://appgrooves.com' + btn['data-url-ajax'],
                                headers={'X-Requested-With': 'XMLHttpRequest'}).json()
            soup = BeautifulSoup(data['data']['preview_html'], 'lxml')

        Final.extend(all_data)

    df = pd.DataFrame(Final)
    return df


# Function to do sentiment analysis, generate wordcloud, distribution of compound values, and percentage of positive, negative and neutral reviews

def analyse(df):
    from nltk.corpus import wordnet

    def get_wordnet_pos(pos_tag):
        if pos_tag.startswith('J'):
            return wordnet.ADJ
        elif pos_tag.startswith('V'):
            return wordnet.VERB
        elif pos_tag.startswith('N'):
            return wordnet.NOUN
        elif pos_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    import string
    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    def clean_text(text):
        # lower text
        text = text.lower()
        # tokenize text and remove puncutation
        text = [word.strip(string.punctuation) for word in text.split(" ")]
        # remove words that contain numbers
        text = [word for word in text if not any(c.isdigit() for c in word)]
        # remove stop words
        stop = stopwords.words('english')
        text = [x for x in text if x not in stop]
        # remove empty tokens
        text = [t for t in text if len(t) > 0]
        # pos tag text
        pos_tags = pos_tag(text)
        # lemmatize text
        text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
        # remove words with only one letter
        text = [t for t in text if len(t) > 1]
        # join all
        text = " ".join(text)
        return (text)

    # clean text data
    df["Review"] = df["Review"].apply(lambda x: clean_text(x))

    # add sentiment analysis columns
    sia = SentimentIntensityAnalyzer()

    df["sentiments"] = df["Review"].apply(lambda x: sia.polarity_scores(x))
    df = pd.concat([df.drop(['sentiments'], axis=1), df['sentiments'].apply(pd.Series)], axis=1)

    # add number of characters column
    df["nb_chars"] = df["Review"].apply(lambda x: len(x))

    # add number of words column
    df["nb_words"] = df["Review"].apply(lambda x: len(x.split(" ")))


    # wordcloud function
    def show_wordcloud(data, title=None):
        wordcloud = WordCloud(
            background_color='white',
            max_words=200,
            max_font_size=40,
            scale=3,
            random_state=42
        ).generate(str(data))

        fig = plt.figure(1, figsize=(20, 20))
        plt.axis('off')
        if title:
            fig.suptitle(title, fontsize=20)
            fig.subplots_adjust(top=2.3)

        plt.imshow(wordcloud)
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    # print wordcloud
    st.subheader("WordCloud of all the reviews")
    st.text("Most words used in the reviews analysed")
    show_wordcloud(df["Review"])

        # Compound value of positive, negative and neutral values
    df['compound'] = df['Review'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # distribution of compound values from -1 to 1
    plt.hist(df['compound'], color='blue', edgecolor='black',
             bins=[-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # Add labels
    st.subheader('Distribution of Compound values')
    st.text("Number of values (0 to 50) vs Range (-1 to 1)")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


    # positive, negative, neutral concentration %
    pos_review = [j for i, j in enumerate(df['Review']) if df['compound'][i] > 0.2]
    neu_review = [j for i, j in enumerate(df['Review']) if 0.2 >= df['compound'][i] >= -0.2]
    neg_review = [j for i, j in enumerate(df['Review']) if df['compound'][i] < -0.2]

    positive = format(len(pos_review) * 100 / len(df['Review']))
    neutral = format(len(neu_review) * 100 / len(df['Review']))
    negative = format(len(neg_review) * 100 / len(df['Review']))

    st.subheader("Percentage of Positive, Neutral and Negative Reviews in: " + str(len(df.index)) + " Reviews")
    #piechart of the share of positive, neutral and negative reviews
    labels = 'Neutral', 'Negative', 'Positive'
    sizes = [neutral, negative, positive]
    separated = (0, 0.1, 0)
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', explode=separated)
    plt.axis('equal')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


    st.text("Percentage of Positive (Green section in the piechart) Reviews: " + positive + "%")
    st.text("Percentage of Neutral (Blue section in the piechart) Reviews: " + neutral + "%")
    st.text("Percentage of Negative (Orange section in the piechart) Reviews: " + negative + "%")


# streamlit header
st.title("Kotak 811 app Review Analysis")
df1 = mouthshut()
df2 = bankbazaar()
df3 = creditkaro()
df4 = appgrooves()
df = pd.concat([df1, df2, df3, df4], ignore_index=True, sort=False)

a = st.sidebar.radio("Website Options:", ("Mouthshut", "BankBazaar", "Creditkaro", "Appgrooves", "All Websites Combined"))
if a == "Mouthshut":
    st.header("Mouthshut Analysis")
    st.text("Number reviews used for this analysis: " + str(len(df1.index)))
    analyse(df1)
elif a == "BankBazaar":
    st.header("Bankbazaar Analysis")
    st.text("Number reviews used for this analysis: " + str(len(df2.index)))
    analyse(df2)
elif a == "Creditkaro":
    st.header("Creditkaro Analysis")
    st.text("Number reviews used for this analysis: " + str(len(df3.index)))
    analyse(df3)
elif a == "Appgrooves":
    st.header("Appgrooves Analysis")
    st.text("Number reviews used for this analysis: " + str(len(df4.index)))
    analyse(df4)
elif a == "All Websites Combined":
    st.header("All the websites combined")
    st.text("Number reviews used for this analysis: " + str(len(df.index)))
    analyse(df)