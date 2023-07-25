import dash
import re
import datetime
import dash_bootstrap_components as dbc
from langchain.embeddings.openai import OpenAIEmbeddings
import numpy as np
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import requests
import openai
import plotly.express as px
import pandas as pd 
from bs4 import BeautifulSoup
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import DBSCAN


# Set up the app
app = dash.Dash(__name__)
app.title = 'FilterNews'
# OpenAI API credentials
#api = OpenAIMultiClient(endpoint="chats", data_template={"model": "gpt-3.5-turbo"})
openai.api_key = "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"
openai.organization = "org-BhRo4Lrkq90rSVIXVc82DeqN"
# Emotion options
emotion_options = [
    {"label": "Joy", "value": "joy"},
    {"label": "Anger", "value": "anger"},
    {"label": "Disgust", "value": "disgust"},
    {"label": "Fear", "value": "fear"},
    {"label": "Neutral", "value": "neutral"},
    {"label": "Sadness", "value": "sadness"},
    {"label": "Surprise", "value": "surprise"},
]


# App layout
app.layout = html.Div(style={"background-color": "#6EF8D8", 'font-family': 'Lucida Handwriting, cursive','padding': '300px'},
    children=[dcc.Tabs(id="tabs",
            value="tab-sentiment",
            children=[dcc.Tab(
                    label="Emotion filtering",
                    value="tab-sentiment",
                    children=[
        
                html.H1("Filter news by emotions!"),
                html.Div(
                    children=[
                        html.H3("Enter a news topic:"),
                        dcc.Input(
                            id="input-topic-1",
                            type="text",
                            placeholder="Enter a news topic",
                        ),
                        html.H3("Select time range:"),
                        dcc.DatePickerRange(
                            id="date-picker-range-1",
                            start_date_placeholder_text="Start Date",
                            end_date_placeholder_text="End Date",
                            display_format="YYYY-MM-DD",
                        ),
                        html.H3("Select emotion category:"),
                        dcc.Dropdown(
                            id="emotion-dropdown",
                            options=emotion_options,
                            value="joy",
                            searchable=False,
                            clearable=False,
                        ),
                        html.Button("Search", id="search-button-1", n_clicks=0,style={"font-size": "20px",
        "padding": "10px 20px","background-color": "#F4C2C2","border-radius": "5px","margin-top":"10px"}),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Div(id="output-container-1"),
            ],
            style={"max-width": "600px", "margin": "0 auto"},
        )
    ,
    dcc.Tab(
                    label="Event filtering",
                    value="tab-clustering",
                    children=[
                        html.H1("Filter news by events!"),
                        
                html.Div(
                    children=[
                        html.H3("Enter a news topic:"),
                        dcc.Input(
                            id="input-topic-2",
                            type="text",
                            placeholder="Enter a news topic",
                        ),
                        html.H3("Select time range:"),
                        dcc.DatePickerRange(
                            id="date-picker-range-2",
                            start_date_placeholder_text="Start Date",
                            end_date_placeholder_text="End Date",
                            display_format="YYYY-MM-DD",
                        ),
                        html.Button("Search", id="search-button-2", n_clicks=0,style={"font-size": "20px",
        "padding": "10px 20px","background-color": "#F4C2C2","border-radius": "5px","margin-left":"10px"}),
                    ],
                    style={"margin-bottom": "20px"},
                ),
                html.Div(id="output-container-2"),
                    ], style={"max-width": "600px", "margin": "0 auto"},
                ),
            ],
        ),
    ]
)
                    
 

def split_text_into_chunks(text, chunk_size, overlap):
    tokens = text.split()
    total_tokens = len(tokens)
    chunks = []
    
    start_idx = 0
    while start_idx < total_tokens:
        end_idx = min(start_idx + chunk_size, total_tokens)
        chunk = tokens[start_idx:end_idx]
        chunks.append(' '.join(chunk))
        start_idx += chunk_size - overlap
    
    return chunks

def generate_radar_chart(probabilities):
    df = pd.DataFrame(dict(
    r=probabilities,
    theta=['Joy', 'Anger', 'Disgust', 'Fear', 'Neutral', 'Sadness', 'Surprise']))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
        width=450,  # Adjust the width
        height=350,plot_bgcolor="#6EF8D8")  # Adjust the height)
   
    return fig


def perform_sentiment_analysis(text):
   
    prompt = "This is a sentiment analysis task. You will give a score distribution in decimals that sum up to 1 of the following text across all of 7 sentiment categories: joy, anger, disgust, fear, neutral, sadness, or surprise in this format Joy: 0.3,Anger: 0.1,Disgust: 0.1,Fear: 0.1,Neutral: 0.1,Sadness: 0.1, Surprise: 0.2"
    chunk_size = 2800
    overlap = 0  # No overlap in this example

    chunks = split_text_into_chunks(text, chunk_size, overlap)
    
    emotion_scores = []
    for chunk in chunks:
        chunk_prompt = f"{prompt}\n\n{chunk}"

        response = openai.ChatCompletion.create(
           model="gpt-3.5-turbo",
        messages=[
            { "role": "user", "content": chunk_prompt}
        ],
        temperature=0.0,
        )
        #print(response)
        # Extract the emotion probabilities from the OpenAI API response
        emotions = response.choices[0].message.content.strip()
        #print(emotions)
        emotion_probabilities = re.findall(r"\b\d+(\.\d+)?\b", emotions)

        emotion_scores.append([float(probability) for probability in emotion_probabilities])
        
    #print(emotion_scores)
    # Calculate the average score for each emotion label across all chunks
    #print(json.dumps(emotion_scores, indent=2))
    if emotion_scores:
     average_scores = [sum(scores[i] for scores in emotion_scores) / len(emotion_scores) for i in range(len(emotion_scores[0]))]

    # Define the emotion labels
     emotion_labels = ['joy', 'anger', 'disgust', 'fear', 'neutral', 'sadness', 'surprise']

    # Find the label with the highest average score
     max_label = emotion_labels[average_scores.index(max(average_scores))]
     print("analysis done")
     return max_label,average_scores
    else:
        return None 


def scrape_article_content(url):
    try:
     response = requests.get(url)
     if response.status_code == 200:
        # Assuming articles are in HTML format
        # You may need to adjust the code for different article formats
        # Here, we use a simple method by considering all text within paragraph tags
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text() for p in paragraphs])
        return content
     else:
        return None
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")
        return None


@app.callback(
    Output("output-container-1", "children"),
    [Input("search-button-1", "n_clicks")],
    [
        dash.dependencies.State("input-topic-1", "value"),
        dash.dependencies.State("date-picker-range-1", "start_date"),
        dash.dependencies.State("date-picker-range-1", "end_date"),
        dash.dependencies.State("emotion-dropdown", "value"),
    ],
)


def search_news_and_analyze_(n_clicks, topic, start_date, end_date, emotion_category):
      print(f"{n_clicks=} {topic=} {start_date=} {end_date=} {emotion_category=}")
      
      if n_clicks > 0 and topic and start_date and end_date and emotion_category:
        # Make API request to retrieve news articles
        api_key = "7a1ee67d2e3f42588dce5d99f99bcfa4"
        url = 'https://newsapi.org/v2/everything'
        params = {
        'q': topic,
       'apiKey': api_key,
       'from': start_date,
        'to': end_date,'language': 'en'
         }
        response = requests.get(url,params=params)
        print("a")
        data = response.json()
        #print(data)

        # Process the response data
        articles = data.get("articles", [])
        if articles:
            urls = [article["url"] for article in articles]
            contents=[]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for result in executor.map(scrape_article_content,urls):
                    contents.append(result)    
            results = []
            for i in range(len(contents)):
                print(contents[i])
            for i in range(len(contents)):
                if contents[i]==None:
                     print("Scraping failed; skipping to next article")
                     continue

                #print(content)

                if contents[i]:
                    # Perform sentiment analysis
                    sentiment_category = perform_sentiment_analysis(contents[i])
                    if sentiment_category[0].lower() == emotion_category.lower():
                        radar_chart = generate_radar_chart(sentiment_category[1])
                        print("something found")
                        results.append(
                            html.Div(
                                className="article-box",
                                children=[
                                    html.H4(articles[i]["title"]),
                                    html.P(html.A(articles[i]["url"], href=articles[i]["url"], target="_blank")),
                                    dcc.Graph(figure=radar_chart),
                                ],
                                style={
                                    "border": "1px solid black",
                                    "padding": "10px",
                                    "margin-bottom": "10px",'font-family':'Arial,serif',"overflow": "hidden"
                                },
                            )
                        )
        else:
            results = html.Div("No articles found.")

        return results

      return ""


@app.callback(
    Output("output-container-2", "children"),
    [Input("search-button-2", "n_clicks")],
    [
        dash.dependencies.State("input-topic-2", "value"),
        dash.dependencies.State("date-picker-range-2", "start_date"),
        dash.dependencies.State("date-picker-range-2", "end_date"),
    ],
)

def search_news_and_cluster(n_clicks, topic, start_date, end_date):
   if n_clicks > 0 and topic and start_date and end_date :
        # Make API request to retrieve news articles
        api_key = "7a1ee67d2e3f42588dce5d99f99bcfa4"
        url = 'https://newsapi.org/v2/everything'
        params = {
        'q': topic,
       'apiKey': api_key,
       'from': start_date,
        'to': end_date,'language': 'en'
         }
        response = requests.get(url,params=params)
        data = response.json()

        articles = data.get("articles", [])
        if articles: 
            df = []
            for article in articles:
                df.append(
                            {
                                "title": article["title"],
                                "url": article["url"],
                                "date": article["publishedAt"],"description":article["description"]
                            }
                        )
            df = pd.DataFrame(df)
            df=df.dropna()
            df=df.drop_duplicates(subset='title').reset_index(drop=True)
            docs = [
                row["title"] + "\n\n" + row["description"]
                for _,row in df.iterrows()]
            embeddings = OpenAIEmbeddings(openai_api_key="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",chunk_size=1000).embed_documents(docs)
            dbscan = DBSCAN(eps=0.1, min_samples=2,metric='cosine').fit(embeddings)
            df["cluster_label"] =dbscan.labels_ 

            results = []
            for label in df["cluster_label"].unique():
                cluster_df = df[df["cluster_label"] == label]
                # Sort articles by publication date
                cluster_df = cluster_df.sort_values("date")
                articles_str = "\n".join([
                f"{article['title']}\n{article['description']}\n"
                 for  _,article in cluster_df.iterrows()])
                x="Using the following articles, write a topic title that summarizes them without the Title:"
          
                response = openai.ChatCompletion.create(
           model="gpt-3.5-turbo",
        messages=[
            { "role": "user", "content": x+articles_str}
        ],
        temperature=0.0,
        )
                title = response.choices[0].message.content.strip()

                articles = []
                for _, row in cluster_df.iterrows():
                    articles.append(
                        html.Div(
                            className="article-box",
                            children=[
                                html.P(datetime.datetime.strptime(row['date'], "%Y-%m-%dT%H:%M:%SZ").date()),
                                html.H4(row["title"]),
                                html.P(html.A(row["url"], href=row["url"], target="_blank")),
                                
                            ],
                            style={
    
                                "padding": "5px",
                                "margin-bottom": "5px",'font-family':'Arial,serif'
                            },
                        )
                    )

                results.append(
                    html.Div(
                        className="cluster-box",
                        children=[
                            html.H3(title),
                            *articles,
                        ],
                        style={"margin-bottom": "20px","margin-right": "20px", "max-width": "400px","display": "inline-block",
                    "vertical-align": "top",'font-family':'Arial,serif',"border": "2px solid black","overflow": "hidden"},
                    )
                )

            return results
   
if __name__ == "__main__":
    app.run_server(debug=True)



