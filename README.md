# FilterNews-app
 FilterNews - A News Filtering and Clustering Web Application\
Introduction\
 The FilterNews project is a web application that leverages Natural Language Processing (NLP) tools to filter and cluster newspaper articles. The goal is to allow users to easily find and read news articles that match their emotions and interests, as well as to group related articles based on events.
Overview\
The FilterNews web application offers two main features:\
Emotion-based News Filtering: Users can enter a query topic and a time range for the articles' publication date. The application then filters the news articles based on the user's desired emotion category. The emotions considered are: joy, anger, sadness, fear, neutral, disgust, and surprise. The emotion classification is performed using the OpenAI GPT-3.5-turbo model.\
Event-based News Clustering: Users can again input a query topic and a time range, but this time the application clusters the news articles based on distinct events. The clustering is done using the DBSCAN algorithm, with OpenAI embeddings for titles and descriptions.
Emotion-based News Filtering\
Initially, I experimented with HuggingFace's Distilroberta-base pretrained emotion classification model. However, due to its slow performance for a large number of articles, I decided to use the OpenAI API with the GPT-3.5-turbo model, a large language model (LLM) with good capabilities for sentiment analysis.
Backend Pipeline\
1). Querying Articles: The News API is used to fetch articles' information based on the user's entered query topic and time range. The results are organized into a dataframe.\
2). Web Scraping: BeautifulSoup is utilized to extract the text content of the articles using their URLs. Threading is implemented to scrape the contents concurrently for faster processing.\
3). Emotion Classification: The GPT-3.5-turbo model performs emotion classification for the seven basic emotions on each article's content. A challenge arose due to the model's token limit of 4096 tokens, which some articles' content exceeded. To address this, the content was divided into chunks of approximately 3000 tokens, and a score distribution of scores that sum up to 1 across all emotion categories was requested for each chunk. For example, the format of scores would be as follows: Joy: 0.3, Anger: 0.1, Disgust: 0.1, Fear: 0.1, Neutral: 0.1, Sadness: 0.1, Surprise: 0.2. The scores were then averaged across all chunks to predict the emotion category for the entire text based on which had the highest score. Regular expressions (Regex) were used to extract scores from the response in cases where the format differed. Additionally, an "overlap" of chunks argument was introduced that can be set to greater than 0 so as to maintain context continuity between adjacent chunks for a more accurate sentiment analysis. \
4). Display Results: Articles matching the selected emotion category are presented in separate boxes, along with a radar chart illustrating the score distribution across all emotion categories. The process takes approximately 3 minutes to complete.\
Threading and RateLimitError\
I tried optimizing the emotion classification process using threading to handle multiple API requests simultaneously. However, this approach resulted in encountering RateLimitError for certain queries, as sending too many API requests at once triggered rate limiting. Thus, the threading approach was not used to ensure stable performance.\
Event-based News Clustering\
Initially, named entity extraction in combination with Spacy's similarity feature were considered for clustering articles based on events. However, this approach didn't capture semantic similarity effectively, so OpenAI embeddings were used for titles and descriptions instead. The DBSCAN clustering algorithm, utilizing cosine similarity, was preferred over KMeans for its ability to determine the number and size of clusters automatically.\
Clustering Process\
1). Embedding Generation: OpenAI embeddings are generated for the titles and descriptions of the articles to capture semantic similarity between texts.\
2). DBSCAN Clustering: The DBSCAN algorithm clusters the articles based on their embeddings, with the epsilon parameter set to determine the maximum distance between sample vectors assigned to the same cluster. Tuning the epsilon parameter around 0.1 yielded the most accurate results.\
3). Display Results: Each cluster is presented in a separate box, with articles sorted chronologically to provide a narrative of the events within the event topic. The OpenAI API is finally used to generate suitable titles for each cluster.\
Frontend\
The web application's frontend is developed using Python's Plotly Dash library, offering two tabs for the two tools. \
Conclusion\
The FilterNews web application provides a user-friendly and efficient way to filter news articles based on what they feel like reading that day and to cluster related articles based on events. The project's potential applications range from personalized news recommendations to event detection and monitoring real-time news trends. With further enhancements and scalability, FilterNews could become a valuable tool for news consumers seeking a tailored and organized news browsing experience. 

