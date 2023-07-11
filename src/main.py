import os
import sys
import json
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic

from utils import S3Connector, clean_dataframe, filter_headlines, check_dir
from sentiment_utils import BertSentiment, vader_sentiment, analyze_topic_sentiment, analyze_df_sentiment


def analyze_news(sentiment_model_type, sentiment_model_name, data_dir, src_data_dir, target, **config):
    """
    Function to run the data analysis on the "news" dataset. Here the dataset is split into a brand 
    specific subset by querying the headlines, e.g., "southwest airlines", and a competitors subset 
    by querying the rest of the headlines, e.g., "airlines". The following processing steps occur:

        1. Create brand-specific and competitors datasets
        2. Train a UMAP/HDBscan model for topics for each dataset
        3. Measure the sentiment for each topic in each dataset

    This produces two models which can be analyzed to find the most appropriate negative and positive
    topics in the news for the brand, and for the market in general. Based on a qualitative assessment
    of the topic model's performance, it can be applied to future data. 

    Parameters:
        sentiment_model_type (str): Type of sentiment model, vader, or BERT.
        sentiment_model_name (str): Name of Huggingface model if BERT based sentiment
        src_data_dir (str): Local path to dir with dataset parquet files. 
        target (str): Name of the brand to partition data, e.g., "southwest airlines"    
    """

    # load parquet to df and perform some cleaning on it
    #   Note: placeholder for more rigorous cleaning
    data_paths = os.path.join(data_dir, src_data_dir)
    input_df = pd.read_parquet(data_paths)
    cleaned_df = clean_dataframe(input_df, ["body", "headline"])

    # generate comparison datasets: headlines containing "southwest airlines"
    # and exclusive set of headlines containing "airlines"
    #   Note: string case not considered
    on_brand_df, competitor_df = filter_headlines(target, cleaned_df, "headline")
    on_brand_text = list(on_brand_df['body'])
    competitor_text = list(competitor_df['body'])

    # cluster with BerTopic model using CountVectorizer instead of an embedding model
    # EDA suggested this is a better approach
    # Heuristics for stop words
    stop_words = list(text.ENGLISH_STOP_WORDS.union(["southwest airlines", "southwest", "airlines", "flight", "airline"]))
    vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words=stop_words)

    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        language='english', calculate_probabilities=True,
        verbose=True
    )    

    # Running Huggingface Sentiment model for 1000s of samples (running multiple times
    # if tokens in text exceed 512) was slow on my local machine. Batching would speed up. 
    # Reverting to Vader
    if sentiment_model_type == "BERT":
        sentiment_model = BertSentiment(sentiment_model_name)
        sentiment_model = sentiment_model.infer
    
    if sentiment_model_type == "Vader":
        sentiment_model = vader_sentiment

    on_brand_df, brand_topic_df, brand_topic_model = analyze_df_sentiment(
                                                        sentiment_model, topic_model, on_brand_text, on_brand_df)
    competitor_df, competitor_topic_df, competitor_topic_model = analyze_df_sentiment(
                                                        sentiment_model, topic_model, competitor_text, competitor_df)


    # save model to location
    # Note: should come from config file
    brand_model_dir = "models/news_analysis/on_brand/"
    competitor_model_dir = "models/news_analysis/competitors/"
    check_dir(brand_model_dir)
    check_dir(competitor_model_dir)

    brand_topic_model.save(os.path.join(brand_model_dir, "brand_model.pkl"), serialization="pickle")
    brand_topic_df.to_csv(os.path.join(brand_model_dir, "brand_model_results.csv"))

    # model metadata, including information about source data, processing methods, model methods, etc
    # should be saved as a model card accompanying the model
    competitor_topic_model.save(os.path.join(competitor_model_dir, "competitor_model.pkl"), serialization="pickle")
    competitor_topic_df.to_csv(os.path.join(competitor_model_dir, "competitor_model_results.csv"))

    # competitor_df, on_brand_df contain the original data with predicted topics and predicted sentiment
    # which can be saved to an appropriate s3 location

def analyze_social(sentiment_model_type, sentiment_model_name, data_dir, src_data_dir, target, **config):
    """
    Function to run the data analysis on the "social" dataset. Here the socials data is first filtered
    for social interactions relevant to the brand. An analysis of the social media data is performed
    as follows:

        1. Filter social media posts containing the brand in their body
        2. Extract the 1000 most impactful (by sorting for audience views)
        3. Measure the sentiment for each post
        4. Split the top k posts by sentiment into "positive", "neutral", and "negative"
        5. Topic model each of the sentiment-based data splits

    This produces 3 models which can be analyzed to look at what are the most impactful positive and negative
    topics currently being discussed. 

    Parameters:
        sentiment_model_type (str): Type of sentiment model, vader, or BERT.
        sentiment_model_name (str): Name of Huggingface model if BERT based sentiment
        src_data_dir (str): Local path to dir with dataset parquet files. 
        target (str): Name of the brand to partition data, e.g., "southwest airlines"    
        **top_k (int): number of posts to process when sorted for audience views 
    """    
    # load parquet to df and perform some cleaning on it
    #   Note: placeholder for more rigorous cleaning
    data_paths = os.path.join(data_dir, src_data_dir)
    input_df = pd.read_parquet(data_paths)
    cleaned_df = clean_dataframe(input_df, ["text", "social_stats"])

    # filter by field
    on_brand_df, _ = filter_headlines(target, cleaned_df, "text")

    social_stats = list(on_brand_df["social_stats"])
 
    # extract audience visits from stats
    for idx, stat in enumerate(social_stats):
        try:
            audience_visits = json.loads(stat)["audience_visits"]
        except KeyError:
            audience_visits = None
        social_stats[idx] = audience_visits

    # pull out top 1000 most impactful (by audience visits) data
    on_brand_df["audience_visits"] = social_stats
    on_brand_df.dropna(subset=['audience_visits'], inplace=True)
    on_brand_df.sort_values(['audience_visits'], ascending=False, inplace=True)
    topk_socials = on_brand_df.iloc[0:config["task"]["top_k"]]
    topk_socials_text = list(topk_socials["text"])


    # Running Huggingface Sentiment model for 1000s of samples (running multiple times
    # if tokens in text exceed 512) was slow on my local machine. Batching would speed up. 
    # Reverting to Vader
    if sentiment_model_type == "BERT":
        sentiment_model = BertSentiment(sentiment_model_name)
        sentiment_model = sentiment_model.infer
    
    if sentiment_model_type == "Vader":
        sentiment_model = vader_sentiment

    # calculate sentiment values on top1k socials
    sentiments = []
    for text in topk_socials_text:
        sentiment = sentiment_model(text)
        sentiments.append(sentiment)

    topk_socials["sentiment"] = sentiments

    # split into positive, neutral, and negative splits
    vp = topk_socials[topk_socials['sentiment'].between(0.6, 1.0, inclusive='right')]
    n = topk_socials[topk_socials['sentiment'].between(0.4, 0.6, inclusive='both')]
    vn = topk_socials[topk_socials['sentiment'].between(0, 0.4, inclusive='left')]

    # find topics in each category
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        language='english', calculate_probabilities=True,
        verbose=True
    )

    # Save Very Negative Model data and results
    # Note: sentiment is unnecessarily being run twice on each sample
    v_neg_model_dir = "models/social_analysis/v_neg/"
    check_dir(v_neg_model_dir)
    _, v_neg_summary_df, topic_model = analyze_df_sentiment(sentiment_model, topic_model, list(vn['text']), vn, field="text")
    topic_model.save(os.path.join(v_neg_model_dir, "v_neg_model.pkl"), serialization="pickle")
    v_neg_summary_df.to_csv(os.path.join(v_neg_model_dir, "v_neg_results.csv"))


    # Save Very Positive Model data and results
    v_pos_model_dir = "models/social_analysis/v_pos/"
    check_dir(v_pos_model_dir)
    _, v_pos_summary_df, topic_model = analyze_df_sentiment(sentiment_model, topic_model, list(vp['text']), vp, field="text")
    topic_model.save(os.path.join(v_pos_model_dir, "v_pos_model.pkl"), serialization="pickle")
    v_pos_summary_df.to_csv(os.path.join(v_pos_model_dir, "v_pos_results.csv"))




if __name__ == "__main__":

    # load config data. Can be done more appropriately with argparse
    with open(sys.argv[1]) as f:
        config_data = json.load(f)

    bucket_name = config_data["data_src"]["bucket_name"]
    s3_folder = config_data["data_src"]["s3_folder"]
    data_dir = config_data["data_src"]["data_dir"]
    sentiment_model_type = config_data["task"]["sentiment_model_type"]  
    sentiment_model_name = config_data["task"]["sentiment_model"]
    src_data_dir = config_data["data_src"]["src_data_dir"]
    target = config_data["task"]["target_brand"]

    # list files in s3, check if downloaded, and download locally
    # s3_connector = S3Connector()
    # s3_file_list = s3_connector.list_files(bucket_name, s3_folder)
    # files_exist = True
    # for file in s3_file_list:
    #     if not os.path.exists(os.path.join(data_dir, file)):
    #         files_exist = False
    # if not files_exist:
    #     s3_connector.download_s3_folder(bucket_name, s3_folder, data_dir)

    # run task
    locals()[config_data["task"]["name"]](sentiment_model_type, sentiment_model_name, data_dir, src_data_dir, target, **config_data)


