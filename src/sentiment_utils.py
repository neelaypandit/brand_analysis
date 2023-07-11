from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig, TFAutoModel 
from torch.nn import Softmax
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import statistics

class BertSentiment(): 
    """
    This object contains methods for running a huggingface based sentiment model. Note no method for batching sequences 
    to the model at this time. 

    Attributes
    ----------
    model_name : str
        Name of the model in Huggingface
    truncation : Bool
        Whether to truncate the sentence vector to (512, typical for BERT). If False, and if the text is greater 
        than 512 tokens, infer() will cycle through the full tokenized vector 512 tokens at a time and calculate an average inference score.
    """

    def __init__(self, model_name, truncation=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.max_length=512
        self.truncation = truncation

    def infer(self, text):
        averages=[]
        inputs = self.tokenizer(text, truncation=self.truncation, max_length=self.max_length, return_tensors = "pt")
        length=len(inputs['input_ids'][0])
        
        while length>0:
            if length>self.max_length:
                next_inputs={k: (i[0][self.max_length:]).reshape(1,len(i[0][self.max_length:])) for k, i in inputs.items()}
                inputs={k: (i[0][:self.max_length]).reshape(1,len(i[0][:self.max_length])) for k, i in inputs.items()}
            else:
                next_inputs=False
            output_p = self.model(**inputs)
            smax = Softmax(dim=-1)
            probs = smax(output_p.logits)
            probs = probs.flatten().detach().numpy()
            prob_pos = probs[1]
            averages.append(prob_pos)
            
            if next_inputs:
                inputs=next_inputs
            else:
                break
            length=len(inputs['input_ids'][0])
        average = np.average(averages)
        return average
    


def vader_sentiment(sentence):
    """
    Placeholder Docstring.
    Runs Vader sentiment analysis [0:1] representing positivity. 
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(sentence)["compound"]
    return np.interp([sentiment], (-1,1), (0,1))[0]


def analyze_topic_sentiment(df, sentiment_model, topic, field):
    """
    Placeholder Docstring.
    Given a sentiment model, and df with a "topic" field, calculates the average sentiment 
    for that topic. 
    """    
    topic_docs = df[df['topic'] == topic]
    docs = list(topic_docs[field])
    sentiment_vals = []
    for doc in docs:
        sentiment_vals.append(sentiment_model(doc))
    avg_sentiment = statistics.mean(sentiment_vals)
    return avg_sentiment

def analyze_df_sentiment(sentiment_model, topic_model, text, df, field="body", train=True):
    """
    Placeholder Docstring.
    Given a sentiment model and topic model, trains a topic model, labels the data, and 
    calculates the average sentiment per topic. 
    """    
    if train:
        topics, probs = topic_model.fit_transform(text)
    
    freq = topic_model.get_topic_info()
    df["topic"] = topics
    
    topic_sentiments = []
    for topic in range(-1, len(freq)-1):
        topic_sentiment = analyze_topic_sentiment(df, sentiment_model, topic, field)
        topic_sentiments.append(topic_sentiment)
    freq['average_sentiment'] = topic_sentiments
    
    return df, freq, topic_model