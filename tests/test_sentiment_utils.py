"""
    Example use of testing using pytest
"""

import pytest
from transformers import pipeline
from src.sentiment_utils import BertSentiment


@pytest.fixture
def texts():
    neg_text = "i hate this."

    pos_text = "i love this."

    return [neg_text, pos_text]


def test_bert_sentiment(texts):
    model_name = "siebert/sentiment-roberta-large-english"
    HF_model_T = BertSentiment(model_name, truncation=True)
    HF_model_nT = BertSentiment(model_name, truncation=False)

    sentiment_model = pipeline(model=model_name)

    # check if negative behavior is correct
    negative_pipeline_score = 1-sentiment_model(texts[0])[0]["score"]
    assert HF_model_T.infer(texts[0]) < 0.5
    assert abs(HF_model_T.infer(texts[0]) - negative_pipeline_score) < 0.001
    assert HF_model_T.infer(texts[0]) == HF_model_nT.infer(texts[0])    

    # check if negative behavior is correct
    positive_pipeline_score = sentiment_model(texts[0])[0]["score"]
    assert HF_model_T.infer(texts[1]) > 0.5
    assert abs(HF_model_T.infer(texts[1]) - positive_pipeline_score) < 0.001
    assert HF_model_T.infer(texts[1]) == HF_model_nT.infer(texts[1])
