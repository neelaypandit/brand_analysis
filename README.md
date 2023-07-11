# Brand Analysis

This project demonstrates my approach for how a brand (a major airline) might want to understand their presence on different media platforms. The two underlying modelling approaches are topic modeling and sentiment analysis. 

For topic modeling, UMAP/HDBscan provides a powerful dimensionality reduction and density based clustering approach on sentence vectors. For vectorization, count vectorizer is simple but effective in generating relevant topics, although using embedding models from Huggingface might provide better results. Other approaches such as LDA, HDP, NMF, are also possibly worth exploring, and provide some quantitative measures of inter/intra topic performance, i.e., coherence, perplexity. 

For sentiment analysis, finetuned BERT-based models provide some of the best results for various sentiment related tasks. But sentiment is a complex problem and very sensitive to the domain of the training data. For example models trained on IMDB data will perform poorly on Twitter datasets or product review datasets. A similar diversity of input data is present here: news vs reddit vs twitter, etc. 

This is a rough project structure with two types of analyses, one on the "news" data and one on the "social" data. A few missing things from the project: type hints, complete class/function documentation, refactoring, performance considerations, handling data properly, testing, proper model hygiene, and requirements.

# News Analysis

This analysis looks at how an airline, Southwest Airlines, can understand what are the different things associated with their brand, that are being talked about in the news (over an arbitrary timespan, the one found in the dataset). Additionally, what are the different things that are being talked about their industry in general?

Approach: The approach here is to create two datasets by querying against news headlines. The first query for "Southwest Airlines" creates the brand-specific dataset. The second query is run against the elements discarded from the first query, and is for "airlines", and represents the competitors/industry dataset. Complex pattern matching is not implemented here. In order to determine if the topics being discussed a positive or negative, sentiment analysis using a BERT based model is run and a summary is produced, giving a list of topics, their description, and the average sentiment per topic. 

Results: 
Competitor/Market analysis caught some interesting topics, some observations:
    - captured many individuals/specific events with high polarity: negative news on an incident with anthony bass, a Spirit Airlines attendant's viral tiktok video, police arresting a person on American Airlines, an individual urinating on a flight on American Airlines, etc. 
    - capturing financial events: positive news on Austrian Airlines/Lufthansa group restructure, quarterly reports etc. 

Southwest analysis:
    - negative news on delays and cancelled flights in general.
    - postive news on companion pass and rewards program
    - highly negative news involving a passenger named Savannah Chrisley
    - highly negative news involving a crying baby
    - somewhat positive news regarding company financial health

Results found in /models/news_analysis/


# Social Analysis
This analysis looks at what the positive and negative things that are being said on social media with the largest impact. "impact" is inferred here by audiance views, but can be inferred through various different ways. 

Approach: The approach here is to first find where "Southwest Airlines" is being mentioned in social media. Then to limit analysis to the most impactful posts/comments/tweets, the stats metadata is used to sort by audience_views. Taking the 1000 most viewed data points, sentiment analysis is performed using a BERT-based huggingface model. Then topic modelling is done on negative and positive social media interactions separately. 

Results:
Topics amongst positive and highly impactful social media interactions:
    - companion pass and other airline rewards
    - promotions and deals
Topics amongst negative and highly impactful social media interactions:
    - a baby crying video
    - savannah chrisley
    - major financial loss after December cancellation chaos
    - data connection issues resulting from a firewall failure
    - a poorly recieved billboard campaign
    - CEO pay hike
    - delays, cancellations, holiday meltdown
    - others

# To Run:
To analyze "news" data: python src/main.py configs/news_analysis.json   
To analyze "social" data: python src/main.py configs/social_analysis.json  

To test: pytest .

