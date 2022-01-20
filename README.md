# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Web APIs & NLP


## Problem Statement

Men In Black is a association that researches and promotes extraterrestrial knowledge. In recent years, the fans looked out for the information all over the topics with majority of them into the space postings. This resulted in the number of fans dwindled and shifted to space to lookout for extraterrestrial answers. As requested by a company MIB to reddit data science team, they want us to identify the key words that can best capture the attention of aliens fans so as to differentiate itself from space. This would facilitate MIB's marketing effort on social media, events and podcasts.

Therefore, the goal of the project is to discover the key words that best differentiate aliens fans from space fans in reddit posts.

Three classification models, Naive Bayes, Logistic Regression and Random Forest will be developed to assist with the problem statement.

The success of the model will be assessed based on its precision and F1 score on unseen test data.

There are 395K aliens fans and 19.5 million space fans respectively on reddit. These fans come from all over the world. The models that will be developed are capable of accurately classify the two subreddits. There are enough unique posts in each subreddit to identify key words to achieve our research goal. Therefore, the scope of the project is appropriate.

This project is important to MIB and the relevant associations to increase our fan base using the key words identified to develop a successful marketing campaign. The secondary stakeholder will be a major aliens clubs, association. Their involvement will benefit them in terms of their club membership, events, shows and podcasts.

Please note that there are 3 files to this project.
- Subreddit Scrapping space and aliens - Part 1
- Data Cleaning and EDA.ipynb - Part 2
- Modeling - Part 3


### Datasets
* [`aliens.csv`](./data/aliens.csv): aliens set extracted from subreddit r/aliens (1500 posts)
* [`space.csv`](./data/space.csv): space set extracted from subreddit r/space. (1500 posts)
* [`df.csv`](./data/df.csv): df set that combined aliens and space posts (3000 posts in total)
* [`dfall.csv`](./data/dfall.csv): Taken from df.csv, dfall set is after EDA was done.
* [`dfmaster.csv`](./data/dfmaster.csv): Taken from dfall set, dfmaster set is the final set after preprocessing done and to be used for Modeling.


### Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---|
|word_length| int64| space, aliens| total word length for title column|
|subreddit|object|space, aliens, df|subreddit name|
|title|object|space, aliens, df|The title of the submission|
|selftext|object|space, aliens, df| markdown formatted content for a text submission|
|id |object|space, aliens, df, dfall, dfmaster|ID of the subreddit|
|score| int64| space, aliens, df, dfall, dfmaster| no of upvotes minus the no of downvotes
|upvote_ratio| float64| space, aliens, df, dfall, dfmaster| percentage of upvotes
|num_comments| int64| space, aliens, df, dfall, dfmaster| number of comments on the submission
|text_length| int64| dfall, dfmaster| total word length for text column|
|text| object| dfall, dfmaster| combination of title and selftext|
|is_aliens|int64|dfall, dfmaster|classification of subreddit|


### Data collection
* Using [Pushshift's](https://github.com/pushshift/api) API, posts from two subreddits r/space and r/aliens were extracted using Pushshift. As there is an limit per extraction of 100 posts, each retrieval is only 100 post and a for loop created to obtain 1500 posts and built in interval between each retrieval so that the site does not treat this retrieval as a hack.

* We will be using 1500 posts from each of the subreddit, space and aliens in order to have a good significant data collection for analysis. The data will be used to determine any relationship between the two.


### Data Cleaning and EDA
* There are a total of 1500 rows and 82 columns for space set and 1500 rows and 83 columns for aliens set. 7 meaningful columns to show some correlation with the text was retained and the rest were dropped.

* Distribution of the title word length was examined and both subreddits space and aliens are stewed towards the right. This is commonly found in posting forums. The average of both subreddits title length is 12-13. Also, the 75 percentile is very close to each other at 14-16.

* Only selftext column has missing data and it is filled with "" so there is no null data. Wording like [removed] and [deleted] was also noted and removed with "". Lastly, duplicate titles and selftext are also cleaned up.

* For the model to study the words more efficiently, we combined the title and selftext into a single column called text.
- Histograms of upvote ratio showed the peak at 1.0. However, we will keep the rest of the data as it reflects the other upvote ratio.

- Histogram of score showed the graph stewed to the right. However, we will keep in as it may have significant impact on the results.

- Histogram of num comments showed the graph stewed to the right. However, we will keep in as it may have significant impact on the results.

- Histogram for text length showed the graph stewed to the right. As this is important, we will examine the outliners and using IQR model to remove the outliners using “maximum”: Q3 + 1.5IQR and “minimum”: Q1 -1.5IQR. We also discovered an obvious outliners after 700 text_length. To keep the distribution proper, data more than 3 std deviation was also removed such as the outliner >700.

A function is created to clean the text for the NLP modeling. The function includes:
- make all lowercase
- remove non-letters
- remove HTML
- remove website hyperlink
- remove words with 2 or fewer letters such as 'is', 'as', 'by' which do not contribute to the modeling
- remove whitespace
- remove emoji using a function demoji
- remove emoticons :)


### Preprocessing
For preprocessing, tokenize and lemmatization were used. The aim of lemmatization is to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma. Although stemming is faster, it chops words without knowing the context of the word in given sentences. Lemmatization on the other hand is slower as compared to stemming but it knows the context of the word before proceeding. It is a rule-based approach. Lemmatization provides better results by performing an analysis that depends on the word's part-of-speech and producing real, dictionary words. As we are examining aliens and space which requires substainable and meaningful words, Lemmatization is used only.

Stopwords are also in use so that we can remove the common words from the list.Replace the ordinal columns with numeric sequencial values for modelling evaluation.

Using word cloud, we analysed the following:
- The popular words used are ufo, theory, mission, nasa, universe, human, earth, moon, star, etc
- Based on the above EDA, there is enough information to answer the problem statement with the provided data.
- During the inspection of data, summary statistics were generated, it shows there are relevant data to address the problem statement.
- During the EDA stage, some interesting relationships between variables were identified. Also popular words were uncovered. Therefore, there is enough information at hand to address the problem statement.

I will explore CountVectorize and TF-IDF word statistics before including into the modeling for optimisation. Comparison was done for CountVectorize and Tfid-Vectorize using ngram 1,3 and ngram 2,2.

ngram(1,3)
- For space subreddit, words such as telescope, nasa, launch and webb all appeared in both top 5 for CountVectorize and Tfid-Vectorize. It is frequently associated with space related matters. Webb is related to James Webb Space Telescope and hence it appeared in Top 5.

- For aliens subreddit, words such as ufo, think, human all appeared in both top 5 for CountVectorize and Tfid-Vectorize. It is frequently associated with aliens related matters. The word think would suggest that it is all in personal opinions without factual evidence as what aliens appeared to the world.

ngram(2,2)
- For space subreddit, the top 5 words are the same using CountVectorize and Tfid-Vectorize It is frequently associated with space related matters.

- For aliens subreddits, words such as youtube com and com watch all appeared in both top 5 using CountVectorize and Tfid-Vectorize It is frequently associate with aliens related matters.


### Modeling

In this section, Train/Test Split was performed twice. That is why we split twice and the proportion distributed is train - 60%, validation - 20%, test - 20%. This is to ensure there is no data leakage and to make sure, X_test and y_test remains untouched. The feature X used the text column as a combination of title and selftext since we are doing NLP modeling. Y is the Target is_aliens.

The baseline accuracy is the percentage of the majority class, regardless of whether it is 1 or 0. It serves as the benchmark for our model to beat.
For every post we select, there is 45% chance that it is under subreddit aliens.

I will explore 3 models. They are Naive Bayes, Logistic Regression and Random Forest. I will run each model twice, first time using CountVertorizer and second time using Term Frequency-Inverse Document Frequency (TF-IDF). I will provide a short explaination of each model and evaluate its performance success/downfalls.

- Naive Bayes
The Naive Bayes classification algorithm is a classificiation modeling techinque which relies on Bayes Theorem. It makes one simplifying assumption that features are independent of one another. The advantages of Naive Bayes are it is easy to calculate probabilities and returns empirically accurate result. The disadvantage is the assumption of feature independence is unrealistic, especially in the case of text data. The predicted probabilites can be quite bad.

- Logistic Regression
Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables. Logistic regression uses the logit link to bend our line of best fit. This allows us to predict between 0 and 1 for any value of inputs.

logit (P(Y=1))=β0+β1X1+β2X2+⋯+βpXp

The advantage of logistic regression are the coefficients are interpretable and it shares similar properties to linear regression.
The disadvantage is the assumption of linearity between the dependent variable and independent variables. Logistic Regression requires average or no mulitcollinearity between independent variables.

- Random Forest
They use a modified tree learning algorithm that selects, at each split in the learning process, a random subset of the features. This process is sometimes called the random subspace method.

The reason for doing this is the correlation of the trees in an ordinary bootstrap sample: if one or a few features are very strong predictors for the response variable (target output), these features will be used in many/all of the bagged decision trees, causing them to become correlated. By selecting a random subset of features at each split, we counter this correlation between base trees, strengthening the overall model.

Random forests, a step beyond bagged decision trees, are very widely used classifiers and regressors. They are relatively simple to use because they require very few parameters to set and they perform pretty well.


- With these 3 models, we will have sufficient info to make an assessment.


### Conclusion and Recommendations

Best Model selected for test data is Logistic Regression with TF-IDF as it offers the best validation score, the train score are also in a good range and does not show overfitting as compared to Random Forest or too low for Naive Bayes. The difference in the train and validate data for Logistic Regression with TF-IDF is also reasonably under 10.

|                   Model                    | Train |  Val  |  Diff | Precision |  F1 |
|--------------------------------------------|-------|-------|-------|-----------|-----|
|      Naive Bayes with CountVectorize       | 92.7% | 87.9% |  -4.8 |   87.3%   | 88% |
|          Naive Bayes with TF-IDF           | 95.2% | 87.9% |  -7.3 |   89.1%   | 88% |
|  Logistic Regression with CountVectorize   | 97.5% | 86.8% | -10.7 |   88.8%   | 87% |
|      Logistic Regression with TF-IDF       | 95.7% | 88.3% |  -7.4 |    92%    | 88% |
| RandomForestClassifier with CountVectorize | 99.9% | 87.9% | -12.0 |   90.6%   | 88% |
|     RandomForestClassifier with TF-IDF     | 99.5% | 86.0% | -13.5 |   89.4%   | 86% |


The ROC curve shows the trade-off between sensitivity (or TPR) and specificity (1 – FPR). Classifiers that give curves closer to the top-left corner indicate a better performance. As a baseline, a random classifier is expected to give points lying along the diagonal (FPR = TPR). The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test. AUC is the area under the ROC curve.
When AUC = 1, then the classifier is able to perfectly distinguish between all the Positive and the Negative class points correctly. If, however, the AUC had been 0, then the classifier would be predicting all Negatives as Positives, and all Positives as Negatives.

When 0.5<AUC<1, there is a high chance that the classifier will be able to distinguish the positive class values from the negative class values. This is so because the classifier is able to detect more numbers of True positives and True negatives than False negatives and False positives.

In the ROC curve graph, LogisticRegression TF-IDF/CVEC and Random Forest TF-IDF showed the highest percentage of 0.94 although the rest of the models are also very close second at 0.93. This showed that all the models have consistent results.

Logistic Regression with TF-IDF was chosen as its validation score has the best accuracy of 88.3%. The diff between train and set is also very close among all the models and it showed the consistency.

|              Model              | Train |  Val  |  Test | Precision |  F1 |
|---------------------------------|-------|-------|-------|-----------|-----|
| Logistic Regression with TF-IDF | 95.7% | 88.3% | 86.2% |   92.4%   | 86% |


The top coefficient words for aliens and space are in chronological order.

* aliens postings
1) alien
2) ufo
3) human
4) guy
5) think
6) believe
7) disclosure
8) thought
9) strange
20) grey

* space postings
1) space
2) nasa
3) launch
4) telescope
5) star
6) moon
7) jswt
8) satellite
9) nebula
20) rocket

Many of the words appeared in both the subreddits and that is why some postings are classified incorrectly. Words like planet, earth and life are all related to both subreddits.

On the misses, many of the incorrect predictions are due to the no meaning postings and also postings in the wrong subreddit. For example,
1) kkkkkkkkkkk has no meaning on alien subreddit.
2) alian was spelled wrongly so it is not correctly classified, one posting on alien word posted in space subreddit.
3) posting with strong correlation like aliens and space are both in the posting,

Test Set
- The model has a high mean accuracy score (95.7% for training data) and (86.2% for test data).
- The mean accuracy score for training is higher than mean accuracy for test data.
- The model has a consistent high accuracy of 86.2% test and 88.3% for val data.
- It also has a resonable precision (i.e the percent of my predictions were correct) score of 92% for aliens.
- The test set performed less than the validation set by 2.1%. It means that the results are consistent and good.

|              Model              | Baseline | Train | Validate |  Test |
|---------------------------------|----------|-------|----------|-------|
| Logistic Regression with TF-IDF |   45%    | 95.7% |  88.3%   | 86.2% |

* Interpretation of the use of model Logistic Regression TF-IDF.

The baseline score is based on the linear line of aliens and space postings and it is 45% chance of chosing alien as the posting.

1) TF-IDF penalize common words and give rare words more influence. This is relevant to our problem statement, space posts and aliens posts tends to use different terms. Space post might commonly use the word "launch", "jwst" because they are some of the most popular words associated with space. On the other hand, Aliens post might commonly find the word "UFO" commonly associated with aliens. Therefore, penalize common words and giving more influence to rare words would help us in our classification problem.

2) Comparing the model scores of CountVectorizer and TfidfVectorizer, in all instances models under TfidfVectorizer scores better except Random Forest. It could be because of its modeling technique of using strong predictors such as ufo to correlate. However, in the misses analysis, we also notice the word ufo appear in the space posting, this could be one of the reasons why Random Forest TF-IDF is lower.

3) Why Logistic Regression? Logistic Regression gives a high mean accuracy score for both Train data 95.7%, Vaidation data of 88.3% and Test data of 86.2%. This means that the model generalises well and scores well on unseen data. It returns high test score that are consistence with the Validation Score. In addition, it also scores much better than the baseline score of 45%

4) Logistic Regression allows me to interpret model coefficients as indicators of feature importance. For example, the presence of the word "ufo" increases by 1, the post is about 488 times as likely to be a aliens post. Likewise for the word 'ufo' where it is 42 times. It is logical to think of ufo word when aliens comes to mind. It almost always comes in pair.

5) Based on reading the posts in reddit, aliens fans heavily discussed in ufo related incidents such as involving human, believe, disclosure and think. Aliens fans always include their own views to any incidents as aliens is not offically factual and they would like to speculate.

6) On the other hand, words such as space, nasa, launch, moon, jswt all are considered rare words and indicates the post is from space instead of aliens.

7) The interpretation of coefficients allows us to make useful recommendations for our problem statement and allow us to target specific marketing for our client.

8) Logistic Regression however has its own limitation. It assumes independence of independent Variables and the independent variables X1...Xm are linearly related to the logit of the probability but not always the case. Nevertheless, it proves to be a good model to use.

Our research team had successfully answered the original problem statement, which is to find the key words that best differentiate aliens fans from space fans in reddit posts. We have also use these key words to derive marketing strategies below.

* Marketing Strategies
1) Search Engine Optimisation (SEO)
Based on our research through reddit post and our logistic model, keys words such as Alien, UFO, human guy, think, believe, disclosure, thought and strange are the words that best indicate a alien post. This means that, aliens fanatics and hard-core fans use such words when talking ahout aliens topics in the discussions. In essence, MIB can include these words during their web improvement process to boost online presence and improves search engine optimisation (SEO).

2) Keep the Marketing Message precise and appealing
Based on our research, aliens posts (mean of 162 text length) are generally longer than space posts (mean 125 text length). There should have some specific contents that get alot more response than the standard distibution text length. Higher distribution are at 40-60 Aliens postings have consistent score regardless of text length as long as the contect is meaningful. Social media platforms would be a good channel for MIB's markeing effort. For example Twitter. A meaningful and appealing twit can start a long message of communications.

3) Marketing Message Needs to be Meaningful
Research result on the number of comments showed that the scoring are more spread out. As long as the content is meaningful, aliens fans will continue to discuss and participate in the discussions. In addition, the number of comments in aliens posts are much longer than space post. This means that there are still alot of core aliens fans to keep the discussion going.

MIB can use the key words identified in our model to create meaningful discussion topics. For example, a discussion on human abduction by aliens. Besides that, we also discovered they like to include their own opinions and views with word such as believe, think from the word cloud. MIB should look for past information and resurfaced the discussion and if there is any new findings. That will surely arouse their interest again.

In summary, MIB can use the recommendations above to better market and so as to win back more fans. This important to ensure aliens fans does not continue to dwindle. Once we built up the fan base, MIB can then organise events, activities and maybe membership to ensure it has more funding and sponsorship such as ticket sales and merchandise. Secondary stakeholders such as aliens associations will also benefit in terms of their club membership, events, shows, podcasts and awareness to the public.

* Next step forward

A larger dataset will be used to further improve the accuracy. In the future, a larger dataset could be gathered. After EDA cleanig, there are about 1122 aliens posts and 1347 for space. Exploring a larger dataset and different posts dates will help us to better analysis.

More models could be explored, such as catboost, KNN, topic classification and so on. We should continue to look out for new models for better prediction. May be neural networks can be looked into as well.

From the stakeholders benefits, they now have a better understanding of the words weightage through word cloud. Future research can combine both qualitative and quantitative methods. Our data science research can be supported by interviews or focus groups. This would allow us to discover more meaning behind our findings. For example, we can invite reputable alien fans to discuss on the keys and its mode. This would help us derive meaning from the fans perspective.

Our model is limited to the corpus of texts obtained from scrapping aliens and space reddit APIs such as pushshift or PRAW. It would be better if the model can learn on its own, constantly update new text. This would allow the model to stay relevant longer.
Currently, our model only analyse aliens and space texts. In the future, the model can be expanded to include other subreddit related to aliens.
