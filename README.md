<h1 align="center">Sentiment Classification of Apple and Google Tweets Using NLP</h1>

![tweet tweet](https://github.com/user-attachments/assets/7cdd5791-be28-48d8-8f88-12e02c652d07)

## Overview
Customer sentiment is an integral part of commerce that drives company profits, influences brand reputation, and dictates strategic positioning in today's world. A bulk of that sentiment is found online on social media platforms such as Twitter in the form of  positive, negative, or neutral product reviews. This project analyzes Twitter sentiment about Apple and Google products using Natural Language Processing and advanced supervised modeling techniques to track their public perception. Knowledge on customer sentiment will influence Apple and Google's efforts in sales marketing, influencing company reputation, and improving their products for better customer engagement.

This repository contains:
1. A python notebook containing the NLP project code
2. A ReadMe file describing the project
3. A 'figures' file where the plots and images used in this project and presentation are located
4. A tableau visualization located [here](https://public.tableau.com/app/profile/edna.maina/viz/TwitterSentimentAnalysis_17567937827180/Dashboard1?publish=yes )
5. A .ptt file containing the project's presentation
6. .python-version, Procfile, app.py, model.pkl and requirements.txt files for deployment of the model. 

## Business Understanding

### Stakeholders
Apple and Google Marketing Teams

### Business Problem
Apple and Google products fail to elicit a strong positive reaction from consumers with the bulk of Twitter sentiment being neutral. It is vital that these companies know what causes the lack of an emotional reaction from users so that they can tailor their products for mass positive impact. Furthermore, they must understand the value in customer sentiment through their tweets to prevent a loss in market share and deteriorating brand reputations.

To address this, we aim to:  

  1. Identify key factors influencing Twitter sentiment.
  2. Build a model that can rate the sentiment of a tweet based on its content.
  3. Provide actionable insights that enable Google and Apple to deploy targeted marketing campaigns.

### Data Understanding
The dataset used for this NLP project was obtained from CrowdFlower via [data.world](https://data.world/crowdflower/brands-and-product-emotions) and contains 9000 tweets categorized as either positive, neutral, or negative. 
It contains information such as:
  1. Tweet text
  2. Emotion in tweet is directed at
  3. Is there an emotion directed at a brand or product

The above columns will be simplified to; Tweet, Device, and Emotion respectively for ease of analysis.

### Exploratory Data Analysis
The dataset is heavily imbalanced, which will dictate our choice of modeling and hyperparameter tuning methods. After renaming the columns, checking for duplicates and missing values, conducting feature engineering, text pre-processing, and tokenization, the following emerged:

#### a) Distribution of Emotions
<img width="626" height="433" alt="Distribution of Emotions" src="https://github.com/user-attachments/assets/81d0c80a-4efe-40c4-aa37-6915451154e8" />

**Insights:**    
- From the bar chart above, we can see the distribution of emotions in the dataset. 
- Most of the tweets express Neutral, followed by Positive and Negative emotions as the least frequent. 
- This distribution gives us insights into the overall sentiment of the tweets which will guide our analysis and model training.

#### b) Brand Vs. Emotion Distribution
<img width="1017" height="547" alt="Brand Vs  Emotion Distribution" src="https://github.com/user-attachments/assets/a9aa3b34-0807-454c-a1ef-1bbc64eddc9e" />

**Insights:**  
- Apple: Most tweets are positive, followed by negative, with very few neutral mentions.
- Google: Tweets are also mostly positive, but at a much lower volume compared to Apple.
- Both: Very few tweets mention both brands, with sentiment fairly balanced but minimal.
- Unknown: The majority of tweets fall into this category, and almost all are neutral, which heavily skews the overall dataset.
- Apple attracts more public attention and stronger sentiment (both positive and negative), while Google has fewer mentions.
- The dominance of neutral tweets in the “Unknown” group highlights a large portion of data that may not be brand-specific.

#### c) Wordcloud for Positive Vs. Negative Tweets
<img width="640" height="352" alt="Wordcloud Pos Vs Neg Tweets" src="https://github.com/user-attachments/assets/333f5757-d34f-42ef-8382-b06908ca2c54" />
<img width="640" height="352" alt="Wordcloud Neg Tweets" src="https://github.com/user-attachments/assets/373dca36-f1d4-4cd0-a510-e298ef347025" />

**Insights:**  
- The word clouds show the most frequent words in positive and negative tweets.
- In both cases, terms like ipad, iphone, google, and apple dominate
- But negative tweets highlight words such as not, need, and think, whereas positive tweets emphasize words like great, awesome, and free.

#### d) Apple Tweets
<img width="1415" height="525" alt="Apple tweets" src="https://github.com/user-attachments/assets/047f8247-fe82-4b16-bada-927c3ba47354" />

**Insights:**  
- Positive Apple Tweets:
  - Words like “austin”, “new”, “launch”, and “awesome” show excitement around Apple events and products.
  - Positive tweets highlight enthusiasm, innovation, and user satisfaction.

- Negative Apple Tweets:
  - Frequent terms like “battery”, “help”, and “headache” point to frustrations with usability and product issues.
  - Negative tweets focus on technical problems and unmet expectations.
  - 
#### e) Google Tweets
<img width="1415" height="525" alt="Google tweets" src="https://github.com/user-attachments/assets/a70e8d8d-7ca0-4482-aa11-4e1145f6780b" />

**Insights:**  
- Positive Tweets
  - Words like “party,” “social,” “network,” “new,” “map,” “marissa,” “mayer,” “circles” suggest:
  - Excitement about new features or products (e.g., Google+ Circles)
  - Positive mentions of Google executives (Marissa Mayer)
  - Interest in social networking and mapping tools

- Negative Tweets
  - Words like “circle,” “product,” “need,” “app,” “launched,” “major,” “network” imply:
  - Frustration with new product launches
  - Criticism of apps or features
  - Possibly unmet expectations or usability issues

### Modeling
We utilized 2 models:
1. **Binary Logistic Model.** Functioned as our baseline model using only positive and negative tweets for a start. The model is well suited for binary classification problems and aided in classifying the tweets' sentiments
2. **Multiclassifier Model.**
Using GridSearchCV, pipelines, and vectorizers such as CountVectorizer and TfidfVectorizer for text feature extraction, we assessed the following supervised learning models:
  - Logistic Regression
  - Linear Support Vector Classifier (LinearSVC)
  - Multinomial Naive Bayes
  - Random Forest
  - Gradient Boosting

In conjuction with this, we tuned the following hyperparameters:
  - Vectorizers: n-gram range and minimum document frequency (min_df).
  - Logistic Regression: regularization strength C.
  - LinearSVC: regularization strength C.
  - Multinomial Naive Bayes: smoothing parameter alpha.
  - Random Forest: number of estimators (n_estimators) and tree depth (max_depth).
  - Gradient Boosting: number of estimators (n_estimators) and learning rate.

Our validaton strategy was **StratifiedKFold cross-validation (5 folds)**

### Evaluation
The primary scoring metric used was F1 Macro, which is more appropriate than accuracy for imbalanced datasets because it weights each class equally. This approach allowed us to systematically compare models, select the best hyperparameters, and mitigate issues of class imbalance through SMOTE.

#### a) Results
After performing hyperparameter tuning with GridSearchCV (5-fold cross-validation), the following best configuration was identified:

Model: Logistic Regression
Vectorizer: TF-IDF
Vectorizer Parameters: ngram_range=(1, 2), min_df=1
Classifier Parameters: C=10, class_weight="balanced", max_iter=500
The best cross-validation score (using macro F1) was:

**Best CV Score: 0.5671**

This indicates that the tuned Logistic Regression model with TF-IDF vectorization and bigram features achieved a reasonable performance in balancing precision and recall across all classes.  
Since we used F1-macro as the scoring metric, the result reflects performance across both frequent and less frequent classes, making it suitable for our imbalanced dataset.

We used accuracy to measure the overall performance of the model. However, since the dataset is imbalanced, accuracy alone could be misleading. To address this, we included macro F1, which gives equal weight to each class, ensuring minority classes are fairly evaluated. Additionally, ROC-AUC was used to capture the model’s ability to distinguish between classes across different thresholds.

#### b) Classification Report
Classification Report (per class):

- **Class 0 (Negative Tweets)**

  - Precision: 0.43 → When the model predicts "negative," it's correct 43% of the time.
  
  - Recall: 0.37 → The model only identifies 37% of actual negative tweets, missing many of them.
  
  - F1-score: 0.40 → Weak performance, showing the model struggles with minority/negative class.

- **Class 1 (Positive Tweets)**

  - Precision: 0.76 → Predictions for "positive" are correct 76% of the time.
  
  - Recall: 0.74 → The model captures most positive tweets (strong recall).
  
  - F1-score: 0.75 → Strong, consistent performance; this class dominates the dataset, so the model learns it best.
  
- **Class 2 (Neutral Tweets)**

  - Precision: 0.57 → Predictions are moderately correct.
  
  - Recall: 0.61 → The model captures 61% of neutral tweets.
   
#### c) ROC Curves
<img width="691" height="547" alt="ROC" src="https://github.com/user-attachments/assets/b766e39a-efbc-4ed5-aec9-799591cae732" />

- **Class 0 (Negative) = 0.85**: This shows strong separation between negative and non-negative classes. This shows the negative class is usually harder to detect in imbalanced datasets. 
- **Class 1 (Neutral) = 0.74**: This indicates moderate separation between neutral and non-neutral classes. The neutral class is somewhat easier to detect but still presents challenges in an imbalanced dataset.  
- **Class 2 (Positive) = 0.76**: This shows good separation between positive and non-positive classes. The positive class is generally easier to detect but can still be affected by class imbalances.  

#### d) Macro and Weighted AUC
- **Macro AUC: 0.78**

  - This is the average AUC across all classes, treating them equally (regardless of class size).

  - A value close to 0.8 indicates that the model has good overall discriminative ability to distinguish between positive, negative, and neutral tweets.

  - The macro value is particularly important here because it shows the model is not just biased toward the majority class but also has some capability with minority classes.

- **Weighted AUC: 0.75**

  - This AUC is weighted by class support (larger classes contribute more).

  - The slightly lower score compared to macro AUC reflects the dataset imbalance — since the majority (positive tweets) dominates, the model’s weaker performance on negative tweets pulls the weighted AUC down.


### Recommendations
**1. Enhance Product Support & Communication**
- Negative sentiment tweets often highlight recurring customer frustrations (e.g., device performance issues, app bugs, or unclear updates). Apple and Google should proactively address these concerns by strengthening customer support responsiveness (e.g by increasing the number of customer support representatives and using FAQs/chatbots to reduce wait times).

**2. Leverage Positive Sentiment for Marketing**
- Positive tweets can be repurposed for marketing campaigns, testimonials, or case studies. Highlighting satisfied customer stories builds brand trust, reinforces customer loyalty and attracts new customers to buy the products.
  
**3. Improve Product Development Based on Feedback**
- Frequent negative mentions about specific product features (e.g., iOS updates, Android battery life) can guide R&D priorities. Systematically feeding sentiment insights into product development cycles will help align products with user expectations.
  
**4. Monitor Competitor Sentiment for Strategic Positioning**
- Since the project compares Apple and Google, differences in sentiment trends reveal competitive advantages. For example, if Apple has more praise for design but Google wins on affordability, both companies can refine positioning strategies accordingly.

**5. Strengthen Crisis Management with Real-Time Monitoring**
- Spikes in negative sentiment (e.g., after product recalls, service outages, or controversial announcements) can be detected early through automated sentiment dashboards. Both firms should integrate this into PR strategies to manage crises swiftly.

**Contributors:**
* [Keith Tongi](https://github.com/Tkei-54) - keith.tongi@student.moringaschool.com
* [Kevin Karanja](https://github.com/tingly-amua) -  kevin.karanja@student.moringaschool.com
* [Jacob Abuon](https://github.com/abuonodindo2030) - jacob.abuon@student.moringaschool.com
* [Edna Maina](https://github.com/Julie-t) - edna.maina@student.moringaschool.com
* [Edgar Muturi](https://github.com/edgarmuturi) - edgar.muturi@student.moringaschool.com
* [Charity Mwangangi](https://github.com/CharityPM) - charity.mwangangi@student.moringaschool.com

