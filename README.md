# Disaster Response Pipeline Project
This project leverages disaster response data provided by. The dataset contains pre-labeled tweets and messages from disaster events. An ETL pipeline is built to read in and transform the data sets, and loads the transformed data into a sqlite database. A natural language processing ML pipeline is fitted and trained to assist with classifiying the disaster response messages. Finally, a Flask app is used to (a) display plots related to message genre and categorization, and (b) to classifiy disaster repsonse messages in real time.

## Udacity Data Science Nanodegree Project
#### Marc Mascaro
#### March 2023

## Libraries Used
- Project uses Python 3.9, Anaconda distribution

| Standard   | Natural Language Toolkit  | Sci-kit Learn          |
| -----------|:-------------------------:| ----------------------:|
| sys        | pos_tag                   | classification_report  |
| os         | sent_tokenize             | test_train_split       |
| re         | word_tokenize             | CountVectorizer        |
| pandas     | stopwords                 | TfidfTransformer       |
| numpy      | WordNetLemmatizer         | Pipeline               |
| sqlalchemy |                           | MultiOutputClassifier  |
| pickle     |                           | DecisionTreeClassifier |
| flask      |                           | AdaBoostClassifier     |
| plotly     |                           |                        |

## Code directory structure
#### data
- **disaster_messages.csv** - disaster messages and message genre
- **disaster_categories.csv** - disater message categories
- **process_data.csv** - ETL pipeline that ingests and transforms disaster message and category data, and loads transformed data into a sqlite database
- **DisasterResponse.db** - output from ETL phase

#### models
- **train_classifier.py** - 
- **classifier.pkl** - fitted, trained NLP  ML pipeline model

#### app

