# Disaster Response Pipeline Project

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
- **disaster_messages.csv** - 
- **disaster_categories.csv** -
- **process_data.csv** - ETL pipeline that ingests and transforms disaster message and response data, and loads transformed data into a sqlite database
- **DisasterResponse.db** - output from ETL phase

#### models

#### app

