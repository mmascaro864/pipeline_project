# Disaster Response Pipeline Project
This project leverages disaster response data provided by [appen](https://appen.com/) (formerly Figure8). The dataset contains pre-labeled tweets and messages from disaster events. An ETL pipeline is built to read in and transform the data sets, and loads the transformed data into a sqlite database. A natural language processing ML pipeline is fitted and trained to assist with classifiying the disaster response messages. Finally, a Flask app is used to (a) display plots related to message genre and categorization, and (b) to classifiy disaster repsonse messages in real time.

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
- **process_data.py** - ETL pipeline that ingests and transforms disaster message and category data, and loads transformed data into a sqlite database
- **DisasterResponse.db** - output from ETL phase

#### models
- **train_classifier.py** - reads in the saved sqlite database file; build NLP ML pipeline to train classifier model
- **classifier.pkl** - fitted, trained NLP  ML pipeline model

#### app
- **run.py** -  Flask app that renders visualizations based on message data stored in sqlite database, and loads saved NLP ML pipeline model to classify disaster reponse messages entered via web page.
- **master.html** - web page renders plotly visualizations and accepts disaster response messages as user input
- **go.html** - web page that displays disaster repsonse message classification based on user input into master.html form

## Project Execution
1. **ETL Pipeline** - to run the ETL pipeline from the **data** directory, enter: `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
2. **ML Pipeline** - to run the ML pipeline from the **models** directory, enter: `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`
3. **Web Page** - to render the web page, from the **app** directory, enter: `python run.py`. When prompted, point your browser to: http://0.0.0.0:3001/ or http://localhost:3001/.

## Installation
Clone git repository: `git clone https://github.com/mmascaro864/pipeline_project.git`

## Acknowledgements
- [Udacity](https://www.udacity.com/) - Data Science Nanodegree Program
- [appen](https://appen.com/) - Appen, formerly Figure8, provider of reliable training data, for providing the data sets for this project
