# import required libraries
import sys
import os
import re
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import pickle as pk
import nltk
#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

# import Natural Language Toolkit Libraries
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# import scikit learn libraries
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def load_data(database_filepath):
    '''
    load_data:
        - Loads data from sqlite database, creates Pandas dataframe,
        - and creates X, y, and category_names variables to be studied

        In:
            - database_filepath: path to stored sqlite database
        
        Out:
            - X: dataframe column containing message data
            - y: dataframe columns containing category data
            - category_names: list of category column names

    '''
    sql_url = 'sqlite:///'+ database_filepath
    table = os.path.basename(database_filepath)
    engine = create_engine(sql_url)
    df = pd.read_sql_table('DisasterResponse.db', engine)

    X = df.message
    y = df[df.columns[4:]]

    category_names = y.columns
    
    return X, y, category_names
    
def tokenize(text):
    '''
    tokenize:
        Split text into words, normalize case, reduce words to their root
        
        In:
            text - message text from disaster dataset
        
        Out:
            clean_tokens - tokenized and lemmatized text
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token and lemmatize text
    clean_tokens = []
    for token in tokens:
        
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    return clean_tokens

def build_model():
    '''
        build_model:
            - Build Natural Language Processing ML pipeline model
            - Processes and transforms text messages and applies a classifier

            In:
                - None

            Out: pipeline model
    '''
    pipeline = Pipeline([
        
        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(AdaBoostClassifier(
                    DecisionTreeClassifier(max_depth = 2), 
                        n_estimators = 10, learning_rate = 1)))
        ]))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
        evaluate_model:
            - Receives fitted model, makes predictions over test dataset, 
            determines overall model accuracy, and outputs classification report

            In:
                - model: NLP ML pipeline model
                - X_test: disaster repsonse messages
                - Y_test: message labels
                - category_names: list of category (column) names

            Out:
                - Screen print:
                    - model accuracy
                    - Classification report
    '''
    # model accuracy
    y_pred = model.predict(X_test)
    overall_accuracy = (y_pred == Y_test).mean().mean()
    print('\nOverall model accuracy: {}'.format(overall_accuracy))

    # classification report
    report = classification_report(Y_test, y_pred, target_names=category_names, output_dict = True)
    df_class_rpt = pd.DataFrame(report).transpose()
    print('\nClassification Report:\n')
    print(df_class_rpt)
    
    return


def save_model(model, model_filepath):
    '''
        save_model:
            - saves fitted, trained model

            In:
                - model: fitted, trained model
                - model_filepath
                
            Out:
                - path to model saved as pickle file
    '''
    pk.dump(model, open(model_filepath, "wb"))

    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()