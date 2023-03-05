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
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

# import Natural Language Toolkit Libraries
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# import scikit learn libraries
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin


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

class Debug(BaseEstimator, TransformerMixin):
    def transform(self, X):
        print(X.shape)
        return X
    
    def fit(self, X, y = None, **fit_params):
        return self 
    
class CaseNormalizer(BaseEstimator, TransformerMixin):
    '''
    CaseNormalizer class: 
        - Feature of machine learning pipeline model
        - custom transformer that transforms a text array to lower case
        and removes white space.

        In: 
            text array
        Out: 
            Pandas Series of transformed text
    '''
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = pd.Series(X).apply(lambda x: x.lower().strip()).values
        X = np.to_array(X)
        return X

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    StartingVerbExtractor class:
        - Feature of machine learning model pipeline
        - Checks whether first word in sentence is a verb

        In: 
            text array
        Out: 
            Pandas DataFrame of tagged text
    '''

    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        import pdb
        pdb.set_trace()
        X_tagged = pd.Series(X).apply(self.starting_verb)
        X_tagged = pd.DataFrame(X_tagged)
        X_tagged['message'] = X_tagged['message'].astype(int)
        print(X_tagged)
        return X_tagged

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

    # initate case_normalizer and normalize text
    case_normalizer = CaseNormalizer()
    normalized_tokens = case_normalizer.transform(tokens)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token and lemmatize text
    clean_tokens = []
    for token in normalized_tokens:
        
        clean_token = lemmatizer.lemmatize(token)
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
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('debug', Debug()),
                ('tfidf', TfidfTransformer()),
                ('debug1', Debug())
            ])),

            ('starting_verb', StartingVerbExtractor()),
            ('debug2', Debug()),
            ('case_normalizer', CaseNormalizer()),
            ('debug3', Debug())
        ])),
    
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    
    '''
    # model accuracy
    y_pred = model.predict(X_test)
    accuracy = (y_pred == Y_test).mean()
    overall_accuracy = (y_pred == Y_test).mean().mean()
    print('Model accuracy by category:\n {}'.format(accuracy))
    print('\nOverall model accuracy: {}'.format(overall_accuracy))

    # classification report
    report = classification_report(Y_test, y_pred, target_names=category_names, output_dict = True)
    df_class_rpt = pd.DataFrame(report).transpose()
    print(df_class_rpt)
    
    return


def save_model(model, model_filepath):
    '''
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