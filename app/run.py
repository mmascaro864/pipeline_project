import json
import plotly
import pandas as pd
import joblib

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # get columns with numeric data
    numeric_df = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    numeric_sum = numeric_df.sum(numeric_only = True)

    # create dataframe from series
    numeric_sum_df = numeric_sum.to_frame()
    
    # reset index and rename columns
    numeric_sum_df.reset_index(inplace=True)
    numeric_sum_df = numeric_sum_df.rename(columns = {'index':'category', 0: 'count'})

    # sort dataframe by count, descending
    numeric_sum_df = numeric_sum_df.sort_values(by = 'count', ascending = False)

    # get top and bottom ten categories in terms of message count
    top_ten = numeric_sum_df[0:10]
    bottom_ten = numeric_sum_df[-11:]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x = top_ten['category'],
                    y = top_ten['count']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories - Top 10',
                'yaxis': {
                    'title': 'Frequency'
                },
                'xaxis': {
                    'title': 'Categories - Top 10'
                }
            }
        },
        {
            'data': [
                Bar(
                    x = bottom_ten['category'],
                    y = bottom_ten['count']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories - Bottom 10',
                'yaxis': {
                    'title': 'Frequency'
                },
                'xaxis': {
                    'title': 'Categories - Bottom 10'
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()