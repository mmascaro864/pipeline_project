# import necessary libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load_data: 
        - load messages and categories csv files
        - create dataframe

        In: 
            - messages.csv
            - categories.scv
        
        Out:
            - merged dataframe
    '''
    # read in data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on = 'id', how = 'left')
    return df


def clean_data(df):
    '''
    clean_data:

        In:
            - dataframe
        
        Out:
            - dataframe
    '''
    # split categories into separate category columns
    categories = df['categories'].str.split(';', expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # extract a list of new column names for categories
    category_colnames = [x[:-2] for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # replace categories column in df with new categories columns
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    # drop duplicates
    df.drop_duplicates(inplace = True)

    # child_alone column is all zeros - remove
    df.drop(['child_alone'], axis = 1, inplace = True)  

    # related column has values 0, 1, and 2. since a high percentage of values are 1, change value = 2 to 1.
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

    return df


def save_data(df, database_filename):
    '''
    save_data: save cleaned data to sqlite database
        In:
            - dataframe
            - database filename

        Out:
            - sqlite database
    '''
    url = 'sqlite:///' + database_filename
    engine = create_engine(url)
    df.to_sql(database_filename, engine, if_exists = 'replace', index=False)  
    
    return  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()