import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    print('-----------start load_data------------')
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df =  pd.merge(messages, categories, on="id")
    print('-----------end load_data------------')
    return df


def clean_data(df):
    print('-----------start clean data------------')
    print('explain categories')
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.head(1).values

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row
    categories.columns = category_colnames[0]
    columns = []
    for column in categories:
        # set each value to be the last character of the string
        value = column
        categories[value] = categories[column].str.strip().str[-1]
        columns.append(value.split('-')[0])
    categories.columns = columns
    df = df.drop('categories', axis=1)
    df['index'] = df.index
    categories['index'] = categories.index
    df = pd.merge(df, categories, on='index', how='inner')
    df = df.drop('index', axis=1)
    df = df.drop(df[df['related'] == '2'].index)
    print('number duplicated', df.duplicated().sum())
    print('drop_duplicates')
    df = df.drop_duplicates()
    print('number duplicated', df.duplicated().sum())
    print('-----------end clean data------------')
    return df
    


def save_data(df, database_filename):
    print('-----------start save_data------------')
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename.replace('.db',''), engine, index=False)
    print('-----------end save_data------------')


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
