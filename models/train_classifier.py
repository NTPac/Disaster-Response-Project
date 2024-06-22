import sys
# import libraries
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
import re
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath: str):
    """Load data

    Args:
        database_filepath (str): filepath

    Returns:
        _type_: X, Y, list categlories
    """
    print('-----------start load_data------------')
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(database_filepath.replace('.db',''), engine)
    related_id = list(df.columns).index("related")
    df = df.drop(df[~df['related'].isin([0,1])].index)
    df = df.drop(df[~df['related'].isin(["0","1"])].index)
    X = df.message
    Y = df[df.columns[related_id:]]
    cols = list(Y.columns)
    print('-----------end load_data------------')
    return X, Y, cols


def tokenize(text):
    print('-----------start tokenize------------')
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    print('-----------end tokenize------------')
    return clean_tokens

def build_model():
    print('-----------start build_model------------')
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    print('-----------end build_model------------')
    return pipeline
    


def evaluate_model(model, X_test, Y_test, category_names):
    print('-----------start evaluate_model------------')
    Y_pred = model.predict(X_test)
    for col in category_names:
        index = category_names.index(col)
        print(col, ':')
        print(classification_report(Y_test[col], Y_pred[:,index], target_names=category_names))
        print('----------------------------------------------------------------------')
    print('-----------end evaluate_model------------')


def save_model(model, model_filepath):
    # Assuming you have a trained model called 'model'
    # and you want to export it as a pickle file

    # Save the model as a pickle file
    print('-----------start save_model------------')
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    print('-----------end save_model------------')


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