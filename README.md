# Disaster Response Project

## Summary
This project is a system for classifying messages during disasters using a classification model with 36 predefined categories (electricity, earthquake, ...) through which appropriate aid is sent. Project includes 3 parts:
1. Data: ETL data, create a suitable data set to train the model
2. Model: Build a model to classify messages during a disaster using a data set created by "Data".
3. App: Use data from the "Data" section and Model from the "Model" section to initialize the message classification system in a disaster

## File Description
~~~~~~~
        Disaster-Response-Project
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- disaster_message.csv
                |-- disaster_categories.csv
                |-- DisasterResponse.db
                |-- process_data.py
          |-- models
                |-- train_classifier.py
          |-- project_materials
                |-- categories.csv
                |-- ETL Pipeline Preparation.ipynb
                |-- messages.csv
                |-- ML Pipeline Preparation.ipynb
                |-- Twitter-sentiment-self-drive-DFE.csv
          |-- README
~~~~~~~
## Installation
Run `pip install requirements.txt`

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3000/

## Licensing, Authors, Acknowledgements
Figure-8, Udacity

### NOTICE: project_materials folder is not necessary for this project to run.
