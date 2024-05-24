import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    EXCEL_DIRECTORY = "path/to/excel/files"
    MODEL_API_URL = os.environ.get('MODEL_API_URL') or 'http://model-api-url'