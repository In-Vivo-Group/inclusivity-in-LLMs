import pandas as pd
import lzma
import json
import numpy as np
import transformers
import torch
from gender_extractor import GenderExtractor
import logging
import vertexai
from vertexai.generative_models import GenerativeModel
import logging


formatter = logging.Formatter('%(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


logger = setup_logger('first','gemini.log')

try:

    data = []
    for i in range(1,4650):
        if i==10:
            break
        try:
            #for data refer to "https://core.ac.uk/documentation/dataset" and replace the './core_2018-03-01_fulltext/{i}.json.xz' with the lolcation
            with lzma.open(f'./core_2018-03-01_fulltext/{i}.json.xz', mode='rt') as file:
                for line in file:
                    data.append(line)
        except:
            pass


    data = [json.loads(i) for i in data]
    df = pd.DataFrame.from_records(data)
    df = df.dropna(subset=['abstract'])
    df = df.drop(['fullText'],axis=1,inplace=False)
    df['datePublished'] = pd.to_datetime(df['datePublished'], format='mixed', errors='coerce')

    df['abstract_length'] = df['abstract'].apply(lambda x: len(x.split(' ')))
    df = df[df['abstract_length']>50]
    df = df[:10000]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

    project_id = 'research-language-bias'


    vertexai.init(project=project_id, location="us-central1")
    model = GenerativeModel(model_name="gemini-1.5-flash-001")

    gemini_abs = []
    print('Starting Gemini')
    for i in range(len(df)):
        if (i==len(df)//2):
            print('Half Gemini Done')
        abs = df.iloc[i,9]
        try:
            response = model.generate_content(f'''Given the scientific document abstract, rewrite this document.
        The document abstract is : {abs}''')

            gemini_abs.append(response.text)
            logger.info((df.iloc[i,2],response.text))
        except:
            gemini_abs.append('')
    print('Gemini Done')

    df['gemini_abs'] = gemini_abs
    df.to_csv('gemini.csv')



except Exception as e:
    print('gemini error: ',e)