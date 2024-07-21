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

from anthropic import AnthropicVertex

formatter = logging.Formatter('%(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


logger = setup_logger('first','claude2.log')

try:

    LOCATION = "us-east5"



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
    df['datePublished'] = pd.to_datetime(df['datePublished'], format='mixed', errors='coerce')

    df['abstract_length'] = df['abstract'].apply(lambda x: len(x.split(' ')))
    df = df[df['abstract_length']>50]
    df = df[2042:10000]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

    project_id = 'research-language-bias'


    client = AnthropicVertex(region=LOCATION, project_id=project_id)


    claude_abs = []
    print('Starting Claude')

    for i in range(len(df)):


        # print(i)
        abs = df.iloc[i,9]
        try:


            message = client.messages.create(
            max_tokens=1024,
            messages=[
            {
            "role": "user",
            "content": f'''Given the scientific document abstract, rewrite this document.
            The document abstract is : {abs}''',
            }
        ],
        model="claude-3-opus@20240229",
        )
        
        
            claude_abs.append(message.content[0].text)
            logger.info((df.iloc[i,1],message.content[0].text ))
        except:
            claude_abs.append('')

    print('Claude Done')

    df['claude_abs'] = claude_abs
    df.to_csv('claude.csv')

except:
    print('Claude Error')