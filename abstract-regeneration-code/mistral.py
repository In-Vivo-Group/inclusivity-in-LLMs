import pandas as pd
import lzma
import json
import numpy as np

import transformers
import torch

from gender_extractor import GenderExtractor

import logging

# try:
        

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
df=df[:10000]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )


model_id = "mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = transformers.BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)


mistral_tokenizer = transformers.AutoTokenizer.from_pretrained(
model_id,
)

mistral_model = transformers.AutoModelForCausalLM.from_pretrained(
model_id,
trust_remote_code=True,
quantization_config=bnb_config,
device_map='auto',
)

formatter = logging.Formatter('%(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


logger = setup_logger('first','mistral.log')



outs = []
for i in range(len(df)):
    try:
        abs = df.iloc[i,9]
        
        prompt = f'''Given the dollowing scientific document abstract, rewrite the document
            document: {abs}

        Give the output in the format "rewritten document": the rewritten document
        '''
            

        message = [
            {"role": "user", "content": f"{prompt}"}
        ]

        encodeds = mistral_tokenizer.apply_chat_template(message, return_tensors="pt")

        model_inputs = encodeds.to(device)
        # mistral_model#.to(device)

        generated_ids = mistral_model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = mistral_tokenizer.batch_decode(generated_ids)
        # print(decoded[0])
        

        raw_model_output = decoded[0]
        
        outs.append(raw_model_output)


        logger.info((df.iloc[i,1],raw_model_output))
    except:
        outs.append('')
    # break

df['mistral'] = outs
df.to_csv('mistral.csv')

# except:
    # print('mistral error')