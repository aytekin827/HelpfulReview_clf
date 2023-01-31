import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd 
from tqdm import tqdm 
from sklearn.model_selection import train_test_split
from glob import glob

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

BATCH_SIZE = 1
MAX_LEN = 256

def load_cate_model(category_name):
    model_path = glob('model/model_{}/results/*.pt'.format(category_name))[-1]
    model = torch.load(model_path)

    model_name = "beomi/kcELECTRA-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device ='cpu'
    model.to(device)

    return model, tokenizer

def analyze_Bert(prod_name, review, model, tokenizer, device='cpu'):
    
    class TestDataset(Dataset):
        def __init__(self, df):
            self.df_data = df
        def __getitem__(self, index):
            # get the sentence from the dataframe
            sentence = self.df_data.loc[index, 'input']
            encoded_dict = tokenizer(
            text = sentence,
            add_special_tokens = True, 
            max_length = MAX_LEN,
            padding = True,
            truncation=True,           # Pad & truncate all sentences.
            return_tensors="pt")

            padded_token_list = encoded_dict['input_ids'][0]
            token_type_id = encoded_dict['token_type_ids'][0]
            att_mask = encoded_dict['attention_mask'][0]
            sample = (padded_token_list, token_type_id , att_mask)
            return sample
        def __len__(self):
            return len(self.df_data)

    review = pd.DataFrame({'review':[review],'prod_name':[prod_name]})
    review['input'] = '제품명 ' + review['prod_name'].fillna('') + ' ' + '리뷰 ' + review['review']

    test_data = TestDataset(review)
    test_dataloader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
        )

    # inference
    print('>> inference start <<')    
    preds = [] 
    model.eval()
    torch.set_grad_enabled(False)
    for batch_id, (input_id,token_type_id,attention_mask) in enumerate(tqdm(test_dataloader)):
        input_id = input_id.long().to(device)
        token_type_id = token_type_id.long().to(device)
        attention_mask = attention_mask.long().to(device)
        outputs = model(input_ids=input_id, token_type_ids=token_type_id, attention_mask=attention_mask)
        out = outputs[0]
        for inp in out:
            preds.append(inp.detach().cpu().numpy())
    Preds_percentage = np.array(preds)
    preds = np.argmax(Preds_percentage)

    return preds, Preds_percentage
