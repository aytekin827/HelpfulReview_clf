import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from adabelief_pytorch import AdaBelief
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import time
import datetime

start = time.time()
starttime = datetime.datetime.now()

# torch 설치 및 환경버전 확인
print('>> torch 버전확인 및 GPU divice 확인 <<')
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
print('-=-'*50,'\n')

# 모델링

# hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
MAX_LEN = 256 
device = torch.device("cuda")
log_interval=200
print('>> hyperparameter <<')
print('batch_size :{}'.format(BATCH_SIZE))
print('NUM_EPOCHS :{}'.format(NUM_EPOCHS))
print('MAX_LEN :{}'.format(MAX_LEN))
print('device :{}'.format(device))
print('log_interval :{}'.format(log_interval))
print('-=-'*30,'\n')

# pretrained Base Model (108M)
model_name = "beomi/kcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2)
print('>> model_name :{} <<'.format(model_name))
print('-=-'*30,'\n')

# optimizer
model.to(device)
optimizer = AdaBelief(model.parameters(), lr=1e-5, eps=1e-16, betas=(0.9,0.999), weight_decouple = True, rectify = True)
print('>> optimizer :{} <<'.format(optimizer))
print('-=-'*30,'\n')

# dataset class definition
class TrainDataset(Dataset):
    def __init__(self, df):
        self.df_data = df
    def __getitem__(self, index):
        # get the sentence from the dataframe
        sentence = self.df_data.loc[index, 'input']
        encoded_dict = tokenizer(
          text = sentence,
          add_special_tokens = True, 
          max_length = MAX_LEN,
          pad_to_max_length = True,
          truncation=True,           # Pad & truncate all sentences.
          return_tensors="pt")

        padded_token_list = encoded_dict['input_ids'][0]
        token_type_id = encoded_dict['token_type_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        target = torch.tensor(self.df_data.loc[index, "label"]) # 감정라벨의 값
        sample = (padded_token_list, token_type_id , att_mask, target) 
        return sample
    def __len__(self):
        return len(self.df_data)
        
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
          pad_to_max_length = True,
          truncation=True,           # Pad & truncate all sentences.
          return_tensors="pt")

        padded_token_list = encoded_dict['input_ids'][0]
        token_type_id = encoded_dict['token_type_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        sample = (padded_token_list, token_type_id , att_mask)
        return sample
    def __len__(self):
        return len(self.df_data)

# accuracy 계산하는 함수
def calc_accuracy(X,Y):
    # 텐서의 최대값 value 와 indices(위치)를 반환
    max_vals, max_indices = torch.max(X, 1) 
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

# EarlyStopper
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.occurred = False

    def __str__(self):
        return "Early stopper = patience :{}, min_delta :{}".format(self.patience, self.min_delta)
    def __repr__(self):
        return "Early stopper = patience :{}, min_delta :{}".format(self.patience, self.min_delta)

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.occurred = True
                return True
        return False

early_stopper = EarlyStopper(patience=3, min_delta=0.005)
print('>> early stop<<')
print(early_stopper)
print('-=-'*30,'\n')
category_name_list = ['브래지어','원피스','티셔츠']
for category_name in category_name_list:
        
    # 데이터 불러오기.
    print('>> 데이터 불러오기 <<')
    FilePath = 'data/review_label/review_label_{}.csv'.format(category_name)
    df = pd.read_csv(FilePath)
    print('데이터 경로: {}'.format(FilePath))
    print('-=-'*30,'\n')

    # 데이터 분리
    # 제품명 + 리뷰 텍스트 인풋으로 사용
    print('>> 데이터분리 및 df 형태 수정 <<')

    df['input'] = '제품명 ' + df['prod_name'].fillna('') + ' ' + '리뷰 ' + df['review']
    data = df['input']
    target = df['label']

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        target,
        test_size = 0.2,
        shuffle = True,
        random_state = 42
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size = 0.2,
        shuffle = True,
        random_state = 42
    )

    train = pd.DataFrame(x_train).join(pd.DataFrame(y_train).astype(int)).reset_index(drop=True)
    val = pd.DataFrame(x_val).join(pd.DataFrame(y_val).astype(int)).reset_index(drop=True)
    test = pd.DataFrame(x_test).join(pd.DataFrame(y_test).astype(int)).reset_index(drop=True)

    print('num total review :',df.shape[0])
    print('len(x_train) :',len(x_train))
    print('len(y_train) :',len(y_train))
    print('len(x_val) :',len(x_val))
    print('len(y_val) :',len(y_val))
    print('len(x_test) :',len(x_test))
    print('len(y_test) :',len(y_test))
    print(train.head(3))
    print('-=-'*30,'\n')

    # dataloader 객체 생성
    train_data = TrainDataset(train)
    val_data = TrainDataset(val)
    test_data = TestDataset(test)

    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
        )
    val_dataloader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
        )
    test_dataloader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
        )

    print('>> dataset 및 dataloader 만들기 객체 확인 <<')
    print(train['input'][0])
    print('-=-'*30,'\n')

    # scheduler 설정
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)       

    early_stopper = EarlyStopper(patience=3, min_delta=0.005)
    print('>> early stop<<')
    print(early_stopper)
    print('-=-'*30,'\n')

    # 학습하기
    print('>> 학습 start <<')
    for e in range(NUM_EPOCHS):
        train_acc = 0.0
        valid_acc = 0.0
        best_acc = 0.0

        model.train()
        torch.set_grad_enabled(True)
        for batch_id, (input_id, token_type_id, attention_mask, label) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            input_id = input_id.long().to(device)
            token_type_id = token_type_id.long().to(device)
            attention_mask = attention_mask.long().to(device)
            label = label.type(torch.LongTensor)
            label = label.to(device)
            outputs = model(input_ids=input_id, token_type_ids=token_type_id, attention_mask=attention_mask, labels=label)
            loss = outputs[0]
            out = outputs[1]
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            
            train_acc += calc_accuracy(out, label)
            if batch_id % log_interval == 0:
                print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
        print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
        
        model.eval()
        with torch.no_grad():
            for batch_id, (input_id, token_type_id, attention_mask, label) in enumerate(tqdm(val_dataloader)):
                input_id = input_id.long().to(device)
                token_type_id = token_type_id.long().to(device)
                attention_mask = attention_mask.long().to(device)
                label = label.type(torch.LongTensor)
                label = label.to(device)
                outputs = model(input_ids=input_id, token_type_ids=token_type_id, attention_mask=attention_mask, labels=label)
                loss = outputs[0]
                out = outputs[1]
                valid_acc += calc_accuracy(out, label)
        print("epoch {} valid acc {}".format(e+1, valid_acc / (batch_id+1)))
        if early_stopper.early_stop(loss):
            print(' ----- Early Stop Occurred! ----- ')
            break

        torch.save(model, 'model/model_{}/KcELECTRA_cred_{}_{}.pt'.format(category_name, e, category_name))

    print('>> 학습 finish <<')    
    print('-=-'*30,'\n')

    # 테스트데이터 검증
    print('>> 테스트 start <<')    
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
    Preds = np.array(preds)
    print('>> 테스트 finish <<')    
    print('-=-'*30,'\n')


    first = []
    second = []
    for i in Preds:
        first.append(i[0]) 
        second.append(i[1]) 
    test['first_%'] = first
    test['second_%'] = second

    results = np.argmax(Preds, axis=1)
    test['Pred']= results

    # 테스트 결과 점수
    print('>> 테스트 점수 <<')    
    print(len(test[test['label']==test['Pred']])/len(test))
    print('-=-'*30,'\n')

    # 오답 출력
    print('>> 오답예시 출력 <<')
    print(test[test['label']!=test['Pred']].head())
    print('-=-'*30,'\n')

    # 결과 저장
    print('>> 결과 저장 <<')
    test.to_csv('model/model_{}/results/KcELECTRA_testdata_{}.csv'.format(category_name, category_name))
    print('-=-'*30,'\n')

    # confusion matrix
    print(">> confusion matrix <<")

    y_true = []
    y_pred = []

    for i in test['label']:
        y_true.append(i)
    for i in test['Pred']:
        y_pred.append(i)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
    print('-=-'*30,'\n')


    # 소모시간 측정
    end = time.time()
    sec = (end - start)
    result_list = str(datetime.timedelta(seconds=sec)).split(".")
    endtime = datetime.datetime.now()
    print('소모시간 : ',result_list[0])

    with open(r'model/model_{}/desc.txt'.format(category_name), 'w', encoding='utf-8') as f:
        f.write('\n시작시간 : {}'.format(starttime))
        f.write('\n')

        f.write('\n카테고리 : {}'.format(category_name))
        f.write('\n')

        f.write('\nhyperparameter')
        f.write('\nbatch_size : {}'.format(BATCH_SIZE))
        f.write('\nnum_epochs : {}'.format(NUM_EPOCHS))
        f.write('\nmax_len : {}'.format(MAX_LEN))
        f.write('\n')

        f.write('\n어떤 pretrained model을 썻는지')
        f.write('\npretrained_model : {}'.format(model_name))
        f.write('\npretrained_model_tokenizer : {}'.format(tokenizer))
        f.write('\noptimizer : {}'.format(optimizer))
        f.write('\n')

        f.write('\n데이터 분리 및 개수')
        f.write('\nlen(x_train) :'.format(len(x_train)))
        f.write('\nlen(y_train) :'.format(len(y_train)))
        f.write('\nlen(x_val) :'.format(len(x_val)))
        f.write('\nlen(y_val) :'.format(len(y_val)))
        f.write('\nlen(x_test) :'.format(len(x_test)))
        f.write('\nlen(y_test) :'.format(len(y_test)))
        f.write('\n')

        f.write('\nearly stopper')
        f.write('\nearly_stopper : {}'.format(early_stopper))     
        f.write('\n')

        f.write('\n학습')
        f.write('\nepoch - {} valid_acc : {}'.format(e+1, valid_acc))
        f.write('\nearly_stop_occured :{}'.format(early_stopper.occurred))
        f.write('\n')
        
        f.write('\n테스트')
        f.write('\n테스트점수 : {}'.format(len(test[test['label']==test['Pred']])/len(test)))
        f.write('\nconfusion matrix :{}'.format(confusion_matrix(y_true, y_pred)))
        f.write('\nclassification report :{}'.format(classification_report(y_true, y_pred, digits=4)))
        f.write('\n')

        f.write('\n마친시간 : {}'.format(endtime))
        f.write('\n소모시간 : {}'.format(result_list[0]))