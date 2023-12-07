import torch

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# dataset
TRAINSET_RATIO = 0.2
# data path
RAW_DATA_PATH = 'data/ccks2019/'
PROCESSED_DATA_PATH = 'data/processed_data/'
SAVED_MODEL_PATH = 'saved_model/'
# model parameter
BASE_MODEL = 'bert-base-chinese'
EMBEDDING_DIM = 768
HIDDEN_DIM = 256
# train parameter
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 50

# tag&label
label_dict = {'药物': 'DRUG',
              '解剖部位': 'BODY',
              '疾病和诊断': 'DISEASES',
              '影像检查': 'EXAMINATIONS',
              '实验室检验': 'TEST',
              '手术': 'TREATMENT'}
label_dict2 = {'DRUG': '药物',
               'BODY': '解剖部位',
               'DISEASES': '疾病和诊断',
               'EXAMINATIONS': '影像检查',
               'TEST': '实验室检验',
               'TREATMENT': '手术'}
model_tag = ('<PAD>', '[CLS]', '[SEP]', 'O', 'B-BODY', 'I-TEST', 'I-EXAMINATIONS',
             'I-TREATMENT', 'B-DRUG', 'B-TREATMENT', 'I-DISEASES', 'B-EXAMINATIONS',
             'I-BODY', 'B-TEST', 'B-DISEASES', 'I-DRUG')
tag2idx = {tag: idx for idx, tag in enumerate(model_tag)}
idx2tag = {idx: tag for idx, tag in enumerate(model_tag)}
LABELS = ['B-BODY', 'B-DISEASES', 'B-DRUG', 'B-EXAMINATIONS', 'B-TEST', 'B-TREATMENT',
          'I-BODY', 'I-DISEASES', 'I-DRUG', 'I-EXAMINATIONS', 'I-TEST', 'I-TREATMENT']
