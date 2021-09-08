# user feature generation: RoBERTa with bio
from transformers import pipeline
import torch
from transformers import *
pretrained_weights = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(pretrained_weights)
feature_extractor = pipeline('feature-extraction', model = RobertaModel.from_pretrained(pretrained_weights), tokenizer = tokenizer, device = 1)

import torch
import json
import numpy
import datetime

user_features = []
f = open('entity_summary.txt')
i = 0
for line in f:
    print(i)
    i += 1
    text = line.strip()
    result = tokenizer(text)
    while len(result['input_ids']) > 500:
        text = text[:int(len(text)/2)]
        result = tokenizer(text)
    feature_temp = torch.tensor(feature_extractor(text))
    feature_temp = torch.mean(feature_temp.squeeze(0), dim=0).unsqueeze(0)
    user_features.append(feature_temp)

user_features = torch.stack(user_features)
torch.save(user_features, 'node_roberta.pt')
temp = torch.load('node_roberta.pt')
print(temp.size())