
''' conspiracy news identification based on event relation graph '''

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'

import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


import pandas as pd
import numpy as np
import json
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import LongformerTokenizer, LongformerModel
import math
import random
import sklearn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import Counter
from scipy.optimize import linear_sum_assignment
import subprocess


''' split method and create train/dev/test file paths '''

split_method = "media_split"
# split_method = "random_split"
print(split_method)

dev_file_paths = []
dev_conspiracy_path = "./LOCO_" + split_method + "_event_graph/dev_conspiracy"
dev_conspiracy_files = os.listdir(dev_conspiracy_path)
for file_i in range(len(dev_conspiracy_files)):
    dev_file_paths.append(dev_conspiracy_path + "/" + dev_conspiracy_files[file_i])
dev_mainstream_path = "./LOCO_" + split_method + "_event_graph/dev_mainstream"
dev_mainstream_files = os.listdir(dev_mainstream_path)
for file_i in range(len(dev_mainstream_files)):
    dev_file_paths.append(dev_mainstream_path + "/" + dev_mainstream_files[file_i])

test_file_paths = []
test_conspiracy_path = "./LOCO_" + split_method + "_event_graph/test_conspiracy"
test_conspiracy_files = os.listdir(test_conspiracy_path)
for file_i in range(len(test_conspiracy_files)):
    test_file_paths.append(test_conspiracy_path + "/" + test_conspiracy_files[file_i])
test_mainstream_path = "./LOCO_" + split_method + "_event_graph/test_mainstream"
test_mainstream_files = os.listdir(test_mainstream_path)
for file_i in range(len(test_mainstream_files)):
    test_file_paths.append(test_mainstream_path + "/" + test_mainstream_files[file_i])

train_file_paths = []
train_conspiracy_path = "./LOCO_" + split_method + "_event_graph/train_conspiracy"
train_conspiracy_files = os.listdir(train_conspiracy_path)
for file_i in range(len(train_conspiracy_files)):
    train_file_paths.append(train_conspiracy_path + "/" + train_conspiracy_files[file_i])
train_mainstream_path = "./LOCO_" + split_method + "_event_graph/train_mainstream"
train_mainstream_files = os.listdir(train_mainstream_path)
for file_i in range(len(train_mainstream_files)):
    train_file_paths.append(train_mainstream_path + "/" + train_mainstream_files[file_i])






''' hyper-parameter '''

MAX_LEN = 2048
num_epochs = 2
batch_size = 1
check_times = 4 * num_epochs

no_decay = ['bias', 'LayerNorm.weight']
longformer_weight_decay = 1e-2
non_longformer_weight_decay = 1e-2

warmup_proportion = 0.1
non_longformer_lr = 1e-4
longformer_lr = 1e-5





''' custom dataset '''

tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

class custom_dataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]

        with open(file_path, "r") as in_json:
            article_json = json.load(in_json)

        if 'conspiracy' in file_path:
            label_article = 1
        else:
            label_article = 0

        label_article = torch.tensor(label_article)


        input_ids = []
        attention_mask = []

        event_words = []
        # event_words[i,:] = [start, end, index of sentence, index of token, prob_event[0], prob_event[1], label_event = 1]
        # input_ids[range(start, end)] can extract the corresponding event word

        stop_flag = 0

        input_ids.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['input_ids'])  # the <s> start token
        attention_mask.extend(tokenizer.encode_plus('<s>', add_special_tokens=False)['attention_mask'])

        for sent_i in range(len(article_json['sentences'])):
            if stop_flag == 1:
                break
            for token_i in range(len(article_json['sentences'][sent_i]['tokens'])):
                token_text = article_json['sentences'][sent_i]['tokens'][token_i]['token_text']
                start = len(input_ids)
                word_encoding = tokenizer.encode_plus(' ' + token_text, add_special_tokens=False)

                if start + len(word_encoding['input_ids']) < MAX_LEN:  # truncate to MAX_LEN
                    input_ids.extend(word_encoding['input_ids'])
                    attention_mask.extend(word_encoding['attention_mask'])
                    end = len(input_ids)
                    if article_json['sentences'][sent_i]['tokens'][token_i]['prob_event'][0] != 0:  # this token participates in event graph
                        if article_json['sentences'][sent_i]['tokens'][token_i]['label_event'] == 1: # this token is an event word
                            event_words.append([start, end, sent_i, article_json['sentences'][sent_i]['tokens'][token_i]['index_of_token'],
                                                article_json['sentences'][sent_i]['tokens'][token_i]['prob_event'][0],
                                                article_json['sentences'][sent_i]['tokens'][token_i]['prob_event'][1],
                                                article_json['sentences'][sent_i]['tokens'][token_i]['label_event']])
                else:
                    stop_flag = 1
                    break

        input_ids.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['input_ids'])  # the </s> end token
        attention_mask.extend(tokenizer.encode_plus('</s>', add_special_tokens=False)['attention_mask'])

        num_pad = MAX_LEN - len(input_ids)
        if num_pad > 0:
            input_ids.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['input_ids'])  # the <pad> padding token
            attention_mask.extend(tokenizer.encode_plus('<pad>' * num_pad, add_special_tokens=False)['attention_mask'])

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        event_words = torch.tensor(event_words)

        event_pairs = []
        # event_pairs[i,:] = [event 1 row in event_words, event 2 row in event_words]
        label_coreference = []
        label_temporal = []
        label_causal = []
        label_subevent = []
        # label_relation[i,:] = [prob_relation[0], prob_relation[1], prob_relation[2], label_relation] of event_pairs[i,:]

        for event_pair_i in range(len(article_json['relation_label'])):
            event_1_index = article_json['relation_label'][event_pair_i]['event_1']['index_of_token']
            event_2_index = article_json['relation_label'][event_pair_i]['event_2']['index_of_token']
            event_1_row_in_event_words = int(torch.argwhere(event_words[:, 3] == event_1_index))
            event_2_row_in_event_words = int(torch.argwhere(event_words[:, 3] == event_2_index))
            event_pairs.append([event_1_row_in_event_words, event_2_row_in_event_words])

            label_coreference.append([article_json['relation_label'][event_pair_i]['prob_coreference'][0],
                                     article_json['relation_label'][event_pair_i]['prob_coreference'][1],
                                     article_json['relation_label'][event_pair_i]['label_coreference']])
            label_temporal.append([article_json['relation_label'][event_pair_i]['prob_temporal'][0],
                                  article_json['relation_label'][event_pair_i]['prob_temporal'][1],
                                  article_json['relation_label'][event_pair_i]['prob_temporal'][2],
                                  article_json['relation_label'][event_pair_i]['prob_temporal'][3],
                                  article_json['relation_label'][event_pair_i]['label_temporal']])
            label_causal.append([article_json['relation_label'][event_pair_i]['prob_causal'][0],
                                article_json['relation_label'][event_pair_i]['prob_causal'][1],
                                article_json['relation_label'][event_pair_i]['prob_causal'][2],
                                article_json['relation_label'][event_pair_i]['label_causal']])
            label_subevent.append([article_json['relation_label'][event_pair_i]['prob_subevent'][0],
                                  article_json['relation_label'][event_pair_i]['prob_subevent'][1],
                                  article_json['relation_label'][event_pair_i]['prob_subevent'][2],
                                  article_json['relation_label'][event_pair_i]['label_subevent']])

        event_pairs = torch.tensor(event_pairs)
        label_coreference = torch.tensor(label_coreference)
        label_temporal = torch.tensor(label_temporal)
        label_causal = torch.tensor(label_causal)
        label_subevent = torch.tensor(label_subevent)

        dict = {"input_ids": input_ids, "attention_mask": attention_mask, "label_article": label_article,
                "event_words": event_words, "event_pairs": event_pairs, "label_coreference": label_coreference,
                "label_temporal": label_temporal, "label_causal": label_causal, "label_subevent": label_subevent}

        return dict




''' input_ids for relation semantic '''

coreference_input_ids = torch.tensor(tokenizer.encode_plus(' coreference', add_special_tokens=False)['input_ids']).view(1, 2).to(device)
coreference_attention_mask = torch.tensor(tokenizer.encode_plus(' coreference', add_special_tokens=False)['attention_mask']).view(1, 2).to(device)
before_input_ids = torch.tensor(tokenizer.encode_plus(' before', add_special_tokens=False)['input_ids']).view(1, 1).to(device)
before_attention_mask = torch.tensor(tokenizer.encode_plus(' before', add_special_tokens=False)['attention_mask']).view(1, 1).to(device)
after_input_ids = torch.tensor(tokenizer.encode_plus(' after', add_special_tokens=False)['input_ids']).view(1, 1).to(device)
after_attention_mask = torch.tensor(tokenizer.encode_plus(' after', add_special_tokens=False)['attention_mask']).view(1, 1).to(device)
overlap_input_ids = torch.tensor(tokenizer.encode_plus(' overlap', add_special_tokens=False)['input_ids']).view(1, 1).to(device)
overlap_attention_mask = torch.tensor(tokenizer.encode_plus(' overlap', add_special_tokens=False)['attention_mask']).view(1, 1).to(device)
cause_input_ids = torch.tensor(tokenizer.encode_plus(' cause', add_special_tokens=False)['input_ids']).view(1, 1).to(device)
cause_attention_mask = torch.tensor(tokenizer.encode_plus(' cause', add_special_tokens=False)['attention_mask']).view(1, 1).to(device)
caused_input_ids = torch.tensor(tokenizer.encode_plus(' caused by', add_special_tokens=False)['input_ids']).view(1, 2).to(device)
caused_attention_mask = torch.tensor(tokenizer.encode_plus(' caused by', add_special_tokens=False)['attention_mask']).view(1, 2).to(device)
contain_input_ids = torch.tensor(tokenizer.encode_plus(' contain', add_special_tokens=False)['input_ids']).view(1, 1).to(device)
contain_attention_mask = torch.tensor(tokenizer.encode_plus(' contain', add_special_tokens=False)['attention_mask']).view(1, 1).to(device)
contained_input_ids = torch.tensor(tokenizer.encode_plus(' contained by', add_special_tokens=False)['input_ids']).view(1, 2).to(device)
contained_attention_mask = torch.tensor(tokenizer.encode_plus(' contained by', add_special_tokens=False)['attention_mask']).view(1, 2).to(device)




''' model '''


class Token_Embedding(nn.Module):

    # input: input_ids, attention_mask, 1 article * number of tokens
    # output: number of tokens * 768, dealing with one article at one time

    def __init__(self):
        super(Token_Embedding, self).__init__()

        self.longformermodel = LongformerModel.from_pretrained('allenai/longformer-base-4096', output_hidden_states=True, )

    def forward(self, input_ids, attention_mask):

        outputs = self.longformermodel(input_ids = input_ids, attention_mask = attention_mask)
        hidden_states = outputs[2]
        token_embeddings_layers = torch.stack(hidden_states, dim=0)  # 13 layer * batch_size (1) * number of tokens * 768
        token_embeddings_layers = token_embeddings_layers[:, 0, :, :] # 13 layer * number of tokens * 768
        token_embeddings = torch.sum(token_embeddings_layers[-4:, :, :], dim = 0) # sum up the last four layers, number of tokens * 768

        return token_embeddings


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.token_embedding = Token_Embedding()

        self.bilstm = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)

        self.conspiracy_1 = nn.Linear(768 * 2, 768, bias=True)
        nn.init.xavier_uniform_(self.conspiracy_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.conspiracy_1.bias)

        self.conspiracy_2 = nn.Linear(768, 2, bias=True)
        nn.init.xavier_uniform_(self.conspiracy_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.conspiracy_2.bias)

        self.event_head_1 = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.event_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.event_head_1.bias)

        self.event_head_2 = nn.Linear(768, 2, bias=True)
        nn.init.xavier_uniform_(self.event_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.event_head_2.bias)

        self.coreference_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.coreference_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.coreference_head_1.bias)

        self.coreference_head_2 = nn.Linear(768, 2, bias=True)
        nn.init.xavier_uniform_(self.coreference_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.coreference_head_2.bias)

        self.temporal_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.temporal_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_1.bias)

        self.temporal_head_2 = nn.Linear(768, 4, bias=True)
        nn.init.xavier_uniform_(self.temporal_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.temporal_head_2.bias)

        self.causal_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.causal_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_1.bias)

        self.causal_head_2 = nn.Linear(768, 3, bias=True)
        nn.init.xavier_uniform_(self.causal_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.causal_head_2.bias)

        self.subevent_head_1 = nn.Linear(768 * 4, 768, bias=True)
        nn.init.xavier_uniform_(self.subevent_head_1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.subevent_head_1.bias)

        self.subevent_head_2 = nn.Linear(768, 3, bias=True)
        nn.init.xavier_uniform_(self.subevent_head_2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.subevent_head_2.bias)

        self.W_gat = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_gat.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.zeros_(self.W_gat.bias)

        self.a_gat = nn.Linear(768 * 2, 1, bias=True)
        nn.init.xavier_uniform_(self.a_gat.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.zeros_(self.a_gat.bias)

        self.W_coreference_r = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_coreference_r.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_coreference_r.bias)

        self.W_coreference_K = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_coreference_K.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_coreference_K.bias)

        self.W_coreference_Q = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_coreference_Q.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_coreference_Q.bias)

        self.W_coreference_V = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_coreference_V.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_coreference_V.bias)

        self.W_before_r = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_before_r.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_before_r.bias)

        self.W_before_K = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_before_K.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_before_K.bias)

        self.W_before_Q = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_before_Q.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_before_Q.bias)

        self.W_before_V = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_before_V.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_before_V.bias)

        self.W_after_r = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_after_r.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_after_r.bias)

        self.W_after_K = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_after_K.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_after_K.bias)

        self.W_after_Q = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_after_Q.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_after_Q.bias)

        self.W_after_V = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_after_V.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_after_V.bias)

        self.W_overlap_r = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_overlap_r.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_overlap_r.bias)

        self.W_overlap_K = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_overlap_K.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_overlap_K.bias)

        self.W_overlap_Q = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_overlap_Q.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_overlap_Q.bias)

        self.W_overlap_V = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_overlap_V.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_overlap_V.bias)

        self.W_cause_r = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_cause_r.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_cause_r.bias)

        self.W_cause_K = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_cause_K.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_cause_K.bias)

        self.W_cause_Q = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_cause_Q.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_cause_Q.bias)

        self.W_cause_V = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_cause_V.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_cause_V.bias)

        self.W_caused_r = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_caused_r.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_caused_r.bias)

        self.W_caused_K = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_caused_K.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_caused_K.bias)

        self.W_caused_Q = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_caused_Q.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_caused_Q.bias)

        self.W_caused_V = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_caused_V.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_caused_V.bias)

        self.W_contain_r = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_contain_r.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_contain_r.bias)

        self.W_contain_K = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_contain_K.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_contain_K.bias)

        self.W_contain_Q = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_contain_Q.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_contain_Q.bias)

        self.W_contain_V = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_contain_V.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_contain_V.bias)

        self.W_contained_r = nn.Linear(768 * 3, 768, bias=True)
        nn.init.xavier_uniform_(self.W_contained_r.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_contained_r.bias)

        self.W_contained_K = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_contained_K.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_contained_K.bias)

        self.W_contained_Q = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_contained_Q.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_contained_Q.bias)

        self.W_contained_V = nn.Linear(768, 768, bias=True)
        nn.init.xavier_uniform_(self.W_contained_V.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.W_contained_V.bias)

        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.softmax_0 = nn.Softmax(dim = 0)
        self.softmax_1 = nn.Softmax(dim = 1)
        self.crossentropyloss = nn.CrossEntropyLoss(reduction='mean')
        self.crossentropyloss_sum = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, input_ids, attention_mask, label_article, event_words, event_pairs, label_coreference, label_temporal, label_causal, label_subevent):

        token_embeddings = self.token_embedding(input_ids, attention_mask) # number of tokens * 768

        # token-level bi-lstm layer
        token_embeddings = token_embeddings.view(1, token_embeddings.shape[0], token_embeddings.shape[1])

        h0 = torch.zeros(2, 1, 384).cuda().requires_grad_()
        c0 = torch.zeros(2, 1, 384).cuda().requires_grad_()

        token_embeddings, (_, _) = self.bilstm(token_embeddings, (h0, c0)) # batch_size 1 * number of tokens * 768
        token_embeddings = token_embeddings[0, :, :] # number of tokens * 768

        # event word embedding
        event_embedding = token_embeddings[event_words[:, 0].long()]

        # distill event identification
        event_scores = self.event_head_2(self.relu(self.event_head_1(event_embedding)))
        event_loss = self.crossentropyloss(event_scores, event_words[:, 4:6])

        # event pair embedding
        event_1_embedding = event_embedding[event_pairs[:, 0]]
        event_2_embedding = event_embedding[event_pairs[:, 1]]
        pair_element_wise_sub = torch.sub(event_1_embedding, event_2_embedding)
        pair_element_wise_mul = torch.mul(event_1_embedding, event_2_embedding)
        event_pair_embedding = torch.cat((event_1_embedding, event_2_embedding, pair_element_wise_sub, pair_element_wise_mul), dim=1)

        # distill coreference relation
        coreference_scores = self.coreference_head_2(self.relu(self.coreference_head_1(event_pair_embedding)))
        coreference_loss = self.crossentropyloss(coreference_scores, label_coreference[:, :2])

        # distill temporal relation
        temporal_scores = self.temporal_head_2(self.relu(self.temporal_head_1(event_pair_embedding)))
        temporal_loss = self.crossentropyloss(temporal_scores, label_temporal[:, :4])

        # distill causal relation
        causal_scores = self.causal_head_2(self.relu(self.causal_head_1(event_pair_embedding)))
        causal_loss = self.crossentropyloss(causal_scores, label_causal[:, :3])

        # distill subevent relation
        subevent_scores = self.subevent_head_2(self.relu(self.subevent_head_1(event_pair_embedding)))
        subevent_loss = self.crossentropyloss(subevent_scores, label_subevent[:, :3])


        # relation embedding
        coreference = torch.mean(self.token_embedding(coreference_input_ids, coreference_attention_mask), dim=0).view(1, 768)
        before = torch.mean(self.token_embedding(before_input_ids, before_attention_mask), dim=0).view(1, 768)
        after = torch.mean(self.token_embedding(after_input_ids, after_attention_mask), dim=0).view(1, 768)
        overlap = torch.mean(self.token_embedding(overlap_input_ids, overlap_attention_mask), dim=0).view(1, 768)
        cause = torch.mean(self.token_embedding(cause_input_ids, cause_attention_mask), dim=0).view(1, 768)
        caused = torch.mean(self.token_embedding(caused_input_ids, caused_attention_mask), dim=0).view(1, 768)
        contain = torch.mean(self.token_embedding(contain_input_ids, contain_attention_mask), dim=0).view(1, 768)
        contained = torch.mean(self.token_embedding(contained_input_ids, contained_attention_mask), dim=0).view(1, 768)

        # edge-aware graph attention network
        tentative_event_embed_coreference = torch.zeros(event_embedding.shape).to(device)
        tentative_event_embed_before = torch.zeros(event_embedding.shape).to(device)
        tentative_event_embed_after = torch.zeros(event_embedding.shape).to(device)
        tentative_event_embed_overlap = torch.zeros(event_embedding.shape).to(device)
        tentative_event_embed_cause = torch.zeros(event_embedding.shape).to(device)
        tentative_event_embed_caused = torch.zeros(event_embedding.shape).to(device)
        tentative_event_embed_contain = torch.zeros(event_embedding.shape).to(device)
        tentative_event_embed_contained = torch.zeros(event_embedding.shape).to(device)

        torch_zeros = torch.zeros(768).to(device)

        new_event_embedding = torch.zeros(event_embedding.shape).to(device)

        # coreference relation
        if torch.argwhere(label_coreference[:, 2] == 1).shape[0] != 0: # there exist coreference relation
            event_index_in_coreference = torch.unique(torch.cat((event_pairs[torch.argwhere(label_coreference[:, 2] == 1)[:, 0]][:, 0],
                                                                 event_pairs[torch.argwhere(label_coreference[:, 2] == 1)[:, 0]][:, 1])))
            for event_i in range(event_index_in_coreference.shape[0]):
                event_index = event_index_in_coreference[event_i]
                connected_events_index = torch.cat((event_pairs[torch.argwhere(label_coreference[:, 2] == 1)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_coreference[:, 2] == 1)[:, 0]][:, 0] == event_index)[:, 0]][:, 1],
                                                    event_pairs[torch.argwhere(label_coreference[:, 2] == 1)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_coreference[:, 2] == 1)[:, 0]][:, 1] == event_index)[:, 0]][:, 0]))
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_coreference_r(torch.cat((this_event_embed.repeat(connected_event_embed.shape[0], 1),
                                                                   coreference.repeat(connected_event_embed.shape[0], 1),
                                                                   connected_event_embed), dim=1))
                tentative_event_embed_coreference[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_coreference_Q(this_event_embed), self.W_coreference_K(relation_feature).t())),
                                                                           self.W_coreference_V(relation_feature))[0, :]

        # temporal relation
        if torch.argwhere(label_temporal[:, 4] == 1).shape[0] != 0:  # there exist before relation
            event_1_index_in_before = torch.unique(event_pairs[torch.argwhere(label_temporal[:, 4] == 1)[:, 0]][:, 0])
            for event_i in range(event_1_index_in_before.shape[0]):
                event_index = event_1_index_in_before[event_i]
                connected_events_index = event_pairs[torch.argwhere(label_temporal[:, 4] == 1)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_temporal[:, 4] == 1)[:, 0]][:, 0] == event_index)[:, 0]][:, 1]
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_before_r(torch.cat((this_event_embed.repeat(connected_event_embed.shape[0], 1),
                                                              before.repeat(connected_event_embed.shape[0], 1),
                                                              connected_event_embed), dim=1))
                tentative_event_embed_before[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_before_Q(this_event_embed), self.W_before_K(relation_feature).t())),
                                                                      self.W_before_V(relation_feature))[0, :]

            event_2_index_in_before = torch.unique(event_pairs[torch.argwhere(label_temporal[:, 4] == 1)[:, 0]][:, 1])
            for event_i in range(event_2_index_in_before.shape[0]):
                event_index = event_2_index_in_before[event_i]
                connected_events_index = event_pairs[torch.argwhere(label_temporal[:, 4] == 1)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_temporal[:, 4] == 1)[:, 0]][:, 1] == event_index)[:, 0]][:, 0]
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_after_r(torch.cat((connected_event_embed,
                                                             before.repeat(connected_event_embed.shape[0], 1),
                                                             this_event_embed.repeat(connected_event_embed.shape[0], 1)), dim=1))
                tentative_event_embed_after[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_after_Q(this_event_embed), self.W_after_K(relation_feature).t())),
                                                                     self.W_after_V(relation_feature))[0, :]

        if torch.argwhere(label_temporal[:, 4] == 2).shape[0] != 0:  # there exist after relation
            event_1_index_in_after = torch.unique(event_pairs[torch.argwhere(label_temporal[:, 4] == 2)[:, 0]][:, 0])
            for event_i in range(event_1_index_in_after.shape[0]):
                event_index = event_1_index_in_after[event_i]
                connected_events_index = event_pairs[torch.argwhere(label_temporal[:, 4] == 2)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_temporal[:, 4] == 2)[:, 0]][:, 0] == event_index)[:, 0]][:, 1]
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_after_r(torch.cat((this_event_embed.repeat(connected_event_embed.shape[0], 1),
                                                             after.repeat(connected_event_embed.shape[0], 1),
                                                             connected_event_embed), dim=1))
                if torch.equal(tentative_event_embed_after[event_index], torch_zeros):
                    tentative_event_embed_after[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_after_Q(this_event_embed), self.W_after_K(relation_feature).t())),
                                                                         self.W_after_V(relation_feature))[0, :]
                else:
                    tentative_event_embed_after[event_index] = torch.mean(torch.cat((tentative_event_embed_after[event_index].view(1, 768),
                                                                                     torch.mm(self.softmax_1(torch.mm(self.W_after_Q(this_event_embed), self.W_after_K(relation_feature).t())),
                                                                                              self.W_after_V(relation_feature)).view(1, 768)), dim = 0), dim = 0)

            event_2_index_in_after = torch.unique(event_pairs[torch.argwhere(label_temporal[:, 4] == 2)[:, 0]][:, 1])
            for event_i in range(event_2_index_in_after.shape[0]):
                event_index = event_2_index_in_after[event_i]
                connected_events_index = event_pairs[torch.argwhere(label_temporal[:, 4] == 2)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_temporal[:, 4] == 2)[:, 0]][:, 1] == event_index)[:, 0]][:, 0]
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_before_r(torch.cat((connected_event_embed,
                                                              after.repeat(connected_event_embed.shape[0], 1),
                                                              this_event_embed.repeat(connected_event_embed.shape[0], 1)), dim=1))
                if torch.equal(tentative_event_embed_before[event_index], torch_zeros):
                    tentative_event_embed_before[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_before_Q(this_event_embed), self.W_before_K(relation_feature).t())),
                                                                          self.W_before_V(relation_feature))[0, :]
                else:
                    tentative_event_embed_before[event_index] = torch.mean(torch.cat((tentative_event_embed_before[event_index].view(1, 768),
                                                                                      torch.mm(self.softmax_1(torch.mm(self.W_before_Q(this_event_embed), self.W_before_K(relation_feature).t())),
                                                                                               self.W_before_V(relation_feature)).view(1, 768)), dim = 0), dim = 0)

        if torch.argwhere(label_temporal[:, 4] == 3).shape[0] != 0: # there exist overlap relation
            event_index_in_overlap = torch.unique(torch.cat((event_pairs[torch.argwhere(label_temporal[:, 4] == 3)[:, 0]][:, 0],
                                                             event_pairs[torch.argwhere(label_temporal[:, 4] == 3)[:, 0]][:, 1])))
            for event_i in range(event_index_in_overlap.shape[0]):
                event_index = event_index_in_overlap[event_i]
                connected_events_index = torch.cat((event_pairs[torch.argwhere(label_temporal[:, 4] == 3)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_temporal[:, 4] == 3)[:, 0]][:, 0] == event_index)[:, 0]][:, 1],
                                                    event_pairs[torch.argwhere(label_temporal[:, 4] == 3)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_temporal[:, 4] == 3)[:, 0]][:, 1] == event_index)[:, 0]][:, 0]))
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_overlap_r(torch.cat((this_event_embed.repeat(connected_event_embed.shape[0], 1),
                                                               overlap.repeat(connected_event_embed.shape[0], 1),
                                                               connected_event_embed), dim=1))
                tentative_event_embed_overlap[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_overlap_Q(this_event_embed), self.W_overlap_K(relation_feature).t())),
                                                                       self.W_overlap_V(relation_feature))[0, :]

        # causal relation
        if torch.argwhere(label_causal[:, 3] == 1).shape[0] != 0:  # there exist cause relation
            event_1_index_in_cause = torch.unique(event_pairs[torch.argwhere(label_causal[:, 3] == 1)[:, 0]][:, 0])
            for event_i in range(event_1_index_in_cause.shape[0]):
                event_index = event_1_index_in_cause[event_i]
                connected_events_index = event_pairs[torch.argwhere(label_causal[:, 3] == 1)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_causal[:, 3] == 1)[:, 0]][:, 0] == event_index)[:, 0]][:, 1]
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_cause_r(torch.cat((this_event_embed.repeat(connected_event_embed.shape[0], 1),
                                                             cause.repeat(connected_event_embed.shape[0], 1),
                                                             connected_event_embed), dim=1))
                tentative_event_embed_cause[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_cause_Q(this_event_embed), self.W_cause_K(relation_feature).t())),
                                                                     self.W_cause_V(relation_feature))[0, :]

            event_2_index_in_cause = torch.unique(event_pairs[torch.argwhere(label_causal[:, 3] == 1)[:, 0]][:, 1])
            for event_i in range(event_2_index_in_cause.shape[0]):
                event_index = event_2_index_in_cause[event_i]
                connected_events_index = event_pairs[torch.argwhere(label_causal[:, 3] == 1)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_causal[:, 3] == 1)[:, 0]][:, 1] == event_index)[:, 0]][:, 0]
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_caused_r(torch.cat((connected_event_embed,
                                                              cause.repeat(connected_event_embed.shape[0], 1),
                                                              this_event_embed.repeat(connected_event_embed.shape[0], 1)), dim=1))
                tentative_event_embed_caused[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_caused_Q(this_event_embed), self.W_caused_K(relation_feature).t())),
                                                                      self.W_caused_V(relation_feature))[0, :]

        if torch.argwhere(label_causal[:, 3] == 2).shape[0] != 0:  # there exist caused relation
            event_1_index_in_caused = torch.unique(event_pairs[torch.argwhere(label_causal[:, 3] == 2)[:, 0]][:, 0])
            for event_i in range(event_1_index_in_caused.shape[0]):
                event_index = event_1_index_in_caused[event_i]
                connected_events_index = event_pairs[torch.argwhere(label_causal[:, 3] == 2)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_causal[:, 3] == 2)[:, 0]][:, 0] == event_index)[:, 0]][:, 1]
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_caused_r(torch.cat((this_event_embed.repeat(connected_event_embed.shape[0], 1),
                                                              caused.repeat(connected_event_embed.shape[0], 1),
                                                              connected_event_embed), dim=1))
                if torch.equal(tentative_event_embed_caused[event_index], torch_zeros):
                    tentative_event_embed_caused[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_caused_Q(this_event_embed), self.W_caused_K(relation_feature).t())),
                                                                          self.W_caused_V(relation_feature))[0, :]
                else:
                    tentative_event_embed_caused[event_index] = torch.mean(torch.cat((tentative_event_embed_caused[event_index].view(1, 768),
                                                                                      torch.mm(self.softmax_1(torch.mm(self.W_caused_Q(this_event_embed), self.W_caused_K(relation_feature).t())),
                                                                                               self.W_caused_V(relation_feature)).view(1, 768)), dim = 0), dim = 0)

            event_2_index_in_caused = torch.unique(event_pairs[torch.argwhere(label_causal[:, 3] == 2)[:, 0]][:, 1])
            for event_i in range(event_2_index_in_caused.shape[0]):
                event_index = event_2_index_in_caused[event_i]
                connected_events_index = event_pairs[torch.argwhere(label_causal[:, 3] == 2)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_causal[:, 3] == 2)[:, 0]][:, 1] == event_index)[:, 0]][:, 0]
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_cause_r(torch.cat((connected_event_embed,
                                                             caused.repeat(connected_event_embed.shape[0], 1),
                                                             this_event_embed.repeat(connected_event_embed.shape[0], 1)), dim=1))
                if torch.equal(tentative_event_embed_cause[event_index], torch_zeros):
                    tentative_event_embed_cause[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_cause_Q(this_event_embed), self.W_cause_K(relation_feature).t())),
                                                                         self.W_cause_V(relation_feature))[0, :]
                else:
                    tentative_event_embed_cause[event_index] = torch.mean(torch.cat((tentative_event_embed_cause[event_index].view(1, 768),
                                                                                     torch.mm(self.softmax_1(torch.mm(self.W_cause_Q(this_event_embed), self.W_cause_K(relation_feature).t())),
                                                                                              self.W_cause_V(relation_feature)).view(1, 768)), dim = 0), dim = 0)

        # subevent relation
        if torch.argwhere(label_subevent[:, 3] == 1).shape[0] != 0:  # there exist contain relation
            event_1_index_in_contain = torch.unique(event_pairs[torch.argwhere(label_subevent[:, 3] == 1)[:, 0]][:, 0])
            for event_i in range(event_1_index_in_contain.shape[0]):
                event_index = event_1_index_in_contain[event_i]
                connected_events_index = event_pairs[torch.argwhere(label_subevent[:, 3] == 1)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_subevent[:, 3] == 1)[:, 0]][:, 0] == event_index)[:, 0]][:, 1]
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_contain_r(torch.cat((this_event_embed.repeat(connected_event_embed.shape[0], 1),
                                                               contain.repeat(connected_event_embed.shape[0], 1),
                                                               connected_event_embed), dim=1))
                tentative_event_embed_contain[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_contain_Q(this_event_embed), self.W_contain_K(relation_feature).t())),
                                                                       self.W_contain_V(relation_feature))[0, :]

            event_2_index_in_contain = torch.unique(event_pairs[torch.argwhere(label_subevent[:, 3] == 1)[:, 0]][:, 1])
            for event_i in range(event_2_index_in_contain.shape[0]):
                event_index = event_2_index_in_contain[event_i]
                connected_events_index = event_pairs[torch.argwhere(label_subevent[:, 3] == 1)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_subevent[:, 3] == 1)[:, 0]][:, 1] == event_index)[:, 0]][:, 0]
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_contained_r(torch.cat((connected_event_embed,
                                                                 contain.repeat(connected_event_embed.shape[0], 1),
                                                                 this_event_embed.repeat(connected_event_embed.shape[0], 1)), dim=1))
                tentative_event_embed_contained[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_contained_Q(this_event_embed), self.W_contained_K(relation_feature).t())),
                                                                         self.W_contained_V(relation_feature))[0, :]

        if torch.argwhere(label_subevent[:, 3] == 2).shape[0] != 0:  # there exist contained relation
            event_1_index_in_contained = torch.unique(event_pairs[torch.argwhere(label_subevent[:, 3] == 2)[:, 0]][:, 0])
            for event_i in range(event_1_index_in_contained.shape[0]):
                event_index = event_1_index_in_contained[event_i]
                connected_events_index = event_pairs[torch.argwhere(label_subevent[:, 3] == 2)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_subevent[:, 3] == 2)[:, 0]][:, 0] == event_index)[:, 0]][:, 1]
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_contained_r(torch.cat((this_event_embed.repeat(connected_event_embed.shape[0], 1),
                                                                 contained.repeat(connected_event_embed.shape[0], 1),
                                                                 connected_event_embed), dim=1))
                if torch.equal(tentative_event_embed_contained[event_index], torch_zeros):
                    tentative_event_embed_contained[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_contained_Q(this_event_embed), self.W_contained_K(relation_feature).t())),
                                                                             self.W_contained_V(relation_feature))[0, :]
                else:
                    tentative_event_embed_contained[event_index] = torch.mean(torch.cat((tentative_event_embed_contained[event_index].view(1, 768),
                                                                                         torch.mm(self.softmax_1(torch.mm(self.W_contained_Q(this_event_embed), self.W_contained_K(relation_feature).t())),
                                                                                                  self.W_contained_V(relation_feature)).view(1, 768)), dim = 0), dim = 0)

            event_2_index_in_contained = torch.unique(event_pairs[torch.argwhere(label_subevent[:, 3] == 2)[:, 0]][:, 1])
            for event_i in range(event_2_index_in_contained.shape[0]):
                event_index = event_2_index_in_contained[event_i]
                connected_events_index = event_pairs[torch.argwhere(label_subevent[:, 3] == 2)[:, 0]][torch.argwhere(event_pairs[torch.argwhere(label_subevent[:, 3] == 2)[:, 0]][:, 1] == event_index)[:, 0]][:, 0]
                this_event_embed = event_embedding[event_index].view(1, 768)
                connected_event_embed = event_embedding[connected_events_index].view(connected_events_index.shape[0], 768)
                relation_feature = self.W_contain_r(torch.cat((connected_event_embed,
                                                               contained.repeat(connected_event_embed.shape[0], 1),
                                                               this_event_embed.repeat(connected_event_embed.shape[0], 1)), dim=1))
                if torch.equal(tentative_event_embed_contain[event_index], torch_zeros):
                    tentative_event_embed_contain[event_index] += torch.mm(self.softmax_1(torch.mm(self.W_contain_Q(this_event_embed), self.W_contain_K(relation_feature).t())),
                                                                           self.W_contain_V(relation_feature))[0, :]
                else:
                    tentative_event_embed_contain[event_index] = torch.mean(torch.cat((tentative_event_embed_contain[event_index].view(1, 768),
                                                                                       torch.mm(self.softmax_1(torch.mm(self.W_contain_Q(this_event_embed), self.W_contain_K(relation_feature).t())),
                                                                                                self.W_contain_V(relation_feature)).view(1, 768)), dim = 0), dim = 0)

        # average of heterogeneous relation edge-aware graph attention network
        for event_i in range(event_embedding.shape[0]):
            this_event_coreference_update = torch.zeros(768).to(device)
            this_event_temporal_update = torch.zeros(768).to(device)
            this_event_causal_update = torch.zeros(768).to(device)
            this_event_subevent_update = torch.zeros(768).to(device)

            if torch.equal(tentative_event_embed_coreference[event_i], torch_zeros):
                coreference_flag = 0
            else:
                coreference_flag = 1
                this_event_coreference_update += tentative_event_embed_coreference[event_i]
            if torch.equal(tentative_event_embed_before[event_i], torch_zeros):
                before_flag = 0
            else:
                before_flag = 1
            if torch.equal(tentative_event_embed_after[event_i], torch_zeros):
                after_flag = 0
            else:
                after_flag = 1
            if torch.equal(tentative_event_embed_overlap[event_i], torch_zeros):
                overlap_flag = 0
            else:
                overlap_flag = 1
            if torch.equal(tentative_event_embed_cause[event_i], torch_zeros):
                cause_flag = 0
            else:
                cause_flag = 1
            if torch.equal(tentative_event_embed_caused[event_i], torch_zeros):
                caused_flag = 0
            else:
                caused_flag = 1
            if torch.equal(tentative_event_embed_contain[event_i], torch_zeros):
                contain_flag = 0
            else:
                contain_flag = 1
            if torch.equal(tentative_event_embed_contained[event_i], torch_zeros):
                contained_flag = 0
            else:
                contained_flag = 1

            temporal_num = before_flag + after_flag + overlap_flag
            causal_num = cause_flag + caused_flag
            subevent_num = contain_flag + contained_flag

            if temporal_num != 0:
                this_event_temporal_update += (tentative_event_embed_before[event_i] + tentative_event_embed_after[event_i] + tentative_event_embed_overlap[event_i]) / temporal_num
            if causal_num != 0:
                this_event_causal_update += (tentative_event_embed_cause[event_i] + tentative_event_embed_caused[event_i]) / causal_num
            if subevent_num != 0:
                this_event_subevent_update += (tentative_event_embed_contain[event_i] + tentative_event_embed_contained[event_i]) / subevent_num

            if (coreference_flag + int(temporal_num != 0) + int(causal_num != 0) + int(subevent_num != 0)) != 0:
                new_event_embedding[event_i] += (this_event_coreference_update + this_event_temporal_update + this_event_causal_update + this_event_subevent_update) / (coreference_flag + int(temporal_num != 0) + int(causal_num != 0) + int(subevent_num != 0))


        # article embedding
        article_embedding = token_embeddings[0].view(1, 768)
        new_article_embedding = torch.zeros(article_embedding.shape).to(device)

        # aggregate event embedding into article embedding via graph attention network
        initial_article = article_embedding.repeat(new_event_embedding.shape[0], 1)
        new_article_embedding += torch.mm((self.softmax_0(self.leakyrelu(self.a_gat(torch.cat((self.W_gat(initial_article), self.W_gat(new_event_embedding)), dim = 1))))).t(), self.W_gat(new_event_embedding))

        final_article_embedding = torch.cat((new_article_embedding, article_embedding), dim = 1) # concatenate article graph embedding and article text embedding

        # conspiracy identification task
        conspiracy_scores = self.conspiracy_2(self.relu(self.conspiracy_1(final_article_embedding)))
        conspiracy_loss = self.crossentropyloss_sum(conspiracy_scores, label_article)

        return conspiracy_scores, conspiracy_loss, event_loss, coreference_loss, temporal_loss, causal_loss, subevent_loss







''' evaluate '''

def evaluate(model, eval_dataloader, verbose):

    model.eval()

    for step, batch in enumerate(eval_dataloader):

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label_article = batch['label_article']
        event_words = batch['event_words'][0]
        event_pairs = batch['event_pairs'][0]
        label_coreference = batch['label_coreference'][0]
        label_temporal = batch['label_temporal'][0]
        label_causal = batch['label_causal'][0]
        label_subevent = batch['label_subevent'][0]

        input_ids, attention_mask, label_article, event_words, event_pairs, label_coreference, label_temporal, label_causal, label_subevent = \
            input_ids.to(device), attention_mask.to(device), label_article.to(device), event_words.to(device), event_pairs.to(device), \
            label_coreference.to(device), label_temporal.to(device), label_causal.to(device), label_subevent.to(device)

        with torch.no_grad():
            conspiracy_scores, conspiracy_loss, event_loss, coreference_loss, temporal_loss, causal_loss, subevent_loss = \
                model(input_ids, attention_mask, label_article, event_words, event_pairs, label_coreference, label_temporal, label_causal, label_subevent)

        decision = torch.argmax(conspiracy_scores, dim = 1).view(conspiracy_scores.shape[0], 1)
        true_label = label_article.view(conspiracy_scores.shape[0], 1)

        if step == 0:
            decision_onetest = decision
            true_label_onetest = true_label
        else:
            decision_onetest = torch.cat((decision_onetest, decision), dim=0)
            true_label_onetest = torch.cat((true_label_onetest, true_label), dim=0)

    decision_onetest = decision_onetest.to('cpu').numpy()
    true_label_onetest = true_label_onetest.to('cpu').numpy()

    if verbose:
        print("Macro: ", precision_recall_fscore_support(true_label_onetest, decision_onetest, average='macro'))
        print("Conspiracy: ", precision_recall_fscore_support(true_label_onetest, decision_onetest, average='binary'))

    macro_F = precision_recall_fscore_support(true_label_onetest, decision_onetest, average='macro')[2]
    conspiracy_F = precision_recall_fscore_support(true_label_onetest, decision_onetest, average='binary')[2]

    return macro_F, conspiracy_F







''' train '''



import time
import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from transformers import logging

logging.set_verbosity_warning()
logging.set_verbosity_error()




seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.use_deterministic_algorithms(True)

model = Model()
model.cuda()
param_all = list(model.named_parameters())
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and ('longformer' in n))], 'lr': longformer_lr, 'weight_decay': longformer_weight_decay},
    {'params': [p for n, p in param_all if ((not any(nd in n for nd in no_decay)) and (not 'longformer' in n))], 'lr': non_longformer_lr, 'weight_decay': non_longformer_weight_decay},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and ('longformer' in n))], 'lr': longformer_lr, 'weight_decay': 0.0},
    {'params': [p for n, p in param_all if ((any(nd in n for nd in no_decay)) and (not 'longformer' in n))], 'lr': non_longformer_lr, 'weight_decay': 0.0}]
# optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-8)

train_dataset = custom_dataset(train_file_paths)
dev_dataset = custom_dataset(dev_file_paths)
test_dataset = custom_dataset(test_file_paths)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_train_steps = num_epochs * len(train_dataloader)
warmup_steps = int(warmup_proportion * num_train_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

best_dev_conspiracy_F = 0
best_dev_macro_F = 0

for epoch_i in range(num_epochs):

    np.random.shuffle(train_file_paths) # shuffle training data
    train_dataset = custom_dataset(train_file_paths)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i, num_epochs))
    print('Training...')

    t0 = time.time()
    total_conspiracy_loss = 0
    total_event_loss = 0
    total_coreference_loss = 0
    total_temporal_loss = 0
    total_causal_loss = 0
    total_subevent_loss = 0
    num_batch = 0

    for step, batch in enumerate(train_dataloader):

        if step % ((len(train_dataloader) * num_epochs) // check_times) == 0 and step != 0:

            elapsed = format_time(time.time() - t0)
            if num_batch != 0:
                avg_conspiracy_loss = total_conspiracy_loss / num_batch
                avg_event_loss = total_event_loss / num_batch
                avg_coreference_loss = total_coreference_loss / num_batch
                avg_temporal_loss = total_temporal_loss / num_batch
                avg_causal_loss = total_causal_loss / num_batch
                avg_subevent_loss = total_subevent_loss / num_batch
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    conspiracy loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_conspiracy_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    event loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_event_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    coreference loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_coreference_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    temporal loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_temporal_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    causal loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_causal_loss))
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    subevent loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_subevent_loss))
            else:
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            total_conspiracy_loss = 0
            total_event_loss = 0
            total_coreference_loss = 0
            total_temporal_loss = 0
            total_causal_loss = 0
            total_subevent_loss = 0
            num_batch = 0

            # evaluate on dev set

            macro_F, conspiracy_F = evaluate(model, dev_dataloader, verbose = 1)
            if conspiracy_F > best_dev_conspiracy_F:
                torch.save(model.state_dict(), "./saved_models/" + split_method + "/best_dev_conspiracy_F.ckpt")
                best_dev_conspiracy_F = conspiracy_F
            if macro_F > best_dev_macro_F:
                torch.save(model.state_dict(), "./saved_models/" + split_method + "/best_dev_macro_F.ckpt")
                best_dev_macro_F = macro_F



        # train

        model.train()

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        label_article = batch['label_article']
        event_words = batch['event_words'][0]
        event_pairs = batch['event_pairs'][0]
        label_coreference = batch['label_coreference'][0]
        label_temporal = batch['label_temporal'][0]
        label_causal = batch['label_causal'][0]
        label_subevent = batch['label_subevent'][0]

        input_ids, attention_mask, label_article, event_words, event_pairs, label_coreference, label_temporal, label_causal, label_subevent = \
            input_ids.to(device), attention_mask.to(device), label_article.to(device), event_words.to(device), event_pairs.to(device), \
            label_coreference.to(device), label_temporal.to(device), label_causal.to(device), label_subevent.to(device)

        optimizer.zero_grad()

        conspiracy_scores, conspiracy_loss, event_loss, coreference_loss, temporal_loss, causal_loss, subevent_loss = \
            model(input_ids, attention_mask, label_article, event_words, event_pairs, label_coreference, label_temporal, label_causal, label_subevent)

        total_conspiracy_loss += conspiracy_loss.item()
        total_event_loss += event_loss.item()
        total_coreference_loss += coreference_loss.item()
        total_temporal_loss += temporal_loss.item()
        total_causal_loss += causal_loss.item()
        total_subevent_loss += subevent_loss.item()
        num_batch += 1

        event_loss.backward(retain_graph = True)
        coreference_loss.backward(retain_graph = True)
        temporal_loss.backward(retain_graph = True)
        causal_loss.backward(retain_graph = True)
        subevent_loss.backward(retain_graph = True)
        conspiracy_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()




    elapsed = format_time(time.time() - t0)
    if num_batch != 0:
        avg_conspiracy_loss = total_conspiracy_loss / num_batch
        avg_event_loss = total_event_loss / num_batch
        avg_coreference_loss = total_coreference_loss / num_batch
        avg_temporal_loss = total_temporal_loss / num_batch
        avg_causal_loss = total_causal_loss / num_batch
        avg_subevent_loss = total_subevent_loss / num_batch
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    conspiracy loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_conspiracy_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    event loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_event_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    coreference loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_coreference_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    temporal loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_temporal_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    causal loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_causal_loss))
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    subevent loss average: {:.3f}'.format(step, len(train_dataloader), elapsed, avg_subevent_loss))
    else:
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

    total_conspiracy_loss = 0
    total_event_loss = 0
    total_coreference_loss = 0
    total_temporal_loss = 0
    total_causal_loss = 0
    total_subevent_loss = 0
    num_batch = 0

    # evaluate on dev set

    macro_F, conspiracy_F = evaluate(model, dev_dataloader, verbose = 1)
    if conspiracy_F > best_dev_conspiracy_F:
        torch.save(model.state_dict(), "./saved_models/" + split_method + "/best_dev_conspiracy_F.ckpt")
        best_dev_conspiracy_F = conspiracy_F
    if macro_F > best_dev_macro_F:
        torch.save(model.state_dict(), "./saved_models/" + split_method + "/best_dev_macro_F.ckpt")
        best_dev_macro_F = macro_F




# best dev conspiracy F model test on test set

print("")
print("Best dev conspiracy F is: {:}".format(best_dev_conspiracy_F))

model.load_state_dict(torch.load("./saved_models/" + split_method + "/best_dev_conspiracy_F.ckpt", map_location=device))
model.eval()
macro_F, conspiracy_F = evaluate(model, test_dataloader, verbose = 1)


# best dev macro model test on test set

print("")
print("Best dev macro F is: {:}".format(best_dev_macro_F))

model.load_state_dict(torch.load("./saved_models/" + split_method + "/best_dev_macro_F.ckpt", map_location=device))
model.eval()
macro_F, conspiracy_F = evaluate(model, test_dataloader, verbose = 1)














# stop here
