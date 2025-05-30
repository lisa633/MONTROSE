import json
import os
import pickle
import random
import re
import sys

import dgl
import nltk
import torch
from prefetch_generator import background

sys.path.append("..")
sys.path.append("../..")
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from typing import List
import numpy as np
from tqdm import tqdm
from functools import reduce
from Data.BiGCN_Data_Utils.twitterloader import BiGCNTwitterSet, CustomDataset, str2timestamp
from torch_geometric.data import Data

Pheme_domain_map = {
  'charliehebdo':0,
  'ferguson':1,
  'germanwings-crash':2,
  'ottawashooting':3,
    'sydneysiege':4
}


def Lemma_Factory():
    lemmatizer = WordNetLemmatizer()

    def lemma(word_tokens):
        tags = nltk.pos_tag(word_tokens)
        new_words = []
        for pair in tags:
            if pair[1].startswith('J'):
                new_words.append(lemmatizer.lemmatize(pair[0], 'a'))
            elif pair[1].startswith('V'):
                new_words.append(lemmatizer.lemmatize(pair[0], 'v'))
            elif pair[1].startswith('N'):
                new_words.append(lemmatizer.lemmatize(pair[0], 'n'))
            elif pair[1].startswith('R'):
                new_words.append(lemmatizer.lemmatize(pair[0], 'r'))
            else:
                new_words.append(pair[0])
        return new_words

    return lemma


class FastBiGCNDataset(BiGCNTwitterSet, CustomDataset):
    def __init__(self, batch_size=20):
        super(FastBiGCNDataset, self).__init__(batch_size=batch_size)

    def labelTensor(self, device=None):
        if device is None:
            device = self.device
        # print("self.data_y",self.data_y)
        # print("------------------------")
        # print(torch.tensor(self.data_y, dtype=torch.float32, device=device))

        return torch.tensor(self.data_y, dtype=torch.float32, device=device)

    def collate_fn_old(self, items):
        tfidf_arr = torch.cat(
            [item[0] for item in items],
            dim=0
        )
        TD_graphs = [item[1] for item in items]
        BU_graphs = [item[2] for item in items]
        labels = [item[3] for item in items]
        topic_labels = [item[4] for item in items]
        num_nodes = [g.num_nodes() for g in TD_graphs]
        big_g_TD = dgl.batch(TD_graphs)
        big_g_BU = dgl.batch(BU_graphs)
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(self.device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(self.device)
        return tfidf_arr, num_nodes, A_TD, A_BU, \
            torch.tensor(labels), torch.tensor(topic_labels)

    def collate_fn(self, items):
        # print("items[0]",items[0])
        tfidf_arr = torch.cat(
            [item.tf_idf for item in items],
            dim=0
        )
        TD_graphs = [item.g_TD for item in items]
        BU_graphs = [item.g_BU for item in items]
        labels = [item.data_y for item in items]
        topic_labels = [item.topic_label for item in items]
        num_nodes = [g.num_nodes() for g in TD_graphs]
        big_g_TD = dgl.batch(TD_graphs)
        big_g_BU = dgl.batch(BU_graphs)
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(self.device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(self.device)
        return tfidf_arr, num_nodes, A_TD, A_BU, \
            torch.tensor(labels), torch.tensor(topic_labels)

    def initGraph(self):
        if not hasattr(self, "g_TD"):
            self.g_TD = {}
        if not hasattr(self, "g_BU"):
            self.g_BU = {}

        for index, d_ID in tqdm(enumerate(self.data_ID)):
            if d_ID in self.g_TD and d_ID in self.g_BU:
                pass
            else:
                g_TD, g_BU = self.construct_graph(index, d_ID)
                self.g_TD[d_ID] = g_TD
                self.g_BU[d_ID] = g_BU

    def initTFIDF(self, tokenizer: TfidfVectorizer):
        self.initGraph()
        for index, d_ID in tqdm(enumerate(self.data_ID)):
            sents = [" ".join(sent) for sent in self.data[d_ID]['text']]
            try:
                self.data[d_ID]['tf-idf'] = tokenizer.transform(sents)
            except:
                print("sents : \n\t", sents)

    def setLabel(self, label: List, idxs: List):
        for k, idx in enumerate(idxs):
            self.data_y[idx] = label[k]

    def __getitem__(self, index):
        d_ID = self.data_ID[index]
        if not hasattr(self, "g_TD"):
            self.g_TD = {}
        if not hasattr(self, "g_BU"):
            self.g_BU = {}
        if not hasattr(self, "lemma_text"):
            self.lemma_text = {}

        if d_ID in self.g_TD and d_ID in self.g_BU:
            g_TD, g_BU = self.g_TD[d_ID], self.g_BU[d_ID]
        else:
            g_TD, g_BU = self.construct_graph(index, d_ID)
            self.g_TD[d_ID] = g_TD
            self.g_BU[d_ID] = g_BU
        tf_idf_arr = self.data[d_ID]['tf-idf'].toarray()
        tf_idf = torch.tensor(tf_idf_arr,
                              device=self.device,
                              dtype=torch.float32)[:self.data_len[index]]
        sents = [" ".join(sent) for sent in self.data[d_ID]['text']]
        return Data(tf_idf=tf_idf, g_TD=dgl.add_self_loop(g_TD), g_BU=dgl.add_self_loop(g_BU),
                    data_y=self.data_y[index], topic_label=self.data[self.data_ID[index]]['topic_label'], text=sents,
                    data_len=self.data_len[index])

#此处需要自己写，参考ABSADataset
class MetaMCMCDataset(FastBiGCNDataset):

    def __init__(self, batch_size=20):
        super(MetaMCMCDataset, self).__init__(batch_size=batch_size)
        self.instance_weights = torch.ones([len(self.data_len)], dtype=torch.float32, device=torch.device('cuda'))
        self._confidence, self._entrophy = None, None
        self.read_indexs:np.array = None

    def set_instance_weights(self, weights_list, idxs_list):
        self.instance_weights[torch.cat(idxs_list)] = torch.squeeze(torch.cat(weights_list))

    def init_instance_weights(self):
        self.instance_weights = torch.ones([len(self.data_ID)], dtype=torch.float32, device=torch.device('cuda'))
        
    def initLabel(self, label):
        assert len(self.read_indexs) == len(label)
        self.data_y = label

    def initConfidence(self, confidence):
        assert len(self.read_indexs) == len(confidence)
        self._confidence = np.array(confidence)

    def initEntrophy(self, entrophy):
        assert len(self.read_indexs) == len(entrophy)
        self._entrophy = entrophy

    def initDomain(self, d_arr):
        assert len(self.read_indexs) == len(d_arr)
        self._domain = d_arr

    def get_instance_weights(self):
        return self.instance_weights

    def load_data_fast(self, data_prefix="~/autodl-tmp/data", min_len=-1):

        super().load_data_fast(data_prefix)
        self.instance_weights = torch.ones([len(self.data_ID)], dtype=torch.float32, device=torch.device('cuda'))
        self._confidence = torch.ones(len(self.data_ID),device=torch.device('cuda'))
        self._entrophy = torch.zeros(len(self.data_ID),device=torch.device('cuda'))
        self.read_indexs = np.arange(len(self.data_ID))

    def split(self, percent=[0.5, 1.0]):
        data_size = len(self.data_ID)
        start_end = [int(item * data_size) for item in percent]
        start_end.insert(0, 0)
        new_idxs = list(random.sample(list(range(data_size)), data_size))
        # new_idxs = list(range(data_size))
        # new_idxs=new_idxs[::-1]

        # if data_size > 5000:
        #     file_path = f'./indomain/pheme_array.pkl'
        #     if os.path.exists(file_path):
        #         with open(file_path, 'rb') as file:
        #             new_idxs = pickle.load(file)
        #     else:
        #         with open(file_path, 'wb') as file:
        #             pickle.dump(new_idxs, file)
        #
        # print("new_idxs", new_idxs)
        # print("new_idxs_len", len(new_idxs))

        rst = [self.__class__() for _ in percent]
        for i in range(len(percent)):
            real_idxs = [self.read_indexs[idx] for idx in new_idxs[start_end[i]:start_end[i + 1]]]
            rst[i].data_ID = [self.data_ID[idx] for idx in new_idxs[start_end[i]:start_end[i + 1]]]
            rst[i].data_len = [self.data_len[idx] for idx in new_idxs[start_end[i]:start_end[i + 1]]]
            rst[i].data_y = [self.data_y[idx] for idx in new_idxs[start_end[i]:start_end[i + 1]]]
            rst[i].data = {ID: self.data[ID] for ID in rst[i].data_ID}
            rst[i].instance_weights = torch.ones([len(rst[i].data_ID)], dtype=torch.float32,
                                                 device=torch.device('cuda'))
            rst[i]._confidence = torch.ones([len(rst[i].data_ID)], dtype=torch.float32,
                                                 device=torch.device('cuda'))
            rst[i]._entrophy= torch.zeros([len(rst[i].data_ID)], dtype=torch.float32,
                                                 device=torch.device('cuda'))

            rst[i].read_indexs = np.arange(len(real_idxs))

        return rst
    

    def __getitem__(self, index):
        d_ID = self.data_ID[index]
        if not hasattr(self, "g_TD"):
            self.g_TD = {}
        if not hasattr(self, "g_BU"):
            self.g_BU = {}
        if not hasattr(self, "lemma_text"):
            self.lemma_text = {}

        if d_ID in self.g_TD and d_ID in self.g_BU:
            g_TD, g_BU = self.g_TD[d_ID], self.g_BU[d_ID]
        else:
            g_TD, g_BU = self.construct_graph(index, d_ID)
            self.g_TD[d_ID] = g_TD
            self.g_BU[d_ID] = g_BU
        sents = [" ".join(sent) for sent in self.data[d_ID]['text']]
        weight = self.instance_weights[index]

        # tf_idf_arr = self.data[d_ID]['tf-idf'].toarray()
        # tf_idf = torch.tensor(tf_idf_arr,
        #                       device=self.device,
        #                       dtype=torch.float32)[:self.data_len[index]]
        return Data(g_TD=dgl.add_self_loop(g_TD), g_BU=dgl.add_self_loop(g_BU),
                     topic_label=self.data[self.data_ID[index]]['topic_label'], text=sents,
                    data_len=self.data_len[index], data_confidence = self._confidence, data_entrophy = self._entrophy,weight=weight, data_y=self.data_y[index], index=index)

    def collate_fn(self, items):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        sents = [item.text for item in items]
        TD_graphs = [item.g_TD for item in items]
        BU_graphs = [item.g_BU for item in items]
        labels = [item.data_y for item in items]
        topic_labels = [item.topic_label for item in items]
        num_nodes = [g.num_nodes() for g in TD_graphs]
        big_g_TD = dgl.batch(TD_graphs)
        big_g_BU = dgl.batch(BU_graphs)
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(device)
        weights = torch.stack([item.weight for item in items])
        idxs = torch.tensor([item.index for item in items], dtype=torch.long,
                            device=device)
        return sents, num_nodes, A_TD, A_BU, weights, idxs, torch.tensor(labels,device=device), torch.tensor(topic_labels,device=device)
    
    def dataset2dataloader(self, dataset: BiGCNTwitterSet, shuffle=False, batch_size=32):
        if shuffle:
            idxs = random.sample(range(len(dataset)), len(dataset)) * 2
        else:
            idxs = [*range(len(dataset))] * 2

        @background(max_prefetch=5)
        def dataloader():
            for start in range(0, len(dataset), batch_size):
                batch_idxs = idxs[start:start + batch_size]
                items = [dataset[index] for index in batch_idxs]
                yield self.collate_fn(items)

        return dataloader()


event_dics = {
    'twitter15': 0,
    'twitter16': 1,
}

#需要添加
@background(max_prefetch=5)
def twitter_data_process(files):
    for file_path in tqdm(files):
        ret = {}
        ss = file_path.split("/")
        data = json.load(open(file_path, mode="r", encoding="utf-8"))
        # 'Wed Jan 07 11:14:08 +0000 2015'
        if data['lang'] == 'en':
            ret[ss[-3]] = {
                'topic_label': event_dics[ss[-4]],
                'event': ss[-4],
                'sentence': [data['text'].lower()],
                'created_at': [str2timestamp(data['created_at'])],
                'tweet_id': [data['id']],
                "reply_to": [data['in_reply_to_status_id']]
            }
            yield ret

#需要添加
class Twitter15(MetaMCMCDataset):
    def __init__(self):
        super(Twitter15, self).__init__()
        self.instance_weights = torch.ones([len(self.data_len)], dtype=torch.float32, device=torch.device('cuda'))

    def read_json_files(self, files):
        for twitter_dict in twitter_data_process(files):
            self.fetch(twitter_dict)

    def dataclear(self, post_fn=1):
        print("data clear:")
        for key, value in tqdm(self.data.items()):
            temp_idxs = np.array(self.data[key]['created_at']).argsort().tolist()
            self.sort_by_timeline(key, temp_idxs)
            self.gather_posts(key, temp_idxs, post_fn)

        for key in self.data.keys():
            self.data_ID.append(key)
        self.data_ID = random.sample(self.data_ID, len(self.data_ID))  # shuffle the data id

        for i in range(len(self.data_ID)):  # pre processing the extra informations
            self.data_len.append(len(self.data[self.data_ID[i]]['text']))

            label_file = f"/data/run01/scz3924/hezj/da-mstf/t1516_text/preprocessed_data/label_15.json"

            label_data = json.load(open(label_file, mode="r", encoding="utf-8"))

            if label_data[self.data_ID[i]] == "non-rumor":
                self.data_y.append([1.0, 0.0, 0.0, 0.0])
            elif label_data[self.data_ID[i]] == 'false':
                self.data_y.append([0.0, 1.0, 0.0, 0.0])
            elif label_data[self.data_ID[i]] == 'true':
                self.data_y.append([0.0, 0.0, 1.0, 0.0])
            elif label_data[self.data_ID[i]] == 'unverified':
                self.data_y.append([0.0, 0.0, 0.0, 1.0])

            # if label_data[self.data_ID[i]] == "non-rumor":
            #     self.data_y.append([1.0, 0.0])
            # elif label_data[self.data_ID[i]] == 'false':
            #     self.data_y.append([0.0, 1.0])
            # elif label_data[self.data_ID[i]] == 'true':
            #     self.data_y.append([0.0, 1.0])
            # elif label_data[self.data_ID[i]] == 'unverified':
            #     self.data_y.append([0.0, 1.0])

    def split(self, percent=[0.5, 1.0]):
        data_size = len(self.data_ID)
        start_end = [int(item * data_size) for item in percent]
        start_end.insert(0, 0)
        new_idxs = list(random.sample(list(range(data_size)), data_size))
        # new_idxs = list(range(data_size))
        # new_idxs=new_idxs[::-1]
        if data_size > 1000:
            file_path = f'./indomain/twitter15_array.pkl'
            if os.path.exists(file_path):
                with open(file_path, 'rb') as file:
                    new_idxs = pickle.load(file)
            else:
                with open(file_path, 'wb') as file:
                    pickle.dump(new_idxs, file)

        print("new_idxs", new_idxs)
        print("new_idxs_len", len(new_idxs))

        rst = [self.__class__() for _ in percent]
        for i in range(len(percent)):
            # if i==0 and data_size > 1000:
            #     # start_end[0] = int(0.8 * data_size)
            #     start_end[1]=int(0.8* data_size)
            # else:
            #     start_end[1] = int(percent[0] * data_size)
            rst[i].data_ID = [self.data_ID[idx] for idx in new_idxs[start_end[i]:start_end[i + 1]]]
            rst[i].data_len = [self.data_len[idx] for idx in new_idxs[start_end[i]:start_end[i + 1]]]
            rst[i].data_y = [self.data_y[idx] for idx in new_idxs[start_end[i]:start_end[i + 1]]]
            rst[i].data = {ID: self.data[ID] for ID in rst[i].data_ID}
            rst[i].instance_weights = torch.ones([len(rst[i].data_ID)], dtype=torch.float32,
                                                 device=torch.device('cuda'))
        return rst


class Twitter16(MetaMCMCDataset):
    def __init__(self):
        super(Twitter16, self).__init__()
        self.instance_weights = torch.ones([len(self.data_len)], dtype=torch.float32, device=torch.device('cuda'))

    def read_json_files(self, files):
        for twitter_dict in twitter_data_process(files):
            self.fetch(twitter_dict)

    def dataclear(self, post_fn=1):
        print("data clear:")
        for key, value in tqdm(self.data.items()):
            temp_idxs = np.array(self.data[key]['created_at']).argsort().tolist()
            self.sort_by_timeline(key, temp_idxs)
            self.gather_posts(key, temp_idxs, post_fn)

        for key in self.data.keys():
            self.data_ID.append(key)
        self.data_ID = random.sample(self.data_ID, len(self.data_ID))  # shuffle the data id

        for i in range(len(self.data_ID)):  # pre processing the extra informations
            self.data_len.append(len(self.data[self.data_ID[i]]['text']))

            label_file = f"/data/run01/scz3924/hezj/da-mstf/t1516_text/preprocessed_data/label_16.json"

            label_data = json.load(open(label_file, mode="r", encoding="utf-8"))

            if label_data[self.data_ID[i]] == "non-rumor":
                self.data_y.append([1.0, 0.0, 0.0, 0.0])
            elif label_data[self.data_ID[i]] == 'false':
                self.data_y.append([0.0, 1.0, 0.0, 0.0])
            elif label_data[self.data_ID[i]] == 'true':
                self.data_y.append([0.0, 0.0, 1.0, 0.0])
            elif label_data[self.data_ID[i]] == 'unverified':
                self.data_y.append([0.0, 0.0, 0.0, 1.0])

            # if label_data[self.data_ID[i]] == "non-rumor":
            #     self.data_y.append([1.0, 0.0])
            # elif label_data[self.data_ID[i]] == 'false':
            #     self.data_y.append([0.0, 1.0])
            # elif label_data[self.data_ID[i]] == 'true':
            #     self.data_y.append([0.0, 1.0])
            # elif label_data[self.data_ID[i]] == 'unverified':
            #     self.data_y.append([0.0, 1.0])

    def split(self, percent=[0.5, 1.0]):
        data_size = len(self.data_ID)
        start_end = [int(item * data_size) for item in percent]
        start_end.insert(0, 0)
        new_idxs = list(random.sample(list(range(data_size)), data_size))
        new_idxs = list(range(data_size))
        new_idxs = new_idxs[::-1]
        if data_size > 600:
            file_path = f'./indomain/twitter16_array.pkl'
            if os.path.exists(file_path):
                with open(file_path, 'rb') as file:
                    new_idxs = pickle.load(file)
            else:
                with open(file_path, 'wb') as file:
                    pickle.dump(new_idxs, file)

        print("new_idxs", new_idxs)
        print("new_idxs_len", len(new_idxs))

        rst = [self.__class__() for _ in percent]
        for i in range(len(percent)):
            # if i==0 and data_size > 600:
            #     # start_end[0] = int(0.8 * data_size)
            #     start_end[1]=int(0.8 * data_size)
            # else:
            #     start_end[1] = int(percent[0] * data_size)
            rst[i].data_ID = [self.data_ID[idx] for idx in new_idxs[start_end[i]:start_end[i + 1]]]
            rst[i].data_len = [self.data_len[idx] for idx in new_idxs[start_end[i]:start_end[i + 1]]]
            rst[i].data_y = [self.data_y[idx] for idx in new_idxs[start_end[i]:start_end[i + 1]]]
            rst[i].data = {ID: self.data[ID] for ID in rst[i].data_ID}
            rst[i].instance_weights = torch.ones([len(rst[i].data_ID)], dtype=torch.float32,
                                                 device=torch.device('cuda'))
        return rst

    def segmente(self, percent=[0.5, 1.0]):
        NR, F, T, U = [], [], [], []
        l1 = l2 = l3 = l4 = 0
        data_size = len(self.data_ID)
        # new_idxs = list(random.sample(list(range(data_size)), data_size))
        new_idxs = list(range(data_size))
        for idx in new_idxs:
            if self.data_y[idx] == [1.0, 0.0, 0.0, 0.0]:
                # rst=self.__class__()
                # rst.data_ID=self.data_ID[idx]
                # rst.data_len=self.data_len[idx]
                # rst.data_y=self.data_y[idx]
                # rst.data={rst.data_ID:self.data[rst.data_ID]}
                NR.append(idx)
                l1 += 1
            elif self.data_y[idx] == [0.0, 1.0, 0.0, 0.0]:
                F.append(idx)
                l2 += 1
            elif self.data_y[idx] == [0.0, 0.0, 1.0, 0.0]:
                T.append(idx)
                l3 += 1
            elif self.data_y[idx] == [0.0, 0.0, 0.0, 1.0]:
                U.append(idx)
                l4 += 1

        print(len(new_idxs))
        print(l1, l2, l3, l4)
        random.shuffle(NR)
        random.shuffle(F)
        random.shuffle(T)
        random.shuffle(U)
        fold0_x_train = []
        fold0_x_test = []
        leng1 = int(l1 * 0.2)
        leng2 = int(l2 * 0.2)
        leng3 = int(l3 * 0.2)
        leng4 = int(l4 * 0.2)

        fold0_x_test.extend(NR[0:leng1])
        fold0_x_test.extend(F[0:leng2])
        fold0_x_test.extend(T[0:leng3])
        fold0_x_test.extend(U[0:leng4])
        fold0_x_train.extend(NR[leng1:])
        fold0_x_train.extend(F[leng2:])
        fold0_x_train.extend(T[leng3:])
        fold0_x_train.extend(U[leng4:])
        random.shuffle(fold0_x_test)
        random.shuffle(fold0_x_train)

        trainset = self.__class__()
        testset = self.__class__()
        testset.data_ID = [self.data_ID[idx] for idx in fold0_x_test]
        testset.data_len = [self.data_len[idx] for idx in fold0_x_test]
        testset.data_y = [self.data_y[idx] for idx in fold0_x_test]
        testset.data = {ID: self.data[ID] for ID in testset.data_ID}
        testset.instance_weights = torch.ones([len(testset.data_ID)], dtype=torch.float32,
                                              device=torch.device('cuda'))

        trainset.data_ID = [self.data_ID[idx] for idx in fold0_x_train]
        trainset.data_len = [self.data_len[idx] for idx in fold0_x_train]
        trainset.data_y = [self.data_y[idx] for idx in fold0_x_train]
        trainset.data = {ID: self.data[ID] for ID in trainset.data_ID}
        trainset.instance_weights = torch.ones([len(trainset.data_ID)], dtype=torch.float32,
                                               device=torch.device('cuda'))
        return trainset, testset

    def load_data_fast(self, data_prefix="../data/train", min_len=-1):
        # data_prefix = data_prefix + "_out"
        super().load_data_fast(data_prefix)
        self.instance_weights = torch.ones([len(self.data_ID)], dtype=torch.float32, device=torch.device('cuda'))


class FastTwitterDataset(BiGCNTwitterSet, CustomDataset):
    def __init__(self, batch_size=20):
        super(FastTwitterDataset, self).__init__(batch_size=batch_size)

    def labelTensor(self, device=None):
        if device is None:
            device = self.device
        return torch.tensor(self.data_y, dtype=torch.float32, device=device)

    def collate_fn(self, items):
        tfidf_arr = torch.cat(
            [item[0] for item in items],
            dim=0
        )
        TD_graphs = [item[1] for item in items]
        BU_graphs = [item[2] for item in items]
        labels = [item[3] for item in items]
        topic_labels = [item[4] for item in items]
        num_nodes = [g.num_nodes() for g in TD_graphs]
        big_g_TD = dgl.batch(TD_graphs)
        big_g_BU = dgl.batch(BU_graphs)
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(self.device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(self.device)
        return tfidf_arr, num_nodes, A_TD, A_BU, \
            torch.tensor(labels,device = self.device), torch.tensor(topic_labels,device = self.device)

    def initGraph(self):
        if not hasattr(self, "g_TD"):
            self.g_TD = {}
        if not hasattr(self, "g_BU"):
            self.g_BU = {}

        for index, d_ID in tqdm(enumerate(self.data_ID)):
            if d_ID in self.g_TD and d_ID in self.g_BU:
                pass
            else:
                g_TD, g_BU = self.construct_graph(index, d_ID)
                self.g_TD[d_ID] = g_TD
                self.g_BU[d_ID] = g_BU

    def initTFIDF(self, tokenizer: TfidfVectorizer):
        self.initGraph()
        for index, d_ID in tqdm(enumerate(self.data_ID)):
            sents = [" ".join(sent) for sent in self.data[d_ID]['text']]
            try:
                self.data[d_ID]['tf-idf'] = tokenizer.transform(sents)
            except:
                print("sents : \n\t", sents)

    def setLabel(self, label: List, idxs: List):
        for k, idx in enumerate(idxs):
            self.data_y[idx] = label[k]

    def __getitem__(self, index):
        d_ID = self.data_ID[index]
        if not hasattr(self, "g_TD"):
            self.g_TD = {}
        if not hasattr(self, "g_BU"):
            self.g_BU = {}
        if not hasattr(self, "lemma_text"):
            self.lemma_text = {}

        if d_ID in self.g_TD and d_ID in self.g_BU:
            g_TD, g_BU = self.g_TD[d_ID], self.g_BU[d_ID]
        else:
            g_TD, g_BU = self.construct_graph(index, d_ID)
            self.g_TD[d_ID] = g_TD
            self.g_BU[d_ID] = g_BU
        tf_idf_arr = self.data[d_ID]['tf-idf'].toarray()
        tf_idf = torch.tensor(tf_idf_arr,
                              device=self.device,
                              dtype=torch.float32)[:self.data_len[index]]
        sents = [" ".join(sent) for sent in self.data[d_ID]['text']]
        return Data(tf_idf=tf_idf, \
                    g_TD=dgl.add_self_loop(g_TD), \
                    g_BU=dgl.add_self_loop(g_BU), \
                    data_y=self.data_y[index], \
                    topic_label=self.data[self.data_ID[index]]['topic_label'], \
                    text=sents, \
                    data_len=self.data_len[index])


def Merge_data(data_set1, data_set2, shuffle=True):
    new_data = data_set1.__class__()
    new_data.data = dict(data_set1.data, **data_set2.data)
    if shuffle:
        new_data_ID_list = data_set1.data_ID + data_set2.data_ID
        new_data.data_ID = random.sample(new_data_ID_list, len(new_data_ID_list))
        new_data._confidence = torch.ones(len(new_data.data_ID),device=torch.device('cuda'))
        new_data._entrophy = torch.zeros(len(new_data.data_ID),device=torch.device('cuda'))
        new_data.read_indexs = np.arange(len(new_data.data_ID))
        new_data.instance_weights = torch.ones([len(new_data.data_ID)], dtype=torch.float32,device=torch.device('cuda'))
        for i in range(len(new_data.data_ID)):  # pre processing the extra informations
            new_data.data_len.append(len(new_data.data[new_data.data_ID[i]]['text']))
            if new_data.data[new_data.data_ID[i]]['label'] == "rumours":
                new_data.data_y.append([0.0, 1.0])
            else:
                new_data.data_y.append([1.0, 0.0])
    else:
        new_data.data_ID = np.concatenate([np.array(data_set1.data_ID),
                                           np.array(data_set2.data_ID)]).tolist()
        new_data.data_len = np.concatenate([np.array(data_set1.data_len),
                                            np.array(data_set2.data_len)]).tolist()
        new_data.read_indexs = np.arange(len(new_data.data_ID))
        new_data._confidence = torch.cat([data_set1._confidence,
                                               data_set2._confidence], dim=0)
        new_data._entrophy = torch.cat([data_set1._entrophy,
                                               data_set2._entrophy], dim=0)
        new_data.data_y = np.concatenate([np.array(data_set1.data_y),
                                          np.array(data_set2.data_y)]).tolist()
        new_data.instance_weights = torch.cat([data_set1.instance_weights,
                                               data_set2.instance_weights], dim=0)
    return new_data


def load_events(events_list: List):
    data_list = []
    for event_dir in events_list:
        if event_dir.split("/")[-1] == "twitter15":
            dataset = Twitter15()
            try:
                dataset.load_data_fast(event_dir)
            except:  # if no caches
                dataset.load_data(event_dir)
                dataset.Caches_Data(event_dir)
        elif event_dir.split("/")[-1] == "twitter16":
            dataset = Twitter16()
            try:
                dataset.load_data_fast(event_dir)
            except:  # if no caches
                dataset.load_data(event_dir)
                dataset.Caches_Data(event_dir)
        else:
            print("here!")
            dataset = MetaMCMCDataset()
            try:
                dataset.load_data_fast(event_dir)
            except:  # if no caches
                dataset.load_event_list([event_dir])
                dataset.Caches_Data(event_dir)
        data_list.append(dataset)

    if len(data_list) > 1:
        final_set = reduce(Merge_data, data_list)
        del dataset, data_list
    elif len(data_list) == 1:
        final_set = data_list[0]
    else:
        raise Exception("Something wrong!")
    return final_set

#source_events, target_events, 0, unlabeled_ratio=-1
def load_data(source_events: List, target_events: List, lt=0, unlabeled_ratio=0.5):
    train_set = load_events(source_events)
    target_set = load_events(target_events)

    if lt != 0:
        lt_percent = lt * 1.0 / len(target_set)
        labeled_target, target_set = target_set.split([lt_percent, 1.0])
    else:
        labeled_target = None
    # val_set, meta_dev, test_set = target_set.split([0.16, 0.2, 1.0])
    val_set, test_set = target_set.split([0.3, 1.0])
    meta_dev = val_set

    if unlabeled_ratio <= 0.0:
        return train_set, labeled_target, val_set, meta_dev, test_set
    else:
        ut_size = int(unlabeled_ratio * len(test_set))
        idxs = random.sample(range(len(test_set)), ut_size)
        unlabeled_target: FastBiGCNDataset = test_set.select(idxs)  # unlabeled target is a subset of test_set
        unlabeled_target.setLabel(
            [[0, 0] for _ in range(len(unlabeled_target))],
            [*range(len(unlabeled_target))]
        )
        return train_set, labeled_target, val_set, test_set, unlabeled_target


def load_data_twitter15(event_dir, lt=0, unlabeled_ratio=0.5):
    dataset = Twitter15()
    try:
        dataset.load_data_fast(event_dir)
    except:  # if no caches
        dataset.load_data(event_dir)
        dataset.Caches_Data(event_dir)

    train_set, target_set = dataset.split([0.8, 1.0])
    # test_set=target_set

    if lt != 0:
        lt_percent = lt * 1.0 / len(target_set)
        labeled_target, target_set = target_set.split([lt_percent, 1.0])
    else:
        labeled_target = None
    val_set, meta_dev, test_set = target_set.split([0.16, 0.2, 1.0])
    # val_set, meta_dev, test_set = target_set.split([0.2, 0.3, 1.0])
    # val_set, test_set = target_set.split([0.2, 1.0])
    # meta_dev = val_set

    if unlabeled_ratio <= 0.0:
        return train_set, labeled_target, val_set, meta_dev, test_set
        # return train_set, labeled_target, None, None, test_set
    else:
        ut_size = int(unlabeled_ratio * len(test_set))
        idxs = random.sample(range(len(test_set)), ut_size)
        unlabeled_target: FastBiGCNDataset = test_set.select(idxs)  # unlabeled target is a subset of test_set
        unlabeled_target.setLabel(
            [[0, 0] for _ in range(len(unlabeled_target))],
            [*range(len(unlabeled_target))]
        )
        return train_set, labeled_target, None, test_set, unlabeled_target


def load_data_twitter16(event_dir, lt=0, unlabeled_ratio=0.5):
    dataset = Twitter16()
    try:
        dataset.load_data_fast(event_dir)
    except:  # if no caches
        dataset.load_data(event_dir)
        dataset.Caches_Data(event_dir)

    train_set, target_set = dataset.split([0.8, 1.0])
    # train_set, target_set = dataset.segmente([0.8, 1.0])

    if lt != 0:
        lt_percent = lt * 1.0 / len(target_set)
        labeled_target, target_set = target_set.split([lt_percent, 1.0])
    else:
        labeled_target = None
    # val_set, meta_dev, test_set = target_set.split([0.16, 0.2, 1.0])
    val_set, test_set = target_set.split([0.2, 1.0])
    meta_dev = val_set
    # test_set = target_set

    if unlabeled_ratio <= 0.0:
        return train_set, labeled_target, val_set, meta_dev, test_set
        # return train_set, None, None, None, test_set
    else:
        ut_size = int(unlabeled_ratio * len(test_set))
        idxs = random.sample(range(len(test_set)), ut_size)
        unlabeled_target: FastBiGCNDataset = test_set.select(idxs)  # unlabeled target is a subset of test_set
        unlabeled_target.setLabel(
            [[0, 0] for _ in range(len(unlabeled_target))],
            [*range(len(unlabeled_target))]
        )
        return train_set, labeled_target, None, test_set, unlabeled_target


def load_events(events_list: List):
    data_list = []
    for event_dir in events_list:
        dataset = MetaMCMCDataset()
        try:
            dataset.load_data_fast(event_dir)
        except:  # if no caches
            dataset.load_event_list([event_dir])
            dataset.Caches_Data(event_dir)
        data_list.append(dataset)

    if len(data_list) > 1:
        final_set = reduce(Merge_data, data_list)
        del dataset, data_list
    elif len(data_list) == 1:
        final_set = data_list[0]
    else:
        raise Exception("Something wrong!")
    return final_set


def load_data_and_TfidfVectorizer(source_events: List, target_events: List, lt=0, unlabeled_ratio=0.5):
    data_list = load_data(source_events, target_events, lt=0, unlabeled_ratio=-1)
    Tf_Idf_twitter_file = "./TfIdf_twitter.pkl"
    if os.path.exists(Tf_Idf_twitter_file):
        with open(Tf_Idf_twitter_file, "rb") as fr:
            tv = pickle.load(fr)
    else:
        lemma = Lemma_Factory()
        corpus = [" ".join(lemma(txt)) for data in [dddata for dddata in data_list if dddata is not None]
                  for ID in data.data_ID for txt in data.data[ID]['text']]
        tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None)
        _ = tv.fit_transform(corpus)
        with open(Tf_Idf_twitter_file, "wb") as fw:
            pickle.dump(tv, fw, protocol=pickle.HIGHEST_PROTOCOL)
    data_len = [str(len(dset)) if dset is not None else '0' for dset in data_list]
    print("data length : ", '/'.join(data_len))
    for idx in range(len(data_list)):
        if data_list[idx] is not None:
            data_list[idx].initTFIDF(tv)
    return (tv,) + tuple(data_list)


def load_data_all(events_list: List, lt=0, unlabeled_ratio=0.5):
    data_set = load_events(events_list)
    train_set, target_set = data_set.split([0.8, 1.0])

    if lt != 0:
        lt_percent = lt * 1.0 / len(target_set)
        labeled_target, target_set = target_set.split([lt_percent, 1.0])
    else:
        labeled_target = None
    val_set, meta_dev, test_set = target_set.split([0.2, 0.3, 1.0])
    # val_set, test_set = target_set.split([0.2, 1.0])
    # meta_dev = val_set

    if unlabeled_ratio <= 0.0:
        return train_set, labeled_target, val_set, meta_dev, test_set
    else:
        ut_size = int(unlabeled_ratio * len(test_set))
        idxs = random.sample(range(len(test_set)), ut_size)
        unlabeled_target: FastBiGCNDataset = test_set.select(idxs)  # unlabeled target is a subset of test_set
        unlabeled_target.setLabel(
            [[0, 0] for _ in range(len(unlabeled_target))],
            [*range(len(unlabeled_target))]
        )
        return train_set, labeled_target, val_set, test_set, unlabeled_target