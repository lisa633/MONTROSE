import sys
sys.path.append('..')

import dgl

from Data.BiGCN_Data_Utils.twitterloader import BiGCNTwitterSet

sys.path.append("../../")
import torch, random, os
import torch.nn as nn
import torch.nn.functional as F
from TrainingEnv import GradientReversal, BaseTrainer, CustomDataset, VirtualModel,VirtualEvaluater
from typing import List, AnyStr
import dgl
from backpack import extend, extensions
from prefetch_generator import background
from torch.utils.data import Dataset
import numpy as np
from Data.BiGCN_Dataloader_out import BiGCNTwitterSet, FastBiGCNDataset, MetaMCMCDataset, load_data, load_data_twitter15
from BaseModel.BiGCN_Utils.RumorDetectionBasic import BaseEvaluator
from BaseModel.BiGCN_Utils.GraphRumorDect import BiGCNRumorDetecV2
from BaseModel.modeling_bert import *
from transformers.models.bert import BertConfig, BertTokenizer
from BaseModel.BiGCN import DomainDiscriminator

class TwitterTransformer(BiGCNRumorDetecV2):
    def __init__(self, sent2vec, prop_model, rdm_cls, **kwargs):
        super(TwitterTransformer, self).__init__(sent2vec, prop_model, rdm_cls)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f"Attribute '{k}' is not a valid attribute of DaMSTF")
            self.__setattr__(k, v)
            
    def AdvDLossAndAcc(self, domain_classifier:nn.Module, batch, labels: torch.Tensor = None,
                            temperature=1.0, label_weight=None, reduction='mean', grad_reverse=True):
        vecs = self.Batch2Vecs(batch)
        if grad_reverse:
            vecs = GradientReversal.apply(vecs)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        vecs = vecs.to(device)
        logits = domain_classifier(vecs)
        preds = F.softmax(logits / temperature, dim=1)
        epsilon = torch.ones_like(preds) * (1e-8)
        preds = (preds - epsilon).abs()
#         print("batch[-1]:",batch[-1])
        if labels is None:
            labels = batch[-1].to(preds.device)
        else:
            labels = labels.to(preds.device)
        if labels.dim() == 2:
            loss, acc = self.expandCrossEntropy(preds, labels, label_weight, reduction)
        elif labels.dim() == 1:
            loss = F.nll_loss(preds.log(), labels, weight=label_weight, reduction=reduction)
            acc_t = ((preds.argmax(dim=1) - labels).__eq__(0).float().sum()) / len(labels)
            acc = acc_t.data.item()
        else:
            raise Exception("weird label tensor!")
        return loss, acc
    
    def AdvPredict(self, batch, temperature=1.0):
#         vecs = self.Batch2Vecs(batch)
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         vecs = vecs.to(device)
#         logits = discriminator(vecs)
#         preds = F.softmax(logits / temperature, dim=1)
        preds = self.predict(batch)
        
        return preds
        
        
   

    def Batch2Vecs(self, batch):
        sentences = [text for s_list in batch[0] for text in s_list]
        assert len(sentences) == sum(batch[1])
        A_TD: torch.Tensor = batch[2].bool().float()
        A_BU: torch.Tensor = batch[3].bool().float()
        adj_mtx = (A_TD + A_BU).__ne__(0.0).float()
        attn_mask = (-10000 * (1.0 - adj_mtx)).unsqueeze(0).unsqueeze(0)
        inputs = self.sent2vec(sentences)
        num_nodes: List = batch[1]
        root_idxs = [sum(num_nodes[:idx]) for idx in range(len(num_nodes))]
        rst = self.prop_model(inputs.unsqueeze(0), attention_mask=attn_mask, root_idxs=root_idxs)
        hiddens = rst[0].squeeze(0)
        return hiddens[root_idxs]

    def meta_lossAndAcc(self, valid_data, temperature, label_weight):
        loss, acc = self.lossAndAcc(valid_data)
        return loss, acc

    def step(self, step_size): 
        for n, p in self.named_parameters():
            p.data = p.data - step_size * p.grad

    def lossAndAcc(self, batch, temperature=1.0, label_weight: torch.Tensor = None, reduction='mean'):

        preds = self.predict(batch, temperature=temperature)
        # 检查是否有可用的GPU
        epsilon = torch.ones_like(preds) * 1e-8
        preds = (preds - epsilon).abs()  # to avoid the prediction [1.0, 0.0], which leads to the 'nan' value in log operation
        labels = batch[-2].to(preds.device)
        loss, acc = self.loss_func(preds, labels, label_weight=label_weight, reduction=reduction)
        return loss, acc
    

    def get_CrossEntropyLoss(self,batch):
        seq_outs = self.Batch2Vecs(batch)
        logits = self.rdm_cls(seq_outs)
        labels = batch[-4].to(logits.device)
        loss_fn = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
        loss=loss_fn(logits,labels)
        return loss
    
    def get_MetaLearningCrossEntropyLoss(self,batch,weights):
        seq_outs = self.Batch2Vecs(batch)
        logits = self.rdm_cls(seq_outs)
        labels = batch[-4].to(logits.device)
        loss_fn = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
        loss=loss_fn(logits,labels)
        return loss
    
    def loss_func(self, preds: torch.Tensor, labels: torch.Tensor, label_weight=None, reduction='none'):
        # print("preds.shape",preds.shape)
        # print("labels.shape", labels.shape)
        if labels.dim() == 3:
            loss, acc = self.expandCrossEntropy(preds, labels, label_weight, reduction)
        elif labels.dim() == 2:
            # labels = labels.unsqueeze(dim=1)
            loss, acc = self.expandCrossEntropy(preds, labels, label_weight, reduction)
        elif labels.dim() == 1:
            loss = F.nll_loss(preds.log(), labels, weight=label_weight, reduction=reduction)
            acc_t = ((preds.argmax(dim=1) - labels).__eq__(0).float().sum()) / len(labels)
            acc = acc_t.data.item()
        else:
            raise Exception("weird label tensor!")
        return loss, acc
    
def obtain_Transformer(bertPath, device=None):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    lvec = SentBert(bertPath)
    prop = GraphEncoder(lvec.bert_config, 2).to(device)
    cls = extend(nn.Linear(768, 2)).to(device)
    BiGCN_model = TwitterTransformer(lvec, prop, cls)
    return BiGCN_model

class GraphEncoder(nn.Module):
    def __init__(self, config, num_hidden_layers):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(num_hidden_layers)])

    def set_aug_type(self, aug_type):
        for i in range(self.config.num_hidden_layers):
            self.layer[i].aug_type = aug_type

    def forward(
            self,
            hidden_states,
            root_idxs,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        assert hidden_states.dim() == 3
        lens = root_idxs + [hidden_states.data.size(1)]
        eye_mtx = torch.eye(len(root_idxs), device=hidden_states.data.device)
        trans_mtx = torch.stack([eye_mtx[jj - 1] for jj in range(1, len(root_idxs) + 1, 1) \
                                 for _ in range(lens[jj - 1], lens[jj])])

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            new_hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            # hidden_states = #torch.matmul(trans_mtx, hidden_states[root_idxs]) \
            #                 + new_hidden_states
            hidden_states = torch.matmul(trans_mtx, hidden_states.squeeze(0)[root_idxs]).unsqueeze(0) \
                            + new_hidden_states

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class SentBert(nn.Module):
    def __init__(self, bertPath):
        super(SentBert, self).__init__()
        self.bert_config = BertConfig.from_pretrained(bertPath, num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained(bertPath)
        # self.model = BertModel.from_pretrained(bertPath, config=self.bert_config).to(torch.device('cuda:0'))
        self.model = nn.DataParallel(
            BertModel.from_pretrained(bertPath, config=self.bert_config).to(torch.device('cuda:0')),
            # device_ids=[0, 1, 2, 3],
            device_ids = [0]
        )

    def text_to_batch_transformer(self, text: List):
        """Turn a piece of text into a batch for transformer model

        :param text: The text to tokenize and encode
        :param tokenizer: The tokenizer to use
        :param: text_pair: An optional second string (for multiple sentence sequences)
        :return: A list of IDs and a mask
        """
        max_len = self.tokenizer.max_len if hasattr(self.tokenizer, 'max_len') else self.tokenizer.model_max_length
        items = [self.tokenizer.encode_plus(sent, text_pair=None, add_special_tokens=True, max_length=max_len,
                                            return_length=False, return_attention_mask=True,
                                            return_token_type_ids=True)
                 for sent in text]
        input_ids = [item['input_ids'] for item in items]
        masks = [item['attention_mask'] for item in items]
        seg_ids = [item['token_type_ids'] for item in items]
        max_length = max([len(i) for i in input_ids])

        input_ids = torch.tensor([(i + [0] * (max_length - len(i))) for i in input_ids], device=torch.device('cuda:0'))
        masks = torch.tensor([(m + [0] * (max_length - len(m))) for m in masks], device=torch.device('cuda:0'))
        seg_ids = torch.tensor([(s + [0] * (max_length - len(s))) for s in seg_ids], device=torch.device('cuda:0'))
        return input_ids, masks, seg_ids

    def forward(self, sents):
        input_ids, masks, seg_ids = self.text_to_batch_transformer(sents)
        encoder_dict = self.model.forward(
            input_ids=input_ids,
            attention_mask=masks,
            token_type_ids=seg_ids
        )
        return encoder_dict.pooler_output  # .unsqueeze(0)

    def save_model(self, model_file):
        torch.save(self.model.module.state_dict(), model_file)

    def load_model(self, pretrained_file):
        self.model.module.load_state_dict(torch.load(pretrained_file))
        
@background(max_prefetch=5)
def DANN_Dataloader(dataset:MetaMCMCDataset, batch_size=32):


    indexs_souce = [random.sample(range(len(domain)), len(domain)) for domain in dataset]
    
    
    bs_domain = batch_size//(len(dataset))#16 在两个领域上平均分配batch
    max_iters = max([len(domain)//bs_domain for domain in dataset])#max_iters = 371
    for iteration in range(max_iters):
        start = iteration*bs_domain
        items = [domain[indexs_souce[d_idx][idx%len(domain)]] \
                    for d_idx, domain in enumerate(dataset) \
                        for idx in range(start, start+bs_domain)]
        yield dataset[0].collate_fn(items)


@background(max_prefetch=3)
def Generator3(labeledSource, unlabeledTarget, labeledTarget, batchSize):
    halfBS = batchSize // 2
    bs2 = halfBS if halfBS < len(labeledTarget) else len(labeledTarget)
    bs1 = batchSize - bs2
    bs3 = batchSize // 3 if batchSize < 3 * len(unlabeledTarget) else len(unlabeledTarget)
    iter1, iter2, iter3 = len(labeledSource) // bs1, \
                          len(labeledTarget) // bs2, \
                          len(unlabeledTarget) // bs3
    maxIters = max([iter1, iter2, iter3])
    idxsS, idxsLT, idxsUT = [], [], []
    for i in range(maxIters + 1):
        if i % iter1 == 0:
            idxsS = random.sample(range(len(labeledSource)), len(labeledSource)) * 2
        if i % iter2 == 0:
            idxsLT = random.sample(range(len(labeledTarget)), len(labeledTarget)) * 2
        if i % iter3 == 0:
            idxsUT = random.sample(range(len(unlabeledTarget)), len(unlabeledTarget)) * 2
        # i * bs1 could be larger than the len(labelSource), thus we need to start from the remainder
        start_LS, start_LT, start_UT = (i * bs1) % len(labeledSource), \
                                       (i * bs2) % len(labeledTarget), \
                                       (i * bs3) % len(unlabeledTarget)
        end_LS, end_LT, end_UT = start_LS + bs1, start_LT + bs2, start_UT + bs3

        items1 = [labeledSource[jj] for jj in idxsS[start_LS:end_LS]]
        items2 = [labeledTarget[jj] for jj in idxsLT[start_LT:end_LT]]
        items3 = [unlabeledTarget[jj] for jj in idxsUT[start_UT:end_UT]]
        batch1 = labeledTarget.collate_fn(items1 + items2)
        batch2 = unlabeledTarget.collate_fn(
            random.sample(items2, batchSize - bs1 - bs3) + items1 + items3
        )
        yield batch1, batch2


@background(max_prefetch=3)
def Generator2(labeledSource, unlabeledTarget, batchSize):
    bs2 = batchSize // 2 if batchSize < 2 * len(unlabeledTarget) else len(unlabeledTarget)
    iter1, iter2 = len(labeledSource) // batchSize, \
                   len(unlabeledTarget) // bs2
    maxIters = max([iter1, iter2])
    idxsS, idxsUT = [], []
    for i in range(maxIters + 1):
        if i % iter1 == 0:
            idxsS = random.sample(range(len(labeledSource)), len(labeledSource)) * 2
        if i % iter2 == 0:
            idxsUT = random.sample(range(len(unlabeledTarget)), len(unlabeledTarget)) * 2
        # i * bs1 could be larger than the len(labelSource), thus we need to start from the remainder
        start_LS, start_UT = (i * batchSize) % len(labeledSource), \
                             (i * bs2) % len(unlabeledTarget)
        end_LS, end_UT = start_LS + batchSize, start_UT + bs2

        items1 = [labeledSource[jj] for jj in idxsS[start_LS:end_LS]]
        items2 = [unlabeledTarget[jj] for jj in idxsUT[start_UT:end_UT]]
        batch1 = labeledSource.collate_fn(items1)
        batch2 = unlabeledTarget.collate_fn(
            random.sample(items1, batchSize - bs2) + items2
        )
        yield batch1, batch2  # batch1 for CE Loss, batch2 for DA Loss


class MMETrainer(BaseTrainer):
    def __init__(self, seed, log_dir, suffix, model_file, class_num, temperature=1.0,
                 learning_rate=5e-3, batch_size=32, Lambda=0.1):
        super(MMETrainer, self).__init__()
        self.seed = seed
        if not os.path.exists(log_dir):
            os.system("mkdir %s" % log_dir)
        else:
            os.system("rm -rf %s" % log_dir)
            os.system("mkdir %s" % log_dir)
        self.log_dir = log_dir
        self.suffix = suffix
        self.model_file = model_file
        self.best_valid_acc = 0.0
        self.min_valid_loss = 1e8
        self.class_num = class_num
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.Lambda = Lambda
        self.valid_step = 0

    def AdvEntrophy(self, trModel: VirtualModel, batch):
        assert hasattr(trModel, "AdvPredict")
        preds = trModel.AdvPredict(batch)
        epsilon = torch.ones_like(preds) * (1e-8)
        preds = (preds - epsilon).abs()
        loss = -1 * self.Lambda * (preds * (preds.log())).sum(dim=1).mean()
        return loss

    def ModelTrain(self, trModel: VirtualModel, labeled_source: CustomDataset, labeled_target: CustomDataset,
                   unlabeled_target: CustomDataset, valid_target: CustomDataset, test_target: CustomDataset,
                   maxEpoch, validEvery=20, learning_rate=None, valid_label=None, test_label=None):
        self.initTrainingEnv(self.seed)
        optim = trModel.obtain_optim(learning_rate)
        best_valid_acc = 0
        best_test_acc=0
        # if os.path.exists(self.model_file):
        #     trModel.load_state_dict(
        #         torch.load(self.model_file)
        #     )
        for epoch in range(maxEpoch):
            for step, (batch1, batch2) in enumerate(
                    Generator3(labeled_source, unlabeled_target, labeled_target, self.batch_size)):
                loss, acc = trModel.lossAndAcc(batch1, temperature=self.temperature)
                trainLoss = loss
                optim.zero_grad()
                trainLoss.backward()
                optim.step()

                HEntrophy = self.AdvEntrophy(trModel, batch2)
                optim.zero_grad()
                HEntrophy.backward()
                optim.step()
                torch.cuda.empty_cache()
                print('####Model Update (%3d | %3d) %3d | %3d ####, loss = %6.8f, entrophy = %6.8f' % (
                    step, len(unlabeled_target), epoch, maxEpoch, loss, HEntrophy.data.item()
                ))
                if (step + 1) % validEvery == 0:
                    acc = self.valid(
                        trModel, valid_target, valid_target.labelTensor() if valid_label is None else valid_label,True,
                        suffix=f"{self.suffix}_valid"
                    )
                    if acc > best_valid_acc:
                        best_valid_acc = acc
                        # torch.save(trModel.state_dict(), self.model_file)
                        trModel.save_model(self.model_file)
                        test_acc=self.valid(
                            trModel, test_target, test_target.labelTensor() if test_label is None else test_label,True,
                            suffix=f"{self.suffix}_Besttest"
                        )
                        if test_acc>best_test_acc:
                            best_test_acc=test_acc
                            print("best_test_acc",best_test_acc)
                        
                    else:
                        test_acc=self.valid(
                            trModel, test_target, test_target.labelTensor() if test_label is None else test_label,True,
                            suffix=f"{self.suffix}_besttest"
                        )
                        if test_acc>best_test_acc:
                            best_test_acc=test_acc
                            print("best_test_acc_un",best_test_acc)
        print("best_valid_acc", best_valid_acc)
        print("best_test_acc", best_test_acc)


    def ModelTrainV2(self, trModel: VirtualModel, discriminator: nn.Module, labeled_source: CustomDataset,
                     labeled_target: CustomDataset,
                     unlabeled_target: CustomDataset, valid_target: CustomDataset, test_target: CustomDataset,
                     maxEpoch=10, validEvery=20, E_Step=1, T_Step=5, lr_G=3e-6,lr_C=5e-5):
        self.initTrainingEnv(10086)
        optim_G = trModel.obtain_optim(lr_G)
        optim_C = discriminator.obtain_optim(lr_C)
        for epoch in range(maxEpoch):
            if labeled_target is None:
                trainLoader = Generator2(labeled_source, unlabeled_target, labeled_target, self.batch_size)
            else:
                trainLoader = Generator3(labeled_source, unlabeled_target, labeled_target, self.batch_size)
                
            for step, (batch1, batch2) in enumerate(trainLoader):
                for da_idx in range(E_Step):
                    HEntrophy = self.AdvEntrophy(trModel, batch2)
                    optim_C.zero_grad()
                    (-1.0 * self.Lambda * HEntrophy).backward()
                    optim_C.step()
                    print('####Domain Adversarial %3d [%3d] %3d | %3d #### T_Entrophy = %6.8f' % (
                        da_idx, step, epoch, maxEpoch, HEntrophy.data.item()
                    ))

                for td_idx in range(T_Step):
                    loss, acc = trModel.lossAndAcc(batch1, self.temperature)
                    optim_G.zero_grad()
                    loss.backward()
                    optim_G.step()
                    print('****Task Discriminative %3d [%3d] %3d | %3d **** Loss/Acc = %6.8f/%6.8f' % (
                        td_idx, step, epoch, maxEpoch, loss.data.item(), acc
                    ))

                if (step + 1) % validEvery == 0:
                    self.valid(trModel, valid_target, valid_target.labelTensor(), suffix=f"{self.suffix}_valid")
                    self.valid(trModel, test_target, test_target.labelTensor(), suffix=f"{self.suffix}_test")
    
    def trainset2trainloader(self, dataset: BiGCNTwitterSet, batch_size=32,shuffle=False):
        return self.dataset2dataloader(dataset, shuffle, batch_size)

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
    
    # def collate_fn(self, items):
    #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    #     sents = [item.text for item in items]
    #     TD_graphs = [item.g_TD for item in items]
    #     BU_graphs = [item.g_BU for item in items]
    #     labels = [item.data_y for item in items]
    #     topic_labels = [item.topic_label for item in items]
    #     num_nodes = [g.num_nodes() for g in TD_graphs]
    #     big_g_TD = dgl.batch(TD_graphs)
    #     big_g_BU = dgl.batch(BU_graphs)
    #     A_TD = big_g_TD.adjacency_matrix().to_dense().to(device)
    #     A_BU = big_g_BU.adjacency_matrix().to_dense().to(device)
    #     weights = torch.stack([item.weight for item in items])
    #     idxs = torch.tensor([item.index for item in items], dtype=torch.long,
    #                         device=device)
    #     return sents, num_nodes, A_TD, A_BU, torch.tensor(labels), torch.tensor(topic_labels), weights, idxs
    
    def collate_fn(self, items):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
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
        A_TD = big_g_TD.adjacency_matrix().to_dense().to(device)
        A_BU = big_g_BU.adjacency_matrix().to_dense().to(device)

        sents = [item.text for item in items]

        sentences = [text for s_list in sents for text in s_list]
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

        weights = torch.stack([item.weight for item in items]).to(device)
        idxs = torch.tensor([item.index for item in items], dtype=torch.long,
                            device=device)
        assert len(sentences) == sum(num_nodes)

        return tfidf_arr, num_nodes, A_TD, A_BU, \
            torch.tensor(labels), torch.tensor(topic_labels), weights, idxs
    
    def PateroPrint(self, model:TwitterTransformer, testset:MetaMCMCDataset):
        domain_acc_li = []
        task_acc_li = []
        for batch in DANN_Dataloader([testset], batch_size=20):
            vecs = model.Batch2Vecs(batch)
            probs = discriminator(vecs).softmax(dim=1)
            predicted_labels = probs.argmax(dim=1)
            domain_acc = (predicted_labels == batch[-1]).float().mean().item()
            domain_acc_li.append(domain_acc)
            _, acc = model.lossAndAcc(batch)
            task_acc_li.append(acc)

        print("domain_acc:",domain_acc_li)
        print("task_acc:",task_acc_li)
    
if __name__ == '__main__':
    data_dir1 = r"../../../autodl-tmp/data/pheme-rnr-dataset/"
    data_dir2 = r"../../../autodl-tmp/data/t1516/"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1" 

    events_list = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting','sydneysiege','twitter15','twitter16']
    # for domain_ID in range(5):
    domain_ID = 5
    fewShotCnt = 100
    source_events = []
    target_events = []
    for idx,dname in enumerate(events_list):
        if idx != domain_ID:
            if dname == "twitter15" or dname == "twitter16":
                source_events.append(os.path.join(data_dir2, dname))
            else:
                source_events.append(os.path.join(data_dir1, dname))
    for idx, dname in enumerate(events_list):
        if idx == domain_ID:
            if dname=="twitter15" or dname=="twitter16":
                target_events.append(os.path.join(data_dir2, dname))
            else:
                target_events.append(os.path.join(data_dir1, dname))
    
    test_event_name = events_list[domain_ID]
    #train_set, labeled_target, val_set, test_set, unlabeled_target
    source_domain, labeled_target, val_set, test_set, unlabeled_target = load_data(
        source_events, target_events, fewShotCnt, unlabeled_ratio=0.3
    )
    
    logDir = f"../../../autodl-tmp/log/{test_event_name}/"
    
    bertPath = r"../../../autodl-tmp/bert_en"
    model = obtain_Transformer(bertPath)
    source_domain.initGraph()
    
    trainer = MMETrainer(seed = 10086, log_dir = logDir, suffix = f"{test_event_name}_FS{fewShotCnt}", model_file = f"../../../autodl-tmp/pkl/MME/MME_{test_event_name}_FS{fewShotCnt}.pkl", class_num = 2, temperature=0.05,
                 learning_rate=3e-6, batch_size=32, Lambda=0.1)
    
    bert_config = BertConfig.from_pretrained(bertPath,num_labels = 2)
    model_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    discriminator = DomainDiscriminator(hidden_size=bert_config.hidden_size,
                                        model_device = model_device,
                                        learningRate=5e-5,
                                        domain_num=7)
#     model.load_model(f"../../../autodl-tmp/pkl/DANN/DANN_{test_event_name}_FS{fewShotCnt}.pkl")
    trainer.ModelTrainV2(model,discriminator,source_domain,labeled_target,unlabeled_target,val_set,test_set,maxEpoch = 3,validEvery = 50,E_Step=1, T_Step=5,lr_G=2e-7,lr_C=5e-5)
    
    trainer.PateroPrint(model,test_set)

