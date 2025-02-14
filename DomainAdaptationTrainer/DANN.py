import sys
sys.path.append('..')
import os, pickle, random, numpy as np
from TrainingEnv import BaseTrainer, AdversarialModel, CustomDataset, GradientReversal
import torch, torch.nn as nn, torch.nn.functional as F
from prefetch_generator import background
from SAM import SAM
from typing import List, AnyStr
import dgl
from backpack import extend, extensions

from cockpit import Cockpit, CockpitPlotter, quantities
from cockpit.utils import schedules
from Data.BiGCN_Dataloader import BiGCNTwitterSet, FastBiGCNDataset, MetaMCMCDataset, load_data, load_data_twitter15
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
#         print("domain preds:",preds)
#         print("batch[-1]:",batch[-1])
        if labels is None:
            labels = batch[-1].to(preds.device)
        else:
            labels = labels.to(preds.device)
#         print("domain labels:",labels)
        if labels.dim() == 2:
            loss, acc = self.expandCrossEntropy(preds, labels, label_weight, reduction)
        elif labels.dim() == 1:
            loss = F.nll_loss(preds.log(), labels, weight=label_weight, reduction=reduction)
            acc_t = ((preds.argmax(dim=1) - labels).__eq__(0).float().sum()) / len(labels)
            acc = acc_t.data.item()
        else:
            raise Exception("weird label tensor!")
        return loss, acc
   

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
#         print("preds:",preds)
        epsilon = torch.ones_like(preds) * 1e-8
        preds = (preds - epsilon).abs()  # to avoid the prediction [1.0, 0.0], which leads to the 'nan' value in log operation
        labels = batch[-2].to(preds.device)
#         print("labels:",labels)
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
#         print("labels.dim", labels.dim())
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


def Generator1(labeledSource, unlabeledTarget, labeledTarget, batchSize):
    halfBS = batchSize // 2
    bs2 = halfBS if halfBS < len(labeledTarget) else len(labeledTarget)
    bs1 = batchSize - bs2
    bs3 = batchSize // 3 if batchSize < 3 * len(unlabeledTarget) else len(unlabeledTarget)
    iter1, iter2, iter3 = len(labeledSource) // bs1, \
                          len(labeledTarget) // bs2, \
                          len(unlabeledTarget) // bs3
    maxIters = max([iter1, iter2, iter3])

    @background(max_prefetch=5)
    def generator():
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

    return maxIters, generator


def Generator2(labeledSource, unlabeledTarget, batchSize):
    bs2 = batchSize // 2 if batchSize < 2 * len(unlabeledTarget) else len(unlabeledTarget)
    iter1, iter2 = len(labeledSource) // batchSize, \
                   len(unlabeledTarget) // bs2
    maxIters = max([iter1, iter2])

    @background(max_prefetch=5)
    def generator():
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

    return maxIters, generator


def Generator3(labeledSource, unlabeledTarget, batchSize):
    bs2 = batchSize // 2 if batchSize < 2 * len(unlabeledTarget) else len(unlabeledTarget)
    bs1 = batchSize - bs2
    iter1, iter2 = len(labeledSource) // bs1, \
                   len(unlabeledTarget) // bs2
    maxIters = max([iter1, iter2])

    @background(max_prefetch=5)
    def generator():
        idxsS, idxsUT = [], []
        for i in range(maxIters + 1):
            if i % iter1 == 0:
                idxsS = random.sample(range(len(labeledSource)), len(labeledSource)) * 2
            if i % iter2 == 0:
                idxsUT = random.sample(range(len(unlabeledTarget)), len(unlabeledTarget)) * 2
            # i * bs1 could be larger than the len(labelSource), thus we need to start from the remainder

            start_LS, start_UT = (i * bs1) % len(labeledSource), \
                                 (i * bs2) % len(unlabeledTarget)
            end_LS, end_UT = start_LS + bs1, start_UT + bs2

            items1 = [labeledSource[jj] for jj in idxsS[start_LS:end_LS]]
            items2 = [unlabeledTarget[jj] for jj in idxsUT[start_UT:end_UT]]
            batch = unlabeledTarget.collate_fn(
                items1 + items2
            )
            yield batch  # batch1 for CE Loss, batch2 for DA Loss

    return maxIters, generator

def DataIter(labeledSource, unlabeledTarget, labeledTarget=None, batchSize=32):
    if labeledTarget is not None:
        assert len(labeledTarget) > 0
        return Generator1(labeledSource, unlabeledTarget, labeledTarget, batchSize)
    else:
        return Generator2(labeledSource, unlabeledTarget, batchSize)


class DANNTrainer(BaseTrainer):
    def __init__(self, random_seed, log_dir, suffix, model_file, class_num, temperature=1.0,
                 learning_rate=5e-3, batch_size=32, Lambda=0.1):
        super(DANNTrainer, self).__init__()
        if not os.path.exists(log_dir):
            os.system("mkdir {}".format(log_dir))
        self.random_seed = random_seed
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
        assert len(sentences) == sum(num_nodes)

        return tfidf_arr, num_nodes, A_TD, A_BU, \
            torch.tensor(labels), torch.tensor(topic_labels)


    def PreTrainDomainClassifier(self, trModel: nn.Module, discriminator: nn.Module,
                                 labeledSource: CustomDataset, labeledTarget: CustomDataset,
                                 unlabeledTarget: CustomDataset, maxEpoch, learning_rate=5e-3):
        # the rule for early stop: when the variance of the recent 50 training loss is smaller than 0.05,
        # the training process will be stopped
        assert hasattr(trModel, 'AdvDLossAndAcc')
        paras = trModel.grouped_parameters(self.learning_rate) + \
                [{"params": discriminator.parameters(), "lr": self.learning_rate}]
        optim = torch.optim.Adam(paras, lr=self.learning_rate)
        lossList = []
        for epoch in range(maxEpoch):
            maxIters, trainLoader = DataIter(labeledSource, unlabeledTarget, labeledTarget, self.batch_size)
            for step, (_, batch2) in enumerate(trainLoader()):
#                 print("labels",batch2[-1])
                with torch.no_grad():
                    vecs = trModel.Batch2Vecs(batch2)
                logits = discriminator(vecs)
                DLoss = F.cross_entropy(logits, batch2[-1])
                probs = discriminator(vecs).softmax(dim=1) #此处的概率是判断属于哪个域
                predicted_labels = probs.argmax(dim=1)
#                 print("predicted_labels:",predicted_labels)
                DAcc = (predicted_labels == batch2[-1]).float().mean().item()
#                 DLoss, DAcc = trModel.discriminatorLoss(discriminator, batch2, grad_reverse=False)
                optim.zero_grad()
                DLoss.backward()
                optim.step()
                torch.cuda.empty_cache()
                print('####Pre Train Domain Classifier (%3d | %3d) %3d | %3d ####, loss = %6.8f, Acc = %6.8f' % (
                    step, maxIters, epoch, maxEpoch, DLoss.data.item(), DAcc
                ))
                lossList.append(DLoss.data.item())
                if len(lossList) > 20:
                    lossList.pop(0)
                    if np.std(lossList) < 0.05 and np.mean(lossList) < 0.2:
                        return

    def ModelTrain(self, trModel: AdversarialModel, discriminator: nn.Module,
                   labeledSource: CustomDataset, labeledTarget: CustomDataset,
                   unlabeledTarget: CustomDataset, validSet: CustomDataset,
                   testSet: CustomDataset, maxEpoch, validEvery=20, test_label=None):
        assert hasattr(trModel, 'AdvDLossAndAcc')
        print("labeled Source/labeled Target/unlabeled Target: {}/{}/{}".format(len(labeledSource),
                                                                                len(labeledTarget) if labeledTarget is not None else 0,
                                                                                len(unlabeledTarget)))
        paras = trModel.grouped_parameters(self.learning_rate) + \
                [{"params": discriminator.parameters(), "lr": self.learning_rate}]
        optim = torch.optim.Adam(paras, lr=self.learning_rate)
        best_test_acc=0
        for epoch in range(maxEpoch):
            maxIters, trainLoader = DataIter(labeledSource, unlabeledTarget, labeledTarget, self.batch_size)
            for step, (batch1, batch2) in enumerate(trainLoader()):
                loss, acc = trModel.lossAndAcc(batch1)
                DLoss, DAcc = trModel.AdvDLossAndAcc(discriminator, batch2)
                trainLoss = loss + self.Lambda * DLoss
                optim.zero_grad()
                trainLoss.backward()
                optim.step()

                torch.cuda.empty_cache()
                print(
                    '####Model Update (%3d | %3d) %3d | %3d ####, loss/acc = %6.8f/%6.8f, DLoss/DAcc = %6.8f/%6.8f' % (
                        step, maxIters, epoch, maxEpoch, loss.data.item(), acc, DLoss.data.item(), DAcc
                    ))
                if (step + 1) % validEvery == 0:
                    acc_v = self.valid(trModel, validSet, validSet.labelTensor(),True, suffix=f"Valid_{self.suffix}")
                    if acc_v > self.best_valid_acc:
                        print("save model!",self.model_file)
                        trModel.save_model(self.model_file)
                        self.best_valid_acc = acc_v
                        test_acc=self.valid(
                            trModel, testSet, testSet.labelTensor() if test_label is None else test_label,True,
                            suffix=f"BestTest_{self.suffix}"
                        )
                        if test_acc>best_test_acc:
                            best_test_acc=test_acc
                            print("best_test_acc",best_test_acc)
                    else:
                        test_acc=self.valid(
                            trModel, testSet, testSet.labelTensor() if test_label is None else test_label,True,
                            suffix=f"Test_{self.suffix}"
                        )
                        if test_acc>best_test_acc:
                            best_test_acc=test_acc
                            print("best_test_acc_un",best_test_acc)

    def ModelTrainV2(self, trModel: AdversarialModel, discriminator: nn.Module,
                     labeledSource: CustomDataset, labeledTarget: CustomDataset,
                     unlabeledTarget: CustomDataset, validSet: CustomDataset, testSet: CustomDataset = None,
                     maxEpoch=50, validEvery=20, D_Step=5, T_Step=1):
        assert hasattr(trModel, 'AdvDLossAndAcc')
        print("labeled Source/labeled Target/unlabeled Target: {}/{}/{}".format(len(labeledSource),
                                                                                len(labeledTarget) if labeledTarget is not None else 0,
                                                                                len(unlabeledTarget)))
        paras = trModel.grouped_parameters(self.learning_rate) + \
                [{"params": discriminator.parameters(), "lr": self.learning_rate}]
        optim = torch.optim.Adam(paras, lr=self.learning_rate)

        for epoch in range(maxEpoch):
            maxIters, trainLoader = DataIter(labeledSource, unlabeledTarget, labeledTarget, self.batch_size)
            for step, (batch1, batch2) in enumerate(trainLoader()):
                optim.zero_grad()
                for da_idx in range(D_Step):
                    DLoss, DAcc = trModel.AdvDLossAndAcc(discriminator, batch2)
                    optim.zero_grad()
                    (self.Lambda * DLoss).backward()
                    optim.step()
                    print('####Domain Adversarial [%3d] (%3d | %3d) %3d | %3d #### DLoss/DAcc = %6.8f/%6.8f' % (
                        da_idx, step, maxIters, epoch, maxEpoch, DLoss.data.item(), DAcc
                    ))
                optim.step()

                for td_idx in range(T_Step):
                    loss, acc = trModel.lossAndAcc(batch1)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    print('****Task Discriminative [%3d] (%3d | %3d) %3d | %3d **** Loss/Acc = %6.8f/%6.8f' % (
                        td_idx, step, maxIters, epoch, maxEpoch, loss.data.item(), acc
                    ))
                torch.cuda.empty_cache()
                if (step + 1) % validEvery == 0:
                    acc_v = self.valid(trModel, validSet, validSet.labelTensor(), suffix=f"Valid_{self.suffix}")
                    if acc_v > self.best_valid_acc:
                        torch.save(trModel.state_dict(), self.model_file)
                        self.best_valid_acc = acc_v
                        self.valid(trModel, testSet, testSet.labelTensor(), suffix=f"BestTest_{self.suffix}")
                    else:
                        self.valid(trModel, testSet, testSet.labelTensor(), suffix=f"Test_{self.suffix}")
    

class GpDANNTrainer(DANNTrainer):
    def GpDaNNTrain(self, trModel: AdversarialModel, discriminator: nn.Module, labeledSource: CustomDataset,
                    unlabeledTarget: CustomDataset, maxEpoch,learning_rate=2e-5):
        assert hasattr(trModel, "AdvDLossAndAcc")
        paras = trModel.grouped_parameters(learning_rate) + \
                [{"params": discriminator.parameters(), "lr": learning_rate}]
        base_optimizer = torch.optim.SGD
        optim = SAM(paras, base_optimizer, lr=learning_rate, momentum=0.9)

        for epoch in range(maxEpoch):
            maxIters, trainLoader = Generator3(labeledSource, unlabeledTarget, self.batch_size)
            for step, batch in enumerate(trainLoader()):
                loss, acc = trModel.AdvDLossAndAcc(discriminator, batch)
                loss.backward()
                optim.first_step(zero_grad=True)
                loss, acc = trModel.AdvDLossAndAcc(discriminator, batch)
                loss.backward()
                optim.second_step(zero_grad=True)
                print('####Model Update (%3d | %3d) %3d | %3d #### DLoss/DAcc = %6.8f/%6.8f' % (
                    step, maxIters, epoch, maxEpoch, loss.data.item(), acc
                ))
            torch.save(trModel.state_dict(), self.model_file[:-4]+"_GpDaNNTrain.pkl")
    def TaskTrain(self, trModel: AdversarialModel, labeledSource: CustomDataset, labeledTarget: CustomDataset,
                  validSet: CustomDataset, testSet: CustomDataset, maxEpoch,learning_rate=2e-5, validEvery=20, test_label=None):
        print("labeled Source/labeled Target: {}/{}".format(
            len(labeledSource),
            len(labeledTarget) if labeledTarget is not None else 0,
        )
        )
        optim = trModel.obtain_optim(learning_rate)

        for epoch in range(maxEpoch):
            maxIters, trainLoader = Generator3(labeledSource, labeledTarget, self.batch_size)
            for step, (batch1) in enumerate(trainLoader()):
                loss, acc = trModel.lossAndAcc(batch1, temperature=self.temperature)
                optim.zero_grad()
                loss.backward()
                optim.step()
                torch.cuda.empty_cache()
                print('####Model Update (%3d | %3d) %3d | %3d ####, loss/acc = %6.8f/%6.8f' % (
                    step, maxIters, epoch, maxEpoch, loss.data.item(), acc
                ))
                if (step + 1) % validEvery == 0:
                    acc_v = self.valid(trModel, validSet, validSet.labelTensor(), suffix=f"Valid_{self.suffix}")
                    if acc_v > self.best_valid_acc:
                        torch.save(trModel.state_dict(), self.model_file)
                        self.best_valid_acc = acc_v
                        self.valid(
                            trModel, testSet, testSet.labelTensor() if test_label is None else test_label,
                            suffix=f"BestTest_{self.suffix}"
                        )
                    else:
                        self.valid(
                            trModel, testSet, testSet.labelTensor() if test_label is None else test_label,
                            suffix=f"Test_{self.suffix}"
                        )

    def ModelTrain(self, trModel: AdversarialModel, discriminator: nn.Module,
                   labeledSource: CustomDataset, labeledTarget: CustomDataset,
                   unlabeledTarget: CustomDataset, validSet: CustomDataset,
                   testSet: CustomDataset, maxEpoch, validEvery=20, test_label=None):
        print("labeled Source/labeled Target/unlabeled Target: {}/{}/{}".format(len(labeledSource),
                                                                                len(labeledTarget) if labeledTarget is not None else 0,
                                                                                len(unlabeledTarget)))

        # self.GpDaNNTrain(trModel, discriminator, labeledSource, unlabeledTarget, maxEpoch=10, learning_rate=5e-5)
        # self.TaskTrain(trModel, labeledSource, labeledTarget, validSet, testSet, maxEpoch=10, learning_rate=2e-3,
        #                validEvery=validEvery)
        for alternate in range(maxEpoch):
            self.GpDaNNTrain(trModel, discriminator, labeledSource, unlabeledTarget, maxEpoch=3, learning_rate=2e-5)
            self.TaskTrain(trModel, labeledSource, labeledTarget, validSet, testSet, maxEpoch=3, learning_rate=2e-5,
                           validEvery=validEvery, test_label=None)

            
if __name__ == '__main__':
    data_dir1 = r"../../../autodl-tmp/data/pheme-rnr-dataset/"
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0" 

    events_list = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting','sydneysiege']
    # for domain_ID in range(5):
    domain_ID = 4
    fewShotCnt = 100
    source_events = [os.path.join(data_dir1, dname)
                     for idx, dname in enumerate(events_list) if idx != domain_ID]
    # source_events = [os.path.join(data_dir1, dname)
    #                  for idx, dname in enumerate(events_list)]
    target_events = [os.path.join(data_dir1, events_list[domain_ID])]
    test_event_name = events_list[domain_ID]
    #train_set, labeled_target, val_set, test_set, unlabeled_target
    source_domain, labeled_target, val_set, test_set, unlabeled_target = load_data(
        source_events, target_events, fewShotCnt, unlabeled_ratio=0.3
    )
    
    logDir = f"../../../autodl-tmp/log/DANN/{test_event_name}/"
    
    bertPath = r"../../../autodl-tmp/bert_en"
    model = obtain_Transformer(bertPath)
    source_domain.initGraph()
    
    trainer = DANNTrainer(random_seed = 10086, log_dir = logDir, suffix = f"{test_event_name}_FS{fewShotCnt}", model_file = f"../../../autodl-tmp/pkl/DANN/DANN_{test_event_name}_FS{fewShotCnt}.pkl", class_num = 2, temperature=0.05,
                 learning_rate=3e-6, batch_size=32, Lambda=0.1)
    
    bert_config = BertConfig.from_pretrained(bertPath,num_labels = 2)
    model_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    discriminator = DomainDiscriminator(hidden_size=bert_config.hidden_size,
                                        model_device = model_device,
                                        learningRate=5e-5,
                                        domain_num=5)
    trainer.PreTrainDomainClassifier(model,discriminator,source_domain,labeled_target,unlabeled_target,maxEpoch = 2, learning_rate = 3e-6)
    
    trainer.ModelTrain(model,discriminator,source_domain,labeled_target,unlabeled_target,val_set,test_set,maxEpoch = 1,validEvery = 50,test_label=test_set.labelTensor())
    

    
    