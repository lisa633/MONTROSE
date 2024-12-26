import os,random,dgl,json
from BaseModel.BiGCN import DomainDiscriminator
from Data.BiGCN_Dataloader import MetaMCMCDataset,load_data
from transformers.models.bert import BertConfig, BertTokenizer
from openai import OpenAI
import torch, torch.nn as nn
from typing import List
import copy
import time,datetime

from backpack import extend, extensions
from BaseModel.BiGCN_Utils.GraphRumorDect import BiGCNRumorDetecV2
from BaseModel.modeling_bert import *
from transformers.models.bert import BertConfig, BertTokenizer

client = OpenAI(
            api_key="sk-65734579fab943f48234f366e64ad181", 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

event_dics = {
    'charliehebdo': 0,
    'ferguson': 1,
    'germanwings-crash': 2,
    'ottawashooting': 3,
    'sydneysiege': 4
}

def str2timestamp(str_time):
    month = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
             'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
             'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    ss = str_time.split(' ')
    m_time = ss[5] + "-" + month[ss[1]] + '-' + ss[2] + ' ' + ss[3]
    d = datetime.datetime.strptime(m_time, "%Y-%m-%d %H:%M:%S")
    t = d.timetuple()
    timeStamp = int(time.mktime(t))
    return timeStamp

def scan_dir(files,dir_name):
    for item in os.walk(dir_name):
        if len(item[2]) == 0:
            # no file in this dir
            pass
        else:
            for fname in item[2]:
                tmp_path = os.path.join(item[0], fname)
                if tmp_path[-5:] == ".json": # is a json file
                    files.append(tmp_path)
                else:
                    print("Warning: non json format file exists in %s : %s" % (dir_name, tmp_path))

def twitter_data_process(files):
    # print("files:",files)
    data = {}
    for file_path in files:
        print(file_path)
        twitter_dict = {}
        ss = file_path.split("/")
#         print(ss)
        origin_data = json.load(open(file_path, mode="r", encoding="utf-8"))
        # 'Wed Jan 07 11:14:08 +0000 2015'
        if origin_data['lang'] == 'en':
            twitter_dict[ss[-3]] = {
                            'topic_label': event_dics[ss[-5]],
                            'label': ss[-4],
                            'event': ss[-5],
                            'sentence': [origin_data['text'].lower()],
                            'created_at': [str2timestamp(origin_data['created_at'])],
                            'tweet_id': [origin_data['id']],
                            "reply_to": [origin_data['in_reply_to_status_id']]
                           }
            
            for key in twitter_dict.keys():  # use temporary data to organize the final whole data
                if key in data:
                    if twitter_dict[key]['tweet_id'][0] in data[key]['tweet_id']:
                        pass  # sometimes, there are dumlicated posts
                    else:
                        data[key]['tweet_id'].append(twitter_dict[key]['tweet_id'][0])
                        data[key]['sentence'].append(twitter_dict[key]['sentence'][0])
                        data[key]['created_at'].append(twitter_dict[key]['created_at'][0])
                        data[key]['reply_to'].append(twitter_dict[key]['reply_to'][0])
                        
                else:
                    data[key] = twitter_dict[key]

    return data

class Node:
    def __init__(self,index,parent,children,data,model,discriminator):
        self.index = index
        self.parent = parent
        self.children = children
        self.data = data
        self.model = model
        self.discriminator = discriminator
        self.state = False
        self.visits = 0
        self.score = 0.0
        self.copy_data = copy.deepcopy(data)
    def get_child_node(self,all_node_list):
        child_node_list = []
        for node_id in self.children:
            child_node_list.append(all_node_list[node_id])
        return child_node_list
    
    def get_parent_node(self,all_node_list):
        parent_node_list = []
        for node_id in self.parent:
            parent_node_list.append(all_node_list[node_id])
        return parent_node_list[0]
    
    def compute_score(self,data):
        batch = source_domain.collate_fn([data])
        with torch.no_grad():
            vecs = self.model.Batch2Vecs(batch)
        logits = self.discriminator(vecs)
        domain_loss = F.cross_entropy(logits, batch[-1]) 
        return domain_loss

    def expand(self):

        prompt_sent = self.copy_data.text[self.index]
        # target_sent = random.choice(unlabeled_target[random.randint(0,len(unlabeled_target.data_ID))].text)
        target_sent = "Black teenage boys are not men. They are children. Stop referring to a 17 year old as a man. You are killing children. #ferguson"
        try:
            completion = client.chat.completions.create(
            model="qwen-plus", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': 'You are a twitter user. You can rephrase twitter-form reply to make it like replies in the target domain'},
                {'role': 'user', 'content': 'here is an example in target domain: '+target_sent+' The given twitter is:'+prompt_sent+'Please rephrase the given twitter to make it a reply in the target domain without \'Note\''}],
            )
            generate_sents = completion.choices[0].message.content
        except:
            generate_sents = prompt_sent
        self.copy_data.text[self.index] = generate_sents
        self.score = self.compute_score(self.copy_data)
       

    def select(self,root_node, all_node_list):
        best_value = float('-inf')
        child_node_list = self.get_child_node(all_node_list)
        best_child = child_node_list[0]
        for child in child_node_list:
            if child.visits > 0:
                value = child.score/child.visits + 2*math.sqrt(2*math.log(self.visits)/child.visits)
            else:
                value = float('inf')
#             print("value:",value)
            if value > best_value:
                best_value = value
                best_child = child
        best_child.state = True
        return best_child

    def backpropagate(self,all_node_list):
        self.visits += 1
        parent = self.get_parent_node(all_node_list)
        parent.visits += 1
        parent.score += self.score
#         print(len(parent.parent))
        while len(parent.parent) != 0:
            parent = parent.get_parent_node(all_node_list)
            parent.visits += 1
            parent.score += self.score


def mcts(root_node, all_node_list, iterations):
    generate_text = root_node.data["text"]
    print("original_text:",generate_text)
    best_score = root_node.compute_score(root_node.data)
    print("init score:",best_score)
    
    for i in range(iterations):
        print("step:",i)
        explore = []
        explore.append(root_node.index)
        root_node.state = True
        node = root_node
        node.copy_data = copy.deepcopy(root_node.copy_data)
        print("process:",node.copy_data["text"])
        while len(node.children) > 0 and node.state == True:
#             print(len(node.children))
#             print(node.state)
            node = node.select(root_node,all_node_list)
#             print(node.index)
            explore.append(node.index)
        print("explore:",explore)
        node.expand()
        node.backpropagate(all_node_list)
        score = node.compute_score(node.copy_data)
        print("after score:",score)

        if score < best_score:
            print("modify save!")
            best_score = score
            generate_text = node.copy_data["text"]
            root_node.copy_data = node.copy_data
            
    return generate_text
            
        
        

class SentBert(nn.Module):
    def __init__(self, bertPath):
        super(SentBert, self).__init__()
        self.bert_config = BertConfig.from_pretrained(bertPath, num_labels=2)
        self.tokenizer = BertTokenizer.from_pretrained(bertPath)
        # self.model = BertModel.from_pretrained(bertPath, config=self.bert_config).to(torch.device('cuda'))
        self.model = nn.DataParallel(
            BertModel.from_pretrained(bertPath, config=self.bert_config).to(torch.device('cuda')),
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

        input_ids = torch.tensor([(i + [0] * (max_length - len(i))) for i in input_ids], device=torch.device('cuda'))
        masks = torch.tensor([(m + [0] * (max_length - len(m))) for m in masks], device=torch.device('cuda'))
        seg_ids = torch.tensor([(s + [0] * (max_length - len(s))) for s in seg_ids], device=torch.device('cuda'))
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


class TwitterTransformer(BiGCNRumorDetecV2):
    def __init__(self, sent2vec, prop_model, rdm_cls, **kwargs):
        super(TwitterTransformer, self).__init__(sent2vec, prop_model, rdm_cls)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f"Attribute '{k}' is not a valid attribute of DaMSTF")
            self.__setattr__(k, v)
   

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
        print("preds device:",preds.device)
        labels = batch[-2].to(preds.device)
        loss, acc = self.loss_func(preds, labels, label_weight=label_weight, reduction=reduction)
        return loss, acc
    

    def get_CrossEntropyLoss(self,batch):
        seq_outs = self.Batch2Vecs(batch)
        logits = self.rdm_cls(seq_outs)
        labels = batch[-2].to(logits.device)
        loss_fn = extend(torch.nn.CrossEntropyLoss(reduction="mean"))
        loss=loss_fn(logits,labels)
        return loss
    
    def get_MetaLearningCrossEntropyLoss(self,batch,weights):
        seq_outs = self.Batch2Vecs(batch)
        logits = self.rdm_cls(seq_outs)
        labels = batch[-2].to(logits.device)
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


if __name__ == '__main__':
    data_dir1 = r"../../autodl-tmp/data/test/"
    data_dir2 = r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_new/ferguson"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0" 

    events_list = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting','sydneysiege']
    # for domain_ID in range(5):
    domain_ID = 1
    fewShotCnt = 0
    source_events = [os.path.join(data_dir1, dname)
                     for idx, dname in enumerate(events_list) if idx != domain_ID]
    # source_events = [os.path.join(data_dir1, dname)
    #                  for idx, dname in enumerate(events_list)]
    target_events = [os.path.join(data_dir1, events_list[domain_ID])]
    test_event_name = events_list[domain_ID]
    #train_set, labeled_target, val_set, test_set, unlabeled_target
    source_domain, labeled_target, val_set, test_set, unlabeled_target = load_data(
        source_events, target_events, fewShotCnt, unlabeled_ratio=0
    )
    bertPath = r"../../autodl-tmp/bert_en"
    bert_config = BertConfig.from_pretrained(bertPath,num_labels = 2)
    model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = obtain_Transformer(bertPath)
    source_domain.initGraph()
    if os.path.exists(f"../../autodl-tmp/pkl/GpDANN/{test_event_name}/BiGCN_{test_event_name}.pkl"):
        model.load_model(f"../../autodl-tmp/pkl/GpDANN/{test_event_name}/BiGCN_{test_event_name}.pkl")

    discriminator = DomainDiscriminator(hidden_size=bert_config.hidden_size,
                                    model_device = model_device,
                                    learningRate=2e-5,
                                    domain_num=5)
    if os.path.exists(f"../../autodl-tmp/pkl/GpDANN/DomainDiscriminator_{test_event_name}.pkl"):
        discriminator.load_state_dict(
            torch.load(f"../../autodl-tmp/pkl/GpDANN/DomainDiscriminator_{test_event_name}.pkl")
        )
    else:
        for epoch in range(3):
            discriminator(model, source_domain, unlabeled_target, max_step=500)
    source_data_copy = copy.deepcopy(source_domain)

    generate_dict = {}

    for i,d_ID in enumerate(source_data_copy.data_ID):
        data = source_data_copy[i]
        tree = data.g_TD
        nodes = tree.nodes()
        print("nodes:",nodes)
        s,d = tree.remove_self_loop().edges()
        print("s,d:",s,d)
        nodes_list = nodes.cpu().tolist()
        source = s.cpu().tolist()
        des = d.cpu().tolist()
        child_lists = [[des[k] for k, nodes_id in enumerate(source) if nodes_id == n] for n in nodes_list]
        parent_lists = [[source[j] for j, nodes_id in enumerate(des) if nodes_id == n] for n in nodes_list]
        class_node_list = [Node(s,parent_lists[s],child_lists[s],data,model,discriminator) for s,n in enumerate(nodes_list)]
        for node in class_node_list:
            if len(node.parent)==0:
                root_node = node
        generate_sent = mcts(root_node, class_node_list, 10)
        generate_dict[d_ID] = generate_sent
   
    files = []
    for event_path in source_events:
        scan_dir(files, event_path)

    print(files[0])


    source_data = twitter_data_process(files)
    s_copy = copy.deepcopy(source_data)

    for key,value in source_data.items():
        new_key = int(str(key)+str(random.randint(0, 9)))
        s_copy[new_key] = {}
        s_copy[new_key]['topic_label'] = event_dics[test_event_name]
        s_copy[new_key]['label'] = value['label']
        s_copy[new_key]['event'] = value['event']
        s_copy[new_key]['tweet_id'] = value['tweet_id']
        s_copy[new_key]['sentence'] = generate_dict[key]
        print(s_copy[new_key]['sentence'])
        s_copy[new_key]['reply_to'] = value['reply_to']
        s_copy[new_key]['created_at'] = value['created_at']
    
    gen_target = MetaMCMCDataset()
    gen_target.data = s_copy
    gen_target.dataclear()
    event_dir = os.path.join(data_dir1,"qwen_gen_from_source",test_event_name)
    print(event_dir)
    gen_target.Caches_Data(event_dir)
    




