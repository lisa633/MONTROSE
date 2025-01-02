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

domain_ID = 0

def construct_graph(temp_dict):
    tIds_dic = {}
    dup_cnt = 0
    dup_idxs = []
    data_len = len(temp_dict["text"])

    for idx, ID in enumerate(temp_dict["tweet_id"]):
        if ID in tIds_dic:
            data_len -= 1
            dup_cnt += 1
            dup_idxs.append(idx)
        else:
            tIds_dic[ID] = idx - dup_cnt

    for i, idx in enumerate(dup_idxs):
        temp_dict["tweet_id"].pop(idx-i)
        temp_dict["reply_to"].pop(idx-i)
        temp_dict[d_ID]["text"].pop(idx-i)
        temp_dict[d_ID]["sentence"].pop(idx-i)
        temp_dict[d_ID]["created_at"].pop(idx-i)

    edges = [(src_idx, tIds_dic[dst_ID] if dst_ID in tIds_dic else 0)
                for src_idx, dst_ID in enumerate(temp_dict["reply_to"][:data_len])]
    src = np.array([item[0] for item in edges])
    dst = np.array([item[1] for item in edges])
    g_TD = dgl.graph((dst, src), num_nodes=data_len)
    g_BU = dgl.graph((src, dst), num_nodes=data_len)
    return g_TD, g_BU

def compute_loss(temp_dict,model,discriminator):

    gen_data = MetaMCMCDataset()
    timestamp = int(time.time())
    temp_id = str(timestamp)
    gen_data.data[temp_id] = temp_dict
    gen_data.dataclear(merge = False)

    batch = source_domain.collate_fn([gen_data[0]])
    with torch.no_grad():
        vecs = model.Batch2Vecs(batch)
    logits = discriminator(vecs)
    topic_label = torch.tensor([domain_ID],device=torch.device('cuda'))
    domain_loss = F.cross_entropy(logits, topic_label) 
    return domain_loss

def compute_score(temp_dict,model,discriminator):
    gen_data = MetaMCMCDataset()
    timestamp = int(time.time())
    temp_id = str(timestamp)
    gen_data.data[temp_id] = temp_dict
    gen_data.dataclear(merge = False)
    batch = source_domain.collate_fn([gen_data[0]])
    with torch.no_grad():
        vecs = model.Batch2Vecs(batch)
    logits = discriminator(vecs)
    value = logits[0, domain_ID].item()
    return value

def compute_confidence(temp_dict,model,discriminator):
    gen_data = MetaMCMCDataset()
    timestamp = int(time.time())
    temp_id = str(timestamp)
    gen_data.data[temp_id] = temp_dict
    gen_data.dataclear(merge = False)
    batch = source_domain.collate_fn([gen_data[0]])
    with torch.no_grad():
        vecs = model.Batch2Vecs(batch)
    probs = discriminator(vecs).softmax(dim=1)
    softmax_value = probs[:,domain_ID]
    return softmax_value.item()


def delete_node_and_descendants(node_id,all_node_list):
    delete_list = []
    delete_list.append(node_id)
    node = all_node_list[node_id]
    for child_id in node.children[:]:
        delete_list.append(child_id)
        # node.children.remove(child_id)
        delete_node_and_descendants(child_id,all_node_list)
    return delete_list


class Node:
    def __init__(self,index,parent,children):
        self.index = index
        self.parent = parent
        self.children = children
        self.state = False
        self.visits = 0
        self.score = 0.0
        
    def get_child_node(self,all_node_list):
        child_node_list = []
#         print("children:",self.children)
#         print("len:",len(all_node_list))
        for node_id in self.children:
            if node_id<len(all_node_list):
                child_node_list.append(all_node_list[node_id])
        return child_node_list
    
    def get_parent_node(self,all_node_list):
        parent_node_list = []
#         print("parent:",self.parent)
#         print("len:",len(all_node_list))
        for node_id in self.parent:
            if node_id<len(all_node_list):
                parent_node_list.append(all_node_list[node_id])
        return parent_node_list[0]
    
    def expand_modify(self,temp_dict,model,discriminator):

        prompt_sent = temp_dict["sentence"][self.index]
        target_sent = random.choice(unlabeled_target[random.randint(0,len(unlabeled_target.data_ID)-1)].text)
#         target_sent = "Black teenage boys are not men. They are children. Stop referring to a 17 year old as a man. You are killing children. #ferguson"
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
        print("generate sents:",generate_sents)
        temp_dict["sentence"][self.index] = generate_sents
        temp_dict["text"][self.index] = [s.split(" ") for s in generate_sents]
        self.score = compute_score(temp_dict,model,discriminator)

    def expand_add(self,temp_dict,model,discriminator,all_node_list):

        new_id = int(str(temp_dict["tweet_id"][self.index]) + str(random.randint(0, 9)))
        prompt_sent = temp_dict["sentence"][self.index]
        target_sent = random.choice(unlabeled_target[random.randint(0,len(unlabeled_target.data_ID)-1)].text)
#         target_sent = "Black teenage boys are not men. They are children. Stop referring to a 17 year old as a man. You are killing children. #ferguson"
        try:
            completion = client.chat.completions.create(
            model="qwen-plus", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': 'You are a twitter user. You can generate twitter-form reply to make it like replies in the target domain'},
                {'role': 'user', 'content': 'here is an example in target domain: '+target_sent+' The given twitter is:'+prompt_sent+'Please generate a target-domain-form reply to the given twitter without \'Note\''}],
            )
            generate_sents = completion.choices[0].message.content
        except:
            generate_sents = ""
        
        print("generate_sent:",generate_sents)
        if len(generate_sents) > 0:
            start_date = datetime.datetime(2022,1,1)
            end_date = datetime.datetime(2022,12,31)
            random_date = start_date+random.random()*(end_date-start_date)
            time_str = random_date.strftime("%Y-%m-%d %H:%M:%S")
            d = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            t = d.timetuple()  
            create_time = int(time.mktime(t))
            data_len = len(temp_dict["text"])
            temp_dict["text"].append([s.split(" ") for s in generate_sents])
            if len(temp_dict["tweet_id"]) > data_len:
                temp_dict["sentence"][data_len] = generate_sents
                temp_dict["tweet_id"][data_len] = new_id
                temp_dict["reply_to"][data_len] = temp_dict["tweet_id"][self.index]
                temp_dict["created_at"][data_len]= create_time
            else:
                temp_dict["sentence"].append(generate_sents)
                temp_dict["tweet_id"].append(new_id)
                temp_dict["reply_to"].append(temp_dict["tweet_id"][self.index])
                temp_dict["created_at"].append(create_time)
            

        self.score = compute_score(temp_dict,model,discriminator)

    def expand_delete(self,temp_dict,model,discriminator,all_node_list):
        delete_list = []
        if len(self.parent) > 0:
            delete_list = delete_node_and_descendants(self.index,all_node_list)
#             print("delete_list:", delete_list)
            temp_sentence = [sentence for i,sentence in enumerate(temp_dict["sentence"]) if i not in delete_list]
            temp_text = [text for i,text in enumerate(temp_dict["text"]) if i not in delete_list]
            temp_id = [t_id for i,t_id in enumerate(temp_dict["tweet_id"]) if i not in delete_list]
            temp_reply = [reply for i,reply in enumerate(temp_dict["reply_to"]) if i not in delete_list]
            temp_creat = [creat for i,creat in enumerate(temp_dict["created_at"]) if i not in delete_list]
            temp_dict["sentence"] = temp_sentence
            temp_dict["text"] = temp_text
            temp_dict["tweet_id"] = temp_id
            temp_dict["reply_to"] = temp_reply
            temp_dict["created_at"] = temp_creat
            self.score = compute_score(temp_dict,model,discriminator)
    
        else:
            print("cannot delete root node!")
        
        return delete_list
            
       

    def select(self,all_node_list):
        best_value = float('-inf')
#         print("child:",self.children)
        child_node_list = self.get_child_node(all_node_list)
        best_child = child_node_list[0]
        for child in child_node_list:
            if child.visits > 0:
                value = child.score/child.visits + 2*math.sqrt(2*math.log(self.visits)/child.visits)
            else:
                value = float('inf')
            if value > best_value:
                best_value = value
                best_child = child
        return best_child

    def backpropagate(self,all_node_list):
        self.visits += 1
        if len(self.parent) > 0:
            parent = self.get_parent_node(all_node_list)
            parent.visits += 1
            parent.score += self.score

            while len(parent.parent) != 0:
                parent = parent.get_parent_node(all_node_list)
                parent.visits += 1
                parent.score += self.score
                
class PseudoDataset(MetaMCMCDataset):
    expand_idxs:List=None # indicating which instances will be used to expand the meta validation set
    expand_logits:List=None # confidence on the expanded examples
    valid_indexs:List=None # some pseudo instances will be used to expand the meta validation set, we use a list
                            # 'valid_indexs' to mark which instances are preserved for constrcuting meta training set.
    logits:torch.Tensor=None # confidence on all unlabeled samples
    instance_weights:torch.Tensor=None

    def update_indexs(self, new_indexs:List):
        if self.expand_idxs is None:
            self.valid_indexs = new_indexs
        else:
            self.valid_indexs = [idx for idx in new_indexs if not idx in self.expand_idxs]

    def reinit_indexs(self):
        new_idxs = [*range(len(self._label))]
        return self.update_indexs(new_idxs)
                
def PseudoLabeling(model, dataset:PseudoDataset, temperature = 1.0):
    TD_graphs = [copy.deepcopy(dataset[idxs].g_TD).add_self_loop() for idxs in range(len(dataset))]
    BU_graphs = [copy.deepcopy(dataset[idxs].g_BU).add_self_loop() for idxs in range(len(dataset))]
    texts = [["".join(dataset[idxs]["text"][j]) for j in range(dataset[idxs].data_len)] for idxs in range(len(dataset))]

    num_nodes = [g.num_nodes() for g in TD_graphs]
    big_g_TD = dgl.batch(TD_graphs)
    big_g_BU = dgl.batch(BU_graphs)
    A_TD = big_g_TD.adjacency_matrix().to_dense().to(torch.device('cuda'))
    A_BU = big_g_BU.adjacency_matrix().to_dense().to(torch.device('cuda'))
    with torch.no_grad():
        logits = model.predict(
            (texts, num_nodes, A_TD, A_BU),
            temperature=temperature
        )
        confidence, label_idx = torch.max(logits, dim=1) #confidence保存最大概率，label_idx为该概率对应的类别
        print("confidence:",confidence)
        print("label_idx:",label_idx)
        class_num = 5
        eye = torch.eye(class_num, device=torch.device('cuda')) #创建一个形状为 (self.class_num, self.class_num) 的单位矩阵
        weak_label =eye[label_idx].cpu().tolist() #将最高置信度的类别转化为one-hot形式作为伪标签
        dataset.initLabel(weak_label)
        dataset.initConfidence(confidence.cpu().tolist())
        
def ComputeDomainConfidence(discriminator,model,dataset):
    batch = source_domain.collate_fn(dataset)
    with torch.no_grad():
        vecs = model.Batch2Vecs(batch)
    probs = discriminator(vecs).softmax(dim=1)
    softmax_value = probs[:,domain_ID]
    print(probs)
    print(softmax_value)
        


def mcts(root_node, all_node_list, iterations, temp_dict, model, discriminator):
#     best_score = compute_loss(temp_dict, model, discriminator)
#     print("init score:",best_score)
    print("before:")
    for node in all_node_list:
        print("index:",node.index)
        print("children:",node.children)
        print("parent:",node.parent)
    
    for i in range(iterations):
        print("step:",i)
        origin_dict = copy.deepcopy(temp_dict)
        origin_all_node_list = copy.deepcopy(all_node_list)
        explore = []
        node = root_node

        while len(node.children) > 0 and node.state == True:
            node = node.select(all_node_list)
            explore.append(node.index)
        print("explore:",explore)
        action = random.choice([0,1,2])
        if action == 0:
            print("modify!")
            node.expand_modify(temp_dict,model,discriminator)
            
            node.state = True
            node.backpropagate(all_node_list)
#             score = compute_loss(temp_dict, model, discriminator)
#             print("score:",score)

#             if score < best_score:
#                 best_score = score
#                 print("modify saved")
#             else:
#                 temp_dict = origin_dict
        elif action == 1:
            print("add!")
            node.expand_add(temp_dict,model,discriminator,all_node_list)
            node.state = True
            node.backpropagate(all_node_list)
#             score = compute_loss(temp_dict, model, discriminator)
#             print("score:",score)
            
            new_node = Node(len(temp_dict["text"])-1,[node.index],[])
            all_node_list.append(new_node)
            all_node_list[node.index].children.append(new_node.index)

#             if score < best_score:
#                 best_score = score
#                 print("modify saved")
#                 new_node = Node(len(temp_dict["text"])-1,[node.index],[])
#                 all_node_list.append(new_node)
#                 all_node_list[node.index].children.append(len(temp_dict["text"])-1)
                

#             else:
#                 temp_dict = origin_dict   

            
            
        else:
            print("delete!")
            delete_list = node.expand_delete(temp_dict,model,discriminator,all_node_list)
            if len(delete_list)>0:
                node.state = True
                node.backpropagate(all_node_list)
#             score = compute_loss(temp_dict, model, discriminator)
#             print("score:",score)
#                 print("delete index:",node.index)
#                 print("parent:",node.parent)

                parent_node = node.get_parent_node(all_node_list)
#                 print("parent:",parent_node.index)
#                 print("before parent children:",all_node_list[parent_node.index].children)
                if node.index in parent_node.children:
                    all_node_list[parent_node.index].children.remove(node.index)
#                 print("after parent children:",all_node_list[parent_node.index].children)
                    
                new_node_list = [n for n in all_node_list if n.index not in delete_list]
                all_node_list = new_node_list
                change_dict = {}
                for i,no in enumerate(all_node_list):
                    no_index = no.index
                    if no_index != i:
                        change_dict[no.index] = i
                        no.index = i
#                 print("change_dict:",change_dict)
                for nod in all_node_list:
                    new_children = []
                    for child_id in nod.children:
                        if child_id in change_dict.keys():
                            new_children.append(change_dict[child_id])
                        else:
                            new_children.append(child_id)
                    new_parent = []
                    for parent_id in nod.parent:
                        if parent_id in change_dict.keys():
                            new_parent.append(change_dict[parent_id])
                        else:
                            new_parent.append(parent_id)
      
                    all_node_list[nod.index].children = new_children
                    all_node_list[nod.index].parent = new_parent
                    

#             if score < best_score:
#                 best_score = score
#                 print("modify saved")
#                 g_TD, g_BU = construct_graph(temp_dict)

#                 tree = g_TD
#                 nodes = tree.nodes()
#                 s,d = tree.remove_self_loop().edges()
#                 nodes_list = nodes.cpu().tolist()
#                 source = s.cpu().tolist()
#                 des = d.cpu().tolist()
#                 child_lists = [[des[k] for k, nodes_id in enumerate(source) if nodes_id == n] for n in nodes_list]
#                 parent_lists = [[source[j] for j, nodes_id in enumerate(des) if nodes_id == n] for n in nodes_list]
#                 all_node_list = [Node(s,parent_lists[s],child_lists[s]) for s,n in enumerate(nodes_list)]
                       
#             else:
#                 temp_dict = origin_dict

        print("after:")
        for nodes in all_node_list:
            print("index:",nodes.index)
            print("children:",nodes.children)
            print("parent:",nodes.parent)

        
        confidence = compute_confidence(temp_dict, model, discriminator) 
        print("confidence:",confidence)
        if confidence > 0.6:
            return temp_dict
    return temp_dict
            
        
        

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
    data_dir1 = r"../../autodl-tmp/data/pheme-rnr-dataset/"
#     data_dir1 = r"../../autodl-tmp/data/test/"
    
    os.environ['CUDA_VISIBLE_DEVICES'] = "0" 

    events_list = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting','sydneysiege']
    # for domain_ID in range(5):

    fewShotCnt = 0
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
    gen_target = MetaMCMCDataset()
    gen_target.data = {}
#     gen_target = copy.deepcopy(source_domain)

    for i,d_ID in enumerate(source_domain.data_ID[:500]):
        temp_dict = source_domain.data[d_ID]
        temp_dict["text"] = [s.split(" ") for s in temp_dict["sentence"]]
        g_TD, g_BU = construct_graph(temp_dict)

        tree = g_TD
        nodes = tree.nodes()
        s,d = tree.remove_self_loop().edges()
        nodes_list = nodes.cpu().tolist()
        source = s.cpu().tolist()
        des = d.cpu().tolist()
        child_lists = [[des[k] for k, nodes_id in enumerate(source) if nodes_id == n] for n in nodes_list]
#         print("child:",child_lists)
        parent_lists = [[source[j] for j, nodes_id in enumerate(des) if nodes_id == n] for n in nodes_list]
#         print("parent:",parent_lists)
        class_node_list = [Node(s,parent_lists[s],child_lists[s]) for s,n in enumerate(nodes_list)]
        for node in class_node_list:
            if len(node.parent)==0:
                root_node = node
        gen_dict = mcts(root_node, class_node_list, 50, temp_dict, model, discriminator)
        gen_target.data[d_ID] = gen_dict
#         gen_target.data[d_ID]['text'] = [s.split(" ") for s in generate_sent]
#         gen_target.data[d_ID]['topic_label'] = domain_ID
# #         gen_target.data[d_ID]['label'] = source_domain.data[d_ID]['label']
# #         gen_target.data[d_ID]['event'] = source_domain.data[d_ID]['event']
# #         gen_target.data[d_ID]['sentnece'] = generate_sent
# #         gen_target.data[d_ID]['created_at'] = source_domain.data[d_ID]['created_at']
# #         gen_target.data[d_ID]['reply_to'] = source_domain.data[d_ID]['reply_to']
# #         gen_target.data[d_ID]['tweet_id'] = source_domain.data[d_ID]['tweet_id']

    gen_target.dataclear()
    
            
    event_dir = os.path.join(data_dir1,"qwen_gen_from_source",test_event_name)
    print(event_dir)
    gen_target.Caches_Data(event_dir)
    
    PseudoLabeling(model, gen_target)
    
    ComputeDomainConfidence(discriminator,model,gen_target)
        

    
    

    




