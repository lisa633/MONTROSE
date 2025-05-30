# from transformers import TextGenerationPipeline, GPT2Tokenizer, GPT2LMHeadModel, pipeline
import os

import tqdm
import json
import time,datetime

import torch
import copy
import random
from openai import OpenAI
import math

from Data.BiGCN_Dataloader import MetaMCMCDataset

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


data_dir1 = r"../../autodl-tmp/data/pheme-rnr-dataset/"
os.environ['CUDA_VISIBLE_DEVICES'] = "0" 

events_list = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting','sydneysiege']
domain_ID = 3
fewShotCnt = 100
source_events = [os.path.join(data_dir1, dname)
                    for idx, dname in enumerate(events_list) if idx != domain_ID]
target_events = [os.path.join(data_dir1, events_list[domain_ID])]
test_event_name = events_list[domain_ID]

generate_ratio = 0.3

files = []
for event_path in target_events:
    scan_dir(files, event_path)

print(files[0])


target_data = twitter_data_process(files)



target_data_copy = copy.deepcopy(target_data)
print("befor:",len(target_data))

for key,value in target_data.items():
    new_key = int(str(key)+str(random.randint(0, 9)))
    print(new_key)
    reply_num = len(value["sentence"])
    temp_list = [k for k in range(reply_num)]
    temp_idxs = random.sample(temp_list, math.ceil(reply_num*generate_ratio))
    target_data_copy[new_key] = {}
    target_data_copy[new_key]['topic_label'] = target_data[key]["topic_label"]
    target_data_copy[new_key]['label'] = target_data[key]["label"]
    target_data_copy[new_key]['event'] = target_data[key]["event"]
    target_data_copy[new_key]['tweet_id'] = target_data[key]["tweet_id"]
    target_data_copy[new_key]['sentence'] = target_data[key]["sentence"]
    target_data_copy[new_key]['created_at'] = target_data[key]["created_at"]
    target_data_copy[new_key]['reply_to'] = target_data[key]["reply_to"]
                  
    for i in temp_idxs:
        text = value["sentence"][i]
        client = OpenAI(
            api_key="sk-65734579fab943f48234f366e64ad181", 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        try:
            completion = client.chat.completions.create(
            model="qwen-plus", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': 'You are a twitter user. You can generate twitter-form reply when given a twitter'},
                {'role': 'user', 'content': 'here is the twitter:'+text+'Please generate the reply to it as a twitter user without \'Note\''}],
            )
            generate_sents = completion.choices[0].message.content
        except:
            generate_sents = ""
        if len(generate_sents) > 0:
            print(generate_sents)
            gen_id =  int(str(target_data[key]["tweet_id"][i])+str(random.randint(0, 9)))
            start_date = datetime.datetime(2022,1,1)
            end_date = datetime.datetime(2022,12,31)
            random_date = start_date+random.random()*(end_date-start_date)
            time_str = random_date.strftime("%Y-%m-%d %H:%M:%S")
            d = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            t = d.timetuple()  
            create_time = int(time.mktime(t))
    #             new_reply = random.choice(target_data[key]["reply_to"])
            new_reply = target_data[key]["tweet_id"][i]
            target_data_copy[new_key]['tweet_id'].append(gen_id)
            target_data_copy[new_key]["sentence"].append(generate_sents)
            target_data_copy[new_key]['created_at'].append(create_time)
            target_data_copy[new_key]['reply_to'].append(new_reply)
print("after:",len(target_data_copy))           
print("success!")
            
# print(target_data_copy["498254340310966273"])

gen_target = MetaMCMCDataset()

gen_target.data = target_data_copy

gen_target.dataclear()

event_dir = os.path.join(data_dir1,"qwen_gen",test_event_name)
print(event_dir)
gen_target.Caches_Data(event_dir)

