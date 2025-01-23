from Data.BiGCN_Dataloader import MetaMCMCDataset, load_data_all, load_data_twitter15, Merge_data

from functools import reduce
import os

data_dir1 = r"../../autodl-tmp/data/pheme-rnr-dataset/"
events_list = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting','sydneysiege','twitter15','twitter16']
domain_ID = 5
test_event_name = events_list[domain_ID]

data_dirs = [r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/0/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/1/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/2/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/3/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/4/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/5/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/6/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/7/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/8/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/9/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/10/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/11/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/12/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/13/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/14/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/15/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/16/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/17/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/18/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/19/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/20/",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/21/"]


data_list = []
for dirs in data_dirs:
    data_dir = os.path.join(dirs,test_event_name)
    gen_dataset = MetaMCMCDataset()
    gen_dataset.load_data_fast(data_dir)
    gen_dataset.dataclear()
    data_list.append(gen_dataset)

new_unlabeled_target = reduce(Merge_data,data_list)


event_dir = os.path.join(data_dir1,"qwen_gen_from_source",test_event_name)
print(event_dir)
new_unlabeled_target.Caches_Data(event_dir)




