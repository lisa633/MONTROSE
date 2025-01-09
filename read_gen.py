from Data.BiGCN_Dataloader import MetaMCMCDataset, load_data_all, load_data_twitter15, Merge_data

from functools import reduce
import os

data_dirs = [r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/0",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/1",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/2",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/3",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/4",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/5",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/6",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/7",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/8",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/9",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/10",r"../../autodl-tmp/data/pheme-rnr-dataset/qwen_gen_from_source/qwen_gen_from_source/11"]

data_list = []
for data_dir in data_dirs:
    gen_dataset = MetaMCMCDataset()
    gen_dataset.load_data_fast(data_dir)
    gen_dataset.dataclear()
    data_list.append(gen_dataset)

new_unlabeled_target = reduce(Merge_data,data_list)

data_dir1 = r"../../autodl-tmp/data/pheme-rnr-dataset/"
events_list = ['charliehebdo', 'ferguson', 'germanwings-crash', 'ottawashooting','sydneysiege']
domain_ID = 0
test_event_name = events_list[domain_ID]

event_dir = os.path.join(data_dir1,"qwen_gen_from_source",test_event_name)
print(event_dir)
new_unlabeled_target.Caches_Data(event_dir)
