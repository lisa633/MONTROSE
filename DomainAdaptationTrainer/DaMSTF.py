import fitlog, torch, random
import torch.nn as nn, torch.nn.functional as F, numpy as np
from torch.utils.data import Dataset, DataLoader
from TrainingEnv import VirtualModel, CustomDataset, AdversarialModel, BaseTrainer
from DomainAdaptationTrainer.Utils.meta_learning_framework import MetaVirtualModel, MetaLearningFramework
from typing import List, AnyStr
from sklearn.metrics import precision_recall_fscore_support
from prefetch_generator import background
import os


class DomainDiscriminator(VirtualModel):
    def __init__(self, hidden_size, model_device, learningRate, domain_num):
        super(DomainDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(nn.Linear(hidden_size, hidden_size * 2),
                                           nn.ReLU(),
                                           nn.Linear(hidden_size * 2, domain_num)).to(model_device)

        self.device = model_device
        self.learning_rate = learningRate

    def grouped_parameters(self, learning_rate):
        if learning_rate is None:
            learning_rate = self.learning_rate
        return [{
            'params': self.discriminator.parameters(),
            'lr': learning_rate
        }]

    def forward(self, vecs):
        return self.discriminator(vecs)


class DaMSTF_Model(MetaVirtualModel, AdversarialModel):
    pass


class PseudoDataset(CustomDataset):
    expand_idxs: List = None  # indicating which instances will be used to expand the meta validation set
    expand_logits: List = None  # confidence on the expanded examples
    valid_indexs: List = None  # some pseudo instances will be used to expand the meta validation set, we use a list
    # 'valid_indexs' to mark which instances are preserved for constrcuting meta training set.
    logits: torch.Tensor = None  # confidence on all unlabeled samples
    instance_weights: torch.Tensor = None

    def update_indexs(self, new_indexs: List):
        if self.expand_idxs is None:
            self.valid_indexs = new_indexs
        else:
            self.valid_indexs = [idx for idx in new_indexs if not idx in self.expand_idxs]

    def reinit_indexs(self):
        new_idxs = [*range(len(self._label))]
        return self.update_indexs(new_idxs)


@background(max_prefetch=5)
def meta_training_loader(source_domains: List, pseudo_target: PseudoDataset, batch_size=32, source_ratio=0.5):
    indexs_souce = [random.sample(range(len(domain)), len(domain)) for domain in source_domains]
    indexs_target = random.sample(pseudo_target.valid_indexs, len(pseudo_target.valid_indexs))
    bs_source = int(batch_size * source_ratio) // (len(source_domains))
    bs_target = batch_size - bs_source * len(source_domains)
    print(bs_source, bs_target)
    max_iters = max([len(domain) // bs_source for domain in source_domains] + [len(pseudo_target) // bs_target])
    for iteration in range(max_iters):
        start_source = iteration * bs_source
        items_source = [domain[indexs_souce[d_idx][idx % len(domain)]] \
                        for d_idx, domain in enumerate(source_domains) \
                        for idx in range(start_source, start_source + bs_source)]
        start_target = iteration * bs_target
        items_target = [pseudo_target[indexs_target[idx]] for idx in range(start_target, start_target + bs_target)]
        yield pseudo_target.collate_fn(
            items_source + items_target
        )


@background(max_prefetch=5)
def DANN_Dataloader(domain_list: List, batch_size=32):
    indexs_souce = [random.sample(range(len(domain)), len(domain)) for domain in domain_list]
    bs_domain = batch_size // (len(domain_list))
    print("bs_domain",bs_domain)
    print("(len(domain_list)",(len(domain_list)))
    max_iters = max([len(domain) // bs_domain for domain in domain_list])
    for iteration in range(max_iters):
        start = iteration * bs_domain
        items = [domain[indexs_souce[d_idx][idx % len(domain)]] \
                 for d_idx, domain in enumerate(domain_list) \
                 for idx in range(start, start + bs_domain)]
        yield domain_list[0].collate_fn(items)


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
            batch1 = labeledTarget.collate_raw_batch(items1 + items2)
            batch2 = unlabeledTarget.collate_raw_batch(
                random.sample(items2, batchSize - bs1 - bs3) + items1 + items3
            )
            yield batch1, batch2

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
        return Generator3(labeledSource, unlabeledTarget, batchSize)

class DaMSTF_Trainer(MetaLearningFramework, BaseTrainer):
    ### Domain Adapatation Settings
    model: DaMSTF_Model = None
    domain_discriminator: nn.Module = None
    labeled_target: CustomDataset = None
    class_num = 2

    ### General Training Parameters ###
    lr4model = 5e-5  # learning rate for updating the model's parameters
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    max_batch_size = 32
    grad_accum_cnt = 1
    valid_every = 100

    ### Meta-Learning Parameters ###
    train_mix_ratio = 0.5
    train_mix_annealing = False
    valid_mix_ratio = 0.5
    valid_mix_annealing = False
    num_of_meta_valid_samples = 100
    lr4weight = 0.1  # learning rate for updating the hyperparameters
    meta_step = 10  # update the hyperparameters with 'meta_step' steps

    ### Meta-Constructor Parameters ###
    topK = 0.1  # A dynamic threshold for determining which pseudo instances will be taken to expand
    # the meta validation set. In detail, the instances whose maximum prediction entropy is
    # greater than 'confidence_th' will be selected.
    expand_size = 100

    ### DaNN Parameters ###
    gStep = 1
    dStep = 10
    G_lr = None  # optional
    D_lr = None  # optional

    ### self-training parameters
    threshold = 0.7

    ### Logging Parameters ###
    model_file = "./tmp.pkl"
    marker: AnyStr = "DaMSTF"  # mark the current trainer to output the customized results

    def __init__(self, random_seed, log_dir, suffix, model_file, class_num, temperature=1.0,
                 learning_rate=5e-3, batch_size=32, Lambda=0.1,discriminator=None):
        super(DaMSTF_Trainer, self).__init__()
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
        self.domain_discriminator=discriminator

    def NNInitialization_DANN(self, source_domains: List, unlabeled_target: PseudoDataset, max_epoch=1,
                              max_iterate=10000):
        D_optim = self.domain_discriminator.obtain_optim(self.lr4model if self.D_lr is None else self.D_lr)
        G_optim = self.model.obtain_optim(self.lr4model if self.G_lr is None else self.G_lr)
        D_optim.zero_grad()
        G_optim.zero_grad()
        step = 0
        for epoch in range(max_epoch):
            maxIters, trainLoader = DataIter(source_domains, unlabeled_target, None, self.batch_size)
            # train_loader = DANN_Dataloader(source_domains + [unlabeled_target], batch_size=self.max_batch_size)
            for batch in trainLoader:
                step += 1
                ELoss, EAcc = self.model.advLossAndAcc(self.domain_discriminator, batch, labels=batch[-1])
                ELoss.backward()
                if step % self.grad_accum_cnt == 0:
                    D_optim.step()
                    G_optim.step()
                    G_optim.zero_grad()
                    D_optim.zero_grad()
                print('#### DANN Step %3d [%3d, %3d] ####, loss = %6.8f, Acc = %6.8f' % (
                    step, epoch, max_epoch, ELoss.data.item(), EAcc
                ))
                torch.cuda.empty_cache()
                if step > max_iterate:
                    return

    def NNInitialization(self, source_domains: List, unlabeled_target: PseudoDataset, max_epoch=1,
                         max_iterate=10000):
        D_optim = self.domain_discriminator.obtain_optim(self.lr4model if self.D_lr is None else self.D_lr)
        G_optim = self.model.obtain_optim(self.lr4model if self.G_lr is None else self.G_lr)
        step = 0
        for epoch in range(max_epoch):
            maxIters, trainLoader = DataIter(source_domains, unlabeled_target, None, self.batch_size)
            # train_loader = DANN_Dataloader(source_domains + [unlabeled_target], batch_size=self.max_batch_size)
            for step,batch in enumerate(trainLoader()):
                # step += 1
                for d_idx in range(self.dStep):
                    D_optim.zero_grad()
                    ELoss, EAcc = self.model.advLossAndAcc(self.domain_discriminator, batch, labels=batch[-3])
                    ELoss.backward()
                    D_optim.step()
                    print('#### D Step (%3d , %3d) %3d [%3d, %3d] ####, loss = %6.8f, Acc = %6.8f' % (
                        d_idx, self.dStep, step, epoch, max_epoch, ELoss.data.item(), EAcc
                    ))

                for g_idx in range(self.gStep):
                    G_optim.zero_grad()
                    ELoss, EAcc = self.model.advLossAndAcc(self.domain_discriminator, batch, labels=batch[-3])
                    ELoss.backward()
                    print('#### G Step (%3d , %3d) %3d [%3d, %3d] ####, loss = %6.8f, Acc = %6.8f' % (
                        g_idx, self.gStep, step, epoch, max_epoch, ELoss.data.item(), EAcc
                    ))
                    G_optim.step()
                    torch.cuda.empty_cache()
                if step > max_iterate:
                    return

    def MetaConstruction(self, pseudo_set: PseudoDataset):
        logits = pseudo_set.logits
        assert logits.dim() == 2
        # ******* sample new idxs **************************#
        valid_logits = logits[pseudo_set.valid_indexs]
        topK = int(len(valid_logits) * self.topK)
        entropy = (valid_logits * (valid_logits.log().neg())).sum(dim=1)
        V, I = entropy.neg().topk(topK)
        idxs = torch.tensor(pseudo_set.valid_indexs)[I.cpu()].tolist()
        if pseudo_set.expand_idxs is None:
            pseudo_set.expand_idxs = idxs
        else:
            # ******** remove out some expand idxs**************#
            th = V.neg()[-1].data.item()
            expand_logits = logits[pseudo_set.expand_idxs]
            expand_entropy = (expand_logits * (expand_logits.log().neg())).sum(dim=1)
            R = torch.arange(len(expand_logits), device=expand_logits.device)[
                expand_entropy.__gt__(th)
            ]
            remove = torch.tensor(pseudo_set.expand_idxs)[R.cpu()].tolist()
            new_expand_idxs = [idx for idx in pseudo_set.expand_idxs if idx not in remove]
            pseudo_set.expand_idxs = new_expand_idxs + idxs
        # ******** Integration ************#
        pseudo_set.valid_indexs = [idx for idx in pseudo_set.valid_indexs if not (idx in pseudo_set.expand_idxs)]
        # some remove idxs may have the confidence higher than the threshold, but the numbel of such type instances is
        # small. Thus, directly discarding these 'remove' idxs will not affect DaMSTF's performance

    def sampling_meta_valid_batch(self, pseudo_target):
        sampling_size = self.num_of_meta_valid_samples
        items = []
        if self.labeled_target is not None:
            with torch.no_grad():
                logits = self.model.dataset_logits(self.labeled_target)
            loss = F.nll_loss(logits.log(),
                              self.labeled_target.labelTensor(device=logits.device).argmax(dim=1),
                              reduction='none')
            lt_size = min(int(sampling_size * self.valid_mix_ratio), len(self.labeled_target))
            sampling_size = sampling_size - lt_size
            items = [self.labeled_target[m_idx.data.item()] for m_idx in torch.multinomial(
                loss, lt_size, replacement=False, generator=None
            )]
        expand_idxs = random.sample(pseudo_target.expand_idxs, min(len(pseudo_target.expand_idxs), sampling_size))
        expand_items = [pseudo_target[idx] for idx in expand_idxs]
        items.extend(expand_items)

        if sampling_size <= self.max_batch_size:
            return pseudo_target.collate_fn(items)
        else:
            return [
                pseudo_target.collate_fn(items[start:min(start + self.max_batch_size, len(items))])
                for start in range(0, len(items), self.max_batch_size)
            ]

    def batch_metalearning(self, model: MetaVirtualModel, optim, valid_batch, train_batch_list: List,valid_set,epoch,max_epoch,step,source_domains):
        # init_state_dict = model.paras_dict()
        init_state_dict = model.state_dict()
        # optim = model.obtain_optim(self.lr4model)
        # optim.zero_grad()
        # assert len(train_batch_list) == self.grad_accum_cnt
        sum_loss, sum_acc = 0.0, 0.0
        train_step = 0
        best_valid_acc, counter = 0.0, 0
        sum_loss, sum_acc = 0.0, 0.0
        valid_every=20
        loss_lists = []
        weights_list, idxs_list = [], []
        for train_batch in train_batch_list:
            counter += 1
            weights, idxs = train_batch[-2].float(), train_batch[-1]
            idxs_list.append(idxs)
            for mstep in range(self.meta_step):
                print(f"mstep {mstep} | {self.meta_step}")
                loss_list = model.lossList(train_batch, temperature=1.0, label_weight=None)
                u = weights.sigmoid()
                loss = (u * loss_list).sum()
                loss.backward()
                optim.step()  # theta hat
                optim.zero_grad()
                rst = self.darts_approximation(model, valid_batch, train_batch, temperature=1.0, label_weight=None)
                # u_grads = -1 * rst.weight_grad.float()
                u_grads = 1 * rst.weight_grad.float()
                w_grads = u_grads * u * (1 - u)
                normed_wGrads = -1 * (w_grads / (w_grads.norm(2) + 1e-10))
                weights = weights - self.lr4weight * normed_wGrads
                # model.load_paras_dict(init_state_dict)
                model.load_state_dict(init_state_dict)
            weights_list.append(weights)
            loss_list, acc = model.lossAndAcc(train_batch, temperature=1.0, label_weight=None)
            normed_weights = weights / (weights.sum())
            loss = (normed_weights * loss_list).sum()
            loss.backward()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            sum_loss += loss.data.item()
            sum_acc += acc
            train_step += 1
            if step % self.grad_accum_cnt == 0:
                optim.step()
                optim.zero_grad()
                mean_loss, mean_acc = sum_loss / self.grad_accum_cnt, sum_acc / self.grad_accum_cnt
                loss_lists.append(mean_loss)
                if len(loss_lists) > 20:
                    loss_lists.pop(0)
                print(
                    '%6d  [%3d | %3d], loss/acc = %6.8f/%6.7f, loss_mean/std=%6.7f/%6.7f, best_valid_acc:%6.7f ' % (
                        counter, epoch, max_epoch,
                        mean_loss, mean_acc, np.mean(loss_lists), np.std(loss_lists),
                        best_valid_acc))
                sum_loss, sum_acc = 0.0, 0.0
            # if counter % (valid_every * self.grad_accum_cnt) == 0 and valid_set is not None:
            if step % (self.valid_every * self.grad_accum_cnt) == 0 and valid_set is not None:
                v_acc = self.model.valid(valid_set, valid_set.labelTensor(), suffix=f"{self.marker}_valid")
                if v_acc > best_valid_acc:
                    best_valid_acc = v_acc
                    model.save_model(self.model_file)
                    # torch.save(self.model.state_dict(), f'{self.model_file}')
                    # self.model.valid(test_set, test_label, suffix=f"{self.marker}_valid_test")
            old_weights = source_domains.instance_weights.clone()
            source_domains.instance_weights[torch.cat(idxs_list)] = torch.squeeze(torch.cat(weights_list))
            self.print_divergence(source_domains.instance_weights,
                                    old_weights)

    def Merge_data(self,data_set1, data_set2):
        new_data = data_set1.__class__()
        new_data.data = dict(data_set1.data, **data_set2.data)
        new_data.data_ID = np.concatenate([np.array(data_set1.data_ID),
                                        np.array(data_set2.data_ID)]).tolist()
        new_data.data_len = np.concatenate([np.array(data_set1.data_len),
                                            np.array(data_set2.data_len)]).tolist()
        new_data.data_y = np.concatenate([np.array(data_set1.data_y),
                                        np.array(data_set2.data_y)]).tolist()
        new_data.instance_weights = torch.cat([data_set1.instance_weights,
                                            data_set2.instance_weights], dim=0)
        return new_data
    
    def dataset2dataloader(self, dataset, shuffle=False, batch_size=32):
        if shuffle:
            idxs = random.sample(range(len(dataset)), len(dataset)) * 2
        else:
            idxs = [*range(len(dataset))] * 2

        @background(max_prefetch=5)
        def dataloader():
            for start in range(0, len(dataset), batch_size):
                batch_idxs = idxs[start:start + batch_size]
                items = [dataset[index] for index in batch_idxs]
                yield dataset.collate_fn(items)

        return dataloader()
            



    def ModelRetraining(self, source_domains: List, pseudo_target: PseudoDataset,
                        valid_set: CustomDataset, test_set: CustomDataset, test_label: torch.Tensor,
                        max_epoch=100, update_meta_set_every=20):
        # model_optim = torch.optim.Adam([
        #     {'params': self.model.parameters(), 'lr': self.lr4model}
        # ])
        optim = self.model.obtain_optim(self.lr4model * 1.0 / self.grad_accum_cnt)
        optim.zero_grad()
        step = 0
        v_batch = None
        best_acc = 0.0
        sum_loss, sum_acc = 0.0, 0.0
        train_step = 0
        best_valid_acc, counter = 0.0, 0
        sum_loss, sum_acc = 0.0, 0.0
        loss_lists = []
        data_list=[]
        data_list.append(source_domains)
        data_list.append(pseudo_target)
        from functools import reduce
        final_set = reduce(self.Merge_data, data_list)
        for epoch in range(max_epoch):
        
            trainLoader=self.dataset2dataloader(final_set,True,self.batch_size)

            # maxIters, trainLoader = DataIter(source_domains, pseudo_target, None, self.batch_size)
            # train_loader = DANN_Dataloader(source_domains + [unlabeled_target], batch_size=self.max_batch_size)
            weights_list, idxs_list = [], []
            step=0
            # for step,train_batch in enumerate(trainLoader()):
            for train_batch in trainLoader:
                if step==0:
                    v_batch = self.sampling_meta_valid_batch(pseudo_target)
                
                counter += 1
                if step % update_meta_set_every == 0:
                    v_batch = self.sampling_meta_valid_batch(pseudo_target)
                init_state_dict = self.model.state_dict()
                weights, idxs = train_batch[-2].float(), train_batch[-1]
                idxs_list.append(idxs)
                
                for mstep in range(self.meta_step):
                    if mstep!=0:
                        self.model.load_state_dict(init_state_dict)
                    print(f"mstep {mstep} | {self.meta_step}")
                    loss_list = self.model.lossList(train_batch, temperature=1.0, label_weight=None)
                    u = weights.sigmoid()
                    loss = (u * loss_list).sum()
                    loss.backward()
                    optim.step()  # theta hat
                    optim.zero_grad()
                    rst = self.darts_approximation(self.model, v_batch, train_batch, temperature=1.0, label_weight=None)
                    # u_grads = -1 * rst.weight_grad.float()
                    u_grads = 1 * rst.weight_grad.float()
                    w_grads = u_grads * u * (1 - u)
                    normed_wGrads = -1 * (w_grads / (w_grads.norm(2) + 1e-10))
                    weights = weights - self.lr4weight * normed_wGrads
                weights_list.append(weights)
                loss_list, acc = self.model.lossAndAcc(train_batch, temperature=1.0, label_weight=None)
                normed_weights = weights / (weights.sum())
                loss = (normed_weights * loss_list).sum()
                loss.backward()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                sum_loss += loss.data.item()
                sum_acc += acc
                train_step += 1
                if counter % self.grad_accum_cnt == 0:
                    optim.step()
                    optim.zero_grad()
                    mean_loss, mean_acc = sum_loss / self.grad_accum_cnt, sum_acc / self.grad_accum_cnt
                    loss_lists.append(mean_loss)
                    if len(loss_lists) > 20:
                        loss_lists.pop(0)
                    print(
                        '%6d  [%3d | %3d], loss/acc = %6.8f/%6.7f, loss_mean/std=%6.7f/%6.7f, best_valid_acc:%6.7f ' % (
                            counter, epoch, max_epoch,
                            mean_loss, mean_acc, np.mean(loss_lists), np.std(loss_lists),
                            best_valid_acc))
                    sum_loss, sum_acc = 0.0, 0.0
                # if counter % (valid_every * self.grad_accum_cnt) == 0 and valid_set is not None:
                if counter % (self.valid_every * self.grad_accum_cnt) == 0 and valid_set is not None:
                    v_acc = self.model.valid(valid_set, valid_set.labelTensor(), suffix=f"{self.marker}_valid")
                    if v_acc > best_valid_acc:
                        best_valid_acc = v_acc
                        self.model.save_model(self.model_file)
                        # torch.save(self.model.state_dict(), f'{self.model_file}')
                    self.model.valid(test_set, test_label, suffix=f"{self.marker}_valid_test")
                old_weights = final_set.instance_weights.clone()
                # print("source_domains.instance_weights:",len(final_set.instance_weights))
                # print("idxs_list",idxs_list)
                # print("idxs_list_len",len(idxs_list))
                final_set.instance_weights[torch.cat(idxs_list)] = torch.squeeze(torch.cat(weights_list))
                self.print_divergence(source_domains.instance_weights,
                                        old_weights)
                step+=1
            # self.model.valid(test_set, test_label, suffix=f"{self.marker}_valid_test")

    def PseudoLabeling(self, data: PseudoDataset, batch_size=20):
        if data.valid_indexs is None:
            data.valid_indexs = [*range(len(data))]
        if data.read_indexs is None:
            data.read_indexs=np.arange(len(data))
        with torch.no_grad():
            pred_tensor = self.model.dataset_logits(data, batch_size=batch_size)
            data.logits = pred_tensor
            weak_label = (pred_tensor > (1.0 / self.class_num)).long().cpu().tolist()
            data.setLabel(weak_label, data.read_indexs)

            valid_indexs = torch.arange(len(pred_tensor), device=pred_tensor.device)[
                pred_tensor.max(dim=1)[0].__gt__(self.threshold)
            ].cpu().tolist()
            data.update_indexs(valid_indexs)

    def Training(self, model, source_domains: List, unlabeled_target: PseudoDataset, valid_set: CustomDataset,
                 test_set: CustomDataset,test_label=None,
                 max_iterate=100, labeled_target=None):
        self.labeled_target=labeled_target
        self.model = model
        if self.valid_mix_annealing or self.train_mix_annealing:
            omega = np.pi / max_iterate
            init_valid_mix_a = (self.valid_mix_ratio - 0.5) / 2.0
            init_train_mix_a = (self.train_mix_ratio - 0.5) / 2.0

        for iterate in range(max_iterate):
            print("iterate",iterate)
            self.PseudoLabeling(unlabeled_target)
            self.MetaConstruction(unlabeled_target)
            self.NNInitialization(source_domains, unlabeled_target, max_epoch=1, max_iterate=100)
            self.ModelRetraining(source_domains, unlabeled_target, valid_set, test_set,
                                 test_label.argmax(dim=1), max_epoch=10)
            self.model.load_model(self.model_file)
            self.model.valid(test_set, test_label, suffix=f"{self.marker}_testfinal")
            if self.valid_mix_annealing:
                self.valid_mix_ratio = 0.7 + init_valid_mix_a * np.cos((iterate + 1) * omega)  # [0.5, 0.9]
            if self.train_mix_annealing:
                self.train_mix_ratio = 0.7 + init_train_mix_a * np.cos((iterate + 1) * omega)  # [0.5, 0.9]


class ABL1(DaMSTF_Trainer):
    def Training(self, source_domains: List, unlabeled_target: PseudoDataset, valid_set: CustomDataset,
                 test_set: CustomDataset,
                 max_iterate=100):
        for iterate in range(max_iterate):
            self.PseudoLabeling(unlabeled_target)
            self.MetaConstruction(unlabeled_target)
            self.NNInitialization(source_domains, unlabeled_target, max_epoch=1, max_iterate=400)
            self.ModelRetraining(source_domains, unlabeled_target, valid_set, test_set,
                                 test_set.labelTensor(), max_epoch=5)
            self.model.valid(test_set, test_set.labelTensor(), suffix=f"{self.marker}_test")


class ABL2(DaMSTF_Trainer):
    def Training(self, source_domains: List, unlabeled_target: PseudoDataset, valid_set: CustomDataset,
                 test_set: CustomDataset,
                 max_iterate=100):
        for iterate in range(max_iterate):
            self.PseudoLabeling(unlabeled_target)
            self.MetaConstruction(unlabeled_target)
            self.NNInitialization(source_domains, unlabeled_target, max_epoch=1, max_iterate=400)
            self.ModelRetraining(source_domains, unlabeled_target, valid_set, test_set,
                                 test_set.labelTensor(), max_epoch=5)
            self.model.valid(test_set, test_set.labelTensor(), suffix=f"{self.marker}_test")
