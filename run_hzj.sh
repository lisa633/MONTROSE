#!/usr/bin/env bash
module load cuda/11.6
module load anaconda/2020.11
source activate mcmc
# python HierarchicalTransformer.py > /data/run01/scz3924/hezj/da-mstf/ottawashooting_0323_1.log 2>&1
# python HierarchicalTransformer_acc.py > /data/run01/scz3924/hezj/da-mstf/ottawashooting_0324_fit.log 2>&1
# python HierarchicalTransformer_meta.py > /data/run01/scz3924/hezj/da-mstf/meta_hist0324_800.log 2>&1
python HierarchicalTransformer_meta.py > /data/run01/scz3924/hezj/da-mstf/fit_hist0324_800.log 2>&1
# python HierarchicalTransformer_acc_fergus.py > /data/run01/scz3924/hezj/da-mstf/fergus_0324_fit.log 2>&1
# python HierarchicalTransformer_out15.py > /data/run01/scz3924/hezj/da-mstf/twitter15_0318.log 2>&1
# python HierarchicalTransformer_out16.py > /data/run01/scz3924/hezj/da-mstf/twditter16_0318.log 2>&1
