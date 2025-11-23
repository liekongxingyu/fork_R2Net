#!/bin/bash/
# For Testing
# 2x
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_BIX2.pt --test_only --save_results --chop --save 'R2NET_Set5' --testpath ../LR/LRBI --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'R2NETplus_Set5' --testpath ../LR/LRBI --testset Set5

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_BIX2.pt --test_only --save_results --chop --save 'R2NET_Set14' --testpath ../LR/LRBI --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'R2NETplus_Set14' --testpath ../LR/LRBI --testset Set14

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_BIX2.pt --test_only --save_results --chop --save 'R2NET_B100' --testpath ../LR/LRBI --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'R2NETplus_B100' --testpath ../LR/LRBI --testset BSD100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_BIX2.pt --test_only --save_results --chop --save 'R2NET_Urban100' --testpath ../LR/LRBI --testset Urban100

CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model R2NET --n_feats 64 --pre_train ../trained_models/R2Net_BIX2.pt --test_only --save_results --chop --self_ensemble --save 'R2NETplus_Urban100' --testpath ../LR/LRBI --testset Urban100

