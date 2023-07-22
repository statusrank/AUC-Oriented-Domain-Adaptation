CUDA_VISIBLE_DEVICES=7 python3 ours_multi_lable.py \
    --data ChestXray \
    --source  ChestX-ray14_train \
    --target  CheXpert_train \
    --da_method AUC_DA \
    --lr=5e-3 \
    --warm_up_epoch=3 \
    --decay_epoch=1 \
    --beta1=1e-3 \
    --beta2=1e-3 \
    --lr-decay=0.98 \
    --loss_type=log_loss \
    --epsilon=0 \
    --class_thres=0.5 \
    --data_save=save_final

