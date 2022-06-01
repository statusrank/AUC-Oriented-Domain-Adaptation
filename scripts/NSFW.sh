CUDA_VISIBLE_DEVICES=6 python3 train_ours.py \
    --data NSFW \
    --source drawings_0.01 \
    --target neutral_0.01 \
    --da_method AUC_DA \
    --lr=5e-3 \
    --warm_up_epoch=20 \
    --decay_epoch=1 \
    --trade-off=1.0 \
    --beta1=1e-3 \
    --beta2=1e-2 \
    --lr-decay=0.98 \
    --loss_type=log_loss \
    --epsilon=0.0 \
    --data_save=save_best


CUDA_VISIBLE_DEVICES=4 python3 train_ours.py \
    --data NSFW \
    --source neutral_0.01 \
    --target drawings_0.01 \
    --da_method AUC_DA \
    --lr=5e-3 \
    --warm_up_epoch=8 \
    --decay_epoch=1 \
    --trade-off=1.0 \
    --beta1=1e-2 \
    --beta2=1e-2 \
    --lr-decay=0.98 \
    --loss_type=log_loss \
    --epsilon=1e-4 \
    --data_save=save_best

