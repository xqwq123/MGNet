_path=./pi3k_data/all_train.csv --dataset_type=classification  --epochs=20 --num_folds=1 --gpu=0 --seed=13 --split_type=random --save_dir=$save_target_dir --metric=auc

task_names=("delta")
for task_name in "${task_names[@]}"; do
    python train.py \
    --encoder_checkpoint_path=./results/MVGNet_2024-02-29_1/fold_0/model_0/model.pt \
    --val_data_path=./pi3k_data/${task_name}_val.csv \
    --test_data_path=./pi3k_data/${task_name}_test.csv \
    --data_path=./pi3k_data/${task_name}_train.csv --dataset_type=classification --epochs=20 --num_folds=5 --gpu=0 --seed=13 --split_type=random --save_dir=./fine_tuning/${task_name} --metric=auc
done
