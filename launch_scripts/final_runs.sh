#!/bin/bash

# Runs all experiments listed at the bottom. Each experiment consists of a
# given number of repetitions using a particular name and arguments.
# Each single repetition checks if it was already run or is currently being
# run, creates a lockfile and trains the network. To distribute runs between
# multiple GPUs, run this script multiple times with different
# CUDA_VISIBLE_DEVICES. To distribute runs between multiple hosts, run this
# script multiple times with a shared output directory (via NFS).

here="${0%/*}"
outdir="$here/../checkpoints"

train_seed() {
	name="$1"
	seed="$2"
	fold=
	for arg in "${@:3}"; do
		if [[ "$arg" == "--fold="* ]]; then
			fold="${arg#*=}"
			break
		fi
	done
	if [ -z "$fold" ]; then
		echo "$name, seed=$seed"
		lockfile="$outdir/$name S$seed"
	else
		echo "$name, seed=$seed, fold=$fold"
		lockfile="$outdir/$name S$seed fold$fold"
	fi
	if [ ! -f "$lockfile"*.ckpt ] && [ ! -f "$lockfile.lock" ]; then
		echo "$HOSTNAME: $CUDA_VISIBLE_DEVICES" > "$lockfile.lock"
		python3 "$here"/train.py --logger wandb --name "$name --seed $seed" "${@:3}" && rm "$lockfile.lock" || echo "failed" >> "$lockfile.lock"
	fi
}

train_seeds() {
	name="$1"
	seeds="$2"
	for (( seed=0; seed<$seeds; seed++ )); do
		train_seed "$name" "$seed" "${@:3}"
	done
}

train() {
    name="$1"
    train_seeds "$name" 3 "${@:2}"
}

train_cv() {
	name="$1"
	for (( fold=0; fold<8; fold++)); do
		train_seed "$name" 0 --fold=$fold "${@:2}"
	done
}

# final model
train "final" $input $augments $model $loss $training

# final model with 8-fold CV
train_cv "final-cv" $input $augments $model $loss $training

# final model, no val
train "final-noval" $input $augments $model $loss $training --val_datasets=""

# removing pitch augmentation
train "no-pitch" $input $augments $model $loss $training "--augmentations=tempo(20,4)"

# removing tempo augmentation
train "no-tempo" $input $augments $model $loss $training "--augmentations=pitch(-5,+6)"

# removing span masking
train "no-mask" $input $augments $model $loss $training "--frontend_augmentations="

# using plain BCE
train "plain-bce" $input $augments $model $training "--loss=beat:BCE_pos(0,0);beat:BCE_neg(0,0);downbeat:BCE_pos(0,0);downbeat:BCE_neg(0,0)" --widen_target_mask_loss=0 --compute_pos_weight

# using plain BCE and no pos weights
train "plain-bce-no-posweight" $input $augments $model $training "--loss=beat:BCE_pos(0,0);beat:BCE_neg(0,0);downbeat:BCE_pos(0,0);downbeat:BCE_neg(0,0)" --widen_target_mask_loss=0 --manual_pos_weight=beat:1,downbeat:1

# using standard task heads
train "no-sum-head" $input $augments $model $loss $training --task_heads="*:Linear"

# removing the frontend transformers
train "no-frontend-tf" $input $augments $model $loss $training --extra_frontend="CustomConv2d(1,C,32,4,3,s,4,BN,gelu,C,64,2,3,s,2,BN,gelu,C,128,2,3,s,2,BN,gelu,C,256,2,3,s,2,BN,gelu,L)"

# using the 80bins frontend
train "80bins" $input $augments $model $loss $training --input_enc="wav_mel(80,30,11000,log1pfix,3,bn,1024)" --extra_frontend="CustomConv2d(1,C,64,3,3,MP,3,gelu,D,0.1,C,128,12,1,MP,3,gelu,D,0.1,C,512,3,3,MP,3,gelu)"

# without oversampling
train "no-oversampling" $input $augments $model $loss $training --lenght_based_oversampling_factor=0 --max_epochs=286

# (only use the Hung et al datasets?)
train "only-hung" $input $augments $model $loss $training --train_datasets=hung
train "only-hung-noval" $input $augments $model $loss $training --train_datasets=hung --val_datasets=""


# small model experiments
train "small h128" $input $augments $model $loss $training --n_heads=4 --n_hidden=128 --dim_feedforward=512
train "small h128-noval" $input $augments $model $loss $training --n_heads=4 --n_hidden=128 --dim_feedforward=512 --val_datasets=""