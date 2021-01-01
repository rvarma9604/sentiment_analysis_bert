#!/bin/bash

set -eu

stage=6

# stage 0 - preprocess data 
# stage 1 - token preparation for BERT
# stage 2 - train model phase 1 for LSTM
# stage 3 - average model parameters
# stage 4 - train model entire model
# stage 5 - average model parameters
# stage 6 - evaluate
# stage 7 - compare

# Load deps
. path.sh
. cmd.sh
. parse_options.sh || exit

# datasets
train_csv=../data/train.csv
test_csv=../data/test.csv


# preprocess data
preprocess_dir=exp/preprocess
if [ $stage -le 0 ]; then
    echo "Processing task performed at $preprocess_dir"
    if [ -d $preprocess_dir ]; then
        echo "$preprocess_dir already exists."
        echo " if you want to retry, delete it."
        exit 1
    fi
    work=$preprocess_dir/.work
    mkdir -p $work
    $process_cmd $work/process.log \
        python -u ../driver/preprocess.py $train_csv $test_csv \
            $preprocess_dir \
            --stage 0 --rem_stop True \
        || exit 1
fi

# tokenize data
token_dir=exp/tokens
stop_word=
if [ $stage -le 1 ]; then
    echo "Tokenizing task performed at $token_dir"
    if [ -d $token_dir ]; then
        echo "$token_dir already exists."
        echo " if you want to retry, delete it."
        exit 1
    fi
    work=$token_dir/.work
    mkdir -p $work
    $process_cmd $work/tokens.log \
        python -u ../driver/bert_tokens.py \
            $preprocess_dir/${stop_word}preprocessed_train.csv \
            $preprocess_dir/${stop_word}preprocessed_test.csv \
            $token_dir \
            --stage 0 --max_len 64 \
            || exit 1

    for max_len in 128 256 512; do
        $process_cmd $work/tokens.log \
            python -u ../driver/bert_tokens.py \
                $preprocess_dir/${stop_word}preprocessed_train.csv \
                $preprocess_dir/${stop_word}preprocessed_test.csv \
                $token_dir \
                --stage 2 --max_len $max_len \
                || exit 1
    done
fi

# train LSTMs
train_phase1_dir=exp/train_phase1
if [ $stage -le 2 ]; then
#    echo "Training Phase 1 at $train_phase1_dir"
#    if [ -d $train_phase1_dir ]; then
#        echo "$train_phase1_dir already exists."
#        echo " if you want to retry, delete it."
#        exit 1
#    fi
    work=$train_phase1_dir/.work
    mkdir -p $work
    $train_cmd $work/train.log  \
        python -u ../models/local_train_phase1.py \
            $token_dir/train_pad_64_tokens.pkl \
            $token_dir/train_att_64_tokens.pkl \
            $preprocess_dir/train_labels.pkl \
            $token_dir/test_pad_64_tokens.pkl \
            $token_dir/test_att_64_tokens.pkl \
            $preprocess_dir/test_labels.pkl \
            $train_phase1_dir \
            --epochs 100 --batch_size 32 --max_len 64 \
            || exit 1
fi

average_start=91
average_end=100
average_id_1=model_${average_start}-${average_end}.pt
if [ $stage -le 3 ]; then
    models=`eval echo $train_phase1_dir/snapshot-{$average_start..$average_end}.pt`
    work=$train_phase1_dir/.work
    mkdir -p $work
    $process_cmd $work/average.log \
        python -u ../models/model_averaging.py \
            $average_id_1 \
            $train_phase1_dir \
            $models \
            || exit 1
fi

# train entire model
train_phase2_dir=exp/train_phase2
if [ $stage -le 4 ]; then
#    echo "Training Phase 2 at $train_phase2_dir"
#    if [ -d $train_phase2_dir ]; then
#        echo "$train_phase2_dir already exists."
#        echo " if you want to retry, delete it."
#        exit 1
#    fi
    work=$ train_phase2_dir/.work
    mkdir -p $work
    $train_cmd $work/train.log  \
        python -u ../models/local_train_phase2.py \
            $token_dir/train_pad_64_tokens.pkl \
            $token_dir/train_att_64_tokens.pkl \
            $preprocess_dir/train_labels.pkl \
            $token_dir/test_pad_64_tokens.pkl \
            $token_dir/test_att_64_tokens.pkl \
            $preprocess_dir/test_labels.pkl \
            $train_phase2_dir \
            --init $train_phase1_dir/$average_id_1 \
            --epochs 3 --batch_size 32 --max_len 64 \
            || exit 1
fi
#exit

# compare
compare_dir=exp/compare
if [ $stage -le 6 ]; then
    work=$compare_dir/.work
    mkdir -p work
    $train_cmd $work/train.log \
        python -u ../models/svm_log_reg.py \
            $preprocess_dir/preprocessed_train.csv \
            $preprocess_dir/preprocessed_test.csv \
            $compare_dir \
            || exit 1
fi
exit
compare_dir=exp/compare
if [ $stage -le 3 ]; then
#    echo "Training Phase 1 at $train_phase1_dir"
#    if [ -d $train_phase1_dir ]; then
#        echo "$train_phase1_dir already exists."
#        echo " if you want to retry, delete it."
#        exit 1
#    fi
    work=$compare_dir/.work
    mkdir -p $work
    $train_cmd $work/train.log  \
        python -u ../models/train_phase1.py \
            $token_dir/train_pad_64_tokens.pkl \
            $token_dir/train_att_64_tokens.pkl \
            $preprocess_dir/train_labels.pkl \
            $token_dir/test_pad_64_tokens.pkl \
            $token_dir/test_att_64_tokens.pkl \
            $preprocess_dir/test_labels.pkl \
            $compare_dir \
            --epochs 3 --batch_size 32 --max_len 64 \
            || exit 1
fi
exit


average_start=15
average_end=25
if [ $stage -le 3 ]; then
    models=`eval echo $train_phase1_dir/snapshot-{$average_start..$average_end}.pt`
    work=$train_phase1_dir/.work
    mkdir -p $work
    $process_cmd $work/average.log \
        python -u ../models/model_averaging.py \
            model_${average_start}-${average_end}.pt \
            $train_phase1_dir \
            $models \
            || exit 1
fi

# train the entire model
train_phase2_dir=exp/train_phase2
if [ $stage -le 4 ]; then
#    echo "Training Phase 1 at $train_phase1_dir"
#    if [ -d $train_phase1_dir ]; then
#        echo "$train_phase1_dir already exists."
#        echo " if you want to retry, delete it."
#        exit 1
#    fi
    work=$train_phase2_dir/.work
    mkdir -p $work
    $train_cmd $work/train.log  \
        python -u ../models/train_phase2.py \
            $token_dir/train_pad_64_tokens.pkl \
            $token_dir/train_att_64_tokens.pkl \
            $preprocess_dir/train_labels.pkl \
            $token_dir/test_pad_64_tokens.pkl \
            $token_dir/test_att_64_tokens.pkl \
            $preprocess_dir/test_labels.pkl \
            $train_phase2_dir \
            $train_phase1_dir/model_${average_start}-${average_end}.pt \
            --epochs 4 --batch_size 32 --max_len 64 \
            || exit 1
fi
exit


# evaluate model
eval_dir=exp/eval
if [ $stage -le 3 ]; then
    echo "Evaluation results at $eval_dir"
    work=$eval_dir/.work
    mkdir -p $work
    $infer_cmd $work/eval.log \
        python -u ../models/evaluate.py \
            $token_dir/train_pad_64_tokens.pkl \
            $token_dir/train_att_64_tokens.pkl \
            $eval_dir \
            $train_phase1_dir/snapshot-1.pt \
            --name train --batch_size 32
fi
