#!/bin/sh
appname=plome_finetune
test_path=./datas/test.txt
init_bert=./datas/pretrained_plome

py_dim=32
multi_task=1
sk_or_py="all"
gpuid=0
keep_prob=0.9
output_dir=./${appname}_output
init_bert_path=$init_bert
max_sen_len=180
batch_size=32
epoch=10
learning_rate=5e-5 #5e-6  #3e-5

echo "multi_task=$multi_task"
echo "appname=$appname"
echo "init_bert=$init_bert"
mkdir $output_dir
python3 csc_inference.py --py_dim $py_dim --gpuid $gpuid --test_path $test_path --output_dir $output_dir --max_sen_len $max_sen_len --batch_size $batch_size --learning_rate $learning_rate --epoch $epoch --keep_prob $keep_prob --init_bert_path $init_bert_path --multi_task $multi_task --sk_or_py $sk_or_py


