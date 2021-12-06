# 导入Flask类
from flask import jsonify
from flask import Flask
from flask import request
import tensorflow as tf
import os
from bert_tagging import BertTagging,DataProcessor
import numpy as np
import tokenization
from pinyin_tool import PinyinTool
from collections import namedtuple
import collections
import datetime

InputFeatures = namedtuple('InputFeature', ['input_ids', 'input_mask', 'segment_ids', 'stroke_ids'])


def convert_single_sentence(sentence,max_sen_len,tokenizer,pytool,sktool):
    sentence = list(sentence.strip().replace(' ',''))
    # Account for [CLS] and [SEP] with "- 2"
    if len(sentence) > max_sen_len - 2:
        tokens = sentence[0:(max_sen_len - 2)]
    else:
        tokens = sentence

    _tokens = []
    # _labels = []
    # _lmask = []
    segment_ids = []
    stroke_ids = []
    _tokens.append("[CLS]")
    # _lmask.append(0)
    # _labels.append(labels[0])
    segment_ids.append(0)
    stroke_ids.append(0)
    for token in tokens:
        _tokens.append(token)
        # _labels.append(label)
        # _lmask.append(1)
        segment_ids.append(pytool.get_pinyin_id(token))
        stroke_ids.append(sktool.get_pinyin_id(token))
    input_ids = tokenizer.convert_tokens_to_ids(_tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_sen_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        stroke_ids.append(0)
        # _labels.append(labels[0])
        # _lmask.append(0)

    assert len(input_ids) == max_sen_len
    assert len(input_mask) == max_sen_len
    assert len(segment_ids) == max_sen_len
    assert len(stroke_ids) == max_sen_len

    # label_ids = [label_map.get(l, label_map['UNK']) for l in _labels]

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        stroke_ids=stroke_ids
    )
    return feature
    # return input_ids,input_mask,segment_ids,stroke_ids

def get_py_seq(token_seq,tokenid_pyid):
    ans = []
    for t in list(token_seq):
        pyid = tokenid_pyid.get(t, 1)
        ans.append(pyid)
    ans = np.asarray(ans, dtype=np.int64)
    return ans

def decode_record(record):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "input_ids": tf.FixedLenFeature([180], tf.int64),
        "input_mask": tf.FixedLenFeature([180], tf.int64),
        "segment_ids": tf.FixedLenFeature([180], tf.int64),
        "stroke_ids": tf.FixedLenFeature([180], tf.int64)
    }


    example = tf.parse_single_example(record, name_to_features)

    #int64 to int64
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int64(t)
        example[name] = t
    input_ids = example['input_ids']
    input_mask = example['input_mask']
    segment_ids = example['segment_ids']
    stroke_ids = example['stroke_ids']
    # label_ids = example['label_ids']
    # lmask = example['lmask']
    # py_labels = tf.py_func(_get_py_seq, [label_ids], [tf.int64])

    return input_ids, input_mask, segment_ids, stroke_ids

def data_process(sentences,max_sen_len,vocab_file,batch_size):

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
    label_map = tokenizer.vocab
    label_list = {}
    for key in tokenizer.vocab:
        label_list[tokenizer.vocab[key]] = key

    py_dict_path = './pinyin_data/zi_py.txt'
    py_vocab_path = './pinyin_data/py_vocab.txt'
    sk_dict_path = './stroke_data/zi_sk.txt'
    sk_vocab_path = './stroke_data/sk_vocab.txt'

    pytool = PinyinTool(py_dict_path=py_dict_path, py_vocab_path=py_vocab_path, py_or_sk='py')
    sktool = PinyinTool(py_dict_path=sk_dict_path, py_vocab_path=sk_vocab_path, py_or_sk='sk')

    py_label_list = {v: k for k, v in pytool.vocab.items()}

    tokenid_pyid = {}
    tokenid_skid = {}
    for key in tokenizer.vocab:
        tokenid_pyid[tokenizer.vocab[key]] = pytool.get_pinyin_id(key)
        tokenid_skid[tokenizer.vocab[key]] = sktool.get_pinyin_id(key)

    ###DataProcessor init 结束
    # input_ids,input_masks,segment_ids,stroke_ids = [],[],[],[]
    features_ = []
    #print('****'*10)
    #print(sentences[:10])
    for sentence in sentences:
        feature = convert_single_sentence(sentence,max_sen_len,tokenizer,pytool,sktool)
        create_int_feature = lambda values: tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["stroke_ids"] = create_int_feature(feature.stroke_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        features_.append(tf_example.SerializeToString())
    dataset = tf.data.Dataset.from_tensor_slices(features_)
    dataset = dataset.map(decode_record, num_parallel_calls=10)
    dataset = dataset.batch(batch_size).prefetch(16)
    return dataset,label_list,py_label_list


def inference(sentences,model):

    if type(sentences) is str:
        sentences = [sentences]
    # gpuid = FLAGS.gpuid
    max_sen_len = 180
    # test_file = FLAGS.test_path
    # out_dir = FLAGS.output_dir
    batch_size = 16
    # init_bert_dir = FLAGS.init_bert_path
    vocab_file = './datas/pretrained_plome/vocab.txt'

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # keep_prob = FLAGS.keep_prob
    # print('test_file=', test_file)

    dataset,label_list,py_label_list = data_process(sentences,max_sen_len,vocab_file,batch_size)
    test_num = len(sentences)

    iterator = dataset.make_one_shot_iterator()
    input_ids, input_masks, segment_ids, stroke_ids = iterator.get_next()

    (pred_loss, pred_probs, gold_probs, gold_mask, py_probs, py_one_hot_labels, fusion_prob) = \
        model.create_model(input_ids, input_masks, segment_ids, stroke_ids, lmask = None, labels = None, py_labels = None, is_training=False)

    all_inputs, all_golds, all_preds = [], [], []
    all_fusino_preds = []

    steps = test_num // batch_size if test_num % batch_size == 0 else int(test_num / batch_size) + 1
    for step in range(steps):
        inputs, gmask, fusion_pred = sess.run([input_ids,gold_mask,fusion_prob])

        print('input_ids:',input_ids.shape[-1],'inputs:',inputs.shape,'gmask',gmask.shape,'fusion_pred:',fusion_pred.shape)
        print(type(gmask),type(fusion_pred))
        nums = inputs.shape[0]
        if nums == batch_size:
            #gmask = np.reshape(gmask, (batch_size, max_sen_len))

            #fusion_pred = np.reshape(fusion_pred, (batch_size, max_sen_len, len(label_list)))
            fusion_pred = np.argmax(fusion_pred, axis=2)
        else:
            #gmask = np.reshape(gmask, (nums, max_sen_len))

            #fusion_pred = np.reshape(fusion_pred, (nums,max_sen_len, len(label_list)))
            fusion_pred = np.argmax(fusion_pred, axis=2)


        for k in range(nums):
            for j in range(max_sen_len):
                if gmask[k][j] == 0: continue
                all_inputs.append(inputs[k][j])
                if model.multi_task is True:
                    all_fusino_preds.append(fusion_pred[k][j])

    all_inputs = [label_list[k] for k in all_inputs]

    all_fusino_preds = [label_list[k] for k in all_fusino_preds]

    return all_inputs,all_fusino_preds

def csc_postprepare(inputs,fusion_preds):

    assert len(inputs) == len(fusion_preds)
    raw,csc,tmp1,tmp2 = [],[],[],[]

    for i,(input,pred) in enumerate(zip(inputs,fusion_preds)):
        if input == '[CLS]':
            if tmp1 == []:
                continue
            else:
                raw.append(tmp1)
                csc.append(tmp2)
                tmp1,tmp2 = [],[]
        else:
            tmp1.append(input)
            tmp2.append(pred)

        if i == len(inputs) - 1:
            raw.append(tmp1)
            csc.append(tmp2)

    return raw,csc

bert_config_path = './datas/pretrained_plome/bert_config.json'
pyid2seq = np.load('./plome_finetune_sess_output/pyid2seq.npy')
skid2seq = np.load('./plome_finetune_sess_output/skid2seq.npy')
zi_py_matrix = np.load('./plome_finetune_sess_output/zi_py_matrix.npy')
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
input_ids = tf.placeholder(dtype=tf.int64,shape=[None, 180])
input_mask = tf.placeholder(dtype=tf.int64,shape=[None, 180])
segment_ids = tf.placeholder(dtype=tf.int64,shape=[None, 180])
stroke_ids = tf.placeholder(dtype=tf.int64,shape=[None, 180])
# lmask = tf.placeholder(dtype=tf.int64,shape=[None, 180])
# label_ids = tf.placeholder(dtype=tf.int64,shape=[None, 180])
# py_labels = tf.placeholder(dtype=tf.int64,shape=[None, 180])

model = BertTagging(bert_config_path, num_class=21128, pyid2seq=pyid2seq, skid2seq=skid2seq, py_dim=32, max_sen_len=180, py_or_sk='all',  keep_prob=0.9,
                    zi_py_matrix=zi_py_matrix, multi_task=True)
(pred_loss, pred_probs, gold_probs, gold_mask, py_probs, py_one_hot_labels, fusion_prob) = \
    model.create_model(input_ids, input_mask, segment_ids, stroke_ids, lmask = None, labels = None, py_labels = None, is_training=False)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./plome_finance_finetune_output')

# saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
saver.restore(sess, ckpt.model_checkpoint_path)

'''
checkpoint 转pb
'''
if not os.path.exists('./export_model/1'):

    model_version = 1
    work_dir = './export_model'

    export_path_base = work_dir
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(model_version)))

    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # 定义输入变量
    tensor_input_ids = tf.saved_model.utils.build_tensor_info(input_ids)
    tensor_input_mask = tf.saved_model.utils.build_tensor_info(input_mask)
    tensor_segment_ids = tf.saved_model.utils.build_tensor_info(segment_ids)
    tensor_stroke_ids = tf.saved_model.utils.build_tensor_info(stroke_ids)
    # tensor_lmask = tf.saved_model.utils.build_tensor_info(lmask)
    # tensor_label_ids = tf.saved_model.utils.build_tensor_info(label_ids)
    # tensor_py_labels = tf.saved_model.utils.build_tensor_info(py_labels)

    tensor_gold_masks = tf.saved_model.utils.build_tensor_info(gold_mask)
    tensor_fusion_probs = tf.saved_model.utils.build_tensor_info(fusion_prob)

    # 构建过程
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'input_ids': tensor_input_ids,
                    'input_mask': tensor_input_mask,
                    'segment_ids': tensor_segment_ids,
                    'stroke_ids': tensor_stroke_ids
                    },

            outputs={'gold_masks': tensor_gold_masks,'fusion_probs':tensor_fusion_probs
                     },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict': prediction_signature
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True
    )

    builder.save()

    print('Done exporting!')

graph = tf.get_default_graph()
# 实例化，可视为固定格式
app = Flask(__name__)

# route()方法用于设定路由；类似spring路由配置
@app.route('/helloworld')
def hello_world():
    return 'Hello, World!'

@app.route('/csc_server', methods=['POST','GET'])
def csc_inference():
    with graph.as_default():
        query = request.json
        print(query)
        print(type(query['query']))
        starttime = datetime.datetime.now()
        all_inputs,all_fusino_preds = inference(query['query'],model)
        raw,csc = csc_postprepare(all_inputs,all_fusino_preds)

        raw = [''.join(line) for line in raw]
        csc = [''.join(line) for line in csc]

        res = {}
        for i in range(len(raw)):
            index = 'NUM_{}'.format(i)
            res[index] = {'原句':raw[i],'纠错后':csc[i]}
        endtime= datetime.datetime.now()
        print(endtime - starttime)
        return jsonify({'result':res})


if __name__ == '__main__':
    # app.run(host, port, debug, options)
    # 默认值：host="127.0.0.1", port=5000, debug=False
    app.run(host="0.0.0.0", port=8888)



