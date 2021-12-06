import tensorflow as tf
import os
from bert_tagging import BertTagging,DataProcessor
import numpy as np
import tokenization
from pinyin_tool import PinyinTool
from collections import namedtuple
import collections

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
    # _tokens.append("[SEP]")
    # segment_ids.append(0)
    # stroke_ids.append(0)
    # _labels.append(labels[0])
    # _lmask.append(0)
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
    ans = np.asarray(ans, dtype=np.int32)
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

    #int32 to int32
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int32:
            t = tf.to_int32(t)
        example[name] = t
    input_ids = example['input_ids']
    input_mask = example['input_mask']
    segment_ids = example['segment_ids']
    stroke_ids = example['stroke_ids']
    # label_ids = example['label_ids']
    # lmask = example['lmask']
    # py_labels = tf.py_func(_get_py_seq, [label_ids], [tf.int32])

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

    # PYID2SEQ = pytool.get_pyid2seq_matrix()
    # SKID2SEQ = sktool.get_pyid2seq_matrix()

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
    dataset = dataset.batch(batch_size).prefetch(2)
    return dataset,label_list,py_label_list


#        input_id,input_mask,segment_id,stroke_id = convert_single_sentence(sentence,max_sen_len,tokenizer,pytool,sktool)
#        input_ids.append(input_id)
#        input_masks.append(input_mask)
#        segment_ids.append(segment_id)
#        stroke_ids.append(stroke_id)
#    input_ids,input_masks,segment_ids,stroke_ids = np.array(input_ids,dtype=np.int32),np.array(input_masks,dtype=np.int32),np.array(segment_ids,dtype=np.int32),np.array(stroke_ids,dtype=np.int32)
#    dataset = tf.data.Dataset.from_tensor_slices(
#        {
#            "input_ids": input_ids,
#            "input_masks":input_masks,
#            "segment_ids":segment_ids,
#            "stroke_ids":stroke_ids
#        }
#    )
#    # dataset = dataset.map(decode_record, num_parallel_calls=10)
#    dataset = dataset.batch(batch_size).prefetch(2)
#    return dataset,label_list,py_label_list

def inference(sentences):

    if type(sentences) is str:
        sentences = [sentences]
    # gpuid = FLAGS.gpuid
    max_sen_len = 180
    # test_file = FLAGS.test_path
    # out_dir = FLAGS.output_dir
    batch_size = 2
    # init_bert_dir = FLAGS.init_bert_path
    vocab_file = './datas/pretrained_plome/vocab.txt'

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # keep_prob = FLAGS.keep_prob
    # print('test_file=', test_file)

    bert_config_path = './datas/pretrained_plome/bert_config.json'
    pyid2seq = np.load('./plome_finetune_sess_output/pyid2seq.npy')
    skid2seq = np.load('./plome_finetune_sess_output/skid2seq.npy')
    zi_py_matrix = np.load('./plome_finetune_sess_output/zi_py_matrix.npy')

    # pyid2seq = pyid2seq.astype(np.int32)
    # skid2seq = skid2seq.astype(np.int32)
    # zi_py_matrix = zi_py_matrix.astype(np.int32)



    # test_data_processor = DataProcessor(test_file, max_sen_len, vocab_file, out_dir, label_list=None, is_training=False)
    # test_data = test_data_processor.build_data_generator(batch_size)
    # iterator = test_data.make_one_shot_iterator()
    # input_ids, input_mask, segment_ids, stroke_ids, lmask, label_ids, py_labels = iterator.get_next()

    dataset,label_list,py_label_list = data_process(sentences,max_sen_len,vocab_file,batch_size)
    test_num = len(sentences)

    iterator = dataset.make_one_shot_iterator()
    input_ids, input_masks, segment_ids, stroke_ids = iterator.get_next()

    model = BertTagging(bert_config_path, num_class=21128, pyid2seq=pyid2seq, skid2seq=skid2seq, py_dim=32, max_sen_len=180, py_or_sk='all',  keep_prob=0.9,
                        zi_py_matrix=zi_py_matrix, multi_task=True)
    (pred_loss, pred_probs, gold_probs, gold_mask, py_probs, py_one_hot_labels, fusion_prob) = \
        model.create_model(input_ids, input_masks, segment_ids, stroke_ids, lmask = None, labels = None, py_labels = None, is_training=False)

    # label_list = test_data_processor.label_list
    # py_label_list = test_data_processor.py_label_list


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./plome_finance_finetune_output')
        saver.restore(sess, ckpt.model_checkpoint_path)


        all_inputs, all_golds, all_preds = [], [], []
        all_py_golds, all_py_preds = [], []
        all_fusino_preds = []
        all_inputs_sent, all_golds_sent, all_preds_sent = [], [], []
        all_py_pred_sent, all_py_gold_sent, all_fusion_sent = [], [], []
        all_py_inputs, all_py_inputs_sent = [], []

        for step in range(test_num // batch_size):
            inputs, py_inputs, loss_value, preds, golds, gmask, py_pred, py_golds, fusion_pred = sess.run([input_ids, segment_ids, pred_loss, pred_probs, gold_probs, gold_mask, py_probs, py_one_hot_labels, fusion_prob])

            #print('inputs:',inputs.shape,'py_inputs:',py_inputs.shape,'preds:',preds.shape,'py_pred:',py_pred.shape,'gmask',gmask.shape,
            #       'fusion_pred:',fusion_pred.shape)
            preds = np.reshape(preds, (batch_size, max_sen_len, len(label_list)))
            preds = np.argmax(preds, axis=2)
            golds = np.reshape(golds, (batch_size, max_sen_len, len(label_list)))
            golds = np.argmax(golds, axis=2)
            gmask = np.reshape(gmask, (batch_size, max_sen_len))

            if model.multi_task is True:
                py_pred = np.reshape(py_pred, (batch_size, max_sen_len, 430))
                py_pred = np.argmax(py_pred, axis=2)
                py_golds = np.reshape(py_golds, (batch_size, max_sen_len, 430))
                py_golds = np.argmax(py_golds, axis=2)
                fusion_pred = np.reshape(fusion_pred, (batch_size, max_sen_len, len(label_list)))
                fusion_pred = np.argmax(fusion_pred, axis=2)


            for k in range(batch_size):
                tmp1, tmp2, tmp3, tmps4, tmps5, tmps6, tmps7 = [], [], [], [], [], [], []
                for j in range(max_sen_len):
                    if gmask[k][j] == 0: continue
                    all_golds.append(golds[k][j])
                    all_preds.append(preds[k][j])
                    all_inputs.append(inputs[k][j])
                    tmp1.append(label_list[golds[k][j]])
                    tmp2.append(label_list[preds[k][j]])
                    tmp3.append(label_list[inputs[k][j]])
                    if model.multi_task is True:
                        all_py_inputs.append(py_inputs[k][j])
                        all_py_golds.append(py_golds[k][j])
                        all_py_preds.append(py_pred[k][j])
                        all_fusino_preds.append(fusion_pred[k][j])
                        tmps4.append(str(py_golds[k][j]))
                        tmps5.append(str(py_pred[k][j]))
                        tmps6.append(label_list[fusion_pred[k][j]])
                        tmps7.append(str(py_inputs[k][j]))


                all_golds_sent.append(tmp1)
                all_preds_sent.append(tmp2)
                all_inputs_sent.append(tmp3)
                if model.multi_task is True:
                    all_py_pred_sent.append(tmps4)
                    all_py_gold_sent.append(tmps5)
                    all_fusion_sent.append(tmps6)
                    all_py_inputs_sent.append(tmps7)

        all_golds = [label_list[k] for k in all_golds]
        all_preds = [label_list[k] for k in all_preds]
        all_inputs = [label_list[k] for k in all_inputs]

        all_fusino_preds = [label_list[k] for k in all_fusino_preds]
        all_py_inputs = [py_label_list.get(int(k), k) for k in all_py_inputs]
        all_py_golds = [py_label_list.get(int(k), k) for k in all_py_golds]
        all_py_preds = [py_label_list.get(int(k), k) for k in all_py_preds]

        ###
        #print('all_inputs:',all_inputs)
        #print('\n')
        #print('all_golds:',all_golds)
        #print('\n')
        #print('all_py_inputs:',all_py_inputs[:5])
        #print('\n')
        #print('all_preds:',all_preds)
        #print('\n')
        #print('all_fusino_preds:',all_fusino_preds)
        #print('\n')
        #print('all_py_preds:',all_py_preds)

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

    #print('#####'*10)
    #print(raw)
    #print('\n')
    #print(csc)
    return raw,csc

#if __name__ == '__main__':
#
#    with open('datas/test.txt','r')as f:
#        data = f.readlines()
#
#    queries = []
#    for line in data:
#        query = line.strip().split('\t')[0]
#        queries.append(query)
#
#    # print(queries[0])
#    all_inputs,all_fusino_preds = inference(queries)
#    raw,csc = csc_postprepare(all_inputs,all_fusino_preds)
#    for i,(input,pred) in enumerate(zip(raw,csc)):
#        print(i)
#        print(''.join(input))
#        print(''.join(pred))
#        print('\n')
#
#






