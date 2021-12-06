# output_node_names: loss/output_py_bias
import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
import numpy as np
from bert_tagging import BertTagging,DataProcessor

bert_config_path = './datas/pretrained_plome/bert_config.json'
pyid2seq = np.load('./plome_finetune_sess_output/pyid2seq.npy')
skid2seq = np.load('./plome_finetune_sess_output/skid2seq.npy')
zi_py_matrix = np.load('./plome_finetune_sess_output/zi_py_matrix.npy')
#
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
input_ids = tf.placeholder(dtype=tf.int64,shape=[None, 180])
input_mask = tf.placeholder(dtype=tf.int64,shape=[None, 180])
segment_ids = tf.placeholder(dtype=tf.int64,shape=[None, 180])
stroke_ids = tf.placeholder(dtype=tf.int64,shape=[None, 180])
# lmask = tf.placeholder(dtype=tf.int64,shape=[None, 180])
# label_ids = tf.placeholder(dtype=tf.int64,shape=[None, 180])
# py_labels = tf.placeholder(dtype=tf.int64,shape=[None, 180])
#
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
if not os.path.exists('./export_model/2'):

    model_version = 2
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