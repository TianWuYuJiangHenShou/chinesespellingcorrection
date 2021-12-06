import tensorflow as tf
import os
from bert_tagging import BertTagging,DataProcessor
import numpy as np

def inference(FLAGS):

    gpuid = FLAGS.gpuid
    max_sen_len = FLAGS.max_sen_len
    # train_path = FLAGS.train_path
    test_file = FLAGS.test_path
    out_dir = FLAGS.output_dir
    batch_size = 50
    EPOCH = FLAGS.epoch
    learning_rate = FLAGS.learning_rate
    init_bert_dir = FLAGS.init_bert_path
    learning_rate = FLAGS.learning_rate
    vocab_file = '%s/vocab.txt' % init_bert_dir
    init_checkpoint = '%s/bert_model.ckpt' % init_bert_dir
    bert_config_path = '%s/bert_config.json'% init_bert_dir
    bert_config_path = '%s/bert_config.json'% init_bert_dir

    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    keep_prob = FLAGS.keep_prob
    print('test_file=', test_file)

    bert_config_path = './datas/pretrained_plome/bert_config.json'
    pyid2seq = np.load('./plome_finetune_sess_output/pyid2seq.npy')
    skid2seq = np.load('./plome_finetune_sess_output/skid2seq.npy')
    zi_py_matrix = np.load('./plome_finetune_sess_output/zi_py_matrix.npy')


    test_data_processor = DataProcessor(test_file, max_sen_len, vocab_file, out_dir, label_list=None, is_training=False)
    test_data = test_data_processor.build_data_generator(batch_size)
    iterator = test_data.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, stroke_ids, lmask, label_ids, py_labels = iterator.get_next()
    test_num = test_data_processor.num_examples

    model = BertTagging(bert_config_path, num_class=21128, pyid2seq=pyid2seq, skid2seq=skid2seq, py_dim=32, max_sen_len=180, py_or_sk='all',  keep_prob=0.9,
                        zi_py_matrix=zi_py_matrix, multi_task=True)
    (pred_loss, pred_probs, gold_probs, gold_mask, py_probs, py_one_hot_labels, fusion_prob) = \
        model.create_model(input_ids, input_mask, segment_ids, stroke_ids, lmask, label_ids, py_labels, is_training=False)

    label_list = test_data_processor.label_list
    py_label_list = test_data_processor.py_label_list
   
    print('input_ids:',type(input_ids),input_ids.shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('./plome_finetune_sess_output')
        saver.restore(sess, ckpt.model_checkpoint_path)


        all_inputs, all_golds, all_preds = [], [], []
        all_py_golds, all_py_preds = [], []
        all_fusino_preds = []
        all_inputs_sent, all_golds_sent, all_preds_sent = [], [], []
        all_py_pred_sent, all_py_gold_sent, all_fusion_sent = [], [], []
        all_py_inputs, all_py_inputs_sent = [], []

        for step in range(test_num // batch_size ):
            inputs, py_inputs, loss_value, preds, golds, gmask, py_pred, py_golds, fusion_pred = sess.run([input_ids, segment_ids, pred_loss, pred_probs, gold_probs, gold_mask, py_probs, py_one_hot_labels, fusion_prob])
            
            print('inputs:',inputs.shape,'py_inputs:',py_inputs.shape,'preds:',preds.shape,'py_pred:',py_pred.shape,'gmask',gmask.shape,
                  'fusion_pred:',fusion_pred.shape)
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

        ####
        print('all_inputs:',all_inputs[-100:])
        print('\n')
        print('all_golds:',all_golds[:5])
        print('\n')
        print('all_py_inputs:',all_py_inputs[:5])
        print('\n')
        print('all_py_golds:',all_py_golds[:5])
        print('\n')
        print('all_preds:',all_preds[:5],len(all_preds))
        print('\n')
        print('all_fusino_preds:',all_fusino_preds[-100:],len(all_fusino_preds))
        print('\n')
        print('all_py_preds:',all_py_preds[:5])



if __name__ == '__main__':

    flags = tf.flags

    flags.DEFINE_string("checkpoint_path", './plome_finetune_output', "The packages of checkpoint ")
    flags.DEFINE_string("meta_file",'best.ckpt.meta','the meta file of checkpoint ')

    flags.DEFINE_string("gpuid", '0', "The gpu NO. ")

    ## Optional
    # flags.DEFINE_string("train_path", '', "train path ")
    flags.DEFINE_string("test_path", '', "test path ")
    flags.DEFINE_string("output_dir", '', "out dir ")
    flags.DEFINE_string("init_bert_path", '', "out dir ")
    flags.DEFINE_string("sk_or_py", 'py', "sk_or_py")
    flags.DEFINE_string("label_list", '', 'max_sen_len')
    flags.DEFINE_integer("max_sen_len", 64, 'max_sen_len')
    flags.DEFINE_integer("batch_size", 32, 'batch_size')
    flags.DEFINE_integer("single_text", '0', 'single_text')
    flags.DEFINE_integer("epoch", 2, 'batch_size')
    flags.DEFINE_float("learning_rate", 5e-5, 'filter_punc')
    flags.DEFINE_float("keep_prob", 0.9, 'keep prob in dropout')
    flags.DEFINE_integer("py_dim", 32, 'keep prob in dropout')
    flags.DEFINE_integer("multi_task", 0, 'keep prob in dropout')

    FLAGS= flags.FLAGS
    inference(FLAGS)






