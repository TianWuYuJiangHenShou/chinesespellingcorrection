import tensorflow as tf

sess = tf.Session()
saver = tf.train.import_meta_graph('./plome_finetune_output/best.ckpt.meta') # 加载模型结构
saver.restore(sess, tf.train.latest_checkpoint('./plome_finetune_output/')) # 只需要指定目录就可以恢复所有变量信息

saver.create_model()
print(sess.graph)
