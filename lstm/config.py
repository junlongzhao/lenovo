import tensorflow as tf

tf.app.flags.DEFINE_string("src_file", "data/train.txt", "Training data.")
tf.app.flags.DEFINE_string("src_file_dev", "data/dev.txt", "Training data.")
tf.app.flags.DEFINE_string("voc_file","vocab/train_voc.pickle","voc_train")
tf.app.flags.DEFINE_string("tag_voc","vocab/tag_voc.pickle","tag_voc_train")
tf.app.flags.DEFINE_string("log_path","result/log.txt","log_path")
tf.app.flags.DEFINE_string("adam","Adam","optimizer")
tf.app.flags.DEFINE_integer("embed_size",64 ,"embedding size")
tf.app.flags.DEFINE_integer("vocab_size",30000 ,"size of vocab")
tf.app.flags.DEFINE_integer("hidden_dim",64,"hidden_size")
tf.app.flags.DEFINE_integer("num_tags",9,"number of tags")
tf.app.flags.DEFINE_boolean("CRF",True,"whether use crf")
tf.app.flags.DEFINE_float("clip",5.0,"gradient clipping")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learnning rate")
tf.app.flags.DEFINE_integer("epoch",100," epoch of training")
tf.app.flags.DEFINE_integer("batch_size",96,"samples of each minibatch")
tf.app.flags.DEFINE_float("dropout",0.5,"dropout keep_prob")
tf.app.flags.DEFINE_integer("sentence_length",64,"length of sentence")


FLAGS = tf.app.flags.FLAGS
tf.VERSION