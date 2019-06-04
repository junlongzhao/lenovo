import tensorflow as tf
from app.data_util import *
import os
import pickle
import jieba
import re
from app.TextRCnnWithCNN import TextCNN_with_RCNN
from app.data_util import load_data

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string("traing_data_path","raw/raw5.txt","path of training data") #变量名称，默认值，用法描述
tf.app.flags.DEFINE_string("traing_data_path", "raw/TDs.txt", "path of training data")
tf.app.flags.DEFINE_integer("vocab_size", 10000, "max vocab__size")
tf.app.flags.DEFINE_integer("sentence_len", 128, "max length of sentence")
tf.app.flags.DEFINE_integer("number_classes", 14, "number of label")
tf.app.flags.DEFINE_integer("num_epochs", 12, "number of epoches to run ")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters")
tf.app.flags.DEFINE_integer("batch_size", 96, "number of batches")
tf.app.flags.DEFINE_integer("embedding_size", 128, "embed_size")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate")
tf.app.flags.DEFINE_float("decay_rate", 0.8, "rate of decay for learning rate")
tf.app.flags.DEFINE_string("ckpt_dir", "textcnn_with_rcnn/", "checkpoint location for the model")
filter_sizes = [3, 4, 5, 9, 10, 15, 20, 25]


def classfytrain(content):
    """
    :param content: 前端传过来的数据
    :return:
    """
    # voc_word2index, voc_index2word, voc_label2index, voc_index2label=create_vocabulary(FLAGS.traing_data_path,FLAGS.vocab_size)
    with open("voc_word2index", 'rb') as  f:
        voc_word2index = pickle.load(f)

    print("前端传过来的数据:", content)
    text = ''.join(re.findall(u'[\u4e00-\u9fff]+', content))
    word=jieba.cut(text)
    text=" ".join(word)
    print("text",text)
    predict = load_data(text, voc_word2index)
    print(type(predict))
    print("predict list",predict)
    predictlist=[]
    for i in range(96):  #把数据做成满足batch_size96的形式
     predictlist.append(predict)
    predict=pad_sequences(predictlist,maxlen=FLAGS.sentence_len,value=0)
    print("predict",predict)
    print("predict shape",predict.shape)
    print("predict type",type(predict))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print("in textcnn_with_rcnn ")
        textcnn_with_rcnn = TextCNN_with_RCNN(filter_sizes, FLAGS.num_filters, FLAGS.number_classes,
                                              FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                                              FLAGS.decay_rate, FLAGS.sentence_len,
                                              FLAGS.vocab_size,
                                              FLAGS.embedding_size)  # 这里直接去初始化model,去执行静态图中的节点，但是还没有feed数据进去。
        print("out textcnn_with_rcnn")
        saver = tf.train.Saver(max_to_keep=1)
        print("os.path.exists(FLAGS.ckpt_dir)", os.path.exists(FLAGS.ckpt_dir + "checkpoint"))
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("initializer")
            sess.run(tf.global_variables_initializer())
        print(" in do_eval")
        label=do_eval(sess, textcnn_with_rcnn, predict, FLAGS.batch_size)
        print("out do_eval")
        return label

def do_eval(sess, textcnn_with_rcnn, Testx, batch_size):
    print("in do_eval")
    logits = sess.run([textcnn_with_rcnn.logits], feed_dict={textcnn_with_rcnn.input_x: Testx,textcnn_with_rcnn.dropout_keep_prob:1})
    print("logits:", logits)
    print("sxhhsis ss sss)=============================")
    predict=np.argmax(logits,axis=-1)
    print("predict",predict)
    label_index=predict[0][0]
    with open("voc_index2label", 'rb') as  f:
        voc_index2label = pickle.load(f)
    return voc_index2label[label_index]