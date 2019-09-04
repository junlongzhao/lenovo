import tensorflow as tf
from data_util import *
import os
import pickle
from TextRCnnWithCNN import TextCNN_with_RCNN
from data_util import load_data,load_data_predict

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string("traing_data_path","raw/raw5.txt","path of training data") #变量名称，默认值，用法描述
tf.app.flags.DEFINE_string("traing_data_path", "data/Train_label.csv", "path of training data")
tf.app.flags.DEFINE_string("predict_data_path","data/Test_id_content.csv","path of predict data")
tf.app.flags.DEFINE_integer("vocab_size", 300000, "max vocab__size")
tf.app.flags.DEFINE_integer("sentence_len", 400, "max length of sentence")
tf.app.flags.DEFINE_integer("number_classes", 3, "number of label")
tf.app.flags.DEFINE_integer("num_epochs", 12, "number of epoches to run ")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters")
tf.app.flags.DEFINE_integer("batch_size", 96, "number of batches")
tf.app.flags.DEFINE_integer("embedding_size", 128, "embed_size")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate")
tf.app.flags.DEFINE_float("decay_rate", 0.8, "rate of decay for learning rate")
tf.app.flags.DEFINE_string("ckpt_dir", "textcnn_with_rcnn/", "checkpoint location for the model")
#tf.app.flags.DEFINE_string()
filter_sizes = [3, 4, 5, 9, 10, 15, 20, 25]


def classfytrain():
    """
    :param content:
    :return:
    """
    with open("vocab/word_id.pickle", 'rb') as  f:
        voc_word2index = pickle.load(f)
    print("voc_word2index",voc_word2index)
    Traindata,Testdata=load_data(FLAGS.traing_data_path,voc_word2index)
    TrainX,Trainy=zip(*Traindata)
    print("TrainX",TrainX)
    print("TrainY",Trainy)
    TestX,TestY=zip(*Testdata)
    predict,id=load_data_predict(FLAGS.predict_data_path,voc_word2index)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print("in textcnn_with_rcnn ")
        textcnn_with_rcnn = TextCNN_with_RCNN(filter_sizes, FLAGS.num_filters, FLAGS.number_classes,
                                              FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                                              FLAGS.decay_rate, FLAGS.sentence_len,
                                              FLAGS.vocab_size,
                                              FLAGS.embedding_size)  # 这里直接去初始化model,去执行静态图中的节点，但是还没有feed数据进去。
        saver = tf.train.Saver(max_to_keep=1)
        print("os.path.exists(FLAGS.ckpt_dir)", os.path.exists(FLAGS.ckpt_dir + "checkpoint"))
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            print("in predict")
            do_eval_predict(sess, textcnn_with_rcnn, predict,id)
        else:
            sess.run(tf.global_variables_initializer())
            do_eval(sess, saver,textcnn_with_rcnn,TrainX,Trainy,TestX,TestY)

def do_eval(sess, saver,textcnn_with_rcnn, Trainx,TrainY,TestX,TestY):
    for epoch in range(0, FLAGS.num_epochs):
        print("epoch", epoch)
        TrainNum=len(Trainx)
        for start,end in zip(range(0,TrainNum,FLAGS.batch_size),range(FLAGS.batch_size,TrainNum,FLAGS.batch_size)):
          logits,loss,accuracy_data,_ = sess.run([textcnn_with_rcnn.logits,textcnn_with_rcnn.loss_val,textcnn_with_rcnn.accuracy,textcnn_with_rcnn.train_op], feed_dict={textcnn_with_rcnn.input_x:Trainx[start:end],textcnn_with_rcnn.input_y:TrainY[start:end],textcnn_with_rcnn.dropout_keep_prob:1})
          predict,label=accuracy_data
          accuracy=0
          for i in range(len(label)):
            if predict[i]==label[i]:
               accuracy=accuracy+1
          print("train accuracy",accuracy/FLAGS.batch_size)
        save_path = FLAGS.ckpt_dir + "model.ckpt"
        saver.save(sess, save_path, global_step=epoch)
        do_test(sess,textcnn_with_rcnn,TestX,TestY)

def do_test(sess, textcnn_with_rcnn,TestX,TestY):
    TestNum=len(TestX)
    for start, end in zip(range(0, TestNum, FLAGS.batch_size), range(FLAGS.batch_size, TestNum, FLAGS.batch_size)):
        logits, loss, accuracy_data_test = sess.run(
            [textcnn_with_rcnn.logits, textcnn_with_rcnn.loss_val, textcnn_with_rcnn.accuracy,
             textcnn_with_rcnn.train_op],
            feed_dict={textcnn_with_rcnn.input_x: TestX[start:end], textcnn_with_rcnn.input_y: TestY[start:end],
                       textcnn_with_rcnn.dropout_keep_prob: 1})
        predict,label=accuracy_data_test
        accuracy=0
        for i in range(len(label)):
            if predict[i] == label[i]:
                accuracy = accuracy + 1
        print("test accuracy", accuracy / FLAGS.batch_size)

def  do_eval_predict(sess, textcnn_with_rcnn, predict,id):
     length_predict=len(predict)
     batchsize=FLAGS.batch_size
     predict_result_all=[]
     id_batch_all=[]
     for start,end  in zip(range(0,length_predict,batchsize),range(batchsize,length_predict,batchsize)):
         id_batch=id[start:end]
         logits = sess.run([textcnn_with_rcnn.logits], feed_dict={textcnn_with_rcnn.input_x: predict[start:end], textcnn_with_rcnn.dropout_keep_prob: 1})
         predict_result=np.argmax(logits,axis=-1)
         #print(" predict_result",predict_result[0])
         predict_result_all.append(predict_result[0])
         id_batch_all.append(id_batch)
     print("predict_result_all_length",len(predict_result_all))
     print("predict_result_all_length[0]",len(predict_result_all[0]))
     print("id_batch_all_length",len(id_batch_all))
     print("id_batch_all_length[0]",len(id_batch_all[0]))
     writedoc(predict_result_all,id_batch_all)


def writedoc(Labels, ID):
    for row in range(len(Labels)):
        print("length labels",len(Labels))
        for num in range(len(Labels[row])):
            #label = Labels[row][num]
            print("label", Labels[row][num])
            print("ID", ID[row][num])
            content = ID[row][num] + ',' + Labels[row][num]
            savecontext("submit.csv", content)


def savecontext(filename, contents):
    fh = open(filename, 'a', encoding='utf-8')
    fh.write(contents + "\n")
    fh.close()


if __name__=="__main__":
    classfytrain()