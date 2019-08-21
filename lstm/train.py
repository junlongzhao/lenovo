import config
from model.ner_model import NERModel
from  data_utils import load_data,data_corpus
from config import FLAGS
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def main():

    word2id=load_data(FLAGS.voc_file)
    label2id=load_data(FLAGS.tag_voc)
    train_data=data_corpus(FLAGS.src_file,word2id,label2id)
    dev_data=data_corpus(FLAGS.src_file_dev,word2id,label2id)
    nermodel=NERModel(FLAGS,config,word2id,label2id)
    nermodel.build_model()
    nermodel.train(train_data,dev_data)

if __name__=="__main__":
    main()