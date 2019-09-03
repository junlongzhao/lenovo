import argparse
import torch

parser = argparse.ArgumentParser(description='CNN text classificer')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epoches', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
parser.add_argument('-vocab_path',type=str,default='vocab/word_id.pickle',help="where we store vocab")
parser.add_argument('-vocab_tag_path',type=str,default='vocab/train_tag_voc.pickle',help="where we store vocab tag")
parser.add_argument('-training_data_path',type=str,default='data/train_label.csv',help='where we store train data')
parser.add_argument('-test_data_path',type=str,default='data/Test_id_content.csv',help='where we store test data')
parser.add_argument('-kernel_num',type=int,default=100,help='number of each kind kernel')
parser.add_argument('-kernel_sizes',type=list,default=[3,4,5],help='size to use for convolution')
parser.add_argument('-embed-dim',type=int,default=128,help='embedding_size')
#parser.add_argument('-embed-dim',type=int,default=768,help='embedding_size')
parser.add_argument('-dropout',type=float,default=0.5,help='the possibility of drop out')
parser.add_argument('-parameters_path',type=str,default='parameters/model.pth',help="where we store vocab tag")
parser.add_argument("-ckpt_dir",type=str,default="text_cnn_title_desc_checkpoint/",help="checkpoint location for the model")
args=parser.parse_args()


