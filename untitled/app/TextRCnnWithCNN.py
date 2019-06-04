import tensorflow as tf
import copy
class TextCNN_with_RCNN:
    def __init__(self, filter_sizes, num_filters, num_classes, learning_rate, batch_size, decay_steps, decay_rate,
                 sequence_length, vocab_size, embed_size, initializer=tf.random_normal_initializer(stddev=0.1), multi_label_flag=False):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.learning_rate = learning_rate
        self.filter_sizes = filter_sizes  #it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes) # how many filters totally.
        self.multi_label_flag = multi_label_flag
        self.hidden_size = embed_size
        self.activation = tf.nn.tanh
        # add placeholder (X,label)
        self.input_x = tf.placeholder(tf.int32, [self.batch_size, self.sequence_length], name="input_x")  # X
        #self.input_y = tf.placeholder(tf.int32, [self.batch_size, self.num_classes], name="input_y")  # y:[None,num_classes]
        #self.input_y_multilabel = tf.placeholder(tf.float32, [self.batch_size, self.num_classes],name="input_y_multilabel")  # y:[None,num_classes]. this is for multi-label classification only.
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate
        self.instantiate_weights_cnn()
        self.instantiate_weights_rcnn()
        self.logits=self.inference()
        #self.loss_val=self.loss_multilabel()
        #self.train_op = self.train()
        #self.accuracy=self.accuracy(self.logits,self.input_y_multilabel)
       # self.accuracy = self.accuracy(self.logits)

    def instantiate_weights_cnn(self):
            """define all weights here"""
            with tf.name_scope("projection_cnn"):  # embedding matrix
                # self.Embedding = tf.get_variable("Embedding",shape=[self.vocab_size, self.embed_size],initializer=self.initializer) #[vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
                print("instantiate_weights_cnn in")
                self.W_projection_cnn = tf.get_variable("W_projection_cnn",shape=[self.num_filters_total, self.num_classes],initializer=self.initializer)  # [embed_size,label_size]
                self.b_projection_cnn = tf.get_variable("b_projection_cnn",shape=[self.num_classes])  # [label_size]
                print("instantiate_weights_cnn out")
    def instantiate_weights_rcnn(self):
        print("instantiate_weights_rcnn in")
        self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)
        self.left_side_firstword=tf.get_variable("left_first_word",[self.batch_size,self.embed_size],initializer=self.initializer)
        self.right_side_lastword=tf.get_variable("right_last_word",[self.batch_size,self.embed_size],initializer=self.initializer)
        self.w1=tf.get_variable("w_l",[self.embed_size,self.embed_size],initializer=self.initializer)
        self.wr=tf.get_variable("w_r",[self.embed_size,self.embed_size],initializer=self.initializer)
        self.w_sl=tf.get_variable("w_sl",[self.embed_size,self.embed_size],initializer=self.initializer)
        self.w_sr=tf.get_variable("w_sr",[self.embed_size,self.embed_size],initializer=self.initializer)
        self.w_protection_rcnn=tf.get_variable("W_protection",[self.embed_size*3,self.num_classes],initializer=self.initializer)
        self.b_projection_rcnn=tf.get_variable("b_protection",[self.num_classes],initializer=self.initializer)
        print("instantiate_weights_rcnn out")

    def get_left_context(self,context_left,embedding_previous):
         #context_left[batch_size,embedding_size]
         #embedding_previous[batch_szie,embeding_size]
         left_c=tf.matmul(context_left,self.w1)
         left_e=tf.matmul(embedding_previous,self.w_sl)
         left_h=left_c+left_e
         context_left=self.activation(left_h)
         context_left=tf.reshape(context_left,[-1,self.embed_size])
         return context_left

    def get_right_context(self,context_right,embedding_afterwards):
         right_c=tf.matmul(context_right,self.wr)
         right_e=tf.matmul(embedding_afterwards,self.w_sr)
         right_h=right_c+right_e
         context_right=self.activation(right_h)
         return  context_right


    def inference1(self):
            print("inference1 in")
            self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)  # [None,sentence_length,embed_size]
            self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words,-1)  # [None,sentence_length,embed_size,1). expand dimension so meet input requirement of 2d-conv
            # 2.=====>loop each filter size. for each filter, do:convolution-pooling layer(a.create filters,b.conv,c.apply nolinearity,d.max-pooling)--->
            # you can use:tf.nn.conv2d;tf.nn.relu;tf.nn.max_pool; feature shape is 4-d. feature is a new variable
            pooled_outputs = []
            for i,filter_size in enumerate(self.filter_sizes):
                with tf.name_scope("convolution-pooling-%s" % filter_size):
                   # print("filter_sizes",len(self.filter_sizes))
                    filter = tf.get_variable("filter-%s" % filter_size,[filter_size, self.embed_size, 1, self.num_filters],initializer=self.initializer)
                    conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1],padding="VALID",name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                    # ====>c. apply nolinearity
                    b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                    h = tf.nn.relu(tf.nn.bias_add(conv, b),"relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                    # ====>. max-pooling.  value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                    #                  ksize: A list of ints that has length >= 4.  The size of the window for each dimension of the input tensor.
                    #                  strides: A list of ints that has length >= 4.  The stride of the sliding window for each dimension of the input tensor.
                    pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID',
                                            name="pool")  # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                    pooled_outputs.append(pooled)
                   # print("pooled_outputs",pooled_outputs[0].shape)
                   # print("length of pooled_outputs",len(pooled_outputs))
                    # 3.=====>combine all pooled features, and flatten the feature.output' shape is a [1,None]
                    # e.g. >>> x1=tf.ones([3,3]);x2=tf.ones([3,3]);x=[x1,x2]
                    #         x12_0=tf.concat(x,0)---->x12_0' shape:[6,3]
                    #         x12_1=tf.concat(x,1)---->x12_1' shape;[3,6]
            self.h_pool = tf.concat(pooled_outputs,3)  # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
            print("h_pool_type",self.h_pool.get_shape())
            self.h_pool_flat = tf.reshape(self.h_pool, [-1,self.num_filters_total]) # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)

                # 4.=====>add dropout: use tf.nn.dropout
            with tf.name_scope("dropout_cnn"):
                    self.h_drop = tf.nn.dropout(self.h_pool_flat,keep_prob=self.dropout_keep_prob) #[None,num_filters_total]
                # 5. logits(use linear layer)and predictions(argmax)
            with tf.name_scope("output_cnn"):
                    logits = tf.matmul(self.h_drop,self.W_projection_cnn) + self.b_projection_cnn # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
            print("inference1 shape",logits.shape)
            print("inference1 out")
            return logits

    def conv_layer_with_recurrent(self):
        embeded_words_spilt=tf.split(self.embedded_words,self.sequence_length,axis=1)  #sentence_length 个[batchsize,1,embedding]
        embeded_words_squeeze=[tf.squeeze(x,axis=1) for x in embeded_words_spilt]  #sentence_length个[batchsize,embedding]  type is list
        #print(" embeded_words_squeeze shape",embeded_words_squeeze[0].shape) #sentence 个 [?,embedding]
        embeded_previous=self.left_side_firstword #[batchsize,embedding_size]
        #contex_left_previous=tf.get_variable("contex_left_previous",[self.batch_size,self.embed_size])
        contex_left_previous=tf.zeros((self.batch_size,self.embed_size))
        context_left_list=[]
        for i,embeded_words_current in enumerate(embeded_words_squeeze):
             context_left=self.get_left_context(contex_left_previous,embeded_previous)
             context_left_list.append(context_left)
             embeded_previous=embeded_words_current
             contex_left_previous=context_left
        embeded_words_squeeze2=copy.copy(embeded_words_squeeze) #
        #embeded_words_squeeze2=tf.reversed(embeded_words_squeeze2) #旋转每一个
        embeded_words_squeeze2.reverse()
        #print("embeded_words_squeeze2",embeded_words_squeeze2[0].shape)
        embeded_afterwards=self.right_side_lastword
        context_right_aterwards=tf.get_variable("contex_right_afterwards",[self.batch_size,self.embed_size])
        context_right_list=[]
        for i,embeded_words_current in enumerate(embeded_words_squeeze2):
             context_right=self.get_right_context(context_right_aterwards,embeded_afterwards)
             context_right_list.append(context_right)
             embeded_afterwards=embeded_words_current
             context_right_aterwards=context_right
        outlist=[]
        for index,embeded_words_current in enumerate(embeded_words_squeeze): #[None,embed_size]
            outrepresent=tf.concat([context_left_list[index],embeded_words_current,context_right_list[index]],axis=1)  #[None,embed_size*3]
           # print("outrepresent_of_shape",outrepresent.shape)
            outlist.append(outrepresent) #sentence_length个[None,embed_size*3]
        output=tf.stack(outlist,axis=1) #[None,sentence_length,embed_size*3]`
        return output

    def inference2(self):
        print("inference2 in")
        self.embedded_words=tf.nn.embedding_lookup(self.Embedding,self.input_x) #look for words
        output_conv=self.conv_layer_with_recurrent() #[None,sentence_length,embed_size*3]
        out_pooling=tf.reduce_max(output_conv,axis=1) #[None,embed_size*3]
        #print("out_pooling_shape",out_pooling.shape) #[96,384]
        with tf.name_scope("dropout_rcnn"):
           dropout= tf.nn.dropout(out_pooling,keep_prob=self.dropout_keep_prob)  #[None,embed_size*3]
           logits=tf.matmul(dropout,self.w_protection_rcnn)+self.b_projection_rcnn
           #print("inference2  shape",logits.shape) #inference2 logits shape (96, 26)
           print("inference2 out")
        return logits

    def inference(self):
        weight1=tf.get_variable("weught1",shape=())
        self.p1_weight1=tf.nn.sigmoid(weight1)
        self.p1_weight2=1.0-self.p1_weight1
        logits1=self.inference1()
        logits2=self.inference2()
        logits=logits1*self.p1_weight1+logits2*self.p1_weight2
        #print("logits",logits.shape)(96, 26)
        return logits

    def loss_multilabel(self,l2_lambda=0.00001): #0.0001#this loss function is for multi-label classification
        with tf.name_scope("loss"):
            losses=tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel,logits=self.logits)
            #print("loss before",losses.shape) (96,26)
            loss=tf.reduce_mean(losses,axis=1)
            #print("loss shape",loss.shape) (96,)
            loss=tf.reduce_mean(loss) #()
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            #print("l2_losses_shape",l2_losses.get_shape())
            loss = loss + l2_losses
           # print("loss shape final",loss.shape) ()
        return loss

    def accuracy(self,logits,input_y_multilabel):
        result1 = tf.nn.top_k(logits, 1)
        result2 = tf.nn.top_k(input_y_multilabel, 1)
        #返回最大值，以及两个最大值的下标
        return result1,result2

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        #decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps) 学习率的更新公式
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op