import tensorflow as tf
import numpy as np

def init_params(shape, param):
    if param is None:
        return tf.random_uniform(shape=shape, minval=-1., maxval=1.)
    else: 
        return tf.constant(param)
    
def crossentropy(y, p):
    return -(y*tf.log(p)+(1-y)*tf.log(1-p))

def choose_emb(args, emb, weight):
    assert args.main_emb in ['emb', 'weight', 'copy']
    if args.main_emb == 'emb':
        return emb, weight
    elif args.main_emb == 'weight':
        return weight, emb
    else: 
        return emb, emb
    
def choose_geo_loss(geo_reg_type, loss_l2, loss_xn):
    loss = 0
    if geo_reg_type == 'l2':
        loss = loss_l2
    elif geo_reg_type == 'xn':
        loss = loss_xn
    else: raise ValueError('geo_type unexpected, got {}'.format(args.geo_reg_type))
    return loss

class STSkipgram(object):
    def __init__(self, args, pretrained_sem=None, pretrained_geo=None, pretrained_weight=None, pretrained_time=None):
        self.pretrained_semantic = init_params((args.vocabulary_size, args.sem_dim), pretrained_sem)
        self.pretrained_geo = init_params((args.vocabulary_size, args.geo_dim), pretrained_geo)
        self.pretrained_weight = init_params((args.vocabulary_size, args.sem_dim+args.geo_dim+args.free_dim), pretrained_weight)
        self.pretrained_time = init_params((args.n_timeslot, args.sem_dim), pretrained_time)
        
        # --Placeholders
        self.center_loc   = tf.placeholder(tf.int32, shape=[None], name='center_loc')
        self.label_loc    = tf.placeholder(tf.int32, shape=[None, 1], name='label_loc') #specific shape for sampled_softamx_loss
        self.weight_decay = tf.placeholder(tf.float32, shape=[None], name='weight_decay')
        self.coor_center  = tf.placeholder(tf.float32, shape=[None,2], name='coor_center')
        self.coor_label   = tf.placeholder(tf.float32, shape=[None,2], name='coor_label')
        self.label_t      = tf.placeholder(tf.int32, shape=[None], name='label_t')
        
        # --Weights
        self.softmax_weights= tf.get_variable('weights', initializer=self.pretrained_weight)
        self.softmax_biases = tf.get_variable('bias', initializer=tf.zeros([args.vocabulary_size]), trainable=False)
        self.sem_emb        = tf.get_variable('semantic_emb', initializer=self.pretrained_semantic)
        self.geo_emb        = tf.get_variable('geo_emb', initializer=self.pretrained_geo)
        self.embeddings     = tf.concat([self.sem_emb, self.geo_emb], axis=1)
        self.time_embeddings= tf.get_variable('time_embeddings', initializer=self.pretrained_time)

        if args.free_dim > 0:
            self.free_emb   = tf.get_variable('free_emb', shape=(args.vocabulary_size, args.free_dim), 
                               initializer=tf.keras.initializers.he_uniform())
            self.embeddings = tf.concat([self.embeddings, self.free_emb], axis=1)
        
        # --Retrive Embeddings
        self.main_emb, self.context_emb = choose_emb(args, emb=self.embeddings, weight=self.softmax_weights)
        self.emb_from_sem   = tf.nn.embedding_lookup(self.sem_emb, self.center_loc)
        self.emb_from_geo_x = tf.nn.embedding_lookup(self.geo_emb, self.center_loc)
        self.emb_from_geo_y = tf.nn.embedding_lookup(self.geo_emb, tf.squeeze(self.label_loc)) # due to the special shape of label_loc
        self.emb_from_whole = tf.nn.embedding_lookup(self.main_emb, self.center_loc)
        
        # --Skipgram loss
        self.skipgram_loss = tf.nn.sampled_softmax_loss(
            inputs=self.emb_from_whole, 
            weights=self.context_emb, 
            biases=self.softmax_biases, labels=self.label_loc, 
            num_sampled=args.num_negative_sample, num_classes=args.vocabulary_size)
        self.weighted_skipgram_loss = tf.reduce_mean(self.skipgram_loss*self.weight_decay)

        # --GEO regularizer
        self.euclidean_dis = tf.norm(self.coor_center-self.coor_label,ord='euclidean', axis=-1)
        self.euclidean_sim = tf.exp(-1*self.euclidean_dis*args.geo_temp) #rescale it to (0,1)
        cosine_sim = tf.reduce_mean(
            tf.multiply(tf.nn.l2_normalize(self.emb_from_geo_x, axis=-1),
                        tf.nn.l2_normalize(self.emb_from_geo_y, axis=-1)), axis=-1)
        self.cosine_sim = 0.5*(cosine_sim+1) #rescale it to (0,1)
        self.geo_loss_l2 = args.regulation_weight*tf.losses.mean_squared_error(labels=self.euclidean_sim, 
                                        predictions=self.cosine_sim)
        self.geo_loss_xn = args.regulation_weight*crossentropy(y=self.euclidean_sim, 
                                        p=self.cosine_sim)
        self.geo_loss = choose_geo_loss(args.geo_reg_type, loss_l2=self.geo_loss_l2, loss_xn=self.geo_loss_xn)
        
        # --Temporal regularizer
        emb_dot_time = tf.matmul(self.emb_from_sem, tf.transpose(self.time_embeddings))
        self.time_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.label_t, logits=emb_dot_time))
        
        # --Normalization
        self.normalize_geo_op = tf.assign(self.geo_emb, tf.nn.l2_normalize(self.geo_emb, axis=1))
        self.normalize_sem_op = tf.assign(self.sem_emb, tf.nn.l2_normalize(self.sem_emb, axis=1))
        
        # --Optimization
        global_step_g = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(args.lr, global_step_g, 5000, 0.5, staircase=True) 
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_t = self.optimizer.minimize(self.time_loss)
        self.train_skipgram = self.optimizer.minimize(self.skipgram_loss)
        self.train_geo = self.optimizer.minimize(self.geo_loss)

        # --Summaries
        self.trainable_params = tf.trainable_variables()
        self.all_params = tf.global_variables()