from operator import neg
import os
from turtle import update
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
from model.AbstractRecommender import AbstractRecommender
import tensorflow
from util import timer, tool
from util import learner, l2_loss, inner_product, log_loss
from data import PairwiseSampler, PointwiseSamplerV2
from util.tool import randint_choice
from time import time
from util.cython.arg_topk import arg_topk



class SSM(AbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(SSM, self).__init__(dataset, conf)

        self.conf = conf
        self.model_name = conf["recommender"]
        self.dataset_name = conf["data.input.dataset"]
        self.lr = conf['lr']
        self.reg = conf['reg']
        self.embedding_size = conf['embed_size']
        self.learner = conf["learner"]
        self.dataset = dataset
        self.n_users, self.n_items = self.dataset.num_users, self.dataset.num_items
        
        self.batch_size = conf['batch_size']
        self.test_batch_size = conf['test_batch_size']
        self.epochs = conf["epochs"]
        self.verbose = conf["verbose"]
        self.stop_cnt = conf["stop_cnt"]
        self.init_method = conf["init_method"]
        self.stddev = conf["stddev"]
        self.flag_l2_norm = conf["flag_l2_norm"]

        self.n_layers = conf['n_layers']
        self.side = conf["side"]
        assert self.side in ["user", "item", "both"]
        self.alpha = conf["alpha"]
        self.norm_adj = self.create_adj_mat(self.side, self.alpha)

        self.temp = conf["temp"]
        self.neg_dist = conf['neg_dist']
        if self.neg_dist:
            self.num_negatives = conf["n_negs"]
            self.beta = conf["beta"]
            self.item_popularity = [item**self.beta for item in dataset.item_popularity]

        self.user_pos_train = self.dataset.get_user_train_dict(by_time=False)
        self.all_users = list(self.user_pos_train.keys())
        
        self.best_result = np.zeros([5], dtype=float)
        self.best_epoch = 0
        self.sess = sess

        self.model_str = '#layers=%d-side=%s-alpha=[%.1f,%.1f]-reg%.0e' % (
            self.n_layers,
            self.side,
            self.alpha[0], self.alpha[1],
            self.reg,
        )
        if self.neg_dist:
            self.model_str += "-beta=%.1f-n_neg=%d" % (
                self.beta,
                self.num_negatives,
            )
        self.pretrain = conf["pretrain"]
        if self.pretrain:
            self.epochs = 0
        self.save_flag = conf["save_flag"]
        if self.pretrain or self.save_flag:
            self.tmp_model_folder = conf["proj_path"] + 'model_tmp/%s/%s/%s/' % (
                self.dataset_name, 
                self.model_name,
                self.model_str)
            self.save_folder = conf["proj_path"] + 'dataset/pretrain-embeddings-%s/%s/%s/' % (
                self.dataset_name, 
                self.model_name,
                self.model_str)
            tool.ensureDir(self.tmp_model_folder)
            tool.ensureDir(self.save_folder)

    @timer
    def create_adj_mat(self, side, alpha):
        if side == "user":
            adj_mat = self.dataset.train_matrix
        elif self.side == "item":
            adj_mat = self.dataset.train_matrix.T
        else:
            user_list, item_list = self.dataset.get_train_interactions()
            user_np = np.array(user_list, dtype=np.int32)
            item_np = np.array(item_list, dtype=np.int32)
            ratings = np.ones_like(user_np, dtype=np.float32)
            n_nodes = self.n_users + self.n_items
            tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(n_nodes, n_nodes))
            adj_mat = tmp_adj + tmp_adj.T
        
        rowsum = np.array(adj_mat.sum(1))
        colsum = np.array(adj_mat.sum(0))
        d_inv_left = np.power(rowsum, -alpha[0]).flatten()
        d_inv_left[np.isinf(d_inv_left)] = 0.
        d_mat_inv_left = sp.diags(d_inv_left)
        d_inv_right = np.power(colsum, -alpha[1]).flatten()
        d_inv_right[np.isinf(d_inv_right)] = 0.
        d_mat_inv_right = sp.diags(d_inv_right)
        norm_adj_tmp = d_mat_inv_left.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv_right)

        return adj_matrix

    def _create_variable(self):
        with tf.name_scope("input_data"):
            self.users = tf.placeholder(tf.int32, shape=(None,))
            self.pos_items = tf.placeholder(tf.int32, shape=(None,))
            if self.neg_dist == 1:
                self.neg_items = tf.placeholder(tf.int32, shape=(None,))
            if self.neg_dist == 2:
                if self.num_negatives == 1:
                    self.neg_items = tf.placeholder(tf.int32, shape=(None,))
                else:
                    self.neg_items = tf.placeholder(tf.int32, shape=(None, self.num_negatives))

        with tf.name_scope("embedding_init"):
            self.weights = dict()
            initializer = tf.contrib.layers.xavier_initializer()
            if self.pretrain:
                pretrain_user_embedding = np.load(self.save_folder + 'user_embeddings.npy')
                pretrain_item_embedding = np.load(self.save_folder + 'item_embeddings.npy')
                self.weights['user_embedding'] = tf.Variable(pretrain_user_embedding, 
                                                             name='user_embedding', dtype=tf.float32)  # (users, embedding_size)
                self.weights['item_embedding'] = tf.Variable(pretrain_item_embedding,
                                                             name='item_embedding', dtype=tf.float32)  # (items, embedding_size)
            else:
                self.weights['user_embedding'] = tf.Variable(
                    initializer([self.n_users, self.embedding_size]), name='user_embedding')
                self.weights['item_embedding'] = tf.Variable(
                    initializer([self.n_items, self.embedding_size]), name='item_embedding')

    def build_graph(self):
        self._create_variable()
        with tf.name_scope("inference"):
            if self.n_layers == 1 and self.side in ["user", "item"]:
                self.ua_embeddings, self.ia_embeddings = self._create_SVDpp_embed()
            else:
                self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()

            """
            *********************************************************
            Establish the final representations for user-item pairs in batch.
            """
            self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
            self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
            self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
            self.pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items)
            if self.neg_dist:
                self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)
                self.neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items)

        """
        *********************************************************
        Generate Predictions & Optimize via graph contrastive loss.
        """
        with tf.name_scope("loss"):
            if self.neg_dist == 1:
                self.mf_loss, self.emb_loss = self._create_SSM_loss_V2(
                    self.u_g_embeddings, 
                    self.pos_i_g_embeddings,
                    self.neg_i_g_embeddings,
                )
            elif self.neg_dist == 2:
                if self.num_negatives == 1:
                    self.mf_loss, self.emb_loss = self._create_bpr_loss(
                        self.u_g_embeddings, 
                        self.pos_i_g_embeddings,
                        self.neg_i_g_embeddings,
                    )
                else:
                    self.mf_loss, self.emb_loss = self._create_SSM_loss_V3(
                        self.u_g_embeddings, 
                        self.pos_i_g_embeddings,
                        self.neg_i_g_embeddings,
                    )
            else:
                self.mf_loss, self.emb_loss = self._create_SSM_loss(
                    self.u_g_embeddings, 
                    self.pos_i_g_embeddings,
                )
            self.loss = self.mf_loss + self.emb_loss

        with tf.name_scope("learner"):
            self.opt = learner.optimizer(self.learner, self.loss, self.lr)

        self.saver = tf.train.Saver()

    def _create_SVDpp_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        if self.side == "user":
            u_embeddings = self.weights['user_embedding'] + tf.sparse_tensor_dense_matmul(adj_mat, self.weights['item_embedding'])
            i_embeddings = self.weights['item_embedding']
        else:
            u_embeddings = self.weights['user_embedding']
            i_embeddings = self.weights['item_embedding'] + tf.sparse_tensor_dense_matmul(adj_mat, self.weights['user_embedding'])
        return u_embeddings, i_embeddings

    def _create_lightgcn_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(1, self.n_layers + 1):
            ego_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings, name="sparse_dense_l%d" % k)

            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
    
    def _create_SSM_loss(self, users, pos_items):
        if self.flag_l2_norm[0]:
            batch_user_emb = tf.nn.l2_normalize(users, 1)
            batch_item_emb = tf.nn.l2_normalize(pos_items, 1)
        else:
            batch_user_emb = users
            batch_item_emb = pos_items
        pos_score = tf.reduce_sum(tf.multiply(batch_user_emb, batch_item_emb), axis=1, keep_dims=True)
        ttl_score = tf.matmul(batch_user_emb, batch_item_emb, transpose_a=False, transpose_b=True)
        logits = ttl_score - pos_score
        clogits = tf.reduce_logsumexp(logits / self.temp, axis=1)
        nce_loss = tf.reduce_sum(clogits)
        regularizer = l2_loss(self.u_g_embeddings_pre, self.pos_i_g_embeddings_pre)
        emb_loss = self.reg * regularizer
        return nce_loss, emb_loss

    def _create_SSM_loss_V2(self, users, pos_items, neg_items):
        '''
        negative items are samples from some pre-defined distribtuion
        '''
        if self.flag_l2_norm[0]:
            batch_user_emb = tf.nn.l2_normalize(users, 1)
            batch_pos_item_emb = tf.nn.l2_normalize(pos_items, 1)
            batch_neg_item_emb = tf.nn.l2_normalize(neg_items, 1)
        else:
            batch_user_emb = users
            batch_pos_item_emb = pos_items
            batch_neg_item_emb = neg_items
        pos_score = tf.reduce_sum(tf.multiply(batch_user_emb, batch_pos_item_emb), axis=1)
        neg_score = tf.matmul(batch_user_emb, batch_neg_item_emb, transpose_a=False, transpose_b=True)

        pos_score = tf.exp(pos_score / self.temp)
        neg_score = tf.reduce_sum(tf.exp(neg_score / self.temp), axis=1)
        ttl_score = neg_score + pos_score

        nce_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score + 1e-15))
        regularizer = l2_loss(self.u_g_embeddings_pre, self.pos_i_g_embeddings_pre, self.neg_i_g_embeddings_pre)
        
        emb_loss = self.reg * regularizer

        return nce_loss, emb_loss

    def _create_SSM_loss_V3(self, users, pos_items, neg_items):
        '''
        negative items are samples from some pre-defined distribtuion
        '''
        if self.flag_l2_norm[0]:
            batch_user_emb = tf.nn.l2_normalize(users, 1)
            batch_pos_item_emb = tf.nn.l2_normalize(pos_items, 1)
            batch_neg_item_emb = tf.nn.l2_normalize(neg_items, -1)
        else:
            batch_user_emb = users
            batch_pos_item_emb = pos_items
            batch_neg_item_emb = neg_items
        pos_score = tf.reduce_sum(tf.multiply(batch_user_emb, batch_pos_item_emb), axis=1)
        neg_score = tf.squeeze(tf.matmul(batch_neg_item_emb, tf.expand_dims(batch_user_emb, -1)))   # [batch_size, num_negatives]

        pos_score = tf.exp(pos_score / self.temp)
        neg_score = tf.reduce_sum(tf.exp(neg_score / self.temp), axis=1)
        ttl_score = neg_score + pos_score

        nce_loss = -tf.reduce_sum(tf.log(pos_score / ttl_score + 1e-15))
        regularizer = l2_loss(self.u_g_embeddings_pre, 
                              self.pos_i_g_embeddings_pre, 
                              tf.reshape(self.neg_i_g_embeddings_pre, [-1, self.num_negatives * self.embedding_size]))
        
        emb_loss = self.reg * regularizer

        return nce_loss, emb_loss

    def _create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = inner_product(users, pos_items)
        neg_scores = inner_product(users, neg_items)

        regularizer = l2_loss(self.u_g_embeddings_pre, self.pos_i_g_embeddings_pre, self.neg_i_g_embeddings_pre)

        mf_loss = tf.reduce_sum(log_loss(pos_scores - neg_scores))

        emb_loss = self.reg * regularizer

        return mf_loss, emb_loss

    def train_model(self):
        if self.neg_dist == 2 and self.num_negatives > 1:
            data_iter = PairwiseSampler(self.dataset, neg_num=self.num_negatives, batch_size=self.batch_size, shuffle=True)
        else:
            data_iter = PointwiseSamplerV2(self.dataset, batch_size=self.batch_size, shuffle=True)

        self.logger.info(self.evaluator.metrics_info())
        if self.pretrain:
            buf, _ = self.evaluate()
            self.logger.info("\t\t%s" % buf)
        stopping_step = 0
        for epoch in range(1, self.epochs + 1):
            total_loss, total_emb_loss = 0.0, 0.0
            training_start_time = time()
            if self.neg_dist == 2 and self.num_negatives > 1:
                for bat_users, bat_pos_items, bat_neg_items in data_iter:
                    feed_dict = {self.users: bat_users,
                                 self.pos_items: bat_pos_items,
                                 self.neg_items: bat_neg_items}
                    loss, emb_loss, _ = self.sess.run((self.loss, self.emb_loss, self.opt), feed_dict=feed_dict)
                    total_loss += loss
                    total_emb_loss += emb_loss
            else:
                for bat_users, bat_pos_items in data_iter:
                    feed_dict = {self.users: bat_users,
                                self.pos_items: bat_pos_items,}
                    if self.neg_dist == 1:
                        bat_neg_items = randint_choice(self.n_items, self.num_negatives, p=self.item_popularity)
                        feed_dict.update({self.neg_items: bat_neg_items})
                    if self.neg_dist == 2:
                        bat_neg_items = randint_choice(self.n_items, len(bat_users) * self.num_negatives, p=self.item_popularity)
                        if self.num_negatives > 1:
                            bat_neg_items = np.reshape(bat_neg_items, [len(bat_users), self.num_negatives])
                        feed_dict.update({self.neg_items: bat_neg_items})
                    loss, emb_loss, _ = self.sess.run((self.loss, self.emb_loss, self.opt), feed_dict=feed_dict)
                    total_loss += loss
                    total_emb_loss += emb_loss
            self.logger.info("[iter %d : loss : %.4f = %.4f + %.4f, time: %f]" % (
                epoch, 
                total_loss/data_iter.num_trainings,
                (total_loss - total_emb_loss) / data_iter.num_trainings,
                total_emb_loss / data_iter.num_trainings,
                time()-training_start_time,))
            if epoch % self.verbose == 0 and epoch > self.conf['start_testing_epoch']:
                buf, flag = self.evaluate()
                self.logger.info("epoch %d:\t%s" % (epoch, buf))
                if flag:
                    self.best_epoch = epoch
                    stopping_step = 0
                    self.logger.info("Find a better model.")
                    if self.save_flag:
                        self.logger.info("Save model to file as pretrain.")
                        self.saver.save(self.sess, self.tmp_model_folder)
                else:
                    stopping_step += 1
                    if stopping_step >= self.stop_cnt:
                        self.logger.info("Early stopping is trigger at epoch: {}".format(epoch))
                        break

        self.logger.info("best_result@epoch %d:\n" % self.best_epoch)
        if self.save_flag:
            self.logger.info('Loading from the saved model.')
            self.saver.restore(self.sess, self.tmp_model_folder)
            uebd, iebd = self.sess.run([self.weights['user_embedding'], self.weights['item_embedding']])
            np.save(self.save_folder + 'user_embeddings.npy', uebd)
            np.save(self.save_folder + 'item_embeddings.npy', iebd)
            buf, _ = self.evaluate()
        elif self.pretrain:
            buf, _ = self.evaluate()
        else:
            buf = '\t'.join([("%.4f" % x).ljust(12) for x in self.best_result])
        self.logger.info("\t\t%s" % buf)

    # @timer
    def evaluate(self):
        self._cur_user_embeddings, self._cur_item_embeddings = self.sess.run([self.ua_embeddings, self.ia_embeddings])
        if self.flag_l2_norm[1]:
            self._cur_user_embeddings = self._cur_user_embeddings / np.linalg.norm(self._cur_user_embeddings, axis=-1, keepdims=True)
            self._cur_item_embeddings = self._cur_item_embeddings / np.linalg.norm(self._cur_item_embeddings, axis=-1, keepdims=True)

        flag = False
        current_result, buf = self.evaluator.evaluate(self)
        if self.best_result[1] < current_result[1]:
            self.best_result = current_result
            flag = True
        return buf, flag

    def predict(self, user_ids, candidate_items=None):
        if candidate_items is None:
            user_embed = self._cur_user_embeddings[user_ids]
            ratings = np.matmul(user_embed, self._cur_item_embeddings.T)
        else:
            ratings = []
            user_embed = self._cur_user_embeddings[user_ids]
            items_embed = self._cur_item_embeddings[candidate_items]
            ratings = np.sum(np.multiply(user_embed, items_embed), 1)
        return ratings
