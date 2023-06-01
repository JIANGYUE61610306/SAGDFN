import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from lib import utils
from model.pytorch.model import SAGDFNModel
from model.pytorch.loss import masked_mae_loss, masked_mape_loss, masked_rmse_loss, masked_mse_loss
import pandas as pd
import os
import time
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class SAGDFNSupervisor:
    def __init__(self, save_adj_name, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self.opt = self._train_kwargs.get('optimizer')
        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)
        self.threshold = self._model_kwargs.get('threshold', 0.5)  
        self.emb_dim = self._model_kwargs.get('emb_dim')
        self.save_adj_name = save_adj_name
        self.num_sample = self._train_kwargs.get('num_sample')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter('runs/' + self._log_dir)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']


        # print('Initial, before loading data')
        ### Feas
        if self._data_kwargs['dataset_dir'] == 'data/METR-LA':
            df = pd.read_hdf('./data/metr-la.h5')
        elif self._data_kwargs['dataset_dir'] == 'data/PEMS-BAY':
            df = pd.read_hdf('./data/pems-bay.h5')
        elif self._data_kwargs['dataset_dir'] == 'data/CARPARK':
            df = pd.read_hdf('./data/carpark_05_06.h5')
        else:
            df = pd.read_hdf('./data/carpark_05_06.h5')
        df_s = df
        max_map = df_s.iloc[:,:].max()
        max_value = max_map.values
        self.max_value = max_value


        num_samples = df.shape[1]
        # num_train = round(num_samples * 0.7)
        # df = df[:num_train].values
        # np.random.seed(101)
        df = np.random.rand(num_samples,self.emb_dim) #(# of nodes, embedding dimensions)


        scaler = utils.StandardScaler(mean=df.mean(), std=df.std())
        train_feas = scaler.transform(df)
        self._train_feas = torch.Tensor(train_feas).to(device)
        print(self._train_feas.shape)

        k = self._train_kwargs.get('knn_k')
        knn_metric = 'cosine'
        from sklearn.neighbors import kneighbors_graph
        g = kneighbors_graph(train_feas.T, k, metric=knn_metric)
        g = np.array(g.todense(), dtype=np.float32)
        self.adj_mx = torch.Tensor(g).to(device)
        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder


        # print('Before setup model')
        # setup model
        SAGDFN_model = SAGDFNModel(self._logger, **self._model_kwargs)
        # print('before cuda')
        self.SAGDFN_model = SAGDFN_model.to(device) if torch.cuda.is_available() else SAGDFN_model
        self._logger.info("Model created with threshold: {}".format(self.threshold))

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
        
            self.load_model()
        # print(self._epoch_num )
        # print('After setup model')
    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'SAGDFN_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.SAGDFN_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self):
        self._setup_graph()
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.SAGDFN_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.SAGDFN_model = self.SAGDFN_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                # print(x.shape, y.shape,self._train_feas.shape)
                output = self.SAGDFN_model(x, self._train_feas)
                print(x.shape, y.shape,self._train_feas.shape)
                break

    def train(self, **kwargs):
        # print('Into the train')
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.SAGDFN_model = self.SAGDFN_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            mapes = []
            #rmses = []
            mses = []
            
            l_3 = []
            m_3 = []
            r_3 = []
            l_6 = []
            m_6 = []
            r_6 = []
            l_12 = []
            m_12 = []
            r_12 = []

            for batch_idx, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output, mid_output, adj_save = self.SAGDFN_model(x, self._train_feas, y, 100*50, 100*50)

                loss = self._compute_loss(y, output, torch.Tensor(self.max_value).to(device))
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)
                mapes.append(masked_mape_loss(y_pred, y_true).item())
                mses.append(masked_mse_loss(y_pred, y_true).item())
                #rmses.append(masked_rmse_loss(y_pred, y_true).item())
                losses.append(loss.item())
                
                
                # Followed the DCRNN TensorFlow Implementation
                l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
                m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
                r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
                l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
                m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
                r_6.append(masked_mse_loss(y_pred[5:6], y_true[5:6]).item())
                l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
                m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
                r_12.append(masked_mse_loss(y_pred[11:12], y_true[11:12]).item())
                    

                
            mean_loss = np.mean(losses)
            mean_mape = np.mean(mapes)
            mean_rmse = np.sqrt(np.mean(mses))
            # mean_rmse = np.mean(rmses) #another option
            
            if dataset == 'test':
                
                # Followed the DCRNN PyTorch Implementation
                message = 'Test: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_loss, mean_mape, mean_rmse)
                self._logger.info(message)
                
                # Followed the DCRNN TensorFlow Implementation
                message = 'Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_3), np.mean(m_3),
                                                                                           np.sqrt(np.mean(r_3)))
                self._logger.info(message)
                message = 'Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_6), np.mean(m_6),
                                                                                           np.sqrt(np.mean(r_6)))
                self._logger.info(message)
                message = 'Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_12), np.mean(m_12),
                                                                                           np.sqrt(np.mean(r_12)))
                self._logger.info(message)

            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            return mean_loss, mean_mape, mean_rmse



    def predict(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.SAGDFN_model = self.SAGDFN_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            mapes = []
            #rmses = []
            mses = []

            
            l_3 = []
            m_3 = []
            r_3 = []
            l_6 = []
            m_6 = []
            r_6 = []
            l_12 = []
            m_12 = []
            r_12 = []

            y_truths = []
            y_preds = []
            # start_time = time.time()
            for batch_idx, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output, mid_output, adj_save = self.SAGDFN_model(x, self._train_feas, y, 100*50, 100*50)

                loss = self._compute_loss(y, output, torch.Tensor(self.max_value).to(device))
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)
                mapes.append(masked_mape_loss(y_pred, y_true).item())
                mses.append(masked_mse_loss(y_pred, y_true).item())
                #rmses.append(masked_rmse_loss(y_pred, y_true).item())
                losses.append(loss.item())
                
                y_truths.append(y_true.cpu())
                y_preds.append(y_pred.cpu())
                
                # Followed the DCRNN TensorFlow Implementation
                l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
                m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
                r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
                l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
                m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
                r_6.append(masked_mse_loss(y_pred[5:6], y_true[5:6]).item())
                l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
                m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
                r_12.append(masked_mse_loss(y_pred[11:12], y_true[11:12]).item())

                    

                

            # end_time = time.time()
            # message = 'Test consuming time: {:.1f}s'.format(end_time - start_time)
            # self._logger.info(message)
            mean_loss = np.mean(losses)
            mean_mape = np.mean(mapes)
            mean_rmse = np.sqrt(np.mean(mses))
            # mean_rmse = np.mean(rmses) #another option
            
            y_preds=np.concatenate(y_preds, axis =1)
            y_truths = np.concatenate(y_truths, axis=1)
            
            if dataset in ['test', 'test1', 'test2', 'test3', 'test4']:

                # ### Change the Y_hat and Y dir
                if self._data_kwargs['dataset_dir'] == 'data/CARPARK':
                    np.savez_compressed('./data/y_preds_carpark.npz', y_preds)
                    np.savez_compressed('./data/y_truths_carpark.npz', y_truths)
                # # np.savez_compressed('./data/adj_save.npz', adj_save.cpu())
                    error = y_preds-y_truths

                    # message = 'Test dateset is {}, shape is {}'.format(dataset, y_preds.shape)
                    # self._logger.info(message)


                    error_tensor = torch.from_numpy(error)
                    error_ratio = torch.mean(abs(error_tensor), [1]).cpu().numpy()

                    # df_s = pd.read_hdf('/home_nfs/jiangyue/github/SAGDFN_test/data/carpark_full.h5')
                    # # df_s = pd.read_hdf('/home/user1/data/carpark/DCRNN/carpark_full_DCRNN_type.h5')
                    # max_map = df_s.iloc[:,:].max()
                    # max_value = max_map.values
                    error_scaled = error_ratio*self.max_value
                    a = np.mean(error_scaled,axis=1)
                    message = 'Test MAE for 15 mins 30 mins 60mins prediction: {:.4f}, {:.4f}, {:.4f}'.format(a[2], a[5], a[11])
                    self._logger.info(message)
                if self._data_kwargs['dataset_dir'] == 'data/METR-LA':
                ### Change the Y_hat and Y dir
                    np.savez_compressed('./data/y_preds_METR.npz', y_preds)
                    np.savez_compressed('./data/y_truths_METR.npz', y_truths)
                count = 0
                adj_map = adj_save.cpu()
                # if adj_map.shape[0] == adj_map.shape[1]:
                for t in range(adj_map.shape[0]):
                    for k in range(adj_map.shape[1]):
                        if adj_map[t][k]>self.threshold:
                            count+=1 
                # else:
                #     for t in range(adj_map.shape[0]):
                #         if adj_map[t][0]>self.threshold:
                #                 count+=1 
                # message = 'Total edges in the final adj matrix: {}'.format(count)
                # self._logger.info(message)

                # Followed the DCRNN PyTorch Implementation
                message = 'Test: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_loss, mean_mape, mean_rmse)
                self._logger.info(message)
                
                # Followed the DCRNN TensorFlow Implementation
                message = 'Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_3), np.mean(m_3),
                                                                                           np.sqrt(np.mean(r_3)))
                self._logger.info(message)
                message = 'Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_6), np.mean(m_6),
                                                                                           np.sqrt(np.mean(r_6)))
                self._logger.info(message)
                message = 'Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_12), np.mean(m_12),
                                                                                           np.sqrt(np.mean(r_12)))
                self._logger.info(message)

            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            return mean_loss, mean_mape, mean_rmse



    def _train(self, base_lr,
               steps, patience=200, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=0,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        if self.opt == 'adam':
            optimizer = torch.optim.Adam(self.SAGDFN_model.parameters(), lr=base_lr, eps=epsilon)
        elif self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.SAGDFN_model.parameters(), lr=base_lr)
        else:
            optimizer = torch.optim.Adam(self.SAGDFN_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=float(lr_decay_ratio))

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        loss_hist = []

        for epoch_num in range(self._epoch_num, epochs):
            print("Num of epoch:",epoch_num)
            self.SAGDFN_model = self.SAGDFN_model.train()
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            start_time = time.time()


            for batch_idx, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()
                x, y = self._prepare_data(x, y)
                output, mid_output, adj_save = self.SAGDFN_model(x, self._train_feas, y, batches_seen, batch_idx)
                if (epoch_num % epochs) == epochs - 1:
                    output, mid_output, adj_save = self.SAGDFN_model(x, self._train_feas, y, batches_seen)

                if batches_seen == 0:
                    if self.opt == 'adam':
                        optimizer = torch.optim.Adam(self.SAGDFN_model.parameters(), lr=base_lr, eps=epsilon)
                    elif self.opt == 'sgd':
                        optimizer = torch.optim.SGD(self.SAGDFN_model.parameters(), lr=base_lr)
                    else:
                        optimizer = torch.optim.Adam(self.SAGDFN_model.parameters(), lr=base_lr, eps=epsilon)

                self.SAGDFN_model.to(device)
                


                loss = self._compute_loss(y, output, torch.Tensor(self.max_value).to(device))
                losses.append(loss.item())
                # print('loss shape is: ', loss)
                
                self._logger.debug(loss.item())
                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.SAGDFN_model.parameters(), self.max_grad_norm)

                optimizer.step()
            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")
            end_time = time.time()

            val_loss, val_mape, val_rmse = self.evaluate(dataset='val', batches_seen=batches_seen)
            end_time2 = time.time()
            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, val_mape: {:.4f}, val_rmse: {:.4f}, lr: {:.6f}, ' \
                            '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                    np.mean(losses), val_loss, val_mape, val_rmse,
                                                    lr_scheduler.get_lr()[0],
                                                    (end_time - start_time), (end_time2 - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, test_mape, test_rmse = self.evaluate(dataset='test', batches_seen=batches_seen)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f}, test_mape: {:.4f}, test_rmse: {:.4f}, lr: {:.6f}, ' \
                            '{:.1f}s, {:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                    np.mean(losses), test_loss, test_mape, test_rmse,
                                                    lr_scheduler.get_lr()[0],
                                                    (end_time - start_time), (end_time2 - start_time))
                self._logger.info(message)
            
            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

                test_loss, test_mape, test_rmse = self.predict(dataset='test', batches_seen=batches_seen)
                end_time3 = time.time()
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f}, test_mape: {:.4f}, test_rmse: {:.4f}, lr: {:.6f}, ' \
                            'validation time{:.1f}s, testing time{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                                    np.mean(losses), test_loss, test_mape, test_rmse,
                                                    lr_scheduler.get_lr()[0],
                                                    (end_time2 - end_time), (end_time3 - end_time2))
                self._logger.info(message)
                
            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break
            loss_hist.append(val_loss)
            np.savez_compressed('./data/loss_hist_full.npz', loss_hist = loss_hist)
    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted, max_value):
        if max_value.shape[0]==1918:
            y_true = self.standard_scaler.inverse_transform(y_true)
            y_true = y_true*max_value
            # print('y_true shape is: ', y_true.shape)
            y_predicted = self.standard_scaler.inverse_transform(y_predicted)
            y_predicted = y_predicted*max_value
            # print('y_predicted shape is: ', y_predicted.shape)
            return masked_mae_loss(y_predicted, y_true)/182
        else:
            y_true = self.standard_scaler.inverse_transform(y_true)
            y_predicted = self.standard_scaler.inverse_transform(y_predicted)
            return masked_mae_loss(y_predicted, y_true)