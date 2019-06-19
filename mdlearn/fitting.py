import torch.nn
import torch.utils.data
import numpy as np

class TorchMLPRegressor():
    def __init__(self, insize, outsize, layers, batch_size=None, batch_step=None,
                 args_layer=dict(), args_opt=dict(), is_gpu=True):
        self.insize = insize
        self.outsize = outsize
        self.layers = layers
        self.batch_size = batch_size
        self.batch_step = batch_step
        self.args_layer = args_layer
        self.args_opt = args_opt
        self.is_gpu = is_gpu

    def init_session(self):
        act_class = self.args_layer.pop('activator', torch.nn.SELU)
        loss_class = self.args_opt.pop('loss', torch.nn.MSELoss)

        def init_layer(layer):
            torch.nn.init.normal(layer.weight, std=0.5)
            torch.nn.init.normal(layer.bias, mean=0.01, std=0.001)

        layers = []
        if not self.layers:
            layers.append(torch.nn.Linear(self.insize, self.outsize))
            init_layer(layers[-1])
        else:
            layers.append(torch.nn.Linear(self.insize, self.layers[0]))
            init_layer(layers[-1])
            layers.append(act_class())
            for i in range(1, len(self.layers)):
                layers.append(torch.nn.Linear(self.layers[i - 1], self.layers[i]))
                init_layer(layers[-1])
                layers.append(act_class())
            layers.append(torch.nn.Linear(self.layers[-1], self.outsize))
            init_layer(layers[-1])

        self.regressor = torch.nn.Sequential(*layers)

        if self.is_gpu:
            self.regressor.cuda()

        self.loss = loss_class()
        


    def reset_optimizer(self, optim_dict):
        optim_class = optim_dict['optimizer']
        self.optimizer = optim_class( self.regressor.parameters(), lr=optim_dict['lr'], weight_decay=optim_dict['weight_decay'] )

    def load_data(self, x, y_ref):
        self.dataset = torch.utils.data.TensorDataset(x, y_ref)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def fit_epoch(self,x, y):
        cnt = 0
        total_step = int( len(x) / self.batch_size )
        for i in range(total_step+1):
            self.optimizer.zero_grad()
            if i == total_step:
                x_batch = x[-self.batch_size:]
                y_batch = y[-self.batch_size:]
            else:
                x_batch = x[self.batch_size*i:self.batch_size*(i+1)]
                y_batch = y[self.batch_size*i:self.batch_size*(i+1)]
            loss = self.loss(self.regressor(x_batch), y_batch)
            loss.backward()
            self.optimizer.step()

            cnt += 1

        loss_data = loss.data.numpy() if not self.is_gpu else loss.data.cpu().numpy()

        return cnt, loss_data

    def predict(self, x):
        if type(x)==type(np.array(1)):
            x = torch.Tensor(x)
        if self.is_gpu:
            v_y = self.regressor(x).data.cpu().numpy()
        else:
            v_y = self.regressor(x).data.numpy()
        return v_y

    def predict_batch(self, batch_x):
        """ Predict a lot of data at the same time.
        """
        return self.predict(batch_x)

    def save(self, filename):
        torch.save(self.regressor, filename)

    def load(self, filename):
        if self.is_gpu:
            self.regressor = torch.load(filename)
            self.regressor.cuda()
        else:
            self.regressor = torch.load(filename, lambda storage, loc: storage)
