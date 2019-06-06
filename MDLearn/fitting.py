import torch.nn
import torch.utils.data


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
        act_class = self.args_layer.pop('activator', torch.nn.ELU)
        loss_class = self.args_opt.pop('loss', torch.nn.MSELoss)
        opt_class = self.args_opt.pop('optimizer', torch.optim.Adam)

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
        self.optimizer = opt_class(self.regressor.parameters(), **self.args_opt)

    def load_data(self, x, y_ref):
        self.dataset = torch.utils.data.TensorDataset(torch.Tensor(x), torch.Tensor(y_ref))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def fit_epoch(self):
        cnt = 0
        for i, (x, y) in enumerate(self.dataloader):
            v_x = torch.autograd.Variable(torch.Tensor(x))
            v_y = torch.autograd.Variable(torch.Tensor(y))

            if self.is_gpu:
                v_x, v_y = v_x.cuda(async=True), v_y.cuda(async=True)

            for j in range(self.batch_step):
                self.optimizer.zero_grad()
                loss = self.loss(self.regressor(v_x), v_y)
                loss.backward()
                self.optimizer.step()

            cnt += 1

        loss_data = loss.data.numpy() if not self.is_gpu else loss.data.cpu().numpy()

        return self.batch_step * cnt, loss_data

    def predict(self, x):
        v_x = torch.autograd.Variable(torch.Tensor(x))

        if self.is_gpu:
            v_x = v_x.cuda(async=True)
            v_y = self.regressor(v_x).data.cpu().numpy()
        else:
            v_y = self.regressor(v_x).data.numpy()

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
