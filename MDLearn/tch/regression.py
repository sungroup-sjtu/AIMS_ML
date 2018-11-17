
import torch.nn
import torch.utils.data


class MLPRegressor:

    def __init__(self, insize, outsize, layers, batch_size=None, batch_step=None, args_layer=dict(), args_opt=dict(), is_gpu=False):

        self.layers = layers
        self.batch_step = batch_step
        self.batch_size = batch_size
        self.insize = insize 
        self.outsize = outsize 
        self.layers = layers 
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
                layers.append(torch.nn.Linear(self.layers[i-1], self.layers[i]))
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

    def fit_epoch(self, verbose_step=0):

        if verbose_step != 0:
            print('step\tloss')

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

    def partial_fit(self, steps, x, y_ref):


        v_x = torch.autograd.Variable(torch.Tensor(x))
        v_y = torch.autograd.Variable(torch.Tensor(y_ref))

        if self.is_gpu:
            v_x, v_y = v_x.cuda(), v_y.cuda()

        for i in range(steps):

            self.optimizer.zero_grad()
            loss = self.loss(self.regressor(v_x), v_y)
            loss.backward()
            self.optimizer.step()

    def predict(self, x):

        v_x = torch.autograd.Variable(torch.Tensor(x))

        if self.is_gpu:
            v_x = v_x.cuda(async=True)
            return self.regressor(v_x).data.cpu().numpy()
        else:
            return self.regressor(v_x).data.numpy()

    def save(self, filename):
        torch.save(self.regressor, filename)

    def load(self, filename):
        if self.is_gpu:
            self.regressor = torch.load(filename)
            self.regressor.cuda()
        else:
            self.regressor = torch.load(filename, lambda storage,loc:storage)
