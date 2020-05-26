import torch
import torch.nn as nn


class basic_model(nn.Module):
    def __init__(self, config):
        super(basic_model, self).__init__()

        self.batch_size = config.data.batch_size
        self.input_fc_dim = config.data.input_fc_dim
        self.input_cnn_dim = config.data.input_cnn_dim
        self.output_dim = config.data.output_dim

        self.hidden_dim = config.model.hidden_dim        
        self.activation_func = activation[config.model.activation_func]
        self.activation = self.activation_func()
        self.num_layers = config.model.num_layers
        
        

    def forward(self):
        raise NotImplementedError


class basic_MLP(basic_model):
    def __init__(self, config):
        super(basic_MLP, self).__init__(config)

        self.backbone = nn.Sequential(
            nn.Linear(self.input_fc_dim, self.hidden_dim),
            self.activation_func(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            self.activation_func(),
        )

    def forward(self, x):
        return self.backbone(x)

class basic_CNN(basic_model):
    def __init__(self, config):
        super(basic_CNN, self).__init__(config)

        self.backbone = nn.Sequential(
            nn.Conv2d(self.input_cnn_dim[0], self.hidden_dim, 7, 2),
            self.activation_func(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1),
            self.activation_func(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1),
            self.activation_func(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.backbone(x)

class Encoder(nn.Module):

    def __init__(self, cfg):
        super(Encoder, self).__init__()

        self.cfg = cfg
        self.batch_size = cfg.data.batch_size
        self.input_fc_dim = cfg.data.input_fc_dim
        self.latent_dim = cfg.data.latent_dim

        self.backbone_fc = backbone_fc[cfg.model.backbone_fc](cfg)
        self.backbone_cnn = backbone_cnn[cfg.model.backbone_cnn](cfg)


        self.linear_means = nn.Linear(self.hidden_dim+conv_out_dim, self.latent_dim)
        self.linear_log_var = nn.Linear(self.hidden_dim+conv_out_dim, self.latent_dim)
    def forward(self, state, occupancy, depth):

        fc_in = torch.cat((state, depth), 1)
        fc_out = self.backbone_fc(fc_in)

        cnn_out = self.backbone_cnn(occupancy)
        cnn_out = cnn_out.view(self.batch_size,-1)
        out = torch.cat((fc_out, cnn_out), 1)

        means = self.linear_means(out)
        log_vars = self.linear_log_var(out)

        return means, log_vars

class Decoder(nn.Module):

    def __init__(self, cfg):

        super(Decoder, self).__init__()

        self.cfg = cfg
        self.latent_dim = cfg.data.latent_dim

        self.backbone_fc = backbone_fc[cfg.model.backbone_fc](cfg)

        self.fc_dim = self.latent_dim + self.input_fc_dim

    def forward(self, z, c):

        return x


class CVAE(nn.Module):

    def __init__(self, cfg):
        super(CVAE, self).__init__()
        # some hypeparameters
        self.cfg = cfg
        self.batch_size = cfg.data.batch_size
        self.input_dim = cfg.data.depth_dim + cfg.data.state_dim

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, state, occupancy, depth):

        means, log_var = self.encoder(state, occupancy, depth)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, self.latent_dim])
        z = eps * std + means

        recon_state = self.decoder(z, occupancy, depth)

        return recon_state, means, log_var, z

    def inference(self, n=1, occupancy, depth):

        batch_size = n
        z = torch.randn([batch_size, self.latent_dim])

        recon_x = self.decoder(z, occupancy, depth)

        return recon_x


backbone_fc = {
    "mlp": basic_MLP,
}

backbone_cnn = {
    "basic": basic_CNN,
}

activation = {
    "tanh": nn.Tanh,
    "sigm": nn.Sigmoid,
    "relu": nn.ReLU,
}