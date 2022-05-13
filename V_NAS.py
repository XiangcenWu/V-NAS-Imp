import torch
import torch.nn as nn
from torch.nn.functional import interpolate


def get_params_to_update(model):
    """ Returns list of model parameters that have required_grad=True"""
    params_to_update = []
    for param in model.parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    return params_to_update


class Encoder(nn.Module):
    
    
    def __init__(self, in_c, out_c):
        super().__init__()

        # Block One
        self.block_2d = nn.Sequential(
            nn.Conv3d(in_c, in_c, (3, 3, 1), (1, 1, 1), (1, 1, 0)),
            nn.Conv3d(in_c, out_c, 1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
        )
        # Block Two
        self.block_3d = nn.Sequential(
            nn.Conv3d(in_c, in_c, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.Conv3d(in_c, out_c, 1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
        )
        # Block three
        self.block_p3d = nn.Sequential(
            nn.Conv3d(in_c, in_c, (3, 3, 1), (1, 1, 1), (1, 1, 0)),
            nn.Conv3d(in_c, out_c, (1, 1, 3), (1, 1, 1), (0, 0, 1)),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
        )
        # softmax for alpha
        self.relax = nn.Softmax(dim=0)
        # alpha parameters
        self.alpha = nn.Parameter(torch.zeros((3, )))


    def forward(self, x):
        a1, a2, a3 = self.relax(self.alpha)
        x1 = x + self.block_2d(x)
        x2 = x + self.block_3d(x)
        x3 = x + self.block_p3d(x)
        return a1*x1 + a2*x2 + a3*x3


    def update_alpha(self):
        # set the model parameters requires grad=False
        for paras in self.parameters():
            paras.requires_grad_(False)
        # set the alpha requires grad=True
        self.alpha.requires_grad_(True)


    def update_w(self):
        # set the model parameters requires grad=False
        for paras in self.parameters():
            paras.requires_grad_(True)
        # set the alpha requires grad=True
        self.alpha.requires_grad_(False)





class Decoder(nn.Module):
    # different from encoder which you can choose the input/output num of channel 
    # This decoder will not affect the num of channel to the input tensor
    
    def __init__(self, in_c, out_c):
        super().__init__()

        # output channel should be an even number
        # out_c is not used, add out_c as a input for debugging
        assert out_c == in_c
        c = int(in_c/2)
        
        # 2D
        self.block_2d_left = nn.Sequential(
            nn.Conv3d(in_c, c, 1, 1),
            nn.Conv3d(c, c, (3, 3, 1), (1, 1, 1), (1, 1, 0)),
            nn.Conv3d(c, c, 1, 1),
            nn.BatchNorm3d(c),
            nn.ReLU(),
        )
        self.block_2d_right = nn.Sequential(
            nn.Conv3d(c, c, 1, 1),
            nn.Conv3d(c, c, 1, 1),
            nn.Conv3d(c, c, 1, 1),
            nn.BatchNorm3d(c),
            nn.ReLU(),
        )
        # 3D
        self.block_3d_left = nn.Sequential(
            nn.Conv3d(in_c, c, 1, 1),
            nn.Conv3d(c, c, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.Conv3d(c, c, 1, 1),
            nn.BatchNorm3d(c),
            nn.ReLU(),
        )
        self.block_3d_right = nn.Sequential(
            nn.Conv3d(c, c, 1, 1),
            nn.Conv3d(c, c, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.Conv3d(c, c, 1, 1),
            nn.BatchNorm3d(c),
            nn.ReLU(),
        )
        # P3D
        self.block_p3d_left = nn.Sequential(
            nn.Conv3d(in_c, c, 1, 1),
            nn.Conv3d(c, c, (3, 3, 1), (1, 1, 1), (1, 1, 0)),
            nn.Conv3d(c, c, 1, 1),
            nn.BatchNorm3d(c),
            nn.ReLU(),
        )
        self.block_p3d_right = nn.Sequential(
            nn.Conv3d(c, c, 1, 1),
            nn.Conv3d(c, c, (1, 1, 3), (1, 1, 1), (0, 0, 1)),
            nn.Conv3d(c, c, 1, 1),
            nn.BatchNorm3d(c),
            nn.ReLU(),
        )
        # softmax for alpha
        self.relax = nn.Softmax(dim=0)
        # alpha parameters
        self.alpha = nn.Parameter(torch.zeros((3, )))

    def _cell_forward(self, x, left_mini_block, right_mini_block):
        x1 = left_mini_block(x)
        x2 = right_mini_block(x1)

        o_add = torch.cat((x1, x2), dim=1)
        # concat at the channel dim
        return x + o_add

    def forward(self, x):
        # 
        a1, a2, a3 = self.relax(self.alpha)

        x1 = self._cell_forward(x, self.block_2d_left, self.block_2d_right)
        x2 = self._cell_forward(x, self.block_3d_left, self.block_3d_right)
        x3 = self._cell_forward(x, self.block_p3d_left, self.block_p3d_right)
        return a1*x1 + a2*x2 + a3*x3

    def update_alpha(self):
        # set the model parameters requires grad=False
        for paras in self.parameters():
            paras.requires_grad_(False)
        # set the alpha requires grad=True
        self.alpha.requires_grad_(True)

    def update_w(self):
        # set the model parameters requires grad=False
        for paras in self.parameters():
            paras.requires_grad_(True)
        # set the alpha requires grad=True
        self.alpha.requires_grad_(False)


class Network(nn.Module):
    
    
    def __init__(self, out_channel):
        super().__init__()
        # input = 1 64**3
        # output = 24 32**3
        self.cmd_1 = nn.Sequential(
            nn.Conv3d(1, 24, (5, 5, 1), (2, 2, 1), (2, 2, 0)),
            nn.MaxPool3d((1, 1, 2)),
            nn.Conv3d(24, 24, 1, 1)
        )
        # input = 24 32**3
        # output = 24 32**3
        self.Encoder_1 = nn.Sequential(*self.init_encoder(24, 3))
        # input = 24 32**3
        # output = 30 16**3
        self.cmd_2 = nn.Sequential(
            nn.Conv3d(24, 24, (1, 1, 1), (2, 2, 1), (0, 0, 0)),
            nn.MaxPool3d((1, 1, 2), (1, 1, 2)),
            nn.Conv3d(24, 30, 1, 1)
        )
        # input = 30 16**3
        # output = 30 16**3
        self.Encoder_2 = nn.Sequential(*self.init_encoder(30, 4))
        # input = 30 16**3
        # output = 32 8**3
        self.cmd_3 = nn.Sequential(
            nn.Conv3d(30, 30, (1, 1, 1), (2, 2, 1), (0, 0, 0)),
            nn.MaxPool3d((1, 1, 2), (1, 1, 2)),
            nn.Conv3d(30, 32, 1, 1)
        )
        # input = 32 8**3
        # output = 32 8**3
        self.Encoder_3 = nn.Sequential(*self.init_encoder(32, 6))
        # input = 32 8**3
        # output = 64 4**3
        self.cmd_4 = nn.Sequential(
            nn.Conv3d(32, 32, (1, 1, 1), (2, 2, 1), (0, 0, 0)),
            nn.MaxPool3d((1, 1, 2), (1, 1, 2)),
            nn.Conv3d(32, 64, 1, 1)
        )
        # input = 64 4**3
        # output = 64 4**3
        self.Encoder_4 = nn.Sequential(*self.init_encoder(64, 3))

        #################################Decoder#############################
        self.up_1 = nn.Sequential(
            nn.Conv3d(64, 32, 1, 1),
            nn.Upsample((8, 8, 8), mode='trilinear')
        )

        self.Decoder_1 = nn.Sequential(*self.init_decoder(32))

        self.up_2 = nn.Sequential(
            nn.Conv3d(32, 30, 1, 1),
            nn.Upsample((16, 16, 16), mode='trilinear')
        )

        self.Decoder_2 = nn.Sequential(*self.init_decoder(30))

        self.up_3 = nn.Sequential(
            nn.Conv3d(30, 24, 1, 1),
            nn.Upsample((32, 32, 32), mode='trilinear')
        )

        self.Decoder_3 = nn.Sequential(*self.init_decoder(24))
        self.Decoder_4 = nn.Sequential(*self.init_decoder(24))

        self.up_4 = nn.Sequential(
            nn.Conv3d(24, out_channel, 1, 1),
            nn.Upsample((64, 64, 64), mode='trilinear')
        )

        

        self.cells = [
            self.Encoder_1,
            self.Encoder_2,
            self.Encoder_3,
            self.Encoder_4,
            self.Decoder_1,
            self.Decoder_2,
            self.Decoder_3,
            self.Decoder_4,
        ]
         
    

    def forward(self, x):
        # Skip connection has not been written yet
        x1 = self.cmd_1(x)
        x2 = self.Encoder_1(x1)
        x3 = self.Encoder_2(self.cmd_2(x2))
        x4 = self.Encoder_3(self.cmd_3(x3))
        

        x5 = self.Encoder_4(self.cmd_4(x4))

        x4 = x4 + self.Decoder_1(self.up_1(x5))
        x3 = x3 + self.Decoder_2(self.up_2(x4))
        x2 = x2 + self.Decoder_3(self.up_3(x3))
        x1 = x1 + self.Decoder_4(x2)
        
        
        return self.up_4(x1)


    def update_alpha(self):
        for cell_outer in self.cells:
            for cell in cell_outer:
                cell.update_alpha()


    def update_w(self):
        for cell_outer in self.cells:
            for cell in cell_outer:
                cell.update_w()

    def init_decoder(self, num_features):
        decoder = []
        for _ in range(3):
            decoder.append(Decoder(num_features, num_features))
        return decoder

    def init_encoder(self, num_features, num_encoder):
        encoder = []
        for _ in range(num_encoder):
            encoder.append(Encoder(num_features, num_features))
        return encoder

    def log(self):
        for i in self.cells:
            for j in i:
                print(j.alpha.data)



def check():


    # model = Encoder(3, 3)
    # x = torch.rand(2, 3, 64, 64, 64)
    # # set the weight optimizer
    # model.update_w()
    # w_opt = torch.optim.SGD(get_params_to_update(model), lr=1)

    # # set the alpha optimizer
    # model.update_alpha()
    # a_opt = torch.optim.SGD(get_params_to_update(model), lr=1)



    # model.update_alpha()
    # o = model(x).sum()
    # o.backward()

    # print(model.alpha)
    # print(model.block_3d[0].weight.data)
    # a_opt.step()
    # print(model.block_3d[0].weight.data)
    # print(model.alpha)


    # print('******************************')
    # model.update_w()
    # print(model.alpha)
    # print(model.block_3d[0].weight.data)

    # o = model(x).sum()
    # o.backward()
    # w_opt.step()
    # print(model.alpha)
    # print(model.block_3d[0].weight.data)


    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
    x = torch.rand(2, 1, 64, 64, 64)
    model = Network(2)



    model.update_w()
    w_opt = torch.optim.SGD(get_params_to_update(model), lr=0.00001)

    # set the alpha optimizer
    model.update_alpha()
    a_opt = torch.optim.SGD(get_params_to_update(model), lr=0.011)


    model.update_alpha()
    o = model(x).sum()
    o.backward()

    print(model.Encoder_1[0].alpha)
    print(model.Encoder_1[0].block_2d[0].weight.data[0, 0, 0])
    a_opt.step()
    print(model.Encoder_1[0].block_2d[0].weight.data[0, 0, 0])
    print(model.Encoder_1[0].alpha)


    print('******************************')

    x = torch.rand(2, 1, 64, 64, 64)
    model = Network(2)



    model.update_w()
    w_opt = torch.optim.SGD(get_params_to_update(model), lr=0.1)

    # set the alpha optimizer
    model.update_alpha()
    a_opt = torch.optim.SGD(get_params_to_update(model), lr=0.1)


    model.update_w()
    o = model(x).sum()
    o.backward()

    print(model.Encoder_1[0].alpha)
    print(model.Encoder_1[0].block_2d[0].weight.data[0, 0, 0])
    w_opt.step()
    print(model.Encoder_1[0].block_2d[0].weight.data[0, 0, 0])
    print(model.Encoder_1[0].alpha)

# check()

def check2():
    model = Network(1)
    x = torch.rand(2, 1, 64, 64, 64)
    o = model(x)
    print(o.shape)


# check2()
