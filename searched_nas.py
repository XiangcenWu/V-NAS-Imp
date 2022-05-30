import torch
import torch.nn as nn





class Encoder(nn.Module):
    
    
    def __init__(self, in_c, out_c, idx :int):
        super().__init__()

        if idx == 0:
        # Block One
            self.block = nn.Sequential(
                nn.Conv3d(in_c, in_c, (3, 3, 1), (1, 1, 1), (1, 1, 0)),
                nn.Conv3d(in_c, out_c, 1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(),
            )
        if idx == 1:
        # Block Two
            self.block = nn.Sequential(
                nn.Conv3d(in_c, in_c, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
                nn.Conv3d(in_c, out_c, 1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(),
            )
        if idx == 2:
        # Block three
            self.block = nn.Sequential(
                nn.Conv3d(in_c, in_c, (3, 3, 1), (1, 1, 1), (1, 1, 0)),
                nn.Conv3d(in_c, out_c, (1, 1, 3), (1, 1, 1), (0, 0, 1)),
                nn.BatchNorm3d(out_c),
                nn.ReLU(),
            )


    def forward(self, x):
        
        x1 = x + self.block(x)
        return x1





class Decoder(nn.Module):
    # different from encoder which you can choose the input/output num of channel 
    # This decoder will not affect the num of channel to the input tensor
    
    def __init__(self, in_c, out_c, idx):
        super().__init__()

        # output channel should be an even number
        # out_c is not used, add out_c as a input for debugging
        assert out_c == in_c
        c = int(in_c/2)
        if idx == 0:
            # 2D
            self.block_left = nn.Sequential(
                nn.Conv3d(in_c, c, 1, 1),
                nn.Conv3d(c, c, (3, 3, 1), (1, 1, 1), (1, 1, 0)),
                nn.Conv3d(c, c, 1, 1),
                nn.BatchNorm3d(c),
                nn.ReLU(),
            )
            self.block_right = nn.Sequential(
                nn.Conv3d(c, c, 1, 1),
                nn.Conv3d(c, c, 1, 1),
                nn.Conv3d(c, c, 1, 1),
                nn.BatchNorm3d(c),
                nn.ReLU(),
            )
        if idx == 1:
            # 3D
            self.block_left = nn.Sequential(
                nn.Conv3d(in_c, c, 1, 1),
                nn.Conv3d(c, c, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
                nn.Conv3d(c, c, 1, 1),
                nn.BatchNorm3d(c),
                nn.ReLU(),
            )
            self.block_right = nn.Sequential(
                nn.Conv3d(c, c, 1, 1),
                nn.Conv3d(c, c, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
                nn.Conv3d(c, c, 1, 1),
                nn.BatchNorm3d(c),
                nn.ReLU(),
            )
        if idx == 2:
        # P3D
            self.block_left = nn.Sequential(
                nn.Conv3d(in_c, c, 1, 1),
                nn.Conv3d(c, c, (3, 3, 1), (1, 1, 1), (1, 1, 0)),
                nn.Conv3d(c, c, 1, 1),
                nn.BatchNorm3d(c),
                nn.ReLU(),
            )
            self.block_right = nn.Sequential(
                nn.Conv3d(c, c, 1, 1),
                nn.Conv3d(c, c, (1, 1, 3), (1, 1, 1), (0, 0, 1)),
                nn.Conv3d(c, c, 1, 1),
                nn.BatchNorm3d(c),
                nn.ReLU(),
            )

    def _cell_forward(self, x, left_mini_block, right_mini_block):
        x1 = left_mini_block(x)
        x2 = right_mini_block(x1)

        o_add = torch.cat((x1, x2), dim=1)
        # concat at the channel dim
        return x + o_add

    def forward(self, x):


        x1 = self._cell_forward(x, self.block_left, self.block_right)

        return x1


class PVP(nn.Module):


    def __init__(self, num_in, num_out, size=(96, 96, 96)):
        super().__init__()
        intermediate_out = int(num_in/4)
        self.block_1 = nn.Sequential(
            nn.Conv3d(num_in, intermediate_out, (1, 1, 1), (1, 1, 1)), 
            nn.Upsample(size, mode="trilinear"),
            nn.ReLU()
        )
        self.block_2 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2), (2, 2, 2)),
            nn.Conv3d(num_in, intermediate_out, (1, 1, 1), (1, 1, 1)), 
            nn.Upsample(size, mode="trilinear"),
            nn.ReLU()
        )
        self.block_3 = nn.Sequential(
            nn.MaxPool3d((4, 4, 4), (4, 4, 4)),
            nn.Conv3d(num_in, intermediate_out, (1, 1, 1), (1, 1, 1)), 
            nn.Upsample(size, mode="trilinear"),
            nn.ReLU()
        )
        self.block_4 = nn.Sequential(
            nn.MaxPool3d((8, 8, 8), (8, 8, 8)),
            nn.Conv3d(num_in, intermediate_out, (1, 1, 1), (1, 1, 1)), 
            nn.Upsample(size, mode="trilinear"),
            nn.ReLU()
        )
        self.out = nn.Conv3d(num_in, num_out, (1, 1, 1), (1, 1, 1))

    def forward(self, x):
        x1 = self.block_1(x)
        x2 = self.block_1(x)
        x3 = self.block_1(x)
        x4 = self.block_1(x)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        return self.out(x)


class Network_best(nn.Module):
    
    
    def __init__(self, input=(96, 96, 96), out_channel=3):
        super().__init__()
        self.first_three_op = nn.Sequential(
            nn.Conv3d(1, 24, (7, 7, 1), (2, 2, 1), (3, 3, 0)),
            nn.MaxPool3d((1, 1, 2), (1, 1, 2)),
            nn.Conv3d(24, 40, (1, 1, 3), (1, 1, 1), (0, 0, 1)),
            nn.ReLU()
        ) # (B, 40, 48, 48, 48)

        self.mp_222 = nn.MaxPool3d((2, 2, 2), (2, 2, 2)) # (B, 40, 24, 24, 24)
        
        
        self.Encoder_1 = nn.Sequential(*self.init_encoder(40, 3, [1, 2, 2])) # (B, 40, 24, 24, 24)

        self.cmd_2 = nn.Sequential(*self.init_cmd(40, 60)) # (B, 60, 12, 12, 12)
        self.Encoder_2 = nn.Sequential(*self.init_encoder(60, 4, [1, 1, 0, 0])) # (B, 60, 12, 12, 12)

        self.cmd_3 = nn.Sequential(*self.init_cmd(60, 80)) # (B, 80, 6, 6, 6)
        self.Encoder_3 = nn.Sequential(*self.init_encoder(80, 6, [1, 1, 0, 0, 0, 0])) # (B, 80, 6, 6, 6)

        self.cmd_4 = nn.Sequential(*self.init_cmd(80, 160))
        self.Encoder_4 = nn.Sequential(*self.init_encoder(160, 3, [1, 0, 0]))

        # #################################Decoder#############################
        self.up_1 = nn.Sequential(*self.init_up(160, 80, tuple((int(ti/16) for ti in input)))) # (B, 80, 6, 6, 6)

        self.Decoder_1 = nn.Sequential(*self.init_decoder(80, [0, 2, 0, 0, 0])) # (B, 80, 6, 6, 6)
        self.up_2 = nn.Sequential(*self.init_up(80, 60, tuple((int(ti/8) for ti in input)))) # (B, 60, 12, 12, 12)

        self.Decoder_2 = nn.Sequential(*self.init_decoder(60, [1, 1, 0, 0, 0])) # (B, 60, 12, 12, 12)
        self.up_3 = nn.Sequential(*self.init_up(60, 40, tuple((int(ti/4) for ti in input)))) # (B, 40, 24, 24, 24)

        self.Decoder_3 = nn.Sequential(*self.init_decoder(40, [0, 1, 1, 1, 1])) # (B, 40, 24, 24, 24)



        self.Decoder_4 = nn.Sequential(*self.init_decoder(40, [2, 0, 2, 1, 2])) # (B, 40, 24, 24, 24)
        self.up_4 = nn.Sequential(*self.init_up(40, 40, tuple((int(ti/2) for ti in input)))) # (B, 40, 48, 48, 48)

        self.Decoder_5 = nn.Sequential(*self.init_decoder(40, [2, 0, 1, 0, 1]))

        # change this into a pyramid
        self.pvp = PVP(40, 3, input)



        self.cells = [
            self.Encoder_1,
            self.Encoder_2,
            self.Encoder_3,
            self.Encoder_4,
            self.Decoder_1,
            self.Decoder_2,
            self.Decoder_3,
            self.Decoder_4,
            self.Decoder_5,
        ]



    def forward(self, x):
        x1 = self.first_three_op(x)
        x2 = self.mp_222(x1)
        x3 = self.Encoder_1(x2)
        x4 = self.Encoder_2(self.cmd_2(x3))
        x5 = self.Encoder_3(self.cmd_3(x4))
        x6 = self.Encoder_4(self.cmd_4(x5))
        
        

        x5 = x5 + self.up_1(x6)
        x4 = x4 + self.up_2(self.Decoder_1(x5))
        x3 = x3 + self.up_3(self.Decoder_2(x4))
        x2 = x2 + self.Decoder_3(x3)
        x1 = x1 + self.up_4(self.Decoder_4(x2))
        x = self.Decoder_5(x1)
        
        
        # x = self.pvp(x1)
        return self.pvp(x)


    def init_decoder(self, num_features, idx_list):
        decoder = []
        for _ in range(5):
            decoder.append(Decoder(num_features, num_features, idx_list[_]))
        return decoder

    def init_encoder(self, num_features, num_encoder, idx_list):
        encoder = []
        for _ in range(num_encoder):
            encoder.append(Encoder(num_features, num_features, idx_list[_]))
        return encoder

    def init_cmd(self, num_in, num_out):
        """Conv-Max Pool Down operation in the VNAS paper, this operation will downsample the size of 3d feature map

        Args:
            num_in (int): input number of feature map's channel
            num_out (int):  output number of feature map's channel

        Returns:
            list: list of operations ready to feed into a nn.Seqential module
        """

        cmd = [
            nn.Conv3d(num_in, num_in, (1, 1, 1), (2, 2, 1), (0, 0, 0)),
            nn.MaxPool3d((1, 1, 2), (1, 1, 2)),
            nn.Conv3d(num_in, num_out, 1, 1)
        ]

        return cmd

    def init_up(self, num_in, num_out, output_size):
        return [
            nn.Conv3d(num_in, num_out, 1, 1),
            nn.Upsample(output_size, mode='trilinear')
        ]

    def log(self):
        for i in self.cells:
            for j in i:
                print(j.alpha.data)



class Network_worst(nn.Module):
    
    
    def __init__(self, input=(96, 96, 96), out_channel=3):
        super().__init__()
        self.first_three_op = nn.Sequential(
            nn.Conv3d(1, 24, (7, 7, 1), (2, 2, 1), (3, 3, 0)),
            nn.MaxPool3d((1, 1, 2), (1, 1, 2)),
            nn.Conv3d(24, 40, (1, 1, 3), (1, 1, 1), (0, 0, 1)),
            nn.ReLU()
        ) # (B, 40, 48, 48, 48)

        self.mp_222 = nn.MaxPool3d((2, 2, 2), (2, 2, 2)) # (B, 40, 24, 24, 24)
        
        
        self.Encoder_1 = nn.Sequential(*self.init_encoder(40, 3, [2, 0, 2])) # (B, 40, 24, 24, 24)

        self.cmd_2 = nn.Sequential(*self.init_cmd(40, 60)) # (B, 60, 12, 12, 12)
        self.Encoder_2 = nn.Sequential(*self.init_encoder(60, 4, [0, 0, 1, 1])) # (B, 60, 12, 12, 12)

        self.cmd_3 = nn.Sequential(*self.init_cmd(60, 80)) # (B, 80, 6, 6, 6)
        self.Encoder_3 = nn.Sequential(*self.init_encoder(80, 6, [2, 2, 2, 1, 1, 1])) # (B, 80, 6, 6, 6)

        self.cmd_4 = nn.Sequential(*self.init_cmd(80, 160))
        self.Encoder_4 = nn.Sequential(*self.init_encoder(160, 3, [0, 1, 2]))

        # #################################Decoder#############################
        self.up_1 = nn.Sequential(*self.init_up(160, 80, tuple((int(ti/16) for ti in input)))) # (B, 80, 6, 6, 6)

        self.Decoder_1 = nn.Sequential(*self.init_decoder(80, [1, 1, 1, 1, 1])) # (B, 80, 6, 6, 6)
        self.up_2 = nn.Sequential(*self.init_up(80, 60, tuple((int(ti/8) for ti in input)))) # (B, 60, 12, 12, 12)

        self.Decoder_2 = nn.Sequential(*self.init_decoder(60, [0, 1, 2, 2, 2])) # (B, 60, 12, 12, 12)
        self.up_3 = nn.Sequential(*self.init_up(60, 40, tuple((int(ti/4) for ti in input)))) # (B, 40, 24, 24, 24)

        self.Decoder_3 = nn.Sequential(*self.init_decoder(40, [1, 2, 2, 2, 0])) # (B, 40, 24, 24, 24)



        self.Decoder_4 = nn.Sequential(*self.init_decoder(40, [1, 2, 1, 0, 1])) # (B, 40, 24, 24, 24)
        self.up_4 = nn.Sequential(*self.init_up(40, 40, tuple((int(ti/2) for ti in input)))) # (B, 40, 48, 48, 48)

        self.Decoder_5 = nn.Sequential(*self.init_decoder(40, [1, 1, 2, 2, 0]))

        # change this into a pyramid
        self.pvp = PVP(40, 3, input)



        self.cells = [
            self.Encoder_1,
            self.Encoder_2,
            self.Encoder_3,
            self.Encoder_4,
            self.Decoder_1,
            self.Decoder_2,
            self.Decoder_3,
            self.Decoder_4,
            self.Decoder_5,
        ]



    def forward(self, x):
        x1 = self.first_three_op(x)
        x2 = self.mp_222(x1)
        x3 = self.Encoder_1(x2)
        x4 = self.Encoder_2(self.cmd_2(x3))
        x5 = self.Encoder_3(self.cmd_3(x4))
        x6 = self.Encoder_4(self.cmd_4(x5))
        
        

        x5 = x5 + self.up_1(x6)
        x4 = x4 + self.up_2(self.Decoder_1(x5))
        x3 = x3 + self.up_3(self.Decoder_2(x4))
        x2 = x2 + self.Decoder_3(x3)
        x1 = x1 + self.up_4(self.Decoder_4(x2))
        x = self.Decoder_5(x1)
        
        
        # x = self.pvp(x1)
        return self.pvp(x)


    def init_decoder(self, num_features, idx_list):
        decoder = []
        for _ in range(5):
            decoder.append(Decoder(num_features, num_features, idx_list[_]))
        return decoder

    def init_encoder(self, num_features, num_encoder, idx_list):
        encoder = []
        for _ in range(num_encoder):
            encoder.append(Encoder(num_features, num_features, idx_list[_]))
        return encoder

    def init_cmd(self, num_in, num_out):
        """Conv-Max Pool Down operation in the VNAS paper, this operation will downsample the size of 3d feature map

        Args:
            num_in (int): input number of feature map's channel
            num_out (int):  output number of feature map's channel

        Returns:
            list: list of operations ready to feed into a nn.Seqential module
        """

        cmd = [
            nn.Conv3d(num_in, num_in, (1, 1, 1), (2, 2, 1), (0, 0, 0)),
            nn.MaxPool3d((1, 1, 2), (1, 1, 2)),
            nn.Conv3d(num_in, num_out, 1, 1)
        ]

        return cmd

    def init_up(self, num_in, num_out, output_size):
        return [
            nn.Conv3d(num_in, num_out, 1, 1),
            nn.Upsample(output_size, mode='trilinear')
        ]

    def log(self):
        for i in self.cells:
            for j in i:
                print(j.alpha.data)



if __name__ == "__main__":
    from monai.losses import DiceCELoss, DiceLoss

   
    
    model = Network_worst((64, 64, 64), 3)
    # x = torch.rand(2, 1, 64, 64, 64)
    # o = model(x)
    # print(o.shape)
    x = torch.rand(1, 1, 64, 64, 64)
    y = torch.randint(0, 3, x.shape)
    o = model(x)
    print(o.shape)

