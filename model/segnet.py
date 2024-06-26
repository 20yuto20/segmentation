class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        # Encoder layers

        self.encoder_0 = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))

        self.encoder_1= nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True))

        self.encoder_2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))

         self.encoder_3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True))

        self.encoder_4 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True))

        # Decoder layers

         self.decoder_4 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True))

        self.decoder_3 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True))

        self.decoder_2 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True))

        self.decoder_1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))

        self.decoder_0 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, padding=1),
                                            nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=1))

        self._init_weight()

    def forward(self, x):
        """
        Forward pass `input_img` through the network
        """

        # Encoder

        # Encoder Stage - 1
        dim_0 = x.size()
        x = self.encoder_0(x)
        x, indices_0 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 2
        dim_1 = x.size()
        x = self.encoder_1(x)
        x, indices_1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 3
        dim_2 = x.size()
        x = self.encoder_2(x)
        x, indices_2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 4
        dim_3 = x.size()
        x = self.encoder_3(x)
        x, indices_3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Encoder Stage - 5
        dim_4 = x.size()
        x = self.encoder_4(x)
        x, indices_4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        # Decoder

        #dim_d = x.size()

        # Decoder Stage - 5
        x = F.max_unpool2d(x, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x = self.decoder_4(x)
        #dim_4d = x.size()

        # Decoder Stage - 4
        x = F.max_unpool2d(x, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x = self.decoder_3(x)
        #dim_3d = x.size()

        # Decoder Stage - 3
        x = F.max_unpool2d(x, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x = self.decoder_2(x)
        #dim_2d = x.size()

        # Decoder Stage - 2
        x = F.max_unpool2d(x, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x = self.decoder_1(x)

        # Decoder Stage - 1
        x = F.max_unpool2d(x, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x = self.decoder_0(x)
        
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()