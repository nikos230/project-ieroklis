import torch
import torch.nn as nn



class ConvolutionalDecoder(nn.Module):

    def __init__(self, in_channels, decoder_layers, embed_dim, num_classes, image_size, grid_size):
        super(ConvolutionalDecoder, self).__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.decoder_layers = decoder_layers
        self.num_classes = num_classes
        self.input_image_size = image_size
        self.grid_size = grid_size
        self.kernel_size = 4
        self.final_image_size = self.grid_size * self.kernel_size * len(self.decoder_layers) # predicted final image size based on grid size, kernel size and number of Transposed Convolutions
        self.num_of_extra_convs = 0

        # check if after transposed convoltutions image size is bigger than input size
        # if it is then just use a number of convolutions to make it smaller
        if self.input_image_size != self.final_image_size:
            # based on kernel size = 2
            self.num_of_extra_convs = 1 #self.final_image_size // self.input_image_size


        self.decoder = self.build_decoder()


    def build_decoder(self):
        layers = []
        current_channels = self.embed_dim


        for out_channels in self.decoder_layers: # decoder_layer = [256, 128] or [512, 256, 128, 64], etc...
            layers.append(nn.ConvTranspose2d(current_channels, out_channels, kernel_size=self.kernel_size, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(p=0.1))
            current_channels = out_channels

        # do some extra convs to reduce image size if needed
        for conv in range(0, self.num_of_extra_convs):
            layers.append(nn.Conv2d(current_channels, current_channels, kernel_size=self.kernel_size, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(current_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(p=0.1))

        # append the final layer
        layers.append(nn.Conv2d(current_channels, self.num_classes, kernel_size=1))

        # pass modules into constructor
        return nn.Sequential(*layers)             



    def forward(self, x): 
        #x = self.decoder(x)
        #print(x.shape)
        #exit()
        return self.decoder(x)
               


