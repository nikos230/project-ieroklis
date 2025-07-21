# This ViT model includes only an ViT Encoder without a Decoder
# so later on a Convolutional Decoder can be added

from functools import partial
import torch
import torch.nn as nn  
import timm.models.vision_transformer
from models.ViT.utils import get_2d_sincos_pos_embed
from models.MaskedAutoEncoderViT.utils import get_1d_sincos_pos_embed_from_grid_torch, get_relative_seasonal_embedding
from models.ViT.ConvDecoder import ConvolutionalDecoder
from timm.models.vision_transformer import PatchEmbed



class VisionTransformerConvTemporal(timm.models.vision_transformer.VisionTransformer):

    def __init__(self, global_pool=False, decoder_layers=[256, 128], **kwargs):
        super(VisionTransformerConvTemporal, self).__init__(**kwargs)

        self.input_chanels = kwargs['in_chans']
        self.decoder_layers = decoder_layers
        self.img_size = kwargs['img_size']
        self.embed_dim = kwargs['embed_dim']
        self.patch_size = kwargs['patch_size']
        self.num_classes = kwargs['num_classes']
        self.grid_size = self.img_size // self.patch_size

        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.input_chanels, self.embed_dim)

        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, kwargs['embed_dim'] - 384))

        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm

        # define convoluational decoder class
        # this decoder is defined in ConvDecoder.py
        self.decoder = ConvolutionalDecoder(in_channels=self.input_chanels,
                                            decoder_layers=self.decoder_layers,
                                            embed_dim=self.embed_dim,
                                            num_classes=self.num_classes,
                                            image_size=self.img_size,
                                            grid_size=self.grid_size)


    # define encoder forward method 
    def forward_encoder(self, x, timestamps):
        # get batch size
        B, T, C, H, W = x.shape

        patches = [self.patch_embed(x[:, t]) for t in range(0, T)]
        x = torch.cat(patches, dim=1)  

        # # create timestamps for all embeddings for all time steps
        # ts_embeds = [get_1d_sincos_pos_embed_from_grid_torch(384, timestamps[:, t].float()) for t in range(0, T)]
        # ts_embed = torch.cat(ts_embeds, dim=1).float()

        # ts_embed = ts_embed.reshape(B, T, 384).unsqueeze(2)
        
        # num_patches = x.shape[1] // T
        
        # ts_embed = ts_embed.expand(-1, -1, num_patches, -1)
        # ts_embed = ts_embed.reshape(B, T * num_patches, 384)
        # print(timestamps)
        # exit()
        delta_days = timestamps[:, :, 0] - timestamps[:, 0:1, 0]  # (B, T)
        day_of_year = timestamps[:, :, 1]                         # (B, T)

        ts_embeds = []
        for t in range(T):
            ts_embeds.append(get_relative_seasonal_embedding(delta_days[:, t], day_of_year[:, t], embed_dim=384))

        ts_embed = torch.stack(ts_embeds, dim=1)  # (B, T, D)
        ts_embed = ts_embed.unsqueeze(2)          # (B, T, 1, D)

        num_patches = x.shape[1] // T
        ts_embed = ts_embed.expand(-1, -1, num_patches, -1).reshape(B, T * num_patches, 384)



        pos_embed_repeated = self.pos_embed[:, 1:, :].repeat(1, T, 1)
        pos_embed_repeated = pos_embed_repeated.expand(B, -1, -1)
        
        x = x + torch.cat([pos_embed_repeated, ts_embed], dim=-1) # TODO: check this
       
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # x = x + torch.cat(
        #     [torch.cat([pos_embed_repeated[:, :1, :], pos_embed_repeated[:, 1:, :].repeat(1, 3, 1)], dim=1).expand(ts_embed.shape[0], -1, -1),
        #      ts_embed], dim=-1)
        

        x = self.pos_drop(x)

        # apply transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)
        #     out = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     out = x[:, 0]

        # using this encoder for segmentation
        x = x[:, 1:, :]

        return x


    # define forward decoder method
    def forward_decoder(self, x, tensor_shape):
        B, T, C, H, W = tensor_shape
        
        # reshape from (batch, num_of_patches, embed_dim) to (batch, embed_dim, height, width)

        x = x.view(B, T, self.grid_size * self.grid_size, self.embed_dim)
        x = x.mean(dim=1)
        #x = x[:, 4]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, self.embed_dim, self.grid_size, self.grid_size)
        #print(f'Inital shape: {x.shape}')
        #print(f'Final shape: {self.decoder(x).shape}')
        #exit()
        return self.decoder(x)
    

    # def forward_decoder(self, x, tensor_shape):
    #     B, T, C, H, W = tensor_shape
    #     num_patches = self.grid_size * self.grid_size

    #     # Reshape encoder output
    #     x = x.view(B, T, num_patches, self.embed_dim)

    #     outputs = []

    #     for t in range(T):
    #         x_t = x[:, t]  # (B, P, D)
    #         x_t = x_t.permute(0, 2, 1).contiguous()  # (B, D, P)
    #         x_t = x_t.view(B, self.embed_dim, self.grid_size, self.grid_size)  # (B, D, H, W)

    #         out_t = self.decoder(x_t)  # (B, num_classes, H', W')
    #         outputs.append(out_t)

    #     # Stack outputs: (B, T, num_classes, H, W)
    #     return torch.stack(outputs, dim=1)


    # define main forward method, images first go to ViT encoder and then to convoluational decoder
    def forward(self, x, timestamps):
        tensor_shape = x.shape

        features = self.forward_encoder(x, timestamps)

        feature_map = self.forward_decoder(features, tensor_shape)

        return feature_map
    



class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters, kernel_size, pool_size, 
                 use_batchnorm=True, final_activation=None, dropout_rate=0.3, num_layers=1):
        """
        Parameters:
        - in_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        - out_channels: Number of output channels (e.g., segmentation classes)
        - num_filters: List of filters for each level in the encoder and decoder
        - kernel_size: Convolution kernel size (default is 3x3x3)
        - pool_size: Pooling size for downsampling (default is (1, 2, 2))
        - use_batchnorm: Whether to use batch normalization in the conv blocks (default is True)
        - final_activation: Activation function to apply at the final output (e.g., nn.Sigmoid() or nn.Softmax(dim=1))
        - dropout_rate: Dropout rate applied after GELU activations to prevent overfitting
        """
        super(UNet3D, self).__init__()
        
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.use_batchnorm = use_batchnorm
        self.final_activation = final_activation
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers  
        
        # Encoder
        self.encoders = nn.ModuleList([self.conv_block(in_channels, num_filters[0], dropout_rate=self.dropout_rate, num_layers=self.num_layers)])
        self.pools = nn.ModuleList([nn.MaxPool3d(kernel_size=self.pool_size)])
        for i in range(1, len(num_filters)):
            self.encoders.append(self.conv_block(num_filters[i-1], num_filters[i], dropout_rate=self.dropout_rate, num_layers=self.num_layers))
            self.pools.append(nn.MaxPool3d(kernel_size=self.pool_size))

        # Bottleneck
        self.bottleneck = self.conv_block(num_filters[-1], num_filters[-1] * 2, dropout_rate=self.dropout_rate, num_layers=self.num_layers)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(1, len(num_filters))):
            self.upconvs.append(nn.ConvTranspose3d(num_filters[i]*2, num_filters[i], kernel_size=self.pool_size, stride=self.pool_size))
            self.decoders.append(self.conv_block(num_filters[i]*2, num_filters[i], dropout_rate=self.dropout_rate, num_layers=self.num_layers))
        
        self.upconvs.append(nn.ConvTranspose3d(num_filters[0]*2, num_filters[0], kernel_size=self.pool_size, stride=self.pool_size))
        self.decoders.append(self.conv_block(num_filters[0]*2, num_filters[0], dropout_rate=self.dropout_rate, num_layers=self.num_layers))

        # Final convolution
        self.conv_final = nn.Conv3d(num_filters[0], out_channels, kernel_size=1)


    def conv_block(self, in_channels, out_channels, dropout_rate=0.3, num_layers=1):
        """
        A helper function to create a convolutional block consisting of multiple Conv3D layers, 
        BatchNorm (if enabled), GELU activation, and optional Dropout.
        
        Parameters:
        - in_channels: Number of input channels for the first Conv3D layer.
        - out_channels: Number of output channels for each Conv3D layer.
        - dropout_rate: The dropout rate applied after each GELU.
        - num_layers: Number of Conv3D layers to include in the block.
        """
        layers = []
        
        # Add the first convolutional layer
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=self.kernel_size, padding=(1, 1, 1)))
        if self.use_batchnorm:
            layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.GELU())
        
        # Add subsequent convolutional layers (num_layers - 1 times)
        for _ in range(num_layers - 1):
            layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=self.kernel_size, padding=(1, 1, 1)))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.GELU())
        
        # Add Dropout after GELU of the last layer
        layers.append(nn.Dropout3d(p=dropout_rate))



        return nn.Sequential(*layers)


    def save_gradients(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Encoder path
        enc_outputs = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            enc_outputs.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for upconv, decoder, enc_output in zip(self.upconvs, self.decoders, reversed(enc_outputs)):
            x = upconv(x)
            x = torch.cat([x, enc_output], dim=1)  # Skip connection
            x = decoder(x)

        # Final convolution
        x = self.conv_final(x)

        # Reduce depth dimension (optional, adjust if necessary)
        x = torch.mean(x, dim=2)  # Shape: (batch_size, out_channels, height, width)
        #x = torch.mean(x, dim=[1, 2, 3])

        # Apply final activation (if any)
        if self.final_activation:
            x = self.final_activation(x)

        return x    