# This ViT model includes only an ViT Encoder without a Decoder
# so later on a Convolutional Decoder can be added

from functools import partial
import torch
import torch.nn as nn  
import timm.models.vision_transformer
from models.ViT.utils import get_2d_sincos_pos_embed
from models.ViT.ConvDecoder import ConvolutionalDecoder
from models.ViT.LSTMDecoder import LSTMDecoder
from timm.models.vision_transformer import PatchEmbed



class VisionTransformerConv(timm.models.vision_transformer.VisionTransformer):

    def __init__(self, global_pool=False, decoder_layers=[256, 128], **kwargs):
        super(VisionTransformerConv, self).__init__(**kwargs)

        self.input_chanels = kwargs['in_chans']
        self.decoder_layers = decoder_layers
        self.img_size = kwargs['img_size']
        self.embed_dim = kwargs['embed_dim']
        self.patch_size = kwargs['patch_size']
        self.num_classes = kwargs['num_classes']
        self.grid_size = self.img_size // self.patch_size

        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.input_chanels, self.embed_dim)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.global_pool = global_pool

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm

        # define convoluational decoder class
        # this decoder is defined in ConvDecoder.py
        self.decoder = ConvolutionalDecoder(in_channels=self.embed_dim,
                                            decoder_layers=self.decoder_layers,
                                            embed_dim=self.embed_dim,
                                            num_classes=self.num_classes,
                                            image_size=self.img_size,
                                            grid_size=self.grid_size)


    # define encoder forward method 
    def forward_encoder(self, x):
        # get batch size
        B = x.shape[0]
        
        # convert image to patches
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # pass the paches into the transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # using this encoder for segmentation
        x = x[:, 1:, :]
        
        return x


    # define forward decoder method
    def forward_decoder(self, x):
        B = x.shape[0]

        # reshape from (batch, num_of_patches, embed_dim) to (batch, embed_dim, height', width')
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(B, self.embed_dim, self.grid_size, self.grid_size)
        #print(f'Inital shape: {x.shape}')
        #print(f'Final shape: {self.decoder(x).shape}')
        #exit()
        return self.decoder(x)


    # define main forward method, images first go to ViT encoder and then to convoluational decoder
    def forward(self, x):

        features = self.forward_encoder(x)

        feature_map = self.forward_decoder(features)

        return feature_map
    







class VisionTransformerLSTM(timm.models.vision_transformer.VisionTransformer):

    def __init__(self, global_pool=False, decoder_layers=[256, 128], lstm_hidden_dim=128, lstm_num_layers=1, **kwargs):
        super(VisionTransformerLSTM, self).__init__(**kwargs)

        self.input_chanels = kwargs['in_chans']
        self.decoder_layers = decoder_layers
        self.img_size = kwargs['img_size']
        self.embed_dim = kwargs['embed_dim']
        self.patch_size = kwargs['patch_size']
        self.num_classes = kwargs['num_classes']
        self.grid_size = self.img_size // self.patch_size
        
        # lstm h-params
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, self.input_chanels, self.embed_dim)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True)

        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.global_pool = global_pool

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm

        # define convolutional decoder class
        # this decoder is defined in ConvDecoder.py
        self.decoder = LSTMDecoder(embed_dim=self.embed_dim,
                                   hidden_dim=self.lstm_hidden_dim,
                                   num_layers=self.lstm_num_layers,
                                   num_classes=self.num_classes)



        self.classifier = nn.Sequential(
                                        nn.Linear(kwargs['embed_dim'], 2),
         
                                    )
        


    # define encoder forward method 
    def forward_encoder(self, x):
        # get batch size
        B = x.shape[0]
        
        # convert image to patches
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # pass the paches into the transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # using this encoder for segmentation
        #x = x[:, 1:, :]
        
        return x


    # define forward decoder method
    def forward_decoder(self, x):
        #cls_token = x[:, 0]
        #logits = self.classifier(cls_token)
        #return logits
        return self.decoder(x)


    # define main forward method, images first go to ViT encoder and then to convoluational decoder
    def forward(self, x):
      
        features = self.forward_encoder(x)
        
        logits = self.forward_decoder(features)

        return logits    
    



import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.3, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,          # Input: [batch, seq, features]
            bidirectional=bidirectional
        )
        
        direction_multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_multiplier, 2)  # binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_size]
        
        # Use the last time step output
        out = lstm_out[:, -1, :]  # shape: [batch_size, hidden_size * num_directions]
        out = self.fc(out)        # [batch_size, 1]
        #out = self.sigmoid(out)   # [batch_size, 1]
        return out.squeeze()      # [batch_size]
