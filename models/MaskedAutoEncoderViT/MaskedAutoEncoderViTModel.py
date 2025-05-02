from functools import partial
import torch.nn as nn
import torch
from timm.models.vision_transformer import PatchEmbed, Block
#from utils.



class MaskedAutoEncoderViTModel(nn.Module):
    def __init__(self, 
                img_size,
                patch_size,
                mask_ratio,
                in_chans,
                embed_dim,
                depth,
                num_heads,
                decoder_embed_dim,
                decoder_depth,
                decoder_num_heads,
                mlp_ratio,
                norm_pixel_loss):

        super().__init__()

        self.norm_layer = nn.LayerNorm

        self.mask_ratio = mask_ratio
        self.in_c = in_chans

        # MaskedAutoEncoder specifics
        # Start
        # ==============================================================================
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList([
            Block(embed_dim, 
                num_heads, 
                mlp_ratio, 
                qkv_bias=True, 
                #qk_scale=None, 
                norm_layer=self.norm_layer)
        ])

        self.norm = self.norm_layer(embed_dim)
        # End
        # ==============================================================================


    def random_masking(self, x, mask_ratio):

        N, L, D = x.shape # batch, length, dim



    # define forward method for Encoder part of the model
    def forward_encoder(self, x, mask_ratio):
        # embded patches
        x = self.patch_embed(x)


        # add positional embed class token
        x = x + self.pos_embed[:, 1:, :]
        print(x.shape)

        # masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append mask cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore



    # define main forward method for model
    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)

        print(latent, mask, ids_restore)
        exit(222)

        
        





           
        