 # -----------------------------------------------
 # References:
 # SatMae: https://github.com/sustainlab-group/SatMAE/blob/main/models_mae.py
 # -----------------------------------------------


from functools import partial
import torch.nn as nn
import torch
from timm.models.vision_transformer import PatchEmbed, Block
from models.MaskedAutoEncoderViT.utils import get_2d_sincos_pos_embed



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

        # MaskedAutoEncoder Encoder specifics
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
            for i in range(depth)
        ])

        self.norm = self.norm_layer(embed_dim)
        # End
        # ==============================================================================

        # MaskedAutoEncoder Decoder specofis
        # Start
        # ==============================================================================
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                norm_layer=self.norm_layer)
            for i in range(decoder_depth)    
            ])

        self.decoder_norm = self.norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        # End
        # ==============================================================================

        # extra stuff
        self.norm_pix_loss = norm_pixel_loss

        self.initialize_weights()

    def random_masking(self, x, mask_ratio):

        N, L, D = x.shape # batch, length, dim
        len_keep = int(L * (1 - mask_ratio)) # from 64 pahces we keep only 16 for patch size 4 and mask_ratio = 75%

        noise = torch.rand(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1) # small keep, large remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # see for explanation: stackoverflow: what does 'gather' do in PyTorch in layman terms ?
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) 

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore



    # define patchify method
    def patchify(self, imgs, p, c): # p = patch size, c = number of channels
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))

        return x
    

    # define unpatchily method
    def unpatchify(self, x, p, c):

        # TODO: check this function

        return 0     


    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timms's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_((m.weight))
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    # define forward method for Encoder part of the model
    def forward_encoder(self, x, mask_ratio):
        # embded patches
        x = self.patch_embed(x)

        
        # add positional embed class token
        x = x + self.pos_embed[:, 1:, :]

        # masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append mask cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore


    # define forward method for (ViT) Decoder part of the model
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # predictor projection
        x = self.decoder_pred(x)    
        
        # remove cls token
        x = x[:, 1:, :]
        

        return x


    # define forward loss method
    def forward_loss(self, imgs, pred, mask):
        # convert label (target) image to paches
        target = self.patchify(imgs, self.patch_embed.patch_size[0], self.in_c)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # calculate loss
        loss = (pred - target) ** 2

        # get mean loss across all paches
        loss = loss.mean(dim=-1)    

        # get mean loss on removed patches
        loss = (loss * mask).sum() / mask.sum()

        return loss




    # define main forward method for model
    def forward(self, imgs):
        # forward data to encoder
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)

        # forward data into decoder
        pred = self.forward_decoder(latent, ids_restore)
        
        # forward data to calculate loss
        loss = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask

        
        





           
        