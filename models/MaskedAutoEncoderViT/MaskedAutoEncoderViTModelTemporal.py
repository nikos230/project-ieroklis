 # -----------------------------------------------
 # References:
 # SatMae: https://github.com/sustainlab-group/SatMAE/blob/main/models_mae_temporal.py
 # -----------------------------------------------


import email
from functools import partial
import torch.nn as nn
import torch
from timm.models.vision_transformer import PatchEmbed, Block
from models.MaskedAutoEncoderViT.utils import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid_torch, get_relative_seasonal_embedding



class MaskedAutoEncoderViTModelTemporal(nn.Module):
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
                norm_pixel_loss,
                same_mask):

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
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim - 384), requires_grad=False)

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
        self.same_mask = same_mask
        # End
        # ==============================================================================

        # MaskedAutoEncoder Decoder specifics
        # Start
        # ==============================================================================
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim - 192), requires_grad=False)
        
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

    def random_masking(self, x, mask_ratio, mask=None):

        N, L, D = x.shape # batch, length, dim
        len_keep = int(L * (1 - mask_ratio)) # from 64 pahces we keep only 16 for patch size 4 and mask_ratio = 75%

        noise = torch.rand(N, L, device=x.device)

        if self.same_mask:
            L2 = L // 3
            assert 3 * L2 == L
            noise = torch.rand(N, L2, device=x.device)  # noise in [0, 1]
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_shuffle = [ids_shuffle + i * L2 for i in range(3)]
            ids_shuffle_keep = [z[: ,:int(L2 * (1 - mask_ratio))] for z in ids_shuffle]
            ids_shuffle_disc = [z[: ,int(L2 * (1 - mask_ratio)):] for z in ids_shuffle]
            ids_shuffle = []
            for z in ids_shuffle_keep:
                ids_shuffle.append(z)
            for z in ids_shuffle_disc:
                ids_shuffle.append(z)
            ids_shuffle = torch.cat(ids_shuffle, dim=1)
            # print(ids_shuffle[0])
            # assert False
        else:
            if mask is None:
                # sort noise for each sample
                ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            else:
                ids_shuffle = mask

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore



    # define patchify method
    def patchify(self, imgs, tensor_shape): # p = patch size, c = number of channels
        B, T, C, H, W = tensor_shape
        
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], C, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * C))

        return x
    

    # define unpatchily method
    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))

        return imgs    


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
    def forward_encoder(self, x, timestamps, mask_ratio, mask=None):
        # embded patches
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

        # masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio, mask=mask)
        
        # append mask cls token
        cls_token = self.cls_token #+ self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore


    # define forward method for (ViT) Decoder part of the model
    def forward_decoder(self, x, timestamps, ids_restore, tensor_shape):
        B, T, C, H, W = tensor_shape

        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # ts_embeds = [get_1d_sincos_pos_embed_from_grid_torch(192, timestamps[:, t].float()) for t in range(0, T)]
        # ts_embed = torch.cat(ts_embeds, dim=1).float()

        # ts_embed = ts_embed.reshape(B, T, 192).unsqueeze(2)
        
        # num_patches = x.shape[1] // T
        
        # ts_embed = ts_embed.expand(-1, -1, num_patches, -1)
        # ts_embed = ts_embed.reshape(B, T * num_patches, 192)

        delta_days = timestamps[:, :, 0] - timestamps[:, 0:1, 0]  # (B, T)
        day_of_year = timestamps[:, :, 1]                         # (B, T)
   
        ts_embeds = []
        for t in range(T):
            ts_embeds.append(get_relative_seasonal_embedding(delta_days[:, t], day_of_year[:, t], embed_dim=192))

        ts_embed = torch.stack(ts_embeds, dim=1)  # (B, T, D)
        ts_embed = ts_embed.unsqueeze(2)          # (B, T, 1, D)

        num_patches = x.shape[1] // T
        ts_embed = ts_embed.expand(-1, -1, num_patches, -1).reshape(B, T * num_patches, 192)

        ts_embed = torch.cat([torch.zeros((ts_embed.shape[0], 1, ts_embed.shape[2]), device=ts_embed.device), ts_embed], dim=1)


        # add pos embed
        x = x + torch.cat(
            [torch.cat([self.decoder_pos_embed[:, :1, :], self.decoder_pos_embed[:, 1:, :].repeat(1, T, 1)], dim=1).expand(ts_embed.shape[0], -1, -1),
             ts_embed], dim=-1)
        

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
    def forward_loss(self, imgs, pred, mask, tensor_shape):
        B, T, C, H, W = tensor_shape
        # convert label (target) image to paches

        targets = [self.patchify(imgs[:, t], tensor_shape) for t in range(0, T)]
        target = torch.cat(targets, dim=1).float()


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
    def forward(self, imgs, timestamps, mask=None):
        tensor_shape = imgs.shape
        # forward data to encoder
        latent, mask, ids_restore = self.forward_encoder(imgs, timestamps, self.mask_ratio, mask=mask)

        # forward data into decoder
        pred = self.forward_decoder(latent, timestamps, ids_restore, tensor_shape)
        
        # forward data to calculate loss
        loss = self.forward_loss(imgs, pred, mask, tensor_shape)
        
        return loss, pred, mask

        
        





           
        