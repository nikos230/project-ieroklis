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
        self.current_image_size = self.grid_size
        self.kernel_size = 0



        self.decoder = self.build_decoder()
        


    def build_decoder(self):
        layers = []
        current_channels = self.embed_dim

        
        
        # upscale the image to match the original size
        while self.current_image_size != self.input_image_size:
            layers.append(self.conv_block(current_channels))

            current_channels = current_channels // 2
            self.current_image_size = self.current_image_size * 2 # kernel size = 2 and stride = 2



        # layers.append(nn.ConvTranspose2d(current_channels, 256, kernel_size=2, stride=2, padding=0))
        # layers.append(nn.GroupNorm(num_groups=16, num_channels=256))
        # layers.append(nn.LeakyReLU(inplace=True))

        # layers.append(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        # layers.append(nn.GroupNorm(num_groups=8, num_channels=128))
        # layers.append(nn.LeakyReLU(inplace=True))

        # layers.append(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0))
        # layers.append(nn.GroupNorm(num_groups=16, num_channels=64))
        # layers.append(nn.LeakyReLU(inplace=True))

        # layers.append(nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1))
        # layers.append(nn.GroupNorm(num_groups=8, num_channels=32))
        # layers.append(nn.LeakyReLU(inplace=True))

        # append the final layer and dropout layer
        #layers.append(nn.Dropout2d(p=0.4))
        layers.append(nn.Conv2d(current_channels, self.num_classes, kernel_size=1))

        # pass modules into constructor
        return nn.Sequential(*layers)             


    # def conv_block(self, input_channels):
    #     return nn.Sequential(
    #         nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2, padding=0),
    #         #nn.BatchNorm2d(num_features=input_channels // 2),
    #         nn.GroupNorm(num_groups=16, num_channels=input_channels // 2),
    #         #nn.LeakyReLU(inplace=True),

    #         nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=3, stride=1, padding=1),
    #         nn.GroupNorm(num_groups=8, num_channels=input_channels // 2),
    #         #nn.BatchNorm2d(num_features=input_channels // 2),
    #         nn.LeakyReLU(inplace=True),

    #         nn.Dropout2d(p=0.5)
    #     )

    def conv_block(self, input_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=2, stride=2, padding=0),
            #nn.BatchNorm2d(num_features=input_channels // 2),
            nn.GroupNorm(num_groups=16, num_channels=input_channels // 2),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=input_channels // 2),
            #nn.BatchNorm2d(num_features=input_channels // 2),
            nn.LeakyReLU(inplace=True),

            nn.Dropout2d(p=0.4)
        )


    def forward(self, x): 
        #x = self.decoder(x)
        # print(x.shape)
        # exit()
        return self.decoder(x)




# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ConvolutionalDecoder(nn.Module):

#     def __init__(self, in_channels, decoder_layers, embed_dim, num_classes, image_size, grid_size):
#         super(ConvolutionalDecoder, self).__init__()

#         self.in_channels = in_channels
#         self.embed_dim = embed_dim
#         self.decoder_layers = decoder_layers
#         self.num_classes = num_classes
#         self.input_image_size = image_size
#         self.grid_size = grid_size

#         # Number of skip connections expected
#         self.num_skips = 4  # change according to features collected in encoder

#         # Create fusion conv blocks for skip connections
#         self.fuse_blocks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1),
#                 nn.GroupNorm(num_groups=16, num_channels=embed_dim),
#                 nn.LeakyReLU(inplace=True),
#                 nn.Dropout2d(p=0.4),
#             )
#             for _ in range(self.num_skips - 1)  # one less than number of features
#         ])

#         self.final_conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

#     def forward(self, features):
#         """
#         features: list of tensors from encoder at multiple depths
#         Each tensor shape: (B, num_patches, embed_dim)
#         """
#         B = features[-1].shape[0]

#         # Start from last feature (lowest resolution)
#         x = features[-1]  # (B, N, C)
#         grid_size = int(features[-1].shape[1] ** 0.5)
#         x = x.permute(0, 2, 1).contiguous().view(B, self.embed_dim, grid_size, grid_size)

#         # Process skip connections from deeper to shallower layers
#         for i in reversed(range(len(features) - 1)):
#             skip = features[i]
#             skip_grid = int(skip.shape[1] ** 0.5)
#             skip = skip.permute(0, 2, 1).contiguous().view(B, self.embed_dim, skip_grid, skip_grid)

#             # Upsample x to skip's spatial size if needed
#             x = F.interpolate(x, size=(skip_grid, skip_grid), mode='bilinear', align_corners=False)

#             # Concatenate skip connection features on channel dim
#             x = torch.cat([x, skip], dim=1)  # Channel dim doubled

#             # Fuse concatenated features
#             x = self.fuse_blocks[i](x)

#         # Final conv layer to get desired number of classes

#         x = self.final_conv(x)

#         return x

               


