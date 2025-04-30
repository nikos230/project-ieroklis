import torch as nn




class MaskedAutoEncoderModel(nn.Module):
    def __init__(self, img_size: int | ):