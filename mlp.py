import torch.nn as nn
from einops import rearrange

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channel_first=False):
        """
        :param channel_first: if True, during forward the tensor shape is [B, C, T, J] and fc layers are performed with
                              1x1 convolutions.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        if channel_first:
            self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
            self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        :param channel_first: if True, during forward the tensor shape is [B, C, T, J] and fc layers are performed with
                              1x1 convolutions.
        """
        super().__init__()
        out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        hidden_features = in_features//2
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

        self.fc_channel_down = nn.Linear(in_features, hidden_features)
        self.fc_spatial_down = nn.Linear(17, 17//2)

        self.fc_channel_up = nn.Linear(hidden_features, out_features)
        self.fc_spatial_up = nn.Linear(17//2, 17)

        self.fc_mid_s = nn.Linear(17//2, 17//2)
        self.fc_mid_c = nn.Linear(hidden_features, hidden_features)

        self.fc2 = nn.Linear(hidden_features, out_features)


    def forward(self, x):
        x_s = x
        x_c = x

        # spatial UMLP
        x_s = rearrange(x_s, 'b t j c -> b c t j')
        x_s = self.fc_spatial_down(x_s)
        x_s = self.drop(self.act(x_s))
        x_s = x_s + self.drop(self.act(self.fc_mid_s(x_s)))
        x_s = self.drop(self.fc_spatial_up(x_s))
        x_s = x_s = rearrange(x_s, 'b c t j -> b t j c')

        # channel UMLP
        x_c = self.fc_channel_down(x_c)
        x_c = self.drop(self.act(x_c))
        x_c = x_c + self.drop(self.act(self.fc_mid_c(x_c)))
        x_c = self.drop(self.fc_channel_up(x_c))

        x = x + x_s + x_c

        return x
    

## UMlp
class linear_block(nn.Module):
    def __init__(self, ch_in, ch_out, drop=0.1):
        super(linear_block,self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(ch_in, ch_out),
            nn.GELU(),
            nn.Dropout(drop)
        )
    def forward(self,x):
        x = self.linear(x)
        return x

class UMlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        self.linear512_256 = linear_block(in_features,hidden_features, drop)
        self.linear256_256 = linear_block(hidden_features, hidden_features, drop) 
        self.linear256_512 = linear_block(hidden_features, in_features, drop)

    def forward(self, x):
        # down          
        x = self.linear512_256(x)
        res_256 = x 
        # mid
        x = self.linear256_256(x)
        x = x + res_256
        # up
        x = self.linear256_512(x) 
        return x    