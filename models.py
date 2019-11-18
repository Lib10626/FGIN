import torch
import torch.nn as nn

rdc_text_dim = 1000
z_dim = 7
h_dim = 256

class _param:
    def __init__(self):
        self.rdc_text_dim = rdc_text_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

# reduce to dim of text first
class _netG(nn.Module):
    def __init__(self, text_dim=11083, X_dim=3584):
        super(_netG, self).__init__()
        self.rdc_text = nn.Linear(text_dim, rdc_text_dim)
        self.main = nn.Sequential(nn.Linear(z_dim + rdc_text_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, X_dim),
                                  nn.Tanh())

    def forward(self, z, c):
        rdc_text = self.rdc_text(c)
        input = torch.cat([z, rdc_text], 1)
        output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, y_dim=150, X_dim=3584):
        super(_netD, self).__init__()
        # Discriminator net layer one
        self.D_shared = nn.Sequential(nn.Linear(X_dim, 2*h_dim),
                                      nn.ReLU())
        self.D_new = nn.Linear(h_dim*2,h_dim)
        # ==========================================================
        self.D_gan = nn.Linear(h_dim, 1)
        # Discriminator net branch two: For aux cls loss
        self.D_aux = nn.Linear(h_dim, y_dim)
        # self.D_label = nn.Linear(2*y_dim,y_dim)
        self.D_part = nn.Linear(h_dim,7)

    def forward(self, input):
        new = self.D_shared(input)
        h = self.D_new(new)
        return self.D_gan(h),self.D_aux(h),self.D_part(h)

# In GBU setting, using attribute
class _netG_att(nn.Module):
    def __init__(self, opt, att_dim, X_dim):
        super(_netG_att, self).__init__()
        self.main = nn.Sequential(nn.Linear(opt.z_dim + att_dim, h_dim),
                                  nn.LeakyReLU(),
                                  nn.Linear(h_dim, X_dim),
                                  nn.Tanh())
    def forward(self, z, c):
        input = torch.cat([z, c], 1)
        output = self.main(input)
        return output
