import torch
import torch.nn as nn
from torch.autograd import Variable


# Residual block
class ResBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlockUp, self).__init__()
        # CBN(n_category, n_hidden, num_features)
        self.condBN1 = CBN(128+20, 128+20, in_channels)
        self.conv1 = deconv4x4(in_channels, out_channels, stride)
        self.condBN2 = CBN(128+20, 128+20, out_channels)
        self.conv2 = deconv3x3(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x_clss_tuple):
        x, clss = x_clss_tuple
        # x: batch x ci x h x w
        # class: batch x (128 + 20)
        residual = x
        out = self.condBN1(x, clss)
        out = self.relu(out)
        # self.conv1: deconv4x4 ConvTranspose2d(ci, c0, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        out = self.conv1(out)
        # out: batch x co x 2h x 2w
        out = self.condBN2(out, clss)
        out = self.relu(out)
        # self.conv2: deconv3x3 ConvTranspose2d(c0, c0, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        out = self.conv2(out)
        # out: batch x co x 2h x 2w
        if self.downsample is not None:
            # x: batch x ci x h x w
            # self.downsample: ConvTranspose2d(ci, c0, kernel_size=(2, 2), stride=(2, 2), bias=False)
            residual = self.downsample(x)
            # residual: batch x co x 2h x 2w
        out += residual
        return out

# 3x3 convolution H->H
def deconv3x3(in_channels, out_channels, stride=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# 4x4 convolution H->2H
def deconv4x4(in_channels, out_channels, stride=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                     stride=stride, padding=1, bias=False)



# Residual block
class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlockDown, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(x)
        out = self.conv1(out)
        # out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        # out = self.relu(out)
        return out

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Self attention block
class SelfAttnBlock(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim):
        super(SelfAttnBlock,self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = max(in_dim//8,1) , kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = max(in_dim//8,1) , kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention


class CBN(nn.Module):

    def __init__(self, n_category, n_hidden, num_features, eps=1e-5, momentum=0.9, is_training=True):
        super(CBN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.is_training = is_training

        #Affine transform parameters
        self.gamma = nn.Parameter(torch.Tensor(num_features), requires_grad = True)
        self.beta = nn.Parameter(torch.Tensor(num_features), requires_grad = True)

        #Running mean and variance, these parameters are not trained by backprop
        self.running_mean = nn.Parameter(torch.Tensor(num_features), requires_grad = False)
        self.running_var = nn.Parameter(torch.Tensor(num_features), requires_grad = False)
        self.num_batches_tracked = nn.Parameter(torch.Tensor(1), requires_grad = False)

        #Parameter initilization
        self.reset_parameters()

        #MLP parameters
        self.n_category = n_category
        self.n_hidden = n_hidden

        # MLP used to predict betas and gammas
        self.fc_gamma = nn.Sequential(
            nn.Linear(self.n_category, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.num_features),
            )

        self.fc_beta = nn.Sequential(
            nn.Linear(self.n_category, self.n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_hidden, self.num_features),
            )

        # Initialize weights using Xavier initialization and biases with constant value
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        self.gamma.data.uniform_()
        self.beta.data.zero_()

    def forward(self, input, category_one_hot):

        N, C, H, W = input.size()

        exponential_average_factor = 0.0
        if self.is_training:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked
            else:  # use exponential moving average
                exponential_average_factor = 1 - self.momentum

        # Obtain delta values from MLP
        delta_gamma = self.fc_gamma(category_one_hot)
        delta_beta = self.fc_beta(category_one_hot)

        gamma_cloned = self.gamma.clone()
        beta_cloned = self.beta.clone()

        gamma_cloned = gamma_cloned.view(1,C).expand(N,C).clone()
        beta_cloned = beta_cloned.view(1,C).expand(N,C).clone()

        # Update the values
        gamma_cloned += delta_gamma
        beta_cloned += delta_beta

        # Standard batch normalization
        out, running_mean, running_var = batch_norm(input, self.running_mean, self.running_var, gamma_cloned, beta_cloned,
            self.is_training, exponential_average_factor, self.eps)

        if self.is_training:
            self.running_mean.data = running_mean.data
            self.running_var.data = running_var.data

        return out


def batch_norm(input, running_mean, running_var, gammas, betas,
            is_training, exponential_average_factor, eps):
        # Extract the dimensions
        N, C, H, W = input.size()

        # Mini-batch mean and variance
        input_channel_major = input.permute(1, 0, 2, 3).contiguous().view(input.size(1), -1)
        mean = input_channel_major.mean(dim=1)
        variance = input_channel_major.var(dim=1)

        # Normalize
        if is_training:

            #Compute running mean and variance
            running_mean = running_mean*(1-exponential_average_factor) + mean*exponential_average_factor
            running_var = running_var*(1-exponential_average_factor) + variance*exponential_average_factor

            # Training mode, normalize the data using its mean and variance
            X_hat = (input - mean.view(1,C,1,1).expand((N, C, H, W))) * 1.0 / torch.sqrt(variance.view(1,C,1,1).expand((N, C, H, W)) + eps)
        else:
            # Test mode, normalize the data using the running mean and variance
            X_hat = (input - running_mean.view(1,C,1,1).expand((N, C, H, W))) * 1.0 / torch.sqrt(running_var.view(1,C,1,1).expand((N, C, H, W)) + eps)

        # Scale and shift
        out = gammas.contiguous().view(N,C,1,1).expand((N, C, H, W)) * X_hat + betas.contiguous().view(N,C,1,1).expand((N, C, H, W))

        return out, running_mean, running_var


if __name__ == '__main__':

    model = CBN(2,2,128)
    x = torch.ones([4,128,16,16])
    one_hot = torch.zeros([4,2])
    one_hot[0,0] = 1
    one_hot[1,1] = 1
    one_hot[2,1] = 1
    one_hot[3,0] = 1

    x = Variable(x)
    one_hot = Variable(one_hot)

    print("x before", x.size())
    print("x after", model(x,one_hot).size())
    print(model.state_dict)
