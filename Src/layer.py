import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class Convolution1(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None)))
            return output
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = F.relu(self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None)))
                return output


class Convolution2(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None)))
            return output, input
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = F.relu(self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None)))
                return output, input


class Convolution3(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None))
                return output


class Convolution4(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution4, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None)))
            return output, input
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = F.relu(self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None)))
                return output, input


class Convolution5(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution5, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None))
                return output


class Convolution6(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution6, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.shortcut_weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.shortcut_bn = nn.BatchNorm2d(out_features)
        # self.shortcut_bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # init.constant_(self.bias, 0)
        init.kaiming_normal_(self.shortcut_weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.shortcut_bn.weight, 1)
        # init.constant_(self.shortcut_bias, 0)

    def forward(self, input, gate, shortcut_gate, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights, stride=2, padding=1, bias=None)))
            shortcut_output = self.shortcut_bn(nn.functional.conv2d(input, self.shortcut_weights, stride=2, padding=1, bias=None))
            return output, shortcut_output
        else:
            if i == 0:
                data0 = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                data1 = torch.mean(self.shortcut_weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data0, data1
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = F.relu(self.bn(nn.functional.conv2d(input, new_weights, stride=2, padding=1, bias=None)))
                shortcut_gate = shortcut_gate.view(self.out_features, 1, 1, 1)
                shortcut_new_weights = self.shortcut_weights.mul(shortcut_gate)
                shortcut_output = self.shortcut_bn(nn.functional.conv2d(input, shortcut_new_weights, stride=2, padding=1, bias=None))
                return output, shortcut_output


class Convolution7(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution7, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None))
                return output


class Convolution8(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution8, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None)))
            return output, input
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = F.relu(self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None)))
                return output, input


class Convolution9(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution9, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None))
                return output


class Convolution10(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution10, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.shortcut_weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.shortcut_bn = nn.BatchNorm2d(out_features)
        # self.shortcut_bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # init.constant_(self.bias, 0)
        init.kaiming_normal_(self.shortcut_weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.shortcut_bn.weight, 1)
        # init.constant_(self.shortcut_bias, 0)

    def forward(self, input, gate, shortcut_gate, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights, stride=2, padding=1, bias=None)))
            shortcut_output = self.shortcut_bn(nn.functional.conv2d(input, self.shortcut_weights, stride=2, padding=1, bias=None))
            return output, shortcut_output
        else:
            if i == 0:
                data0 = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                data1 = torch.mean(self.shortcut_weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data0, data1
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = F.relu(self.bn(nn.functional.conv2d(input, new_weights, stride=2, padding=1, bias=None)))
                shortcut_gate = shortcut_gate.view(self.out_features, 1, 1, 1)
                shortcut_new_weights = self.shortcut_weights.mul(shortcut_gate)
                shortcut_output = self.shortcut_bn(nn.functional.conv2d(input, shortcut_new_weights, stride=2, padding=1, bias=None))
                return output, shortcut_output


class Convolution11(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution11, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None))
                return output


class Convolution12(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution12, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None)))
            return output, input
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = F.relu(self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None)))
                return output, input


class Convolution13(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution13, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None))
                return output


class Convolution14(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution14, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.shortcut_weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.shortcut_bn = nn.BatchNorm2d(out_features)
        # self.shortcut_bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # init.constant_(self.bias, 0)
        init.kaiming_normal_(self.shortcut_weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.shortcut_bn.weight, 1)
        # init.constant_(self.shortcut_bias, 0)

    def forward(self, input, gate, shortcut_gate, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights, stride=2, padding=1, bias=None)))
            shortcut_output = self.shortcut_bn(nn.functional.conv2d(input, self.shortcut_weights, stride=2, padding=1, bias=None))
            return output, shortcut_output
        else:
            if i == 0:
                data0 = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                data1 = torch.mean(self.shortcut_weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data0, data1
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = F.relu(self.bn(nn.functional.conv2d(input, new_weights, stride=2, padding=1, bias=None)))
                shortcut_gate = shortcut_gate.view(self.out_features, 1, 1, 1)
                shortcut_new_weights = self.shortcut_weights.mul(shortcut_gate)
                shortcut_output = self.shortcut_bn(nn.functional.conv2d(input, shortcut_new_weights, stride=2, padding=1, bias=None))
                return output, shortcut_output


class Convolution15(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution15, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None))
                return output


class Convolution16(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution16, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = F.relu(self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None)))
            return output, input
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = F.relu(self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None)))
                return output, input


class Convolution17(nn.Module):
    def __init__(self, in_features, out_features):
        super(Convolution17, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(out_features, in_features, 3, 3))
        self.bn = nn.BatchNorm2d(out_features)
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn.weight, 1)
        # nn.init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = self.bn(nn.functional.conv2d(input, self.weights, stride=1, padding=1, bias=None))
            return output
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.out_features, self.in_features * 3 * 3), dim=1)
                return data
            else:
                gate = gate.view(self.out_features, 1, 1, 1)
                new_weights = self.weights.mul(gate)
                output = self.bn(nn.functional.conv2d(input, new_weights, stride=1, padding=1, bias=None))
                return output


class Fully_Connection(nn.Module):
    def __init__(self, in_features, out_features):
        super(Fully_Connection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        # self.bias = Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.weights, 0, 0.01)
        # init.constant_(self.bias, 0)

    def forward(self, input, gate, i, pattern):
        if pattern == 0:
            output = input.mm(self.weights)
            # output.add_(self.bias)
            return output
        else:
            if i == 0:
                data = torch.mean(self.weights.view(self.in_features, self.out_features), dim=1)
                return data
            else:
                gate = gate.view(1, self.in_features).expand(input.size(0), self.in_features)
                new_input = input.mul(gate)
                output = new_input.mm(self.weights)
                # output.add_(self.bias)
                return output
