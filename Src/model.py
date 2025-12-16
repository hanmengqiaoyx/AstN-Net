import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from layer import Convolution1, Convolution2, Convolution3, Convolution4, Convolution5, Convolution6, Convolution7, Convolution8, Convolution9, \
    Convolution10, Convolution11, Convolution12, Convolution13, Convolution14, Convolution15, Convolution16, Convolution17, Fully_Connection


class ResNet18(nn.Module):
    def __init__(self, in_channel=3, c_in0=64, c_in1=128, c_in2=256, c_in3=512, f_in=512, num_classes=10):
        super(ResNet18, self).__init__()
        self.c_in0 = c_in0
        self.c_in1 = c_in1
        self.c_in2 = c_in2
        self.c_in3 = c_in3
        self.f_in = f_in
        self.c_layer1 = Convolution1(in_channel, c_in0)
        self.c_layer2 = Convolution2(c_in0, c_in0)
        self.c_layer3 = Convolution3(c_in0, c_in0)
        self.c_layer4 = Convolution4(c_in0, c_in0)
        self.c_layer5 = Convolution5(c_in0, c_in0)
        self.c_layer6 = Convolution6(c_in0, c_in1)
        self.c_layer7 = Convolution7(c_in1, c_in1)
        self.c_layer8 = Convolution8(c_in1, c_in1)
        self.c_layer9 = Convolution9(c_in1, c_in1)
        self.c_layer10 = Convolution10(c_in1, c_in2)
        self.c_layer11 = Convolution11(c_in2, c_in2)
        self.c_layer12 = Convolution12(c_in2, c_in2)
        self.c_layer13 = Convolution13(c_in2, c_in2)
        self.c_layer14 = Convolution14(c_in2, c_in3)
        self.c_layer15 = Convolution15(c_in3, c_in3)
        self.c_layer16 = Convolution16(c_in3, c_in3)
        self.c_layer17 = Convolution17(c_in3, c_in3)
        self.f_layer18 = Fully_Connection(f_in, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input, gates, pattern):
        if pattern == 0:
            # 1~17
            out = self.c_layer1(input, 0, 0, pattern)
            out, out0 = self.c_layer2(out, 0, 0, pattern)
            out = self.c_layer3(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer4(out, 0, 0, pattern)
            out = self.c_layer5(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer6(out, 0, 0, 0, pattern)
            out = self.c_layer7(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer8(out, 0, 0, pattern)
            out = self.c_layer9(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer10(out, 0, 0, 0, pattern)
            out = self.c_layer11(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer12(out, 0, 0, pattern)
            out = self.c_layer13(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer14(out, 0, 0, 0, pattern)
            out = self.c_layer15(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer16(out, 0, 0, pattern)
            out = self.c_layer17(out, 0, 0, pattern)
            out += out0
            out = self.relu(out)
            out = self.avgpool(out)
            x = torch.flatten(out, 1)
            # 18
            out = self.f_layer18(x, 0, 0, pattern)
            return out
        elif pattern == 1:
            data1 = self.c_layer1(input, 0, 0, pattern)
            data2 = self.c_layer2(input, 0, 0, pattern)
            data3 = self.c_layer3(input, 0, 0, pattern)
            data4 = self.c_layer4(input, 0, 0, pattern)
            data5 = self.c_layer5(input, 0, 0, pattern)
            data6, shortcut_data6 = self.c_layer6(input, 0, 0, 0, pattern)
            data7 = self.c_layer7(input, 0, 0, pattern)
            data8 = self.c_layer8(input, 0, 0, pattern)
            data9 = self.c_layer9(input, 0, 0, pattern)
            data10, shortcut_data10 = self.c_layer10(input, 0, 0, 0, pattern)
            data11 = self.c_layer11(input, 0, 0, pattern)
            data12 = self.c_layer12(input, 0, 0, pattern)
            data13 = self.c_layer13(input, 0, 0, pattern)
            data14, shortcut_data14 = self.c_layer14(input, 0, 0, 0, pattern)
            data15 = self.c_layer15(input, 0, 0, pattern)
            data16 = self.c_layer16(input, 0, 0, pattern)
            data17 = self.c_layer17(input, 0, 0, pattern)
            data18 = self.f_layer18(input, 0, 0, pattern)
            weights = torch.cat((data1, data2, data3, data4, data5, data6, shortcut_data6, data7, data8, data9, data10, shortcut_data10, data11,
                                 data12, data13, data14, shortcut_data14, data15, data16, data17, data18), dim=0).view(1, 1, -1, 64)  # [1, 1, 83, 64]
            return weights
        elif pattern == 2:
            c_gate1 = gates[0:64]
            c_gate2 = gates[64:128]
            c_gate3 = gates[128:192]
            c_gate4 = gates[192:256]
            c_gate5 = gates[256:320]
            c_gate6 = gates[320:448]
            c_shortcut_gate6 = gates[448:576]
            c_gate7 = gates[576:704]
            c_gate8 = gates[704:832]
            c_gate9 = gates[832:960]
            c_gate10 = gates[960:1216]
            c_shortcut_gate10 = gates[1216:1472]
            c_gate11 = gates[1472:1728]
            c_gate12 = gates[1728:1984]
            c_gate13 = gates[1984:2240]
            c_gate14 = gates[2240:2752]
            c_shortcut_gate14 = gates[2752:3264]
            c_gate15 = gates[3264:3776]
            c_gate16 = gates[3776:4288]
            c_gate17 = gates[4288:4800]
            f_gate18 = gates[4800:5312]
            # 1~17
            out = self.c_layer1(input, c_gate1, 1, pattern)
            out, out0 = self.c_layer2(out, c_gate2, 1, pattern)
            out = self.c_layer3(out, c_gate3, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer4(out, c_gate4, 1, pattern)
            out = self.c_layer5(out, c_gate5, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer6(out, c_gate6, c_shortcut_gate6, 1, pattern)
            out = self.c_layer7(out, c_gate7, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer8(out, c_gate8, 1, pattern)
            out = self.c_layer9(out, c_gate9, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer10(out, c_gate10, c_shortcut_gate10, 1, pattern)
            out = self.c_layer11(out, c_gate11, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer12(out, c_gate12, 1, pattern)
            out = self.c_layer13(out, c_gate13, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer14(out, c_gate14, c_shortcut_gate14, 1, pattern)
            out = self.c_layer15(out, c_gate15, 1, pattern)
            out += out0
            out = self.relu(out)
            out, out0 = self.c_layer16(out, c_gate16, 1, pattern)
            out = self.c_layer17(out, c_gate17, 1, pattern)
            out += out0
            out = self.relu(out)
            out = self.avgpool(out)
            x = torch.flatten(out, 1)
            # 18
            out = self.f_layer18(x, f_gate18, 1, pattern)
            return out


class Astrocyte_Network(nn.Module):
    def __init__(self, bias=False):
        super(Astrocyte_Network, self).__init__()
        self.dw_weights0 = Parameter(torch.Tensor(16, 1, 3, 3))
        self.dw_bn0 = nn.BatchNorm2d(16)
        # self.dw_bias0 = Parameter(torch.Tensor(4))
        self.dw_weights1 = Parameter(torch.Tensor(32, 16, 3, 3))
        self.dw_bn1 = nn.BatchNorm2d(32)
        # self.dw_bias1 = Parameter(torch.Tensor(8))
        self.dw_weights2 = Parameter(torch.Tensor(64, 32, 3, 3))
        self.dw_bn2 = nn.BatchNorm2d(64)
        # self.dw_bias2 = Parameter(torch.Tensor(16))
        self.dw_weights3 = Parameter(torch.Tensor(128, 64, 3, 3))
        self.dw_bn3 = nn.BatchNorm2d(128)
        # self.dw_bias3 = Parameter(torch.Tensor(32))
        self.up_sample0 = Parameter(torch.Tensor(128, 64, 2, 2))
        self.up_bn00 = nn.BatchNorm2d(64)
        # self.up_bias00 = Parameter(torch.Tensor(16))
        self.up_weights0 = Parameter(torch.Tensor(64, 128, 3, 3))
        self.up_bn01 = nn.BatchNorm2d(64)
        # self.up_bias01 = Parameter(torch.Tensor(16))
        self.up_sample1 = Parameter(torch.Tensor(64, 32, 3, 2))
        self.up_bn10 = nn.BatchNorm2d(32)
        # self.up_bias10 = Parameter(torch.Tensor(8))
        self.up_weights1 = Parameter(torch.Tensor(32, 64, 3, 3))
        self.up_bn11 = nn.BatchNorm2d(32)
        # self.up_bias11 = Parameter(torch.Tensor(8))
        self.up_sample2 = Parameter(torch.Tensor(32, 16, 3, 2))
        self.up_bn20 = nn.BatchNorm2d(16)
        # self.up_bias20 = Parameter(torch.Tensor(4))
        self.up_weights2 = Parameter(torch.Tensor(16, 32, 3, 3))
        self.up_bn21 = nn.BatchNorm2d(16)
        # self.up_bias21 = Parameter(torch.Tensor(4))
        self.gate_weights = Parameter(torch.Tensor(1, 16, 3, 3))
        self.gate_bn = nn.BatchNorm2d(1)
        # self.gate_bias = Parameter(torch.Tensor(1))
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.dw_weights0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.dw_weights3, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_sample2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights0, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.up_weights2, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.gate_weights, mode='fan_out', nonlinearity='relu')
        init.constant_(self.dw_bn0.weight, 1)
        init.constant_(self.dw_bn1.weight, 1)
        init.constant_(self.dw_bn2.weight, 1)
        init.constant_(self.dw_bn3.weight, 1)
        init.constant_(self.up_bn00.weight, 1)
        init.constant_(self.up_bn01.weight, 1)
        init.constant_(self.up_bn10.weight, 1)
        init.constant_(self.up_bn11.weight, 1)
        init.constant_(self.up_bn20.weight, 1)
        init.constant_(self.up_bn21.weight, 1)
        init.constant_(self.gate_bn.weight, 1)

    def forward(self, weights):
        layer00 = F.relu(self.dw_bn0(nn.functional.conv2d(weights, self.dw_weights0, stride=1, padding=1, bias=None)))
        layer01 = self.maxpool(layer00)
        layer10 = F.relu(self.dw_bn1(nn.functional.conv2d(layer01, self.dw_weights1, stride=1, padding=1, bias=None)))
        layer11 = self.maxpool(layer10)
        layer20 = F.relu(self.dw_bn2(nn.functional.conv2d(layer11, self.dw_weights2, stride=1, padding=1, bias=None)))
        layer21 = self.maxpool(layer20)
        layer3 = F.relu(self.dw_bn3(nn.functional.conv2d(layer21, self.dw_weights3, stride=1, padding=1, bias=None)))  # [1, 16, 9, 8]
        layer40 = F.relu(self.up_bn00(nn.functional.conv_transpose2d(layer3, self.up_sample0, stride=2, bias=None)))
        layer41 = torch.cat((layer20, layer40), dim=1)
        layer42 = F.relu(self.up_bn01(nn.functional.conv2d(layer41, self.up_weights0, stride=1, padding=1, bias=None)))
        layer50 = F.relu(self.up_bn10(nn.functional.conv_transpose2d(layer42, self.up_sample1, stride=2, bias=None)))
        layer51 = torch.cat((layer10, layer50), dim=1)
        layer52 = F.relu(self.up_bn11(nn.functional.conv2d(layer51, self.up_weights1, stride=1, padding=1, bias=None)))
        layer60 = F.relu(self.up_bn20(nn.functional.conv_transpose2d(layer52, self.up_sample2, stride=2, bias=None)))
        layer61 = torch.cat((layer00, layer60), dim=1)
        layer62 = F.relu(self.up_bn21(nn.functional.conv2d(layer61, self.up_weights2, stride=1, padding=1, bias=None)))
        layer_out = F.sigmoid(self.gate_bn(nn.functional.conv2d(layer62, self.gate_weights, stride=1, padding=1, bias=None)))  # [1, 1, 83, 64]
        gates = layer_out.view(-1)
        return gates