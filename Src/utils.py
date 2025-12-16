# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from functools import reduce
from operator import mul
import random
import torch.nn.functional as F
import csv
import os
import sys
import time
import math


term_width = 100
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def set_parameters(meta_networks, grad_list):
    offset = 0
    for name, params in meta_networks.named_parameters():
        weight_shape = params.size()
        weight_flat_size = reduce(mul, weight_shape, 1)
        params.data = grad_list[offset:offset + weight_flat_size].view(*weight_shape)
        offset += weight_flat_size


def get_parameters(meta_learner):
    _loss_grad = []
    for name, params in meta_learner.named_parameters():
        _loss_grad.append(params.view(-1).unsqueeze(1))
    flat_loss_grad = torch.cat(_loss_grad, dim=0).view(1, -1)
    return flat_loss_grad


class CSV_writer():
    def __init__(self, path):
        self.file = open(path,'w',encoding='utf-8')
        self.writer=csv.writer(self.file)
    def Writer(self, content):
        self.writer.writerow([str(content.data.cpu().numpy())])
    def close(self):
        self.file.close()


def reset_parameters(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.uniform_(m.weight, a=-5, b=5)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-5, b=5)


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
