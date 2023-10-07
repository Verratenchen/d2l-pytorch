import random
import torch
from d2l import torch as d2l


# 3.2.1
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = 