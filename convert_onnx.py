from net_models.ssd.ssd_core import SSDNet
import torch
ssd_net = 



torch.onnx.export(net,x,"ssd.onnx",export_params=True,verbose=True)
