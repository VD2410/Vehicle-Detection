import json
import numpy as np
import ssd_net
# with open('/home/vbatvia/VC/Lab/Object detection/cityscapes_samples_labels/konigswinter/konigswinter_000000_000000_gtCoarse_polygons.json','r') as f:
# 	frame_info = json.load(f)
# print(frame_info['objects'][0]['polygon'])
#
# polygons = np.asarray(frame_info['objects'][0]['polygon'],dtype = np.float32)
# left_top = np.min(polygons , axis = 0)
# right_bottom = np.max(polygons , axis = 0)
# print(left_top)
# print(right_bottom)

net = ssd_net.SSD(1)
