import glob, os.path as osp
import cv2
import numpy as np
occu_map_paths= sorted(glob.glob(osp.join('data/maps_test/','*', 'floor_trav_0_v2.png')))
print(len(occu_map_paths))
hs = []
ws = []
for map_path in occu_map_paths:
    
    bitmap = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    free_space = (bitmap >= 250).astype(np.uint8)

    height, width = free_space.shape
    hs.append(height)
    ws.append(width)
    if(height < 1024 or width < 1024):
    	print("h: {}, wid: {}".format(height, width))

hs = np.array(hs)
ws = np.array(ws)
print("min height is {}".format(np.min(hs)))
print("max height is {}".format(np.max(hs)))
print("min width is {}".format(np.min(ws)))
print("max width is {}".format(np.max(ws)))