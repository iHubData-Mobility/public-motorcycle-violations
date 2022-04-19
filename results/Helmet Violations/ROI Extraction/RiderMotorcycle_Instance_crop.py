import pandas as pd
import numpy as numpy
from shapely.geometry import Polygon
import os
from os.path import basename
import cv2
import glob

img_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\WACV paper Implementation\model_testing\test_images"
txt_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\WACV paper Implementation\model_testing\test_images_v4_VL_2C_FG(R_R+M)_txt"
roi_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\WACV paper Implementation\model_testing\test_images_roi_WACV"

def iou(bbox1, bbox2):
    """
    args: 
    bbox1: List of coordinates: [(x1,y1), (x2,y2), (x3,y3)....,(xn,yn)]
    bbox1: List of coordinates: [(x1,y1), (x2,y2), (x3,y3)....,(xn,yn)]
    
    output:
    iou: (float)
    """
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)
    poly3 = poly2.intersection(poly1)
        
    Ar1 = float(poly1.area)
    Ar2 = float(poly2.area)
    Ar_of_int = float(poly3.area)

    iou = Ar_of_int / (Ar1 + Ar2 - Ar_of_int)

    return iou

def union(bbox1, bbox2):
    """
    args: 
    bbox1: List of coordinates: [(x1,y1), (x2,y2), (x3,y3)....,(xn,yn)]
    bbox1: List of coordinates: [(x1,y1), (x2,y2), (x3,y3)....,(xn,yn)]
    
    output:
    iou: (float)
    """
    poly1 = Polygon(bbox1)
    poly2 = Polygon(bbox2)
    poly3 = poly2.union(poly1)

    
    return Polygon(poly3)

def motor_rider_iou(motorcycle, rider):
    x, y, w, h = motorcycle['x'], motorcycle['y'], motorcycle['w'], motorcycle['h']
    motor = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
    x, y, w, h = rider['x'], rider['y'], rider['w'], rider['h']
    rider = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
    return iou(motor, rider)

def motor2_rider_iou(motorcycle_now, motorcycle_before, rider, instance_now, instance_before):
    if (motor_rider_iou(motorcycle_now, rider) > motor_rider_iou(motorcycle_before, rider)):
        return instance_now
    else:
        return instance_before

def get_instance(rider, motorcycle, iou_threshold):
    """
    args:
    rider, motorcycle : pd.DataFrame
    output:
    rider, motorycle : pd.DataFrame with a column named 'instance_id'
    """
    rider['instance_id'] = -1
    motorcycle['instance_id'] = -1

    for i in range(len(motorcycle)):
        motorcycle.iat[i,motorcycle.columns.get_loc('instance_id')] = i
        for j in range(len(rider)):
            if (motor_rider_iou(motorcycle.iloc[i], rider.iloc[j]) > iou_threshold):
                if (rider.iloc[j]['instance_id'] == -1):
                    rider.iat[j,rider.columns.get_loc('instance_id')] = i
                else:
                    instance = int(rider.iloc[j]['instance_id'])
                    instance_final = motor2_rider_iou(motorcycle.iloc[i], motorcycle.iloc[instance], rider.iloc[j], i, instance)
                    rider.iat[j,rider.columns.get_loc('instance_id')] = instance_final
    return rider, motorcycle


def iou2(rider, head):
    xr1, xr2, yr1, yr2 = rider['center_x'] - rider['width']/2, rider['center_x'] + rider['width']/2, rider['center_y']-rider['height']/2, rider['center_y']+rider['height']/2
    xh1, xh2, yh1, yh2 = head['center_x'] - head['width']/2, head['center_x'] + head['width']/2, head['center_y']-head['height']/2, head['center_y']+head['height']/2
    if ((xr2<xh1) or (xr1>xh2) or (yr2<yh1) or (yr1>yh2)):
        overlap = 0
    else:
        x_coor = np.sort(np.array([xr1, xr2, xh1, xh2]))
        y_coor = np.sort(np.array([yr1, yr2, yh1, yh2]))
        width = x_coor[1] - x_coor[2]
        height = y_coor[1]-y_coor[2]
        overlap = width * height
        
    Ar1 = rider['width']*rider['height']
    Ar2 = head['width']*head['height']
    NIOU = round((Ar2 - overlap),4)/round((Ar1+Ar2-overlap),4)
    IOU = round(overlap,4)/round((Ar1+Ar2-overlap),4)
    return IOU, NIOU

def coeff(rider, head):
    IOU, NIOU = iou2(rider, head)
    if (round(NIOU,4)==0):
        val = 10000
    else:
        val = IOU/NIOU
    return (val)


for files in glob.glob(txt_folder+"/*.txt"):
    df = pd.read_csv(files, sep=" ", names=['class_id', 'x', 'y', 'w', 'h'])

    rider = df.loc[df['class_id']==0]
    motorcycle = df.loc[df['class_id']==3]

    rider, motorcycle = get_instance(rider, motorcycle, 0.01)

    path = os.path.join(img_folder, basename(files).split('.')[0] + ".jpg")
    img = cv2.imread(path)

    for i in range(len(motorcycle)):
        motor = motorcycle.loc[motorcycle['instance_id']==i]
        instance = motorcycle.iloc[i]['instance_id']
        ride = rider.loc[rider['instance_id']== instance]

        if (len(ride)==0):
            continue
        
        print(type(motor['x'] + motor['w']/2))
        print(motor['x'] + motor['w']/2)
        print(type(max(ride['x'] + ride['w']/2)))
        xmax = max(float(motor['x'] + motor['w']/2), max(ride['x'] + ride['w']/2))
        xmin = min(float(motor['x'] - motor['w']/2), min(ride['x'] - ride['w']/2))
        ymax = max(float(motor['y'] + motor['h']/2), max(ride['y'] + ride['h']/2))
        ymin = min(float(motor['y'] - motor['h']/2), min(ride['y'] - ride['h']/2))

        w = xmax - xmin
        h = ymax - ymin

        xmax = xmax + 0.05*w
        xmin = xmin - 0.05*w

        ymax = ymax + 0.05 * h
        ymin = ymin - 0.05 * h

        if (xmin < 0):
            xmin=0
        if (xmax >1):
            xmax=1
        if (ymax>1):
            ymax =1
        if(ymin<0):
            ymin =0

        t = int(ymin*img.shape[0])
        l = int(xmin*img.shape[1])
        b = int(ymax*img.shape[0])
        r = int(xmax*img.shape[1])

        if t<0 or l<0 or b<0 or r<0:
            continue
        patch = img[t:b, l:r]
        print(patch)
        cv2.imwrite(os.path.join(roi_folder, basename(files).split('.')[0] +"_"+str(i)+ ".jpg"), patch)