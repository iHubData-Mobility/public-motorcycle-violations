import pandas as pd
from shapely.geometry import Polygon
import glob
import shutil
from os.path import basename
import os

txt_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\Triple_Riding_Dataset_Trap(solidification)\data\test_set_4class_txt"
threshold = 0.01
final_txt_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\Triple_Riding_Dataset_Trap(solidification)\data\test_set_4class_vio_by_rect_txt"
flag_val = 0.08
#test_set_final_txt = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\Triple_Riding_Dataset_Trap(solidification)\data\test_set_final_txt"
#img_path = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\Triple_Riding_Dataset_Trap(solidification)\data\test_set"
#test_set_final =r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\Triple_Riding_Dataset_Trap(solidification)\data\test_set_final"

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

def motor_rider_iou(motorcycle, rider):
  x, y, w, h = motorcycle['x'], motorcycle['y'], motorcycle['w'], motorcycle['h']
  motor = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
  x, y, w, h = rider['x'], rider['y'], rider['w'], rider['h']
  rider = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
  return iou(motor, rider)

num_violations = 0
for files in glob.glob(txt_folder + "/*.txt"):
    df = pd.read_csv(files, sep=" ", names=['class_id', 'x', 'y', 'w', 'h'])
    motor = df.loc[df['class_id']==3]
    rider = df.loc[df['class_id']==0]
    rect = motor

    for i in range(len(motor)):
        m = motor.iloc[i]
        ymin = m['y'] - m['h']/2
        ymax = m['y'] + m['h']/2
        for j in range(len(rider)):
            r = rider.iloc[j]
            if (motor_rider_iou(m, r) > threshold):
                ym = r['y'] - r['h']/2
                yma = r['y'] + r['h']/2
                
                if (ym < ymin):
                    ymin = ym
                if (yma > ymax):
                    ymax = yma
        rect.iat[i, rect.columns.get_loc('y')] = (ymax + ymin )/2
        rect.iat[i, rect.columns.get_loc('h')] = ymax - ymin

    for i in range(len(rect)):
        m = rect.iloc[i]
        num_rider = 0
        for j in range(len(rider)):
            r = rider.iloc[j]
            if (motor_rider_iou(m, r) > flag_val):
                num_rider += 1
        
        if (num_rider>2):
            rect.iat[i, motor.columns.get_loc('class_id')] = 1
            num_violations += 1
            #shutil.copy(files, test_set_final_txt)
            #shutil.copy(os.path.join(img_path, os.path.basename(files).split('.')[0] + ".jpg"), test_set_final)
        else:
            rect.iat[i, motor.columns.get_loc('class_id')] = 0

    rider["class_id"].replace({0: 2}, inplace=True)
    data = pd.concat([rect, rider])    

    with open(os.path.join(final_txt_folder, basename(files)), "w") as f:
        for i in range(len(data)):
            f.write(str(int(data.iloc[i]['class_id'])) + " " + str(data.iloc[i]['x']) + " " + str(data.iloc[i]['y']) + " " + str(data.iloc[i]['w']) + " " + str(data.iloc[i]['h']) + "\n" )

print(num_violations)
