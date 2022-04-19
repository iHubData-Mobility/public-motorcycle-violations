import pandas as pd
from shapely.geometry import Polygon
import glob
import shutil
from os.path import basename
import os
import numpy as np
import pickle

txt_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\Triple_Riding_Dataset_Trap(solidification)\data\Triple_Riding_Trap_Testing_Data_4class_txt"
final_txt_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\Triple_Riding_Dataset_Trap(solidification)\data\Triple_Riding_Trap_Testing_Data_4class_vio_by_trap_txt"
threshold = 0.02
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
    """
    initial_instances = len(motorcycle)
    instances = len(motorcycle)
    for i in range(len(rider)):
        if (rider.iloc[i]['instance_id']==-1):
            rider.iat[i, rider.columns.get_loc('instance_id')] = instances
            for j in range(len(rider)):
                if (rider.iloc[j]['instance_id']==-1):
                    if (motor_rider_iou(rider.iloc[i], rider.iloc[j]) > iou_threshold):
                        rider.iat[j, rider.columns.get_loc('instance_id')] = instances
                elif (rider.iloc[j]['instance_id']>=initial_instances):
                    if (motor_rider_iou(rider.iloc[i], rider.iloc[j]) > iou_threshold):
                        rider.iat[i, rider.columns.get_loc('instance_id')] =  rider.iat[j, rider.columns.get_loc('instance_id')]
                        continue
            r = rider.loc[rider['instance_id']==instances]

            df2 = {'class_id': 0, 'x': r['x'].mean(), 'y': (r['y'] + r['h']/2).mean(), 'w':max(r['x'] + r['w']/2) - min(r['x'] - r['w']/2), 'h': (r['h']/2).mean(), 'instance_id':instances}
            motorcycle = motorcycle.append(df2, ignore_index=True)
            instances+=1
    """
    return rider, motorcycle



def heuristic_on_pred(a, motor, rider_ins):
    no_of_bbox = len(motor) + len(rider_ins)
    if (motor['w']==0):
        no_of_bbox = no_of_bbox - 1

    mean_w = (rider_ins['w'].sum() + motor['w'].sum())/no_of_bbox
    mean_x = (rider_ins['x'].sum() + motor['x'].sum())/no_of_bbox

    if (a[4]<mean_w):
        a[4] = motor['w'].mean()
    if (a[0] < mean_x - mean_w/2):
        a[0] = rider_ins['x'].mean()
    if (a[0] > mean_x + mean_w/2):
        a[0] = rider_ins['x'].mean()
    return a
    
def corner_condition(y, xmax, ymax):
    if (y[0]<0):
        y[0] = 0
    if (y[0]>xmax):
        y[0] = xmax
    if (y[1]<0):
        y[1] = 0
    if (y[1]>ymax):
        y[1] = ymax
    if (y[2]<0):
        y[2] = 0
    if (y[2]>xmax):
        y[2] = xmax
    if (y[3]<0):
        y[3] = 0
    if (y[3]>ymax):
        y[3] = ymax
    if (y[4]<0):
        y[4] = 0
    if (y[4]>xmax):
        y[4] = xmax
    if (y[5]<0):
        y[5] = 0
    if (y[5]>ymax):
        y[5] = ymax
    if (y[6]<0):
        y[6] = 0
    if (y[6]>xmax):
        y[6] = xmax
    if (y[7]<0):
        y[7] = 0
    if (y[7]>ymax):
        y[7] = ymax
    
    return y

def trap_rider(y_, rider):

    y_ = [[y_[0], y_[1]], [y_[2], y_[3]], [y_[4], y_[5]], [y_[6], y_[7]]]
    x, y, w, h = rider['x'], rider['y'], rider['w'], rider['h']
    rider = [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]

    return iou(y_, rider)

num_violations = 0
original_w = 1
original_h = 1

final = []

trapez_model = pickle.load(open(r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Video_Testing\Curriculum\L4\L4_improving_code_peg_ag_motorcycle\data\Trapezium_Prediction_Weights.pickle", 'rb'))
for files in glob.glob(txt_folder + "/*.txt"):
    df = pd.read_csv(files, sep=" ", names=['class_id', 'x', 'y', 'w', 'h'])
    motorcycle = df.loc[df['class_id']==3]
    rider = df.loc[df['class_id']==0]

    rider, motorcycle = get_instance(rider, motorcycle, 0.01)
    y = np.zeros((len(motorcycle), 8))
    num = 0

    for i in range(len(motorcycle)):
        input = []
        motor = motorcycle.iloc[i]
        instance = motor['instance_id']
        rider_ins = rider.loc[rider['instance_id']==instance]
        input.extend([float (motor['x']),float (motor['y']),float (motor['w']),float (motor['h'])])


        if (len(rider_ins)>=3):
            for j in range(len(rider_ins)):
                if (j > 4):
                    continue
                input.extend([float (rider_ins.iloc[j]['x']),float (rider_ins.iloc[j]['y']),float (rider_ins.iloc[j]['w']),float (rider_ins.iloc[j]['h'])])
            x=np.zeros((1,24))
            x[0,:len(input)] = np.array(input).reshape((1,-1))
            predict = trapez_model.predict(x)
            a = predict[0]
            a = heuristic_on_pred(a, motor, rider_ins)

            y[num][0], y[num][1], y[num][2], y[num][3], y[num][4], y[num][5],y[num][6] ,y[num][7]  = (a[0] - a[4]/2)*original_w,(a[5]+x[0][1]-x[0][3]/2)*original_h, (a[0] - a[4]/2)*original_w,(a[2]+x[0][1]+x[0][3]/2)*original_h, (a[0] + a[4]/2)*original_w, (a[3]+x[0][1]+x[0][3]/2)*original_h, (a[0] + a[4]/2)*original_w, (a[6]+x[0][1]-x[0][3]/2)*original_h
            
            y[num] = corner_condition(y[num], original_w, original_h)
            num_riders = 0

            for k in range(len(rider_ins)):
                if (trap_rider(y[num], rider_ins.iloc[k]) > threshold):
                    num_riders += 1
            print(num_riders)
            final.append(num_riders)

            shutil.copy(files, final_txt_folder)
            #shutil.copy(files, test_set_final_txt)
            #shutil.copy(os.path.join(img_path, os.path.basename(files).split('.')[0] + ".jpg"), test_set_final)


        """
        tracker_instance.append([num, len(rider_ins)])
        x=np.zeros((1,24))
        x[0,:len(input)] = np.array(input).reshape((1,-1))
        predict = trapez_model.predict(x)

        a = predict[0]
        a = heuristic_on_pred(a, motor, rider_ins)
        y[num][0], y[num][1], y[num][2], y[num][3], y[num][4], y[num][5],y[num][6] ,y[num][7]  = (a[0] - a[4]/2)*original_w,(a[5]+x[0][1]-x[0][3]/2)*original_h, (a[0] - a[4]/2)*original_w,(a[2]+x[0][1]+x[0][3]/2)*original_h, (a[0] + a[4]/2)*original_w, (a[3]+x[0][1]+x[0][3]/2)*original_h, (a[0] + a[4]/2)*original_w, (a[6]+x[0][1]-x[0][3]/2)*original_h
        
        y[num] = corner_condition(y[num], 1, 1)
        num = num+1
        """
print(len(final))