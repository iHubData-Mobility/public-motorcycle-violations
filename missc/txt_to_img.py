import glob
import cv2
import os
from os.path import basename
import pandas as pd


data_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\Helmet_Data\HELMET_DATASET\test_helmet_data"
txt_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\Helmet_Data\big_bbox+HNH\big_bbox_txt"
dest_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\Helmet_Data\big_bbox+HNH\big_bbox_txt\images"
helmet = 0
no_helmet = 0
rider =0
motorcycles = 0
 
for txt_file in glob.glob(txt_folder + "/*.txt"):
    base = basename(txt_file).split(".")[0]+".jpg"
    filename = os.path.join(data_folder, base)
    image = cv2.imread(filename)
    print(filename)

    obj = pd.read_csv(txt_file, names=["class_id", "x", "y", "w", "h"], sep=" ")
    for num_bbox in range(len(obj)):
        coor_x = obj.iloc[num_bbox]['x']
        coor_y = obj.iloc[num_bbox]['y']
        coor_w = obj.iloc[num_bbox]['w']
        coor_h = obj.iloc[num_bbox]['h']
        class_id = obj.iloc[num_bbox]['class_id']
        y_len = image.shape[0]
        x_len = image.shape[1]
        
        x =coor_x* x_len
        y =coor_y*y_len
        w =coor_w* x_len
        h =coor_h* y_len

        x1 = int(x - (w/2))
        y1 = int(y - (h/2)) 
        x2 = int(x + (w/2)) 
        y2 = int(y + (h/2)) 

        start_point = (x1, y1) 
        end_point = (x2, y2) 

        # Blue color in BGR 
        if (class_id==2):
            color = (0, 0, 255)
            no_helmet += 1
        elif(class_id==1):
            color = (0, 255, 0)
            helmet += 1
        elif(class_id==0):
            color = (255, 0, 0)
            rider += 1
        else:
            color = (100, 100, 100)
            motorcycles +=1
  
        # Line thickness of 2 px 
        thickness = 2
  
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        cv2.rectangle(image, start_point, end_point, color, thickness)        
    cv2.imwrite(os.path.join(dest_folder, base), image)
print("helmet=",helmet )
print("no_helmet=", no_helmet)
print("rider", rider)
print("motorcycle", motorcycles)