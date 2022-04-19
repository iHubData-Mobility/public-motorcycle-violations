import json
import cv2
import os
from matplotlib import pyplot as plt

data_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\data\Triple Rider\Triple Riding Images"
json_file = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\preds\3Step_heurestics\FG\result.json"
dest_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\preds\3Step_heurestics\FG\Triple Rider"

with open(json_file, 'r') as f:
    json_dict = json.load(f)

for num_images in range(len(json_dict)):
    filename = json_dict[num_images]['filename'].split('/')
    basename = filename[-1]
    filename = os.path.join(data_folder,basename)
    print(filename)
    image = cv2.imread(filename)

    obj = json_dict[num_images]['objects']
    for num_bbox in range(len(obj)):
        coor = obj[num_bbox]['relative_coordinates']
        name = obj[num_bbox]['name']
        confidence = str(round(obj[num_bbox]['confidence'], 2)) + "%"
        class_id = obj[num_bbox]['class_id']
        y_len = image.shape[0]
        x_len = image.shape[1]
        
        x =coor['center_x']* x_len
        y =coor['center_y']*y_len
        w =coor['width']* x_len
        h =coor['height']* y_len

        x1 = int(x - (w/2))
        y1 = int(y - (h/2)) 
        x2 = int(x + (w/2)) 
        y2 = int(y + (h/2)) 

        start_point = (x1, y1) 
        end_point = (x2, y2) 
  
        # Blue color in BGR 
        if (class_id==2):
            color = (255, 0, 0)
        if(class_id==1):
            color = (0, 255, 0)
        if(class_id==0):
            color = (0, 0, 255)
        else:
            color = (100, 100, 100)
  
        # Line thickness of 2 px 
        thickness = 2
  
        # Using cv2.rectangle() method 
        # Draw a rectangle with blue line borders of thickness of 2 px 
        cv2.rectangle(image, start_point, end_point, color, thickness)
        cv2.rectangle(image, (x1, y1), (x1 + 100, y1 -30), (255,255,255), -1)
        cv2.putText(image, confidence,(x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)
        
    cv2.imwrite(os.path.join(dest_folder, basename), image)
    plt.imshow(image)
    plt.show()