import json
import os

json_file = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\Helmet_Data\big_bbox+HNH\big_bbox_json\test_helmet_data_rest.json"
dest_folder = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\Helmet_Data\big_bbox+HNH\big_bbox_txt"

with open(json_file, 'r') as f:
    json_dict = json.load(f)


for num_images in range(len(json_dict)):
    print(num_images)
    filename = json_dict[num_images]['filename'].split('/')
    imgname = filename[-1]
    filename = imgname.split('.')[0] + '.txt'
    obj = json_dict[num_images]['objects']
    for num_bbox in range(len(obj)):
        coor = obj[num_bbox]['relative_coordinates']
        name = obj[num_bbox]['name']
        class_id = 0
        file_path = os.path.join(dest_folder, filename)
        with open(file_path,'a') as f:
            f.write(str(int(class_id))+' '+str(coor['center_x'])+' '+str(coor['center_y'])+' '+str(coor['width'])+' '+str(coor['height'])+ '\n')