import glob
import os
def append_multiple_lines(file_name, lines_to_append):
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        appendEOL = False
        # Move read cursor to the start of file.
        file_object.seek(0)
        # Check if file is not empty
        data = file_object.read(100)
        if len(data) > 0:
            appendEOL = True
        # Iterate over each string in the list
        for line in lines_to_append:
            # If file is not empty then append '\n' before first line for
            # other lines always append '\n' before appending line
            if appendEOL == True:
                file_object.write("\n")
            else:
                appendEOL = True
            # Append element at the end of file
            file_object.write(line)

# folder = 'idd_temporal_test_2_riders'
paths = glob.glob(r'C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\WACV paper Implementation\model_testing\test_images_roi_proposed_model/*.*g')
# file_list = []
# for i in range(len(paths)):
#     file_list.append(paths[i])
#     file_list.sort()

for i in range(len(paths)):
    imagename = os.path.basename(paths[i])
    paths[i] = '/content/gdrive/MyDrive/test_images_roi_proposed_model/'+ imagename

append_multiple_lines(r'C:\Users\Dev Agarwal\Desktop\vehicle_detection\Evaluation\WACV paper Implementation\model_testing\test_images_roi_proposed_model\test_images_roi_proposed_model.txt',paths)
