from PIL import Image
import cv2
import pandas as pd
import numpy as np
import time

vlc_path = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Final\data\video\2019-07-19-13-01-17_1.mp4"
det_path = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Final\outputs\detections_2019-07-19-13-01-17_1_interpolated.csv"
dest_path = r"C:\Users\Dev Agarwal\Desktop\vehicle_detection\Final\outputs\interpolated_2019-07-19-13-01-17_1.mp4"
def det_to_vid(video_path, det_path, dest_path):
    df = pd.read_csv(det_path)

    input_size = 416

    vid = cv2.VideoCapture(video_path)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dest_path, codec, fps, (width, height))

    frame_num = 0
    triple_rider_violated = []
    HNH_violated = []

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        if (frame_num==5000):
            triple_rider_violated = []
            HNH_violated = []
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        original_h, original_w, _ = frame.shape

        frame_data = df.loc[df['frame_id']==frame_num]
        for i in range(len(frame_data)):
            d = frame_data.iloc[i]
            if (d['class_name']=="No-Helmet"):
                if (d['track_id'] not in HNH_violated):
                    HNH_violated.append(d['track_id'])
                color = [255, 0, 0]
                cv2.rectangle(frame, (int(d['bbox_0']), int(d['bbox_1'])), (int(d['bbox_2']), int(d['bbox_3'])), color, 4)
                #cv2.putText(frame,  "ID:" + str(d['track_id']),(int(d['bbox_0']), int(d['bbox_1']-10)),0, 0.75, (255,255,255),2)
            if (d['class_name']=="Triple_Rider"):
                if (d['track_id'] not in triple_rider_violated):
                    triple_rider_violated.append(d['track_id'])
                color = [255,165,0]
                pts = np.array([[[d['trapez_0'], d['trapez_1']], [d['trapez_2'], d['trapez_3']], [d['trapez_4'], d['trapez_5']], [d['trapez_6'], d['trapez_7']]]], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True , color, 4)
                #cv2.putText(frame, "ID:"+ str(d['track_id']),(int(d['trapez_0']), int(d['trapez_1']-10)),0, 0.75, (255,255,255),2)

        # calculate frames per second of running detections
        #fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        cv2.rectangle(frame,(0,0),(750,100),(255,165,0),-1)
        cv2.putText(frame, 'Triple-riding Violations: ' + str(len(triple_rider_violated)),(30,65),0, 1.5, (255,255,255),4)
        cv2.rectangle(frame,(original_w - 650,0),(original_w,100),(255,0,0),-1)
        cv2.putText(frame, 'Helmet Violations: ' + str(len(HNH_violated)),(original_w - 550,65),0, 1.5, (255,255,255),4)
        #cv2.putText(frame, 'frame_num : ' + str(frame_num),(original_w-850 + 30,30),0, 1, (0,0,0),2)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Output Video", result)

        out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    return 0



det_to_vid(vlc_path, det_path, dest_path)
