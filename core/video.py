from PIL import Image
import cv2
import pandas as pd
import numpy as np
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import os

flags.DEFINE_string('videos', './data/video/3idiots.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('detections', './outputs/detections/detections_3idiots_interpolated.csv', 'path to detections file(.csv) for the video')
flags.DEFINE_string('outputs', './outputs/detections/3idiots.mp4', 'path to final output video')


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
        

        out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
    print("Video saved at {}".format(dest_path))
    return 0

def interpolation(path):
    detections = pd.read_csv(path)

    data = []
    unique_ids = detections.track_id.unique()
    for ids in unique_ids:
        df = detections.loc[detections['track_id']==ids]
        for i in range(len(df)):
            if (i==0):
                prev_frame = int(df.iloc[i]['frame_id'])
                class_name = df.iloc[i]['class_name']
                prev = df.iloc[0]
                continue

            curr = df.iloc[i]
            curr_frame = int(curr['frame_id'])
            diff = curr_frame - prev_frame

            if (diff == 1):
                prev_frame = curr_frame
                prev = curr
            elif (diff <= 10):
                for j in range(1,diff):
                    interpolation = [
                        str(prev_frame + j),
                        class_name,
                        ids,
                        str(int(prev['trapez_0'] + (curr['trapez_0'] - prev['trapez_0'])*j/diff)),
                        str(int(prev['trapez_1'] + (curr['trapez_1'] - prev['trapez_1'])*j/diff)),
                        str(int(prev['trapez_2'] + (curr['trapez_2'] - prev['trapez_2'])*j/diff)),
                        str(int(prev['trapez_3'] + (curr['trapez_3'] - prev['trapez_3'])*j/diff)),
                        str(int(prev['trapez_4'] + (curr['trapez_4'] - prev['trapez_4'])*j/diff)),
                        str(int(prev['trapez_5'] + (curr['trapez_5'] - prev['trapez_5'])*j/diff)),
                        str(int(prev['trapez_6'] + (curr['trapez_6'] - prev['trapez_6'])*j/diff)),
                        str(int(prev['trapez_7'] + (curr['trapez_7'] - prev['trapez_7'])*j/diff)),
                        str(int(prev['bbox_0'] + (curr['bbox_0'] - prev['bbox_0'])*j/diff)),
                        str(int(prev['bbox_1'] + (curr['bbox_1'] - prev['bbox_1'])*j/diff)),
                        str(int(prev['bbox_2'] + (curr['bbox_2'] - prev['bbox_2'])*j/diff)),
                        str(int(prev['bbox_3'] + (curr['bbox_3'] - prev['bbox_3'])*j/diff))
                    ]
                    data.append(interpolation)
                prev_frame = curr_frame
                prev = curr
    data = pd.DataFrame(data, columns=detections.columns)
    frames = [detections, data]
    result = pd.concat(frames, ignore_index=True)
    dir_path = os.path.dirname(path)
    filename = os.path.basename(path).split(".")[0] + "_interpolated.csv"
    path = os.path.join(dir_path, filename)
    result.to_csv(path , index=False)
    return path

def main(_argv):
    #Hyperparameter
    # Root directory for dataset
    vlc_path = FLAGS.videos
    det_path = FLAGS.detections
    dest_path = FLAGS.outputs
    det_to_vid(vlc_path, det_path, dest_path)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
