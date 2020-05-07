import cv2
import os

frames_num = 0

print("Visualization started!")

# the path of predicted bboxes, take Airport, TColor128, a DataSet, as an example
results_path = "/globalwork/SiamR-CNN/tracking_data/results/TColor128/ThreeStageTracker_0.06_0.3_0.3_0.1_0.9_7.0/Airport_ce.txt"

with open(results_path, "r") as f:
    gt_bboxes = f.readlines()
    frames_num = len(gt_bboxes)
    for i, gt_bbox in enumerate(gt_bboxes):
        x1, y1, x2, y2 = gt_bbox.split(',')
        
        # the path of original images, take TColor128, a DataSet, as an example
        images_path = "/globalwork/data/tc128/Temple-color-128/Airport_ce/img/%04d.jpg" % (i+1)
        
        img = cv2.imread(images_path)
        img = cv2.rectangle(img, (int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))), (0, 255, 0), 3)
        
        # the path of images with bbox
        images_bbox_path = "/globalwork/Visualization/images/%04d.jpg" % (i+1)
        
        cv2.imwrite(images_bbox_path, img)
        
fps = 30
size = (1280, 720)

# the path of video to be generated
video_path = "/globalwork/Visualization/videos/test_airport.mp4"
videowriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

for i in range(frames_num):
    images_bbox_path = "/globalwork/Visualization/images/%04d.jpg" % (i+1)
    img = cv2.imread(images_bbox_path)
    videowriter.write(img)
    
print("Visualization Completed!")
