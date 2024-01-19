import numpy as np
import cv2
import pickle
import pickle
import matplotlib.pyplot as plt
# from util import get_parking_spots_bboxes,empty_or_not
def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

MODEL = pickle.load(open("CustomCONVnet.p", "rb"))
EMPTY = True
NOT_EMPTY = False

def empty_or_not(spot_bgr):

    flat_data = []

    img_resized = cv2.resize(spot_bgr,(69,30))
    img_reshaped=img_resized.reshape(1,img_resized.shape[0],img_resized.shape[1],img_resized.shape[2])
    y_output = MODEL.predict(img_reshaped)

    if y_output[0] >=0.5:
        return EMPTY
    else:
        return NOT_EMPTY

def cal_diff(img1,img2):
    return np.abs(np.mean(img1)-np.mean(img2))

full_mask_path="mask_1920_1080.png"
full_video_path="parking_1920_1080.mp4"

crop_mask_path="mask_crop.png"
crop_video_path="parking_crop.mp4"

cap = cv2.VideoCapture(crop_video_path)
mask=cv2.imread(crop_mask_path)

mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
# mask=cv2.resize(mask,(1080,1920))

# cv2.imshow("mask",mask)


connected_components=cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)
spots=get_parking_spots_bboxes(connected_components)

stop=40
frame_no=0
spots_status=[None for j in spots]
diffs=[None for j in spots]

previous_frame=None

while True:
    ret,frame=cap.read()
    # cv2.resize(frame,(w,h))

    # binary_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # bitwise_and = cv2.bitwise_and(binary_frame, mask)

    if frame_no % stop==0 and previous_frame is not None:
        for spot_indx,spot in enumerate(spots):
            x1,y1,w,h=spot
            bb=frame[y1:y1+h , x1:x1+w , :]
            diffs[spot_indx]=cal_diff(bb,previous_frame[y1:y1+h , x1:x1+w , :])
        print([diffs[j] for j in np.argsort(diffs)][::-1])
        plt.figure()
        plt.hist([diffs[j]/np.amax(diffs) for j in np.argsort(diffs)][::-1])
        if(frame_no==300):
            plt.show()

    if frame_no % stop==0:
        if previous_frame is None:
            arr=range(len(spots))
        else:
            arr=[j for j in np.argsort(diffs) if diffs[j]/np.max(diffs)>0.4][::-1]
            print(len(arr))
        for spot_indx in arr:
            spot=spots[spot_indx]
            x1,y1,w,h=spot
            bb=frame[y1:y1+h , x1:x1+w , :]
            truth_value=empty_or_not(bb)
            spots_status[spot_indx]=truth_value

    if frame_no % stop==0:
        previous_frame=frame.copy()

    for spot_indx,spot in enumerate(spots):
        x1,y1,w,h=spot
        truth=spots_status[spot_indx]
        if truth:
            frame2=cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,0,255),2)
        else:
            frame2=cv2.rectangle(frame,(x1,y1),(x1+w,y1+h),(0,255,0),2)
    cv2.rectangle(frame2,(5,5),(170,25),(0,0,0),-1)
    cv2.putText(frame2,"Empty Spots:-{}/{}".format(str(len(spots_status)-sum(spots_status)),str(len(spots_status))),(10,20),cv2.FONT_HERSHEY_DUPLEX,0.5,(255,255,255))
    cv2.namedWindow("frames",cv2.WINDOW_NORMAL)
    cv2.imshow("frames",frame2)
    # cv2.imshow("parking_video_with mask",bitwise_and)
    frame_no +=1

    if cv2.waitKey(25) & 0xFF==ord("c"):
        break


cap.release()
cv2.destroyAllWindows()
