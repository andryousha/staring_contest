import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

from consts import ID_LIST, SIZE, WAITKEY, COLOR, RATIO_THR

cap = cv2.VideoCapture(1)
detector = FaceMeshDetector(maxFaces=1)
plot_y = LivePlot(640, 360, [20, 50], invert=True)

ratioList = []
blink_counter = 0
counter = 0

while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in ID_LIST:
            cv2.circle(img, face[id], 5, COLOR, cv2.FILLED)

        left_up = face[159]
        left_down = face[23]
        left_left = face[130]
        left_right = face[243]
        lenght_ver, _ = detector.findDistance(left_up, left_down)
        lenght_hor, _ = detector.findDistance(left_left, left_right)

        cv2.line(img, left_up, left_down, (0, 200, 0), 3)
        cv2.line(img, left_left, left_right, (0, 200, 0), 3)

        ratio = int((lenght_ver / lenght_hor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 3:
            ratioList.pop(0)
        ratio_avg = sum(ratioList) / len(ratioList)

        if ratio_avg < RATIO_THR and counter == 0:
            blink_counter += 1
            COLOR = (0, 200, 0)
            counter = 1
        if counter != 0:
            counter += 1
            if counter > 10:
                counter = 0
                COLOR = (255, 0, 255)

        cvzone.putTextRect(img, f'Blink Count: {blink_counter}', (50, 100),
                           colorR=COLOR)

        img_plot = plot_y.update(ratio_avg, COLOR)
        img = cv2.resize(img, SIZE)
        imgStack = cvzone.stackImages([img, img_plot], 2, 1)
    else:
        img = cv2.resize(img, SIZE)
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Image", imgStack)
    cv2.waitKey(WAITKEY)
