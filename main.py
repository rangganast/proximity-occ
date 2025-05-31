import cv2
import numpy as np
from yolov8 import YOLOv8
from multiprocessing import Pool
import os

csrt_tracker = cv2.legacy.TrackerCSRT_create()
csrt_tracker_2 = cv2.legacy.TrackerCSRT_create()
multi_tracker = cv2.legacy.MultiTracker_create()

cap = cv2.VideoCapture("/dev/video3")
# cap = cv2.VideoCapture("/dev/video2")
# cap = cv2.VideoCapture("/dev/video0")

model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.2, iou_thres=0.2)

led_left_result = ""
led_right_result = []
one_byte = []
char_temp = ""

def read_leds(select_box):
    global led_left_result
    global led_right_result
    global one_byte
    global char_temp

    select_box = (int(select_box[0]), int(select_box[1]), int(select_box[2]), int(select_box[3]))
    led_frame = frame[select_box[1]:select_box[1]+select_box[3], select_box[0]:select_box[0]+select_box[2]]

    # image edit (convert to gray => reduce exposure => add threshold)
    alpha = 0.7  # Adjust this value to control the contrast (1.0 means no change)
    beta = 14 # Bias, set to 0.0 for just contrast adjustment

    scaled_image = cv2.convertScaleAbs(led_frame, alpha=alpha, beta=beta)
    # cv2.imshow('scaled_image', scaled_image)

    blurred_image = cv2.GaussianBlur(scaled_image, (1, 1), 0)
    # cv2.imshow('blurred_image', blurred_image)
    img_gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    # img_gray = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('img_gray', img_gray)
    _, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)
    # img_blur = cv2.GaussianBlur(thresh, (5, 5), 2)

    contours, hierarchy = cv2.findContours(np.array(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ma = [[], [], [], []]
    td = []

    lower_bound = 0
    upper_bound = 100

    contours = [contour + [select_box[0], select_box[1]] for contour in contours]
    # print(contours)
    for j, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        # print(x, y, w, h)
        if (lower_bound < w < upper_bound) and (lower_bound < h < upper_bound):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            ma[0].append(x)
            ma[1].append(y)
            a = [x+w/2, y+h/2]
            td.append(a)
            ma[2].append(w)
            ma[3].append(h)

    # get leds only area wide and length
    xmin = min(ma[0])
    xmax = max(ma[0])
    ymin = min(ma[1])
    ymax = max(ma[1])

    bit = []
    # only get second row
    dx = (xmax-xmin)/(8-1)
    dy = (ymax-ymin)/(8-1)
    #only get second row
    c = ymin

    while c <= ymax + 1:
        b = xmin
        while b <= xmax+1:
            for j in td:
                if b <= j[0] <= b+dx and c <= j[1] <= c+dy:
                    d = 1
                    break
                else:
                    d = 0
            bit.append(d)
            b += dx
        c += dy
    
    
    temp = bit[8:16]
    if bit[0:8] == [1, 1, 0, 1, 1, 0, 1, 1]: # led left
        if temp == [0, 0, 0, 0, 0, 0, 0, 0]:
            if len(led_left_result) > 0:
                try:
                    if len(one_byte) > 10:
                        led_left_result += one_byte[0] * 2
                        cv2.putText(frame, f'Receiving text: {led_left_result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        led_left_result += one_byte[0]
                        cv2.putText(frame, f'Receiving text: {led_left_result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    one_byte = []
                except Exception as error:
                    pass
            cv2.putText(frame, f'LED left text: {led_left_result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Convert binary values to an integer
            decimal_value = int(''.join(map(str, temp)), 2)
            # Convert the integer to ASCII character
            char = chr(decimal_value)       
            if (len(one_byte) == 0) or (char in one_byte):
                one_byte.append(char)
            else:
                if len(one_byte) > 10:
                    led_left_result += one_byte[0] * 2
                    cv2.putText(frame, f'Receiving text: {led_left_result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                else:
                    led_left_result += one_byte[0]
                    cv2.putText(frame, f'Receiving text: {led_left_result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                one_byte = []
    elif bit[0:8] == [1, 0, 0, 1, 1, 0, 0, 1]: # led left
        if temp == [0, 0, 0, 0, 0, 0, 0, 0]:
            if len(led_right_result) > 0:
                try:
                    if len(one_byte) >= 11:
                        led_right_result += one_byte[0] * 2
                        # cv2.putText(frame, f'Receiving text: {led_right_result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    else:
                        led_right_result += one_byte[0]
                        # cv2.putText(frame, f'Receiving text: {led_right_result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    if "rs" in led_right_result:
                        led_right_result = ""
                    one_byte = []
                except Exception as error:
                    pass

            cv2.putText(frame, f'LED right text: {led_right_result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # else:
            # Convert binary values to an integer
        decimal_value = int(''.join(map(str, temp)), 2)
        char = ""
        # Convert the integer to ASCII character
        char = chr(decimal_value)
        if char != '\x00':
            if char_temp != char:
                led_right_result.append(char)
                print(led_right_result)
        char_temp = char

        # if (len(one_byte) == 0) or (char in one_byte):
        #     one_byte.append(char)
        # else:
        #     if len(one_byte) >= 18:
        #         print(char)
        #         led_right_result += one_byte[0] * 2
        #         cv2.putText(frame, f'Receiving text: {led_right_result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        #     else:
        #         led_right_result += one_byte[0]
        #         cv2.putText(frame, f'Receiving text: {led_right_result}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        #     one_byte = []

    return thresh

def process_select_box(select_box_):
    select_box_[2] = select_box_[2] - select_box_[0]
    select_box_[3] = select_box_[3] - select_box_[1]
    select_box_ = tuple([int(val) for val in select_box_])
    return select_box_

def process_select_boxes(func, select_boxes):
    return [func(select_box_) for select_box_ in select_boxes]

if __name__ == "__main__":
    i = 0

    while True:
        i += 1
        ret, frame = cap.read()
        if not ret:
            break

        if i == 1:
            select_boxes, scores, class_ids = yolov8_detector(frame)
            select_boxes_processed = process_select_boxes(process_select_box, select_boxes)

            bbox1 = list(select_boxes_processed[0])
            # bbox2 = list(select_boxes_processed[1])

            # Widen the first bounding box by 5 units
            bbox1[0] -= 5  # Decrease x-coordinate by 2.5 to extend to the right
            bbox1[1] -= 5  # Decrease y-coordinate by 2.5 to extend downward
            bbox1[2] += 10    # Increase width by 5 to cover both sides
            bbox1[3] += 10   # Increase height by 5 to cover both top and bottom

            # Widen the second bounding box by 5 units
            # bbox2[0] -= 5  # Decrease x-coordinate by 2.5 to extend to the right
            # bbox2[1] -= 5  # Decrease y-coordinate by 2.5 to extend downward
            # bbox2[2] += 10    # Increase width by 5 to cover both sides
            # bbox2[3] += 10    # Increase height by 5 to cover both top and bottom
            
            multi_tracker.add(csrt_tracker, frame, bbox1)
            # multi_tracker.add(csrt_tracker_2, frame, bbox2)

        if i % 2 == 0:
            select_boxes, scores, class_ids = yolov8_detector(frame)
            select_boxes_processed = process_select_boxes(process_select_box, select_boxes)
            
            bbox1 = list(select_boxes_processed[0])
            # bbox2 = list(select_boxes_processed[1])

            multi_tracker = cv2.legacy.MultiTracker_create()
            csrt_tracker = cv2.legacy.TrackerCSRT_create()
            csrt_tracker_2 = cv2.legacy.TrackerCSRT_create()
            
            # Widen the first bounding box by 5 units
            bbox1[0] -= 5  # Decrease x-coordinate by 2.5 to extend to the right
            bbox1[1] -= 5  # Decrease y-coordinate by 2.5 to extend downward
            bbox1[2] += 10    # Increase width by 5 to cover both sides
            bbox1[3] += 10   # Increase height by 5 to cover both top and bottom

            # Widen the second bounding box by 5 units
            # bbox2[0] -= 5  # Decrease x-coordinate by 2.5 to extend to the right
            # bbox2[1] -= 5  # Decrease y-coordinate by 2.5 to extend downward
            # bbox2[2] += 10    # Increase width by 5 to cover both sides
            # bbox2[3] += 10    # Increase height by 5 to cover both top and bottom
            
            multi_tracker.add(csrt_tracker, frame, bbox1)
            # multi_tracker.add(csrt_tracker_2, frame, bbox2)

        ret_tracker, select_boxes_updated = multi_tracker.update(frame)
        tl1, br1 = (int(select_boxes_updated[0][0]), int(select_boxes_updated[0][1])), (int(select_boxes_updated[0][0] + select_boxes_updated[0][2]), int(select_boxes_updated[0][1] + select_boxes_updated[0][2]))
        # tl2, br2 = (int(select_boxes_updated[1][0]), int(select_boxes_updated[1][1])), (int(select_boxes_updated[1][0] + select_boxes_updated[1][2]), int(select_boxes_updated[1][1] + select_boxes_updated[1][2]))

        if ret_tracker:
            # if select_boxes_updated[0][2] * select_boxes_updated[0][3] > select_boxes_updated[1][2] * select_boxes_updated[1][3]:
            led1_thresh = read_leds(select_boxes_updated[0])
            cv2.rectangle(frame, tl1, br1, (0, 255, 0), 2, 2)
            # cv2.rectangle(frame, tl1, br1, (255, 255, 0), 2, 2)
            cv2.imshow('led1_thresh', led1_thresh)
            # else:
            #     if ("rs" in led_right_result):
            #         led_right_result = ""
            #     led2_thresh = read_leds(select_boxes_updated[1])
            #     cv2.rectangle(frame, tl2, br2, (255, 0, 0), 2, 2)
            #     cv2.imshow('led2_thresh', led2_thresh)
                
            cv2.imshow('frame', frame)


        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            with open("chars.txt", "w") as f:
                f.write("".join(led_right_result))
            cv2.destroyAllWindows()
            break