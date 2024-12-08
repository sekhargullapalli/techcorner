import cv2
from djitellopy import Tello
import time

tello = Tello()

tello.connect()
tello.streamoff()
tello.streamon()
frame_read=tello.get_frame_read()

tello.takeoff()
time.sleep(2)
tello.move('forward',100)
time.sleep(2)


cv2.imwrite("capture1.png",frame_read.frame)
time.sleep(2)

tello.rotate_counter_clockwise(90)
time.sleep(1)
tello.move('back',50)

tello.land()
time.sleep(2)
tello.end()
