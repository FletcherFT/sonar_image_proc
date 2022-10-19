#!/usr/bin/env python3


from acoustic_msgs.msg import SonarImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
from typing import Tuple, Union
import cv2
import numpy as np
import datetime


def proc_sonar(msg: SonarImage, args: Tuple[rospy.Publisher, CvBridge, Union[cv2.VideoWriter, None]]):
    # Get the ranges
    range_vec = np.array(msg.ranges)
    range_ticks = np.linspace(min(msg.ranges), max(msg.ranges), 5)
    num_samples = len(range_vec)
    # Get the bearings
    azi_vec = np.array(msg.azimuth_angles)
    azi_ticks = np.linspace(min(msg.azimuth_angles), max(msg.azimuth_angles), 5)
    num_beams = len(azi_vec)

    azimuth_bounds = (azi_vec.min(), azi_vec.max())
    minus_width = np.floor(num_samples * np.sin(azimuth_bounds[0]))
    plus_width = np.ceil(num_samples * np.sin(azimuth_bounds[1]))
    width = int(plus_width - minus_width)

    originx = np.abs(minus_width)

    img_size = (num_samples, width)

    if width <= 0 or num_samples <= 0:
        return

    newmap = np.zeros(img_size+(2,), np.float32)

    db = (azimuth_bounds[1] - azimuth_bounds[0]) / num_beams


    # need to vectorize this (it works)
    # for x in range(newmap.shape[1]):
    #     for y in range(newmap.shape[0]):
    #         dx = x - originx # dx is the column distance from the current sample to the origin
    #         dy = newmap.shape[0] - y
    #         R = np.sqrt(dx ** 2 + dy ** 2)
    #         azimuth = np.arctan2(dx, dy)
    #         xp = R
    #         yp = (azimuth - azimuth_bounds[0]) / db
    #         newmap[y, x, :] = [yp, xp]

    # This does the job nicely
    x, y = np.meshgrid(np.arange(newmap.shape[1]), np.arange(newmap.shape[0]))
    dx = x - originx
    dy = newmap.shape[0] - y
    R = np.sqrt(dx ** 2 + dy ** 2)
    azimuth = np.arctan2(dx, dy)
    xp = R.copy()
    yp = (azimuth -azimuth_bounds[0]) / db
    newmap[:, :, 0] = yp
    newmap[:, :, 1] = xp

    # Construct a rectified canvas
    rect_canvas = np.frombuffer(msg.intensities, dtype=np.uint8).reshape((num_samples, num_beams)).copy()
    rect_canvas = cv2.applyColorMap(rect_canvas, cv2.COLORMAP_JET)

    # Adding range lines here produces better result
    rect_canvas[::100, :, :] = 255

    # Transform to polar canvas
    polar_canvas = cv2.remap(rect_canvas, newmap, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Annotation time!

    # pad canvas to allow room for annotations
    pad_size = 100
    polar_canvas = np.pad(polar_canvas, ((pad_size, 0), (pad_size, pad_size), (0, 0)))
    height, width = polar_canvas.shape[:2]

    # Draw the azimuth lines and ticks
    radius = height - pad_size
    center = ((width) / 2, height)
    azi = np.linspace(azimuth_bounds[0], azimuth_bounds[1], 5) + np.pi/2
    dy = radius * np.sin(azi)
    dx = radius * np.cos(azi)
    xp = dx + center[0]
    yp = -dy + center[1]
    pt1 = np.array(center, dtype=np.uint16)
    for ptx, pty, a in zip(xp, yp, azi):
        pt2 = np.array((ptx, pty), dtype=np.uint16)
        polar_canvas = cv2.line(polar_canvas, pt1, pt2 , (255, 255, 255))  # draw the line
        textsize, baseline = cv2.getTextSize("{}o".format(int((a - np.pi/2) * 180.0 / np.pi)), cv2.FONT_HERSHEY_PLAIN, 2, 2)
        if (a - np.pi/2) > 0:
            org = pt2 + (-textsize[0], -textsize[1])
        elif (a - np.pi/2) < 0:
            org = pt2 + (0, -textsize[1])
        else:
            org = pt2 + (int(-textsize[0]/2), -textsize[1])
        polar_canvas = cv2.putText(polar_canvas, "{}o".format(int((a - np.pi/2) * 180.0 / np.pi)), org, cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    
    # Draw the range ticks
    range_rings = np.arange(0, radius, 100)
    range_labels = msg.ranges[::100]
    for rr, label in zip(range_rings[1:], range_labels[1:]):
        textsize, baseline = cv2.getTextSize("{:.01f} m".format(label), cv2.FONT_HERSHEY_PLAIN, 2, 2)
        xp = rr * np.cos(-25 * np.pi / 180.0) + width / 2
        yp = rr * np.sin(-25 * np.pi / 180.0) + height + textsize[1]
        org = (int(xp), int(yp))
        polar_canvas = cv2.putText(polar_canvas, "{:.01f} m".format(label), org, cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), thickness=2)
    
    polar_canvas = cv2.resize(polar_canvas, (1280, 720), interpolation=cv2.INTER_AREA)

    t = msg.header.stamp
    dt = datetime.datetime.fromtimestamp(msg.header.stamp.to_time())
    textsize, baseline = cv2.getTextSize(str(dt), cv2.FONT_HERSHEY_PLAIN, 2, 2)
    polar_canvas = cv2.putText(polar_canvas, str(dt), (10, textsize[1]+10), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

    # cv2.imshow("test", polar_canvas)
    # cv2.waitKey(1)

    if args[2] is not None:
        args[2].write(polar_canvas)

    img_msg = args[1].cv2_to_imgmsg(polar_canvas, "bgr8")
    args[0].publish(img_msg)
    
    return


def main():
    rospy.init_node("sonar_labeller")
    pub = rospy.Publisher("labelled_sonar", Image, queue_size=10)
    video_path = rospy.get_param("output_video", None)
    bridge = CvBridge()
    writer = cv2.VideoWriter("/home/fft/Videos/output.mp4", cv2.VideoWriter_fourcc(*"avc1"), 10.0, (1280, 720), True) if video_path is not None else None
    rospy.Subscriber("sonar_image", SonarImage, proc_sonar, (pub, bridge, writer))
    rospy.spin()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()
    return


if __name__=="__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass