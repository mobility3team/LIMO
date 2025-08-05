#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from math import pi

def nothing(x): pass

def create_trackbars():
    cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("KP",         "Settings", 11, 100, nothing)
    cv2.createTrackbar("KD",         "Settings", 7, 100, nothing)
    cv2.createTrackbar("Offset",     "Settings", 258, 400, nothing)
    cv2.createTrackbar("CenterX",    "Settings", 325, 640, nothing)
    cv2.createTrackbar("Speed",      "Settings", 10, 50, nothing)
    cv2.createTrackbar("WindowNum",  "Settings", 8, 16, nothing)
    cv2.createTrackbar("HistThresh", "Settings", 20, 255, nothing)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[클릭] x = {x}, y = {y}")

class LineTracer(Node):
    def __init__(self):
        super().__init__('line_tracer')
        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.last_angular = 0.0
        self.prev_error   = 0.0
        self.integral     = 0.0
        self.prev_left    = None
        self.prev_right   = None
        create_trackbars()

        self.create_subscription(
            Image,
            '/image',
            self.camera_callback,
            10
        )
        self.get_logger().info('LineTracer with Trackbars Started')

    def camera_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().warn(f"imgmsg_to_cv2 실패: {e}")
            return

        if cv_img is None:
            self.get_logger().warn("이미지 None")
            return

        h, w, _ = cv_img.shape

        # 트랙바 값 불러오기
        kp         = cv2.getTrackbarPos("KP", "Settings") / 1000.0
        kd         = cv2.getTrackbarPos("KD", "Settings") / 1000.0
        offset     = cv2.getTrackbarPos("Offset", "Settings")
        center_ideal = cv2.getTrackbarPos("CenterX", "Settings")
        speed      = cv2.getTrackbarPos("Speed", "Settings") / 100.0
        win_n      = max(2, cv2.getTrackbarPos("WindowNum", "Settings"))
        hist_thresh = cv2.getTrackbarPos("HistThresh", "Settings")

        hsv         = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (10, 60, 40), (50, 255, 255))

        src = np.float32([(0, h), (190, 330), (w-190, 330), (w, h)])
        dst = np.float32([(80, h), (80, 0), (w-80, 0), (w-80, h)])
        M   = cv2.getPerspectiveTransform(src, dst)
        warp_mask = yellow_mask.copy()
        binary = (warp_mask > 0).astype(np.uint8)
        color_warp = cv2.cvtColor(warp_mask, cv2.COLOR_GRAY2BGR)

        win_h = h // win_n
        left_pts = []
        right_pts = []

        for i in range(win_n//2):
            top = h - (i+1)*win_h
            bot = h - i*win_h
            window = binary[top:bot, :]
            hist   = np.sum(window, axis=0)
            hist[hist < hist_thresh] = 0
            idxs = np.nonzero(hist)[0]
            mid  = w // 2

            # 왼쪽 차선 추적
            lidx = [x for x in idxs if x < mid]
            if self.prev_left is not None and lidx and abs(np.mean(lidx) - self.prev_left) > 100:
                lidx = []
            if lidx:
                avg_l = int(np.mean(lidx))
                left_pts.append(avg_l)
                cv2.line(color_warp, (avg_l, top), (avg_l, bot), (255, 0, 255), 2)

            # 오른쪽 차선 추적
            ridx = [x for x in idxs if x > mid]
            if self.prev_right is not None and ridx and abs(np.mean(ridx) - self.prev_right) > 100:
                ridx = []
            if ridx:
                avg_r = int(np.mean(ridx))
                right_pts.append(avg_r)
                cv2.line(color_warp, (avg_r, top), (avg_r, bot), (0, 0, 255), 2)

        l_avg = np.mean(left_pts) if left_pts else None
        r_avg = np.mean(right_pts) if right_pts else None

        if l_avg is not None and r_avg is not None:
            center_avg = int((l_avg + r_avg) / 2 + offset)
            self.prev_left = l_avg
            self.prev_right = r_avg
        else:
            t = Twist()
            t.linear.x  = 0.1
            t.angular.z = 50.0
            self.cmd_pub.publish(t)
            return

        error = center_ideal - center_avg
        self.integral += error
        deriv = error - self.prev_error
        self.prev_error = error

        # 곡선 여부 판단
        if abs(error) > 30:
            curve_state = "Curve"
            kp *= 0.2     # 꺾임 강도 2배
            kd *= 0.5
            ang_limit = 1.0
        else:
            curve_state = "Straight"
            ang_limit = 1.0

        pid_out = kp * error + kd * deriv
        ang_z = float(np.clip(pid_out, -ang_limit, ang_limit))
        print(f"[PID] error={error:.2f}, deriv={deriv:.2f}, pid_out={pid_out:.2f}, ang_z={ang_z:.2f}")
        t = Twist()
        t.linear.x  = speed
        t.angular.z = ang_z
        self.cmd_pub.publish(t)
        self.last_angular = ang_z

        cv2.line(color_warp, (center_ideal, 0), (center_ideal, h), (0,255,0), 2)
        cv2.line(color_warp, (center_avg, 0),   (center_avg, h),   (0,255,255), 2)

        cv2.imshow("warp_mask", color_warp)
        cv2.setMouseCallback("warp_mask", mouse_callback)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = LineTracer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
