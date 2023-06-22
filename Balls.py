import numpy as np
import cv2
import random
import math
import time
import LiveRun

# Function 1.
# ball_detection for only ball detection
def ball_detection(x_min_orig, y_min_orig, x_max_orig, y_max_orig):


    image = cv2.imread('/Users/antoneyfernando/Downloads/balls.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=22)

    circle_points = []  # Array to store circle points

    i = 1

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            if check_withinbox(x, y, (x_min_orig, x_max_orig), (y_min_orig, y_max_orig)):
                circle_points.append((x, y))  # Save circle points in the array
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                cv2.putText(image, f'{i}: ({x}, {y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                i += 1

    cv2.imshow('scan', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return circle_points


# Function set 2.
# ** Tracking of coarse and cross in middle
# ** Caculating goal
# ** Caculating different rectangles

def coarse_tracking():
    processed_image = load_and_process_image()
    rectangles = find_rectangles(processed_image)
    x_range_big, y_range_big, x_range_2, y_range_2, goal, x_min_orig, y_min_orig, \
        x_max_orig, y_max_orig = process_rectangles(rectangles)

    final_image = cv2.imread('/Users/antoneyfernando/Downloads/balls1.jpg')

    cv2.imshow("Shapes", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return x_range_big, y_range_big, x_range_2, y_range_2, goal, x_min_orig, y_min_orig, x_max_orig, y_max_orig

def load_and_process_image():
    #'''
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('python webcam screenshot app')
    img_counter = 0

    ret, frame = cam.read()
    if not ret:
        print('failed to grab frame')
        cam.release()
        cv2.destroyAllWindows()
        return []

    img_name = '/Users/antoneyfernando/Downloads/balls.jpg'
    cv2.imwrite(img_name, frame)
    print('screenshot taken')
    img_counter += 1

    cam.release()
    cv2.destroyAllWindows()  # '''



    image = cv2.imread('/Users/antoneyfernando/Downloads/balls.jpg')


    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)


    mask = mask1 + mask2

    red_pixels = cv2.bitwise_and(image, image, mask=mask)

    return red_pixels

def find_rectangles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 50, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []  # We will keep the rectangles here
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) >= 4 and area > 1000:  # We are looking for polygons with at least 4 sides.
            rectangles.append((area, approx))

    # Sort rectangles by largest first
    rectangles.sort(key=lambda x: -x[0])

    return rectangles

def process_rectangles(rectangles):
    saved_image = cv2.imread('/Users/antoneyfernando/Downloads/balls.jpg')

    # Remove the largest rectangle
    rectangles = rectangles[1:]

    shape_1 = rectangles[0][1] if rectangles else None  # Extract the largest remaining rectangle
    shape_2 = rectangles[1][1] if len(rectangles) > 1 else None  # Extract the second largest rectangle

    x_range_big, y_range_big = None, None
    x_range_2, y_range_2 = None, None
    goal = None  # Placeholder for the goal point on the right side

    if shape_1 is not None:
        # Draw the larger shape (shape_1) on the image
        saved_image = cv2.drawContours(saved_image, [shape_1], -1, (0, 255, 0), 3)

        x_min_orig, y_min_orig = np.min(shape_1[:, 0], axis=0)
        x_max_orig, y_max_orig = np.max(shape_1[:, 0], axis=0)

        # Calculate the right-bottom corner point and upper-right corner point
        bottom_right = shape_1[np.argmax(shape_1[:, 0, 0] + shape_1[:, 0, 1])]
        upper_right = shape_1[np.argmax(shape_1[:, 0, 0] - shape_1[:, 0, 1])]

        # Calculate the middle point of the line joining upper right and lower right corners
        goal = ((upper_right[0][0] + bottom_right[0][0]) // 2, (upper_right[0][1] + bottom_right[0][1]) // 2)

        # Calculate the center of the larger rectangle
        M = cv2.moments(shape_1)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Scale the coordinates of the coarse
        scale_factor = 0.80 # Scale down
        shape_1_scaled = shape_1 * scale_factor

        shape_1_scaled += np.array([cX, cY]) - np.array([cX, cY]) * scale_factor

        # Draw the smaller scaled shape
        saved_image = cv2.drawContours(saved_image, [shape_1_scaled.astype(np.int32)], -1, (0, 0, 255), 2)

        # Define x_range and y_range
        x_range_big = (np.min(shape_1_scaled[:, 0, 0]), np.max(shape_1_scaled[:, 0, 0]))
        y_range_big = (np.min(shape_1_scaled[:, 0, 1]), np.max(shape_1_scaled[:, 0, 1]))

        # Mark the goal
        cv2.circle(saved_image, goal, 10, (255, 0, 0), -1)


    if shape_2 is not None:
        # Get the x and y coordinate ranges of shape 2
        x_coordinates = shape_2[:, 0, 0]
        y_coordinates = shape_2[:, 0, 1]
        x_min, x_max = np.min(x_coordinates), np.max(x_coordinates)
        y_min, y_max = np.min(y_coordinates), np.max(y_coordinates)

        # Compute the current width and height of the bounding box
        width = x_max - x_min
        height = y_max - y_min

        # Set the scale factor to increase the bounding box size
        scale_factor = 1.5  # Increase the size by 20%

        # Compute the new width and height based on the scale factor
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Compute the adjustments to the x and y coordinate ranges
        x_adjustment = (new_width - width) // 2
        y_adjustment = (new_height - height) // 2

        # Modify the x and y coordinate ranges
        x_range = (x_min - x_adjustment, x_max + x_adjustment)
        y_range = (y_min - y_adjustment, y_max + y_adjustment)

        # Draw lines for x and y coordinate ranges
        cv2.line(saved_image, (x_range[0], y_range[0]), (x_range[1], y_range[0]), (0, 255, 0), 2)  # horizontal line
        cv2.line(saved_image, (x_range[0], y_range[0]), (x_range[0], y_range[1]), (0, 255, 0), 2)  # vertical line
        cv2.line(saved_image, (x_range[0], y_range[1]), (x_range[1], y_range[1]), (0, 255, 0), 2)  # horizontal line
        cv2.line(saved_image, (x_range[1], y_range[0]), (x_range[1], y_range[1]), (0, 255, 0), 2)  # vertical line

        x_range_2 = x_range
        y_range_2 = y_range

    # Save the processed image to a file
    cv2.imwrite('/Users/antoneyfernando/Downloads/balls1.jpg', saved_image)

    return x_range_big, y_range_big, x_range_2, y_range_2, goal, x_min_orig, y_min_orig, x_max_orig, y_max_orig


# Function set 3.
# ** Finding placement of robot
# ** Calculating the next target of robot ( either goal or ball )
def robot_detection(circle_points, x_range, y_range, x_range_big, y_range_big, goal):
    #'''
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('python webcam screenshot app')
    img_counter = 0

    ret, frame = cam.read()
    if not ret:
        print('failed to grab frame')
        cam.release()
        cv2.destroyAllWindows()
        return []

    # Save the frame
    img_name = '/Users/antoneyfernando/Downloads/balls.jpg'
    cv2.imwrite(img_name, frame)
    print('screenshot taken')
    img_counter += 1

    cam.release()
    cv2.destroyAllWindows()#'''
    try:
        frame = cv2.imread('/Users/antoneyfernando/Downloads/balls.jpg')
        if frame is None:
            print("Error: Image could not be loaded. Check the file path and try again.")
            return None, None
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    except Exception as e:
        print("An error occurred while trying to load or process the image: ", e)
        return None, None

    shortest_ball_position = None
    shortest_ball_color = None

    green_mask, orange_mask = apply_mask(hsv_frame, frame)

    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    frame, green_cx, green_cy = handle_contour(green_contours, frame, (0, 255, 0), "Green")
    frame, orange_cx, orange_cy = handle_contour(orange_contours, frame, (0, 140, 255), "Orange")

    ## choose goal if function returns true
    if returnballs(len(circle_points)):
        shortest_ball_position = goal[0], goal[1]

        quarter_1, quarter_2, quarter_3, quarter_4 = devide_into_four(x_range_big, y_range_big)

        if check_withinbox(green_cx, green_cy, (quarter_1[0], quarter_1[1]), (quarter_1[2], quarter_1[3])):
            shortest_ball_color = "goal_quarter_1"
        elif check_withinbox(green_cx, green_cy, (quarter_2[0], quarter_2[1]), (quarter_2[2], quarter_2[3])):
            shortest_ball_color = "goal_quarter_2"
        elif check_withinbox(green_cx, green_cy, (quarter_3[0], quarter_3[1]), (quarter_3[2], quarter_3[3])):
            shortest_ball_color = "goal_quarter_3"
        else:
            shortest_ball_color = "goal_quarter_4"

    ## choose ball otherwise
    else:
        # Initialize lists for differnt types of balls
        red_line_balls = []
        blue_line_balls = []
        purple_line_balls = []
        black_line_balls = []

        # Iterate over the circle points
        for i, (circle_x, circle_y) in enumerate(circle_points):

            # Check if the current ball is close to boarder or cross
            if check_outofbox(circle_x, circle_y, x_range_big, y_range_big) or \
                    check_withinbox(circle_x, circle_y, x_range, y_range):

                #check if ball needs a route to get to ball or not
                if (line_intersects_segment(green_cx, green_cy, circle_x, circle_y, x_range[0], y_range[0], x_range[0],
                                       y_range[1]) \
                    or line_intersects_segment(green_cx, green_cy, circle_x, circle_y, x_range[0], y_range[1], x_range[1],
                                               y_range[1]) \
                    or line_intersects_segment(green_cx, green_cy, circle_x, circle_y, x_range[1], y_range[1], x_range[1],
                                               y_range[0]) \
                    or line_intersects_segment(green_cx, green_cy, circle_x, circle_y, x_range[1], y_range[0], x_range[0],
                                               y_range[0])) and check_outofbox(circle_x, circle_y, x_range_big, y_range_big):
                    # The ball is marked BLACK
                    black_line_balls.append(i)
                    cv2.line(frame, (green_cx, green_cy), (circle_x, circle_y), (0, 0, 0), 2)
                else:
                    # The ball is marked PURPLE
                    purple_line_balls.append(i)
                    cv2.line(frame, (green_cx, green_cy), (circle_x, circle_y), (128, 0, 128), 2)

            # check if ball needs a route to get to ball or not
            elif (line_intersects_segment(green_cx, green_cy, circle_x, circle_y, x_range[0], y_range[0], x_range[0],
                                       y_range[1]) \
                    or line_intersects_segment(green_cx, green_cy, circle_x, circle_y, x_range[0], y_range[1], x_range[1],
                                               y_range[1]) \
                    or line_intersects_segment(green_cx, green_cy, circle_x, circle_y, x_range[1], y_range[1], x_range[1],
                                               y_range[0]) \
                    or line_intersects_segment(green_cx, green_cy, circle_x, circle_y, x_range[1], y_range[0], x_range[0],
                                               y_range[0])) \
                    and (check_withinbox(circle_x, circle_y, x_range_big, y_range_big)) \
                    or (line_intersects_segment(orange_cx, orange_cy, circle_x, circle_y, x_range[0], y_range[0], x_range[0],
                                       y_range[1]) \
                    or line_intersects_segment(orange_cx, orange_cy, circle_x, circle_y, x_range[0], y_range[1], x_range[1],
                                               y_range[1]) \
                    or line_intersects_segment(orange_cx, orange_cy, circle_x, circle_y, x_range[1], y_range[1], x_range[1],
                                               y_range[0]) \
                    or line_intersects_segment(orange_cx, orange_cy, circle_x, circle_y, x_range[1], y_range[0], x_range[0],
                                               y_range[0])):
                # The ball is marked RED
                red_line_balls.append(i)
                cv2.line(frame, (green_cx, green_cy), (circle_x, circle_y), (0, 0, 255), 2)

            else:
                # The ball is marked BLUE
                blue_line_balls.append(i)
                cv2.line(frame, (green_cx, green_cy), (circle_x, circle_y), (255, 0, 0), 2)

        # PRIORETIZED BALL COLOURS same code just for different ball colours
        if len(blue_line_balls) > 0:
            shortest_ball_index = min(blue_line_balls, key=lambda idx: math.sqrt(
                (green_cx - circle_points[idx][0]) ** 2 + (green_cy - circle_points[idx][1]) ** 2))
            shortest_ball_position = circle_points[shortest_ball_index]
            shortest_ball_color = "blue"


            shortest_ball_distance = math.sqrt(
                (green_cx - shortest_ball_position[0]) ** 2 + (green_cy - shortest_ball_position[1]) ** 2)

            angle_degrees, turn = LiveRun.angle_turn(shortest_ball_position[0] , shortest_ball_position[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)
            shortest_ball_angle = angle_degrees

            if shortest_ball_angle > 180:
                shortest_ball_angle = 360 - shortest_ball_angle

            for idx in blue_line_balls:
                ball_position = circle_points[idx]
                ball_distance = math.sqrt((green_cx - ball_position[0]) ** 2 + (green_cy - ball_position[1]) ** 2)

                angle_degrees, turn = LiveRun.angle_turn(ball_position[0], ball_position[1],
                                                         green_cx, green_cy, orange_cx, orange_cy)
                ball_angle = angle_degrees

                if ball_angle > 180:
                    ball_angle = 360 - ball_angle

                # shortest ball is chossen both based on angle and distance
                if ball_distance < (2 * shortest_ball_distance) and ball_angle < shortest_ball_angle:

                    shortest_ball_position = ball_position
                    shortest_ball_distance = ball_distance
                    shortest_ball_angle = ball_angle

        elif len(red_line_balls) > 0:
            shortest_ball_index = min(red_line_balls, key=lambda idx: math.sqrt(
                (green_cx - circle_points[idx][0]) ** 2 + (green_cy - circle_points[idx][1]) ** 2))
            shortest_ball_position = circle_points[shortest_ball_index]
            shortest_ball_color = "red"

            shortest_ball_distance = math.sqrt(
                (green_cx - shortest_ball_position[0]) ** 2 + (green_cy - shortest_ball_position[1]) ** 2)

            angle_degrees, turn = LiveRun.angle_turn(shortest_ball_position[0], shortest_ball_position[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)
            shortest_ball_angle = angle_degrees

            if shortest_ball_angle > 180:
                shortest_ball_angle = 360 - shortest_ball_angle

            for idx in red_line_balls:
                ball_position = circle_points[idx]
                ball_distance = math.sqrt((green_cx - ball_position[0]) ** 2 + (green_cy - ball_position[1]) ** 2)
                angle_degrees, turn = LiveRun.angle_turn(ball_position[0], ball_position[1],
                                                         green_cx, green_cy, orange_cx, orange_cy)
                ball_angle = angle_degrees

                if ball_angle > 180:
                    ball_angle = 360 - ball_angle

                if ball_distance < (2 * shortest_ball_distance) and ball_angle < shortest_ball_angle:
                    shortest_ball_position = ball_position
                    shortest_ball_distance = ball_distance
                    shortest_ball_angle = ball_angle

        elif len(purple_line_balls) > 0:
            shortest_ball_index = min(purple_line_balls, key=lambda idx: math.sqrt(
                (green_cx - circle_points[idx][0]) ** 2 + (green_cy - circle_points[idx][1]) ** 2))
            shortest_ball_position = circle_points[shortest_ball_index]
            shortest_ball_color = "purple"

            shortest_ball_distance = math.sqrt(
                (green_cx - shortest_ball_position[0]) ** 2 + (green_cy - shortest_ball_position[1]) ** 2)
            angle_degrees, turn = LiveRun.angle_turn(shortest_ball_position[0], shortest_ball_position[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)
            shortest_ball_angle = angle_degrees

            if shortest_ball_angle > 180:
                shortest_ball_angle = 360 - shortest_ball_angle

            for idx in purple_line_balls:
                ball_position = circle_points[idx]
                ball_distance = math.sqrt((green_cx - ball_position[0]) ** 2 + (green_cy - ball_position[1]) ** 2)

                angle_degrees, turn = LiveRun.angle_turn(ball_position[0], ball_position[1],
                                                         green_cx, green_cy, orange_cx, orange_cy)
                ball_angle = angle_degrees

                if ball_angle > 180:
                    ball_angle = 360 - ball_angle

                if ball_distance < (2 * shortest_ball_distance) and ball_angle < shortest_ball_angle:
                    shortest_ball_position = ball_position
                    shortest_ball_distance = ball_distance
                    shortest_ball_angle = ball_angle

            purple_line_balls.append(shortest_ball_index)

        elif len(black_line_balls) > 0:
            shortest_ball_index = min(black_line_balls, key=lambda idx: math.sqrt(
                (green_cx - circle_points[idx][0]) ** 2 + (green_cy - circle_points[idx][1]) ** 2))
            shortest_ball_position = circle_points[shortest_ball_index]
            shortest_ball_color = "red"


            shortest_ball_distance = math.sqrt(
                (green_cx - shortest_ball_position[0]) ** 2 + (green_cy - shortest_ball_position[1]) ** 2)
            angle_degrees, turn = LiveRun.angle_turn(shortest_ball_position[0], shortest_ball_position[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)
            shortest_ball_angle = angle_degrees

            if shortest_ball_angle > 180:
                shortest_ball_angle = 360 - shortest_ball_angle

            for idx in black_line_balls:
                ball_position = circle_points[idx]
                ball_distance = math.sqrt((green_cx - ball_position[0]) ** 2 + (green_cy - ball_position[1]) ** 2)

                angle_degrees, turn = LiveRun.angle_turn(ball_position[0], ball_position[1],
                                                         green_cx, green_cy, orange_cx, orange_cy)
                ball_angle = angle_degrees

                if ball_angle > 180:
                    ball_angle = 360 - ball_angle

                if ball_distance < (2 * shortest_ball_distance) and ball_angle < shortest_ball_angle:
                    shortest_ball_position = ball_position
                    shortest_ball_distance = ball_distance
                    shortest_ball_angle = ball_angle

            black_line_balls.append(shortest_ball_index)

        if shortest_ball_position is not None:
            shortest_ball_x, shortest_ball_y = shortest_ball_position
            cv2.line(frame, (green_cx, green_cy), (shortest_ball_x, shortest_ball_y), (0, 255, 0), 2)

    print(shortest_ball_color)


    cv2.rectangle(frame, (x_range[0], y_range[0]), (x_range[1], y_range[1]), (0, 255, 0), 2)

    cv2.imshow('robot', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return shortest_ball_position, shortest_ball_color


#line_intersects_segment is used to check if there need to be a route to ball
def line_intersects_segment(x1, y1, x2, y2, x3, y3, x4, y4):
    d1 = orientation(x3, y3, x4, y4, x1, y1)
    d2 = orientation(x3, y3, x4, y4, x2, y2)
    d3 = orientation(x1, y1, x2, y2, x3, y3)
    d4 = orientation(x1, y1, x2, y2, x4, y4)

    if (
            (d1 > 0 and d2 < 0 or d1 < 0 and d2 > 0)
            and (d3 > 0 and d4 < 0 or d3 < 0 and d4 > 0)
    ):
        return True

    if d1 == 0 and on_segment(x3, y3, x4, y4, x1, y1):
        return True

    if d2 == 0 and on_segment(x3, y3, x4, y4, x2, y2):
        return True

    if d3 == 0 and on_segment(x1, y1, x2, y2, x3, y3):
        return True

    if d4 == 0 and on_segment(x1, y1, x2, y2, x4, y4):
        return True

    return False
def orientation(x1, y1, x2, y2, x3, y3):
    return (y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1)
def on_segment(x1, y1, x2, y2, x3, y3):
    return (
            min(x1, x2) <= x3 <= max(x1, x2)
            and min(y1, y2) <= y3 <= max(y1, y2)
    )

# coarse is devided into 4
def devide_into_four(x_range_big, y_range_big):
    x_min, x_max = x_range_big
    y_min, y_max = y_range_big

    x_mid = (x_min + x_max) // 2
    y_mid = (y_min + y_max) // 2

    # Define the four quarters
    quarter_1 = (x_min, x_mid, y_min, y_mid)
    quarter_2 = (x_mid, x_max, y_min, y_mid)
    quarter_3 = (x_min, x_mid, y_mid, y_max)
    quarter_4 = (x_mid, x_max, y_mid, y_max)

    return quarter_1, quarter_2, quarter_3, quarter_4

def check_outofbox(circle_x, circle_y, x_range_big, y_range_big):
    # Check if the circle's x-coordinate is outside the x-range of the box
    if circle_x < x_range_big[0] or circle_x > x_range_big[1]:
        return True  # Circle is outside the box in the x-direction

    # Check if the circle's y-coordinate is outside the y-range of the box
    if circle_y < y_range_big[0] or circle_y > y_range_big[1]:
        return True  # Circle is outside the box in the y-direction

    # Circle is within the box
    return False


def check_withinbox(point_x, point_y, x_range_big, y_range_big):
    # Check if the point's x-coordinate is within the x-range of the box
    if x_range_big[0] <= point_x <= x_range_big[1]:
        # Check if the point's y-coordinate is within the y-range of the box
        if y_range_big[0] <= point_y <= y_range_big[1]:
            return True  # Point is within the box

    return False

#Used to Know when to go to goal
global_prev_balls = 11
def returnballs(ball_amount):
    global global_prev_balls

    print(ball_amount)

    if global_prev_balls - ball_amount == 2:
        global_prev_balls = ball_amount
        return True
    elif global_prev_balls < 2:
        if ball_amount == 0:
            return True
        else:
            return False



# Other Helper functions
def apply_mask(hsv_frame, frame):
    # Define the color ranges for red, green, and orange in HSV format
    red_lower = np.array([136, 87, 111])
    red_upper = np.array([180, 255, 255])
    green_lower = np.array([25, 52, 72])
    green_upper = np.array([102, 255, 255])
    orange_lower = np.array([15, 200, 200])  # Lower HSV values for orange
    orange_upper = np.array([22, 255, 255])  # Upper HSV values for orange

    # Create masks for each color using inRange() function
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
    orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)

    # Apply morphological transformations (dilation) to each mask
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    orange_mask = cv2.dilate(orange_mask, kernel)

    return green_mask, orange_mask


def handle_contour(contours, frame, color, label):
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)  # Extract the values of x, y, w, h
        M = cv2.moments(max_contour)
        cX = int(M["m10"] / M["m00"])  # X-coordinate of centroid
        cY = int(M["m01"] / M["m00"])  # Y-coordinate of centroid
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (cX, cY), 5, (0, 0, 0), -1)  # Draw midpoint as a small black circle
        return frame, cX, cY
    else:
        return frame, None, None



