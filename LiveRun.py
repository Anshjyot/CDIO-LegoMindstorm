import numpy as np
import cv2
import math
import Balls
import SSH
import time

## The robot movements are run in this code
def live_feed():

    cam = cv2.VideoCapture(0)
    x_range_big, y_range_big, x_range, y_range, goal, x_min_orig, y_min_orig, \
        x_max_orig, y_max_orig = Balls.coarse_tracking()
    circle_points = Balls.ball_detection(x_min_orig, y_min_orig, x_max_orig, y_max_orig)
    shortest_ball_position, shortest_ball_color \
        = Balls.robot_detection(circle_points, x_range, y_range, x_range_big, y_range_big, goal)

    ssh, channel = SSH.establish_ssh_connection()

    if shortest_ball_color == "blue":

        while True:
            ret, frame = cam.read()
            if not ret:
                print('failed to grab frame')
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_mask, orange_mask = Balls.apply_mask(hsv_frame, frame)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            frame, green_cx, green_cy = Balls.handle_contour(green_contours, frame, (0, 255, 0), 'Green')
            frame, orange_cx, orange_cy = Balls.handle_contour(orange_contours, frame, (0, 165, 255), 'Orange')

            if shortest_ball_position is not None:
                shortest_ball_x, shortest_ball_y = shortest_ball_position

                line_length = np.linalg.norm(
                    np.array([green_cx, green_cy]) - np.array([shortest_ball_x, shortest_ball_y]))

                # caluclate which way to turn and how many degrees
                angle_degrees, turn \
                    = angle_turn(shortest_ball_x, shortest_ball_y, green_cx, green_cy, orange_cx, orange_cy)

                # stop robot after it is at desired point
                if line_length < 60:
                    if(angle_degrees >= 350) or (angle_degrees <= 10):
                        SSH.send_channel_commands(channel, turn)

                    SSH.send_channel_commands(channel, 5)
                    channel.close()
                    ssh.close()
                    cam.release()
                    cv2.destroyAllWindows()
                    break
                # is ball is not at the desired angle go straight otherwise turn
                if (angle_degrees >= 350) or (angle_degrees <= 10):
                    SSH.send_channel_commands(channel, 1)
                else:
                    SSH.send_channel_commands(channel, turn)

    elif shortest_ball_color == "red":
        flag = True
        flag_one = True
        flag_two = flag_three = False
        closest_to_goal = None

        while True:
            ret, frame = cam.read()

            if not ret:
                print('failed to grab frame')
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_mask, orange_mask = Balls.apply_mask(hsv_frame, frame)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            frame, green_cx, green_cy = Balls.handle_contour(green_contours, frame, (0, 255, 0), 'Green')
            frame, orange_cx, orange_cy = Balls.handle_contour(orange_contours, frame, (0, 165, 255), 'Orange')

            if shortest_ball_position is not None:

                ## only run once
                if flag_one:
                    route_shape = np.array([
                        [[x_range_big[0], y_range_big[0]]],
                        [[x_range_big[0], y_range_big[1]]],
                        [[x_range_big[1], y_range_big[1]]],
                        [[x_range_big[1], y_range_big[0]]]
                    ], dtype=np.int32)

                    # Calculate the center
                    M = cv2.moments(route_shape)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # Scale the coordinates
                    scale_factor = 0.7  # Scale down
                    route_shape_scaled = route_shape * scale_factor

                    # Translate the smaller rectangle so that it has the same centroid as the larger rectangle
                    route_shape_scaled += np.array([cX, cY]) - np.array([cX, cY]) * scale_factor

                    distances = [math.dist([green_cx, green_cy], vertex[0]) for vertex in route_shape_scaled]
                    closest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:2]
                    closest_points = [route_shape_scaled[index][0] for index in closest_indices]
                    distances_to_shortest_ball = [math.dist(shortest_ball_position, point) for point in
                                                  closest_points]
                    closest_to_shortest_ball_index = distances_to_shortest_ball.index(
                        min(distances_to_shortest_ball))
                    closest_to_shortest_ball = closest_points[closest_to_shortest_ball_index]
                    closest_to_shortest_ball = closest_to_shortest_ball.astype(int)

                    flag_one = False
                    flag_two = True

                if flag_two:

                    cv2.line(frame, (closest_to_shortest_ball[0], closest_to_shortest_ball[1]), (green_cx, green_cy),
                             (255, 0, 0), 2)

                    line_length = np.linalg.norm(np.array([green_cx, green_cy]) -
                                                 np.array([closest_to_shortest_ball[0], closest_to_shortest_ball[1]]))

                    angle_degrees, turn = angle_turn(closest_to_shortest_ball[0], closest_to_shortest_ball[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)

                    #'''
                    if line_length < 70:
                        SSH.send_channel_commands(channel, 5)
                        flag_three = True
                        flag_two = False
                    elif angle_degrees >= 350 or angle_degrees <= 10:
                        SSH.send_channel_commands(channel, 8)
                    elif flag:
                        flag = False
                        SSH.send_channel_commands(channel, turn)
                           
                if flag_three:
                    closest_to_ball = find_closest_point(route_shape_scaled, shortest_ball_position)
                    closest_to_goal = closest_to_ball

                    cv2.line(frame, (closest_to_ball[0], closest_to_ball[1]), (green_cx, green_cy),
                             (255, 0, 0), 2)
                    
                    line_length = np.linalg.norm(np.array([green_cx, green_cy]) -
                                                 np.array([closest_to_ball[0], closest_to_ball[1]]))

                    angle_degrees, turn = angle_turn(closest_to_ball[0], closest_to_ball[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)

                    print(closest_to_ball[0], closest_to_ball[1], 'first')

                    if line_length < 70:
                        SSH.send_channel_commands(channel, 5)
                        flag_three = False
                    elif angle_degrees >= 350 or angle_degrees <= 10:
                        SSH.send_channel_commands(channel, 8)
                    else:
                        SSH.send_channel_commands(channel, turn)


                if shortest_ball_position == (goal[0], goal[1]) and closest_to_goal is not None:
                    same_quarter = check_same_quarter(green_cx, green_cy,
                                                      (closest_to_goal[0], closest_to_goal[1]), x_range_big, y_range_big)
                    print(closest_to_ball[0], closest_to_ball[1], 'secoun')
                else:
                    #check if the robot is close to the ball
                    same_quarter = check_same_quarter(green_cx, green_cy, shortest_ball_position,
                                                      x_range_big, y_range_big)

                if same_quarter:
                    SSH.send_channel_commands(channel, 5)
                    channel.close()
                    ssh.close()
                    cam.release()
                    cv2.destroyAllWindows()
                    break


            cv2.imshow('Live Feed', frame)


        flag = True
        flag_one = True
        flag_two = flag_three = False

        while True:
            ret, frame = cam.read()

            if not ret:
                print('failed to grab frame')
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_mask, orange_mask = Balls.apply_mask(hsv_frame, frame)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            frame, green_cx, green_cy = Balls.handle_contour(green_contours, frame, (0, 255, 0), 'Green')
            frame, orange_cx, orange_cy = Balls.handle_contour(orange_contours, frame, (0, 165, 255), 'Orange')

            if shortest_ball_position is not None:

                ## only run once
                if flag_one:
                    route_shape = np.array([
                        [[x_range_big[0], y_range_big[0]]],
                        [[x_range_big[0], y_range_big[1]]],
                        [[x_range_big[1], y_range_big[1]]],
                        [[x_range_big[1], y_range_big[0]]]
                    ], dtype=np.int32)

                    # Calculate the centroid of the larger rectangle
                    M = cv2.moments(route_shape)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Scale the coordinates of the larger rectangle to create a smaller rectangle
                    scale_factor = 0.7  # Scale down
                    route_shape_scaled = route_shape * scale_factor

                    # Translate the smaller rectangle so that it has the same centroid as the larger rectangle
                    route_shape_scaled += np.array([cX, cY]) - np.array([cX, cY]) * scale_factor

                    distances = [math.dist([green_cx, green_cy], vertex[0]) for vertex in route_shape_scaled]
                    closest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:2]
                    closest_points = [route_shape_scaled[index][0] for index in closest_indices]
                    distances_to_shortest_ball = [math.dist(shortest_ball_position, point) for point in
                                                  closest_points]
                    closest_to_shortest_ball_index = distances_to_shortest_ball.index(
                        min(distances_to_shortest_ball))
                    closest_to_shortest_ball = closest_points[closest_to_shortest_ball_index]
                    closest_to_shortest_ball = closest_to_shortest_ball.astype(int)

                    flag_one = False
                    flag_two = True

                if flag_two:

                    print('1')

                    cv2.line(frame, (closest_to_shortest_ball[0], closest_to_shortest_ball[1]), (green_cx, green_cy),
                             (255, 0, 0), 2)

                    line_length = np.linalg.norm(np.array([green_cx, green_cy]) -
                                                 np.array([closest_to_shortest_ball[0], closest_to_shortest_ball[1]]))

                    angle_degrees, turn = angle_turn(closest_to_shortest_ball[0], closest_to_shortest_ball[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)

                    # '''
                    if line_length < 100:
                        SSH.send_channel_commands(channel, 5)
                        flag_three = True
                        flag_two = False
                    elif angle_degrees >= 355 or angle_degrees <= 5:
                        SSH.send_channel_commands(channel, 8)
                    elif flag:
                        flag = False
                        SSH.send_channel_commands(channel, turn)

                if flag_three:
                    closest_to_ball = find_closest_point(route_shape_scaled, shortest_ball_position)

                    cv2.line(frame, (closest_to_ball[0], closest_to_ball[1]), (green_cx, green_cy),
                             (255, 0, 0), 2)

                    line_length = np.linalg.norm(np.array([green_cx, green_cy]) -
                                                 np.array([closest_to_ball[0], closest_to_ball[1]]))

                    angle_degrees, turn = angle_turn(closest_to_ball[0], closest_to_ball[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)


                    # Perform the desired actions based on the angle and distance
                    if line_length < 70:
                        SSH.send_channel_commands(channel, 5)
                        flag_three = False
                    elif angle_degrees >= 350 or angle_degrees <= 10:
                        print(angle_degrees)
                        SSH.send_channel_commands(channel, 1)
                    else:
                        SSH.send_channel_commands(channel, turn)

                # check if the robot is close to the ball
                same_quarter = check_same_quarter(green_cx, green_cy, shortest_ball_position, x_range_big, y_range_big)
                if same_quarter:
                    SSH.send_channel_commands(channel, 5)
                    channel.close()
                    ssh.close()
                    cam.release()
                    cv2.destroyAllWindows()
                    break

                # Draw the line segment from the green point to the closest line segment on shape_1_scaled
                cv2.drawContours(frame, [route_shape_scaled.astype(np.int32)], -1, (0, 0, 255), 2)

            cv2.imshow('Live Feed', frame)

    elif shortest_ball_color == "purple":
        while True:
            ret, frame = cam.read()
            if not ret:
                print('failed to grab frame')
                break

            route_shape = np.array([
                [[x_range_big[0], y_range_big[0]]],
                [[x_range_big[0], y_range_big[1]]],
                [[x_range_big[1], y_range_big[1]]],
                [[x_range_big[1], y_range_big[0]]]
            ], dtype=np.int32)

            # Calculate the center
            M = cv2.moments(route_shape)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Scale the coordinates
            scale_factor = 0.7  # Scale down
            route_shape_scaled = route_shape * scale_factor

            # Translate the smaller rectangle so that it has the same centroid as the larger rectangle
            route_shape_scaled += np.array([cX, cY]) - np.array([cX, cY]) * scale_factor

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_mask, orange_mask = Balls.apply_mask(hsv_frame, frame)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            frame, green_cx, green_cy = Balls.handle_contour(green_contours, frame, (0, 255, 0), 'Green')
            frame, orange_cx, orange_cy = Balls.handle_contour(orange_contours, frame, (0, 165, 255), 'Orange')

            if shortest_ball_position is not None:
                shortest_ball_x, shortest_ball_y = shortest_ball_position

                line_length = np.linalg.norm(
                    np.array([orange_cx, orange_cy]) - np.array([shortest_ball_x, shortest_ball_y]))

                angle_degrees, turn \
                    = angle_turn(shortest_ball_x, shortest_ball_y, orange_cx, orange_cy, green_cx, green_cy)

                # stop when the ball is close wnough and use fan
                if line_length < 80:
                    SSH.send_channel_commands(channel, 5)
                    SSH.send_channel_commands(channel, 7)
                    time.sleep(2)
                    channel.close()
                    ssh.close()
                    cam.release()
                    cv2.destroyAllWindows()
                    break
                elif (angle_degrees >= 350) or (angle_degrees <= 10):
                    SSH.send_channel_commands(channel, 4)
                elif (angle_degrees <= 355) or (angle_degrees <= 5) and (line_length > 300):
                    SSH.send_channel_commands(channel, turn)
                ## drive front if the robot is too much out of the box
                elif Balls.check_outofbox(orange_cx, orange_cy, x_range_big, y_range_big) or \
                        Balls.check_withinbox(orange_cx, orange_cy, x_range_big, y_range_big):
                    SSH.send_channel_commands(channel, 8)
                    time.sleep(1)
                    SSH.send_channel_commands(channel, 5)

        cam.release()

    ## depending how far awya the goal is from ball routes are run
    elif shortest_ball_color == "goal_quarter_1" or shortest_ball_color == "goal_quarter_3":

        flag = True
        flag_one = True
        flag_two = flag_three = flag_four = flag_five = False

        while True:
            ret, frame = cam.read()

            if not ret:
                print('failed to grab frame')
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_mask, orange_mask = Balls.apply_mask(hsv_frame, frame)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            frame, green_cx, green_cy = Balls.handle_contour(green_contours, frame, (0, 255, 0), 'Green')
            frame, orange_cx, orange_cy = Balls.handle_contour(orange_contours, frame, (0, 165, 255), 'Orange')

            if shortest_ball_position is not None:

                ## only run once
                if flag_one:
                    route_shape = np.array([
                        [[x_range_big[0], y_range_big[0]]],
                        [[x_range_big[0], y_range_big[1]]],
                        [[x_range_big[1], y_range_big[1]]],
                        [[x_range_big[1], y_range_big[0]]]
                    ], dtype=np.int32)


                    M = cv2.moments(route_shape)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    scale_factor = 0.7  # Scale down
                    route_shape_scaled = route_shape * scale_factor

                    route_shape_scaled += np.array([cX, cY]) - np.array([cX, cY]) * scale_factor

                    flag_one = False
                    flag_two = True

                elif flag_two:

                    if shortest_ball_color == "goal_quarter_1":
                        first_corner = tuple(route_shape_scaled[0][0].astype(int))
                    elif shortest_ball_color == "goal_quarter_3":
                        first_corner = tuple(route_shape_scaled[1][0].astype(int))

                    cv2.line(frame, (green_cx, green_cy), first_corner, (0, 255, 0), 2)

                    line_length \
                        = np.linalg.norm(np.array([green_cx, green_cy]) - np.array([first_corner[0], first_corner[1]]))

                    angle_degrees, turn = angle_turn(first_corner[0], first_corner[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)

                    # '''
                    if line_length < 60:
                        SSH.send_channel_commands(channel, 5)
                        flag_three = flag = True
                        flag_two = False
                    elif angle_degrees >= 355 or angle_degrees <= 5:
                        SSH.send_channel_commands(channel, 8)
                    elif flag:
                        flag = False
                        SSH.send_channel_commands(channel, turn)

                elif flag_three:

                    if shortest_ball_color == "goal_quarter_1":
                        next_corner = tuple(route_shape_scaled[3][0].astype(int))
                    elif shortest_ball_color == "goal_quarter_3":
                        next_corner = tuple(route_shape_scaled[2][0].astype(int))

                    cv2.line(frame, (next_corner[0], next_corner[1]), (green_cx, green_cy),
                             (255, 0, 0), 2)

                    line_length = np.linalg.norm(np.array([green_cx, green_cy]) -
                                                 np.array([next_corner[0], next_corner[1]]))

                    angle_degrees, turn = angle_turn(next_corner[0], next_corner[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)

                    print(line_length, angle_degrees, turn)


                    if line_length < 60:
                        SSH.send_channel_commands(channel, 5)
                        flag_three = False
                        flag_four = flag = True
                    elif angle_degrees >= 350 or angle_degrees <= 10:
                        SSH.send_channel_commands(channel, 8)
                    elif flag:
                        flag = False
                        SSH.send_channel_commands(channel, turn)
                    else:
                        flag = True

                elif flag_four:
                    if shortest_ball_color == "goal_quarter_1":
                        next_corner = tuple(route_shape_scaled[3][0].astype(int))
                    elif shortest_ball_color == "goal_quarter_3":
                        next_corner = tuple(route_shape_scaled[2][0].astype(int))

                    cv2.line(frame, (green_cx, green_cy), (next_corner[0], goal[1]), (255, 0, 0), 2)

                    line_length = np.linalg.norm(np.array([green_cx, green_cy]) -
                                                 np.array([next_corner[0], goal[1]]))

                    angle_degrees, turn = angle_turn(next_corner[0], goal[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)

                    if line_length < 50:
                        SSH.send_channel_commands(channel, 5)
                        flag_four = False
                        flag_five = flag = True
                    elif angle_degrees >= 350 or angle_degrees <= 10:
                        SSH.send_channel_commands(channel, 8)
                    elif flag:
                        flag = False
                        SSH.send_channel_commands(channel, turn)

                elif flag_five:

                    cv2.line(frame, (green_cx, green_cy), (goal[0], goal[1]), (255, 0, 0), 2)

                    angle_degrees, turn = angle_turn(goal[0], goal[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)

                    if angle_degrees >= 355 or angle_degrees <= 7:
                        # run goal code
                        SSH.send_channel_commands(channel, 6)
                        time.sleep(5)
                        cam.release()
                        cv2.destroyAllWindows()
                        break

                    elif flag:
                        flag = False
                        SSH.send_channel_commands(channel, turn)

                cv2.drawContours(frame, [route_shape_scaled.astype(np.int32)], -1, (0, 0, 255), 2)

            cv2.imshow('Live Feed', frame)

    elif shortest_ball_color == "goal_quarter_2" or shortest_ball_color == "goal_quarter_4":
        flag = True
        flag_one = True
        flag_three = flag_four = flag_five = False

        while True:
            ret, frame = cam.read()

            if not ret:
                print('failed to grab frame')
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_mask, orange_mask = Balls.apply_mask(hsv_frame, frame)
            green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            frame, green_cx, green_cy = Balls.handle_contour(green_contours, frame, (0, 255, 0), 'Green')
            frame, orange_cx, orange_cy = Balls.handle_contour(orange_contours, frame, (0, 165, 255), 'Orange')

            if shortest_ball_position is not None:

                ## only run once
                if flag_one:
                    route_shape = np.array([
                        [[x_range_big[0], y_range_big[0]]],
                        [[x_range_big[0], y_range_big[1]]],
                        [[x_range_big[1], y_range_big[1]]],
                        [[x_range_big[1], y_range_big[0]]]
                    ], dtype=np.int32)


                    M = cv2.moments(route_shape)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    scale_factor = 0.7  # Scale down
                    route_shape_scaled = route_shape * scale_factor

                    route_shape_scaled += np.array([cX, cY]) - np.array([cX, cY]) * scale_factor

                    flag_one = False
                    flag_three = True

                elif flag_three:

                    if shortest_ball_color == "goal_quarter_2":
                        next_corner = tuple(route_shape_scaled[3][0].astype(int))
                    elif shortest_ball_color == "goal_quarter_4":
                        next_corner = tuple(route_shape_scaled[2][0].astype(int))

                    line_length = np.linalg.norm(np.array([green_cx, green_cy]) -
                                                 np.array([next_corner[0], next_corner[1]]))

                    angle_degrees, turn = angle_turn(next_corner[0], next_corner[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)

                    if line_length < 70:
                        SSH.send_channel_commands(channel, 5)
                        flag_three = False
                        flag_four = flag = True
                    elif angle_degrees >= 353 or angle_degrees <= 7:
                        SSH.send_channel_commands(channel, 8)
                    elif flag:
                        flag = False
                        SSH.send_channel_commands(channel, turn)
                    else:
                        flag = True

                elif flag_four:

                    if shortest_ball_color == "goal_quarter_2":
                        next_corner = tuple(route_shape_scaled[3][0].astype(int))
                    elif shortest_ball_color == "goal_quarter_4":
                        next_corner = tuple(route_shape_scaled[2][0].astype(int))

                    cv2.line(frame, (green_cx, green_cy), (next_corner[0], goal[1]), (255, 0, 0), 2)

                    line_length = np.linalg.norm(np.array([green_cx, green_cy]) -
                                                 np.array([next_corner[0], goal[1]]))

                    angle_degrees, turn = angle_turn(next_corner[0], goal[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)

                    if line_length < 50:
                        SSH.send_channel_commands(channel, 5)
                        flag_four = False
                        flag_five = flag = True
                    elif angle_degrees >= 353 or angle_degrees <= 7:
                        SSH.send_channel_commands(channel, 8)
                    elif flag:
                        flag = False
                        SSH.send_channel_commands(channel, turn)

                elif flag_five:

                    angle_degrees, turn = angle_turn(goal[0], goal[1],
                                                     green_cx, green_cy, orange_cx, orange_cy)

                    if angle_degrees >= 356 or angle_degrees <= 5:
                        # run goal code
                        SSH.send_channel_commands(channel, 6)
                        time.sleep(5)
                        cam.release()
                        cv2.destroyAllWindows()
                        break

                    elif flag:
                        flag = False
                        SSH.send_channel_commands(channel, turn)

                cv2.drawContours(frame, [route_shape_scaled.astype(np.int32)], -1, (0, 0, 255), 2)

            cv2.imshow('Live Feed', frame)


def check_same_quarter(green_cx, green_cy, shortest_ball_position, x_range_big, y_range_big):
    x_min, x_max = x_range_big
    y_min, y_max = y_range_big

    x_mid = (x_min + x_max) // 2
    y_mid = (y_min + y_max) // 2

    # Determine the quarter of robot
    if green_cx < x_mid:
        if green_cy < y_mid:
            green_quarter = 1
        else:
            green_quarter = 3
    else:
        if green_cy < y_mid:
            green_quarter = 2
        else:
            green_quarter = 4

    # Determine the quarter of shortest_ball_position
    shortest_ball_x, shortest_ball_y = shortest_ball_position
    if shortest_ball_x < x_mid:
        if shortest_ball_y < y_mid:
            shortest_ball_quarter = 1
        else:
            shortest_ball_quarter = 3
    else:
        if shortest_ball_y < y_mid:
            shortest_ball_quarter = 2
        else:
            shortest_ball_quarter = 4

    # Check if they are in the same quarter
    return green_quarter == shortest_ball_quarter

## find which point of the rectangle is cloest to a point
def find_closest_point(route_shape_scaled, ball_position):
    distances = [math.dist(ball_position, vertex[0]) for vertex in route_shape_scaled]
    closest_index = distances.index(min(distances))
    closest_point = route_shape_scaled[closest_index][0]
    closest_point = closest_point.astype(int)
    return closest_point

def angle_turn(shortest_ball_x, shortest_ball_y, green_cx, green_cy, orange_cx, orange_cy):

    # Calculate  vectors from orange to green and orange to the ball
    vector_green = complex(green_cx - orange_cx, green_cy - orange_cy)
    vector_ball = complex(shortest_ball_x - orange_cx, shortest_ball_y - orange_cy)

    # Calculate the angle
    angle_radians = math.degrees(
        math.atan2(vector_ball.imag, vector_ball.real) - math.atan2(vector_green.imag, vector_green.real))

    angle_degrees = (angle_radians + 360) % 360

    #assign the number of which way to turn
    if 0 < angle_degrees < 180:
        turn = 3 #right
    else:
        turn = 2#

    return angle_degrees, turn


