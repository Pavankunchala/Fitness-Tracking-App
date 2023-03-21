import cv2
import mediapipe as mp
import numpy as np
import argparse
import numpy as np
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='Testing for Handstand.')
    parser.add_argument('-v','--video', type=str, default = 'Squats.mp4')
    parser.add_argument('-o','--output', type=str, default =None)
    parser.add_argument('--det',type=float, default = 0.5,help='Detection confidence')
    parser.add_argument('--track',type=float, default =0.5,help='Tracking confidence')
    parser.add_argument('-c','--complexity', type=int, default = 0,help='Complexity of the model options 0,1,2')
    
    return parser.parse_args()


args=parse_args()

line_color = (255,255,255)
# def estimateSpeed(location1, location2):
# 	d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
# 	# ppm = location2[2] / carWidht
# 	ppm = 8.8
# 	d_meters = d_pixels / ppm
# 	#print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
# 	fps = 18
# 	speed = d_meters * fps * 3.6
# 	return speed

start_time = time.time()

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians *180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=4,color = (line_color))
drawing_spec_points = mp_drawing.DrawingSpec(thickness=5, circle_radius=4,color = (line_color))


detection_confidence = args.det
tracking_confidence = args.track

complexity = args.complexity





vid = cv2.VideoCapture(args.video)
#vid = cv2.VideoCapture(0)

width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))
    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
codec = cv2.VideoWriter_fourcc('X','V','I','D')
out = cv2.VideoWriter(args.output, codec, fps, (width, height))

with mp_pose.Pose(
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence,
    model_complexity=complexity,smooth_landmarks = True ) as pose:

    while vid.isOpened():

        success, image = vid.read()

        #image = cv2.rotate(image,cv2.ROTATE_180)

        if not success:
            break
        #image = cv2.resize(image,(0,0),fx = 0.5, fy = 0.5)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
   
        image.flags.writeable = False
        results = pose.process(image)
        eyesVisible = False
        shoulderVisible = False


        
            #code for pose extraction
        try:
            
            landmarks = results.pose_landmarks.landmark

            


            #Check if both eyes are visible.

            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]

            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]

            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            elbow_r = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            wrist_r = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,landmarks[mp_pose.PoseLandmark.NOSE.value].y]


                #Get Tje Corridnates of Hip
            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]




            #Put the Values for visibility 


            #visiblity for Eyes
            landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value].visibility = 0


            #fOR NOSE
            landmarks[mp_pose.PoseLandmark.NOSE.value].visibility = 0

            landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].visibility = 0



            #fOR eAR
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].visibility = 0

            


            
            

            

            #print('LeftEye',left_visible)

        
            

            #Check if both shoulders are visible.
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]


                



                #Midpointts
            midpoint_shoulder_x = (int(shoulder[0] * image_width )+ int(shoulder_r[0] * image_width))/2

            midpoint_shoulder_y = (int(shoulder[1] * image_height )+ int(shoulder_r[1] * image_height))/2

            midpoint_hip_x = (int(left_hip[0] * image_width )+ int(right_hip[0] * image_width))/2
            midpoint_hip_y = (int(left_hip[1] * image_height)+ int(right_hip[1] * image_height))/2



            based_mid_x = int((midpoint_shoulder_x + midpoint_hip_x)/2)
            based_mid_y = int((midpoint_shoulder_y + midpoint_hip_y)/2)

            neck_point_x = (int(nose[0] * image_width )+ int(midpoint_shoulder_x))/2
            neck_point_y = (int(nose[1] * image_height) + int(midpoint_shoulder_y))/2
            


            #print('Hip',left_hip)

            #angles 
            left_arm_angle = int(calculate_angle(shoulder, elbow, wrist))
            right_arm_angle = int(calculate_angle(shoulder_r, elbow_r, wrist_r))
            left_leg_angle = int(calculate_angle(left_hip, left_knee, left_ankle))

            right_leg_angle = int(calculate_angle(right_hip, right_knee, right_ankle))


            
            left_arm_length = np.linalg.norm(np.array(shoulder) - np.array(elbow))

            # ppm = 10.8

            # left_arm_motion = left_arm_angle* left_arm_length

            # left_arm_motion = left_arm_motion/ppm

            #newpoint_left = [left_hip[0] +5,right_hip[0] +5]

            mid_point_x = (int(left_hip[0] * image_width )+ int(right_hip[0] * image_width))/2
            mid_point_y = (int(left_hip[1] * image_height)+ int(right_hip[1] * image_height))/2

            #cv2.circle(image,(int(mid_point_x) ,int(mid_point_y +30 )),15,(0,255,255),-1)


            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility = 0
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility = 0


            cv2.line(image,(int(shoulder[0]* image_width),int(shoulder[1]* image_height)),(int(neck_point_x),int(neck_point_y)),(line_color),3)

            cv2.line(image,(int(shoulder_r[0]* image_width),int(shoulder_r[1]* image_height)),(int(neck_point_x),int(neck_point_y)),(line_color),3)

            cv2.line(image,(int(shoulder[0]* image_width),int(shoulder[1]* image_height)),(int(elbow[0]* image_width),int(elbow[1]* image_height)),(line_color),3)
            cv2.line(image,(int(shoulder_r[0]* image_width),int(shoulder_r[1]* image_height)),(int(elbow_r[0]* image_width),int(elbow_r[1]* image_height)),(line_color),3)

            
                

                #neck to mid point
            cv2.line(image,(int(neck_point_x),int(neck_point_y)),(int(based_mid_x),int(based_mid_y)),(line_color),3,cv2.LINE_4)

                #mid to hips
            cv2.line(image,(int(based_mid_x),int(based_mid_y)),(int(left_hip[0] * image_width ),(int(left_hip[1] * image_height))),(line_color),3,cv2.LINE_8)

            cv2.line(image,(int(based_mid_x),int(based_mid_y)),(int(right_hip[0] * image_width ),(int(right_hip[1] * image_height))),(line_color),3,cv2.LINE_8)


                ##neck point


            cv2.circle(image,(int(neck_point_x),int(neck_point_y)),4,(line_color),5)

                #create new circles at that place
            cv2.circle(image,(int(shoulder[0]* image_width),int(shoulder[1]* image_height)),4,(line_color),3)
            cv2.circle(image,(int(shoulder_r[0]* image_width),int(shoulder_r[1]* image_height)),4,(line_color),3)
                #mid point
            cv2.circle(image,(int(based_mid_x),int(based_mid_y)),4,(line_color),5)



            cv2.rectangle(image,(image_width,0),(image_width-300,250),(0,0,0),-1)
            cv2.putText(image,'Angles',(image_width-300,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)
            cv2.putText(image, 'Left Elbow Angle: ' + str(left_arm_angle), (image_width-290, 70), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Right Elbow Angle: ' + str(right_arm_angle), (image_width-290, 110), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Left Knee Angle: ' + str(left_leg_angle), (image_width-290, 150), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Right Knee Angle: ' + str(right_leg_angle), (image_width-290, 190), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(image, 'Left arm motion: ' + str(left_arm_motion), (image_width-290, 230), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)







                




            


            


            
            


            #print('left eye',left_eye)

            #print('Is Eye Visible',eyesVisible)
            #print('Is Shoulder Visible',shoulderVisible)
        except:
            pass

        
                    #writing angles
        
            #cv2.putText(image,"left elbow" + str(left_arm_angle),(int(image_width - 250),int(40)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
            #cv2.putText(image,str(right_arm_angle),(int(elbow_r[0]* image_width  -40),int(elbow_r[1]* image_height)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,244,244),2,cv2.LINE_AA)


            #cv2.putText(image,str(left_leg_angle),(int(left_knee[0]* image_width + 40),int(left_knee[1]* image_height)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2,cv2.LINE_AA)
            #cv2.putText(image,str(right_leg_angle),(int(right_knee[0]* image_width - 40),int(right_knee[1]* image_height)),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2,cv2.LINE_AA)





        


                


        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            drawing_spec_points,
            
        connection_drawing_spec=drawing_spec)

        fps =1.0/(time.time() - start_time)

        
        
        final_frame = image
        #final_frame = cv2.rotate(final_frame,cv2.ROTATE_180)
        if args.output:
            out.write(final_frame)


        final_frame = cv2.resize(final_frame,(0,0),fx = 0.5,fy = 0.5)

        cv2.imshow('Pose',final_frame)


        if cv2.waitKey(1) & 0xFF == 27:

            break

    vid.release()
    out.release()
    cv2.destroyAllWindows()

        




