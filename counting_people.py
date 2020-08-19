# Usage example:  python3 counting_people.py --video=TownCentreXVID.avi

import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import math
from counting.centroidtracker import CentroidTracker
from counting.trackableobject import TrackableObject
import mysql.connector
import winsound
import datetime
import os
import smtplib 
import threading
import random
import pyopencl as cl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
# Initialize the parameters
confThreshold = 0.6  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
floor=3
door=2
parser = argparse.ArgumentParser(description='Counting People using YOLO in OPENCV')

parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
        
# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3-spp.cfg";
modelWeights = "yolov3-spp.weights";

# load our serialized model from disk
print("[INFO] loading model...")
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

# initialize the video writer
writer = None
 
# initialize the frame dimensions (we'll set them as soon as we read
# the first frame from the video)
W = None
H = None
 
# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
 
# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalDown = 0
totalUp = 0
totalU=0
totalD=0
# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    print(left,top,right,bottom)
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
    # Draw a center of a bounding box
    # frameHeight = frame.get().shape[0]
    # frameWidth = frame.get().shape[1]
    # cv.line(frame, (0, frameHeight//2 - 50), (frameWidth, frameHeight//2 - 50), (0, 255, 255), 2)
    # cv.circle(frame,(left+(right-left)//2, top+(bottom-top)//2), 3, (0,0,255), -1)
        
    coun=0
    counter = []
    if (top+(bottom-top)//2 in range(frameHeight//2 - 2,frameHeight//2 + 2)):
        coun +=1
        #print(coun)

        counter.append(coun)

    label = 'Pedestrians: '.format(str(counter))
    cv.putText(frame, label, (0, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs,mydb):
    shape = frame.get().shape
    frameHeight = shape[0]
    frameWidth = shape[1]

    rects = []

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Class "person"
        if classIds[i] == 0:
            rects.append((left, top, left + width, top + height))
            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            objects = ct.update(rects)
            counting(objects,mydb,left, top, left + width, top + height)
            # drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

def counting(objects,mydb,left, top, right, bottom):
    shape = frame.get().shape
    frameHeight = shape[0]
    frameWidth = shape[1]

    global totalDown
    global totalUp
    global totalU
    global totalD
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)
 
        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)
     
        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
 
            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object

                if direction < 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                    totalUp += 1
                    to.counted = True
                    time_now=datetime.datetime.now()
                    if (totalUp % 3!=0):
                        write_record_to_DB(mydb,time_now-datetime.timedelta(seconds=1),"in")
                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                    totalDown += 1
                    to.counted = True
                    time_now=datetime.datetime.now()
                    if (totalDown % 3!=0):
                        write_record_to_DB(mydb,time_now+datetime.timedelta(seconds=1),"out")
        # store the trackable object in our dictionary
        trackableObjects[objectID] = to
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        
        cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # construct a tuple of information we will be displaying on the
    # frame
    if totalU!=totalUp:
        text = "In Count {}".format(totalUp)
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
        cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        count_new=totalUp-totalU
        totalU=totalUp
        validate_up_count(mydb,count_new,time_now)
    if totalD!=totalDown:
        text = "Down Count {}".format(totalDown)
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
        cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        count_new=totalDown-totalD
        totalD=totalDown
        validate_down_count(mydb,count_new,time_now)
    info = [
        ("Up", totalUp),
        ("Down", totalDown),
    ]
    #to write into file
    f = open("outputfile.txt", "a")   
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    line = "{} , {}, {}\n".format(timestamp, totalUp, totalDown)
    f.write(line)
    f.close()

    # print(totalDown,totalUp)
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv.putText(frame, text, (10, frameHeight - ((i * 20) + 20)),
            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def validate_up_count(mydb,count,time_now):
    global totalU
    global floor
    global door
    X=2
    start_time= time_now - datetime.timedelta(seconds=X)
    end_time=time_now
    sql = "SELECT * FROM identity_status where floor=3 and door=2 and direction='in' and time BETWEEN '"+ str(start_time) + "' AND '"+ str(end_time) + "'"
    #to read from file
    # if os.path.isfile(filename):
    #     with open(filename) as csvfile:
    #         csv_reader = csv.reader(csvfile, delimiter=',')
    #         for row in csv_reader:
    #             current_time=datetime.datetime.strptime(row[0], "%Y%m%d_%H-%M-%S ")
    #             if current_time>= start_time:
    #                 count_list.append(int(row[1]))
    mycursor = mydb.cursor()
    mycursor.execute(sql)
    myresult = mycursor.fetchall()
    mycursor.close()
    mydb.commit()
    print("[INFO] got "+str(len(myresult))+" records when expected was "+str(count))
    if len(myresult)!=count:
        fps = cap.get(cv.CAP_PROP_FPS)
        current_frame_number =(cap.get(cv.CAP_PROP_POS_FRAMES))
        videotime = current_frame_number / fps
        print("[INFO] tailgating happened at  ",videotime)
        timestamp = time_now.strftime("%Y%m%d_%H-%M-%S")
        file_name="G:\\Counting-People-master\\in"+str(timestamp)+".jpg"
        if not cv.imwrite(file_name, frame):
             raise Exception("Could not write image")
        duration = 1000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)
        time_diff=time_now - datetime.timedelta(seconds=5)
        sql="Select distinct a.* from (Select b.* From (SELECT * FROM identity_status where floor=3 and door=2 and direction='in' and time < '"+ str(time_now)+"' Order by time desc LIMIT 1 ) b UNION select * from identity_status where floor=3 and door=2 and direction='in' and time between '"+str(time_now)+"' and '"+str(time_diff)+"')a "
        mycursor = mydb.cursor()
        mycursor.execute(sql)
        myresult = mycursor.fetchall()
        mycursor.close()
        mydb.commit()            
        table_body= ""
        for x in myresult:
            table_row="<tr>\n"+"<td>"+x[1]+"</td>\n"+"<td>"+str(x[2])+"</td>\n"+"<td>"+str(x[3])+"</td>\n"+"<td>"+x[4]+"</td>\n"+"<td>"+str(x[5])+"</td>\n"+"<td>"+str(x[6])+"</td>\n"
            table_body=table_body+table_row
        start_part = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <style type="text/css">
            table {
                background: white;
                border-radius:3px;
                border-collapse: collapse;
                height: auto;
                max-width: 900px;
                padding:5px;
                width: 100%;
                animation: float 5s infinite;
            }
            th {
                color:#D5DDE5;;
                background:#1b1e24;
                border-bottom: 4px solid #9ea7af;
                font-size:14px;
                font-weight: 300;
                padding:10px;
                text-align:center;
                vertical-align:middle;
            }
            tr {
                border-top: 1px solid #C1C3D1;
                border-bottom: 1px solid #C1C3D1;
                border-left: 1px solid #C1C3D1;
                color:#666B85;
                font-size:16px;
                font-weight:normal;
            }
            tr:hover td {
                background:#4E5066;
                color:#FFFFFF;
                border-top: 1px solid #22262e;
            }
            td {
                background:#FFFFFF;
                padding:10px;
                text-align:left;
                vertical-align:middle;
                font-weight:300;
                font-size:13px;
                border-right: 1px solid #C1C3D1;
            }
            </style>
        </head>
        <body>
            You are recieving this email because system has observed tailgating <br>
            Please check the following details and snapshot of the same: <br>
            <table>
            <thead>
                <tr style="border: 1px solid #1b1e24;">
                <th>username</th>
                <th>floor</th>
                <th>door</th>
                <th>department</th>
                <th>timestamp</th>
                <th>direction</th>
                </tr>
            </thead>
            <tbody>
            """
        later_part="""
            </tbody>
            </table>
            <br>
            For more assistance please contact our support team:
            <a href='mailto:counter_people@gmail.com'>counter_people@gmail.com</a>.<br> <br>
            Thank you!
        </body>
        </html>
        """    
        HTML=start_part+table_body+later_part
        thread = threading.Thread(target=send_Email(HTML,file_name))
        thread.start()
        
def write_record_to_DB(mydb,time_now,direction):
    global floor
    global door
    names=['Muskan Nechlani','Sagar Sanghrajka', 'Karan Nayak','Naira Rohida','Hemant Kulkarni','Rhea Haran','Sneha Rohra', 'Arjun Punjabi','Ayu Talreja','Mohan Katyar']
    mycursor=mydb.cursor()
    insert_stmt = (
   "INSERT INTO identity_status(userName, floor, door, department,time, direction)"
   "VALUES (%s, %s, %s, %s, %s, %s)"
                )
    data = (random.choice(names), floor, door, 'Stocks',time_now, direction)
   
    try:
   # Executing the SQL command
        mycursor.execute(insert_stmt, data)   
   # Commit your changes in the database
        mydb.commit()
        print("[INFO] Inserted ",data)

    except Exception as e:
   # Rolling back in case of error
        mydb.rollback()
        print("Exception",e)

def validate_down_count(mydb,count,time_now):
    global totalD
    global floor
    global door
    seconds=0
    while seconds!=3:
        current_time=time_now + datetime.timedelta(seconds=seconds)
        current_time=current_time.replace(microsecond=0)
        sql = "SELECT * FROM identity_status where floor=3 and door=2 and direction='out' and time LIKE '%"+str(current_time)+"%'"
        mycursor = mydb.cursor()
        mycursor.execute(sql)
        myresult = mycursor.fetchall()  
        seconds=seconds+1
        mycursor.close()
        mydb.commit()
        if(len(myresult)!=0):
            break
    #to read from file
    # if os.path.isfile(filename):
    #     with open(filename) as csvfile:
    #         csv_reader = csv.reader(csvfile, delimiter=',')
    #         for row in csv_reader:
    #             current_time=datetime.datetime.strptime(row[0], "%Y%m%d_%H-%M-%S ")
    #             if current_time>= start_time:
    #                 count_list.append(int(row[1]))
   
    print("[INFO] got "+str(len(myresult))+" records when expected was "+str(count))
    if len(myresult)!=count:
        fps = cap.get(cv.CAP_PROP_FPS)
        current_frame_number =(cap.get(cv.CAP_PROP_POS_FRAMES))
        videotime = current_frame_number / fps
        print("[INFO] tailgating happened at  ",videotime)
        timestamp = time_now.strftime("%Y%m%d_%H-%M-%S")
        file_name="G:\\Counting-People-master\\out"+str(timestamp)+".jpg"
        if not cv.imwrite(file_name, frame):
             raise Exception("Could not write image")
        duration = 1000  # milliseconds
        freq = 440  # Hz
        winsound.Beep(freq, duration)
        time_diff=time_now - datetime.timedelta(seconds=5)
        sql="Select distinct a.* from (Select b.* From (SELECT * FROM identity_status where floor=3 and door=2 and direction='out' and time < '"+ str(time_now)+"' Order by time desc LIMIT 1 ) b UNION select * from identity_status where floor=3 and door=2 and direction='out' and time between '"+str(time_now)+"' and '"+str(time_diff)+"')a "
        mycursor = mydb.cursor()
        mycursor.execute(sql)
        myresult = mycursor.fetchall()
        mycursor.close()
        mydb.commit()
        
        table_body= ""
        for x in myresult:
            table_row="<tr>\n"+"<td>"+x[1]+"</td>\n"+"<td>"+str(x[2])+"</td>\n"+"<td>"+str(x[3])+"</td>\n"+"<td>"+x[4]+"</td>\n"+"<td>"+str(x[5])+"</td>\n"+"<td>"+str(x[6])+"</td>\n"
            table_body=table_body+table_row
        start_part = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8" />
            <style type="text/css">
            table {
                background: white;
                border-radius:3px;
                border-collapse: collapse;
                height: auto;
                max-width: 900px;
                padding:5px;
                width: 100%;
                animation: float 5s infinite;
            }
            th {
                color:#D5DDE5;;
                background:#1b1e24;
                border-bottom: 4px solid #9ea7af;
                font-size:14px;
                font-weight: 300;
                padding:10px;
                text-align:center;
                vertical-align:middle;
            }
            tr {
                border-top: 1px solid #C1C3D1;
                border-bottom: 1px solid #C1C3D1;
                border-left: 1px solid #C1C3D1;
                color:#666B85;
                font-size:16px;
                font-weight:normal;
            }
            tr:hover td {
                background:#4E5066;
                color:#FFFFFF;
                border-top: 1px solid #22262e;
            }
            td {
                background:#FFFFFF;
                padding:10px;
                text-align:left;
                vertical-align:middle;
                font-weight:300;
                font-size:13px;
                border-right: 1px solid #C1C3D1;
            }
            </style>
        </head>
        <body>
            You are recieving this email because system has observed tailgating <br>
            Please check the following details and snapshot of the same: <br>
            <table>
            <thead>
                <tr style="border: 1px solid #1b1e24;">
                <th>username</th>
                <th>floor</th>
                <th>door</th>
                <th>department</th>
                <th>timestamp</th>
                <th>direction</th>
                </tr>
            </thead>
            <tbody>
            """
        later_part="""
            </tbody>
            </table>
            <br>
            For more assistance please contact our support team:
            <a href='mailto:counter_people@gmail.com'>counter_people@gmail.com</a>.<br> <br>
            Thank you!
        </body>
        </html>
        """    
        HTML=start_part+table_body+later_part
        thread = threading.Thread(target=send_Email(HTML,file_name))
        thread.start()
          
def send_Email(message,file_name):
    
        subject_template_name = 'Security Alert ';
        img_data = open(file_name, 'rb').read()
        fromaddr = "softcornercummins@gmail.com"
        toaddr = "muskannechlani@gmail.com"
        mail = MIMEMultipart( )
        mail['From'] = fromaddr
        mail['To'] = toaddr
        baseurl="g:/counting-people-master"
        mail['Subject'] = subject_template_name
        mail.attach(MIMEText(message,'html'))
        image = MIMEImage(img_data, name=os.path.basename(file_name))
        mail.attach(image)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(fromaddr, 'softcorner@2020')
        text = mail.as_string()
        server.sendmail(fromaddr, toaddr, text)
        server.quit()

# Process inputs

winName = 'Deep learning object detection in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
outputFile = "yolo_out_py.avi"
if (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv.VideoCapture(10)
print('OpenCL available:', cv.ocl.haveOpenCL())
print(cap.get(cv.CAP_PROP_FPS))
# Get the video writer initialized to save the output video
vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="yolodb"
                )
print(datetime.datetime.now())
while cv.waitKey(1) < 0:
    
    # get frame from the video
    hasFrame, frame = cap.read()
    
    if frame is None:
        print("over")
    else:
        frame = cv.UMat(frame)
        shape = frame.get().shape
        frameHeight = shape[0]
        frameWidth = shape[1]
        cv.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        print(datetime.datetime.now())
        cv.waitKey(3000)
        # Release device
        cap.release()
        break
   


    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))
    print(len(outs))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs,mydb)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes

    vid_writer.write(cv.UMat(frame))

    cv.imshow(winName, frame)