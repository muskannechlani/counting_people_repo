import datetime
import cv2 as cv
import sys
import numpy as np
import os.path
from counting.centroidtracker import CentroidTracker
from counting.trackableobject import TrackableObject
import pyodbc
import winsound
import os
import imutils
import smtplib 
import threading
import random
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
mydb = pyodbc.connect('Driver={SQL Server};'
                      'Server=ILPT812;'
                      'Database=Test;'
                      'Trusted_Connection=yes;')

# Initialize the parameters
confThreshold = 0.6  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
    # Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3-spp.cfg"
modelWeights = "yolov3-spp.weights"
threadLock=threading.Lock()
class myThread (threading.Thread):
    maxRetries=20
    def __init__(self, threadID, src,floor):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = src
        self.floor=floor
        self.door=1
        # self.cap = cv.VideoCapture(src)
        self.net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)
        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackableObjects = {}
        self.totalDown = 0
        self.totalUp = 0
        self.totalU=0
        self.totalD=0
          
    def attemptRead(self,cvVideo):
        threadLock.acquire()
        (isRead,cvImage)=cvVideo.read()
        threadLock.release()
        if isRead==False:
            count=1
            while isRead==False and count<myThread.maxRetries:
                threadLock.acquire()
                (isRead,cvImage)=cvVideo.read()
                threadLock.release()
                # print(self.name+' try no: ',count)
                count+=1
        return (isRead,cvImage)

    # Get the names of the output layers
    def getOutputsNames(self):
        # Get the names of all the layers in the network
       layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
       return [layersNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

# Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self,cap,frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

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
                objects = self.ct.update(rects)
                cv.rectangle(frame, (left, top), (left + width, top + height), (255, 178, 50), 2)
                self.counting(cap,frame,objects,left, top, left + width, top + height)

                #drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    def counting(self,cap,frame,objects,left, top, right, bottom):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = self.trackableObjects.get(objectID, None)
    
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
                        self.totalUp += 1
                        to.counted = True
                        time_now=datetime.datetime.now()
                        if (self.totalUp % 3!=0):
                            self.write_record_to_DB(time_now-datetime.timedelta(seconds=1),"in")
                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] in range(frameHeight//2 - 30, frameHeight//2 + 30):
                        self.totalDown += 1
                        to.counted = True
                        time_now=datetime.datetime.now()
                        if (self.totalDown % 3!=0):
                            self.write_record_to_DB(time_now+datetime.timedelta(seconds=1),"out")
            # store the trackable object in our dictionary
            self.trackableObjects[objectID] = to
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            #text = "ID {}".format(objectID)
            #cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                #cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        # construct a tuple of information we will be displaying on the
        # frame
        if self.totalU!=self.totalUp:
            cv.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)
            count_new=self.totalUp-self.totalU
            self.totalU=self.totalUp
            self.validate_up_count(cap,frame,count_new,time_now)
        if self.totalD!=self.totalDown:
            cv.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)
            count_new=self.totalDown-self.totalD
            self.totalD=self.totalDown
            self.validate_down_count(cap,frame,count_new,time_now)
        info = [
            ("Up", self.totalUp),
            ("Down", self.totalDown),
        ]
        #to write into file
        # f = open("outputfile.txt", "a")   
        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
        # line = "{} , {}, {}\n".format(timestamp, totalUp, totalDown)
        # f.write(line)
        # f.close()

        # print(totalDown,totalUp)
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv.putText(frame, text, (10, frameHeight - ((i * 20) + 20)),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
  
    def write_record_to_DB(self,time_now,direction):
        names=['Muskan N','Sagar S', 'Karan Nayak','Naira Rohida','Hemant Kulkarni','Rhea Haran','Sneha Rohra', 'Arjun Punjabi','Ayu Talreja','Mohan Katyar']
        mycursor=mydb.cursor()
        name=random.choice(names)
        insert_stmt = (
        "INSERT INTO identity_status(username, floor, door, department,time, direction)"
        "VALUES ('"+name+"', '"+str(self.floor)+"', '"+str(self.door)+"', 'Stocks', Convert(datetime2,'"+str(time_now)+"'), '"+direction+"')"
                )
        data = (name, self.floor, self.door, 'Stocks',time_now, direction)
    
        try:
            print("Video - ",self.name)
            mycursor.execute(insert_stmt)   
            # Commit your changes in the database
            mydb.commit()
            print("[INFO] Inserted ",data)

        except Exception as e:
        # Rolling back in case of error
            mydb.rollback()
            print("Exception",e)

    def run(self):
        print( self.name + "  Starting " +str(datetime.datetime.now()))
        cv.namedWindow(self.name, 0)
        cv.resizeWindow(self.name, 700,700)     
        cvVideo = cv.VideoCapture(self.name)
        outputFile = str(self.name)+"_yolo_out_py.avi"
        vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, (round(cvVideo.get(cv.CAP_PROP_FRAME_WIDTH)),round(cvVideo.get(cv.CAP_PROP_FRAME_HEIGHT))))


        while True:
            (isRead,cvImage)=self.attemptRead(cvVideo)
            if isRead==False:
                break
            # rgb = imutils.resize(cvVideo, width=750)
            #cv.resize(rgb, (300, 300))
            blob = cv.dnn.blobFromImage(cvImage, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.getOutputsNames())
            mydb=None
            self.postprocess(cvVideo,cvImage, outs)
            frameHeight = cvImage.shape[0]
            frameWidth = cvImage.shape[1]
            cv.line(cvImage, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)
            cv.imshow(self.name,cvImage)
            vid_writer.write(cvImage)
            key=cv.waitKey(50)
            if key==27:
                break

        cv.destroyWindow(self.name)
        print(self.name + "  Exiting "+ str(datetime.datetime.now()))

    def validate_up_count(self,cap,current_frame,count,time_now):
        
        X=2
        start_time= time_now - datetime.timedelta(seconds=X)
        end_time=time_now
        sql = "SELECT * FROM identity_status where floor=" +str(self.floor)+" and door="+str(self.door)+" and direction='in' and time BETWEEN Convert(datetime2,'"+ str(start_time) + "') AND Convert(datetime2,'"+ str(end_time) + "')"
        mycursor = mydb.cursor()
        mycursor.execute(sql)
        myresult = mycursor.fetchall()
        mycursor.close()
        mydb.commit()
        if len(myresult)!=count:
            print("Video - ",self.name)
        print("[INFO] got "+str(len(myresult))+" records when expected was "+str(count))
        if len(myresult)!=count:
            fps = cap.get(cv.CAP_PROP_FPS)
            current_frame_number =cap.get(cv.CAP_PROP_POS_FRAMES)
            videotime = current_frame_number / fps
            print("[INFO] tailgating happened at  ",videotime)
            timestamp = time_now.strftime("%Y%m%d_%H-%M-%S")
            file_name="C:\\Work\\P\\Project\\counting-people\\in"+str(timestamp)+".jpg"
            if not cv.imwrite(file_name, current_frame):
                raise Exception("Could not write image")
            duration = 1000  # milliseconds
            freq = 440  # Hz
            winsound.Beep(freq, duration)
            time_diff=time_now - datetime.timedelta(seconds=5)
            sql="Select distinct a.* from (Select b.* From (SELECT top 1* FROM identity_status where floor="+str(self.floor)+" and door="+str(self.door)+"  and time < Convert(datetime2,'"+ str(time_now)+"') Order by time desc ) b UNION select * from identity_status where floor="+str(self.floor)+" and door="+str(self.door)+" and direction='in' and time between Convert(datetime2,'"+str(time_now)+"') and Convert(datetime2,'"+str(time_diff)+"'))a "
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
                Please check the following details of person who entered exited before tailgating happened : <br>
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
                Refer the snapshot at the time of tailgating...
                For more assistance please contact our support team:
                <a href='mailto:counter_people@gmail.com'>counter_people@gmail.com</a>.<br> 
                Thank you!
            </body>
            </html>
            """
            HTML=start_part+table_body+later_part
            thread = threading.Thread(target=self.send_Email(HTML,file_name))
            thread.start()
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    
    def validate_down_count(self,cap,current_frame,count,time_now):
       
        seconds=0
        while seconds!=3:
            current_time=time_now + datetime.timedelta(seconds=seconds)
            current_time=current_time.replace(microsecond=0)
            sql = "SELECT * FROM identity_status where floor="+str(self.floor)+" and door="+str(self.door)+" and direction='out' and CONVERT(VARCHAR(25),time,121) LIKE '%"+str(current_time)+"%'"
            mycursor = mydb.cursor()
            mycursor.execute(sql)
            myresult = mycursor.fetchall()  
            seconds=seconds+1
            mycursor.close()
            mydb.commit()
            if(len(myresult)!=0):
                break
      
        if len(myresult)!=count:
            print("Video - ",self.name)
        print("[INFO] got "+str(len(myresult))+" records when expected was "+str(count))
        if len(myresult)!=count:
            fps = cap.get(cv.CAP_PROP_FPS)
            current_frame_number =(cap.get(cv.CAP_PROP_POS_FRAMES))
            videotime = current_frame_number / fps
            print("[INFO] tailgating happened at  ",videotime)
            timestamp = time_now.strftime("%Y%m%d_%H-%M-%S")
            file_name="C:\\Work\\P\\Project\\counting-people\\out"+str(timestamp)+".jpg"
            if not cv.imwrite(file_name, current_frame):
                raise Exception("Could not write image")
            duration = 1000  # milliseconds
            freq = 440  # Hz
            winsound.Beep(freq, duration)
            time_diff=time_now - datetime.timedelta(seconds=5)
            sql="Select distinct a.* from (Select b.* From (SELECT top 1 * FROM identity_status where floor="+str(self.floor)+" and door="+str(self.door)+"  and time < Convert(datetime2,'"+ str(time_now)+"') Order by time desc) b UNION select * from identity_status where floor="+str(self.floor)+" and door="+str(self.door)+" and direction='out' and time between Convert(datetime2,'"+str(time_now)+"') and Convert(datetime2,'"+str(time_diff)+"'))a "
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
                Please check the following details of person who entered exited before tailgating happened : <br>
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
                Refer the snapshot at the time of tailgating...
                For more assistance please contact our support team:
                <a href='mailto:counter_people@gmail.com'>counter_people@gmail.com</a>.<br> 
                Thank you!
            </body>
            </html>
            """
        
            HTML=start_part+table_body+later_part
            thread = threading.Thread(target=self.send_Email(HTML,file_name))
            thread.start()
        print("------------------------------------------------------------------------------------------------------------------------------------------------------------------")
        
    def send_Email(self,message,file_name):
        
            subject_template_name = 'Security Alert '
            img_data = open(file_name, 'rb').read()
            fromaddr = "softcornercummins@gmail.com"
            toaddr = "sbsanghrajka19@gmail.com"
            mail = MIMEMultipart( )
            mail['From'] = fromaddr
            mail['To'] = toaddr
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

def main():
    thread1 = myThread(1,'Video1.mp4',1)
    thread2 = myThread(2,'Video2.mp4',2)
    thread1.start()
    thread2.start()

print("Exiting Main Thread")

if __name__ == '__main__':
    main()