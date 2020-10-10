######################## IMPORTING MODULES ########################
from gtts import gTTS
import os
import speech_recognition as sr
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import time
global model
MODEL_PATH = './model/model.h5'
model = load_model(MODEL_PATH)


######################## VOICE COMMANDS ########################

# greetings = ['hey vision cap','hi vision cap','high vision cap','highvision cap','hello vision cap','hey bhajan cap','hi bhajan cap','high bhajan cap','highbhajan cap','hello bhajan cap','hey vision cab','hi vision cab','high vision cab','highvision cab','hello vision cab','hey bhajan cab','hi bhajan cab','high bhajan cab','highbhajan cab','hello bhajan cab']
currency_voice_commands = ['recognise the currency', 'identify the currency','recognise the note','identify the note','recognise currency', 'identify currency','recognise note','identify note']
object_voice_commands = ['list all objects in front of me', 'list the objects in front of me','list all the objects in front of me','identify the objects in front of me','identify all the objects in front of me']
store_person_commands = ['store face of person in front of me', 'store face of the person in front of me','store the face of person in front of me','store a new face'] 
recognise_person_commands = ['who is standing in front of me', 'who is there in front of me','recognise the person','identify the person']

######################## USEFUL FUNCTIONS ########################

def captureFrames(n):
    # Here we are capturing 30 frames and saving every alternate frame 
    to_be_captured = 2*n
    camera = cv2.VideoCapture(0)
    k = 0
    for i in range(to_be_captured):
        return_value, image = camera.read()
        if i%2 == 0:
            cv2.imwrite(str(k)+'.png', image)
            k += 1
    del(camera)

def sayMyText(mytext):
    # Passing the text and language to the engine, here we have marked slow=False, which tells the module that the converted audio should have a high speed 
    myobj = gTTS(text=mytext, lang='en', slow=False)
    
    # Saving the converted audio in a mp3 file named
    myobj.save("result.mp3")
    
    # Playing the converted file 
    os.system("mpg321 result.mp3")


######################## OBJECT DETECTION ########################

yoloClasses="model/yolov3.txt"
yoloConfig="model/yolov3.cfg"
yoloWeights="model/yolov3.weights"
classes = None
with open(yoloClasses, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def getOutputLayers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
#     label = str(classes[class_id])
#     color = COLORS[class_id]
#     cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
#     cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def predictObjects(filename):
    image = cv2.imread(filename)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    net = cv2.dnn.readNet(yoloWeights, yoloConfig)
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputLayers(net))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    all_objects=set()
    harmful_objects=set()
    for class_id in class_ids:
        className=str(classes[class_id])
        if className in ['fork','knife','scissors']:
            harmful_objects.add(className)
        else:
            all_objects.add(className)
    return [all_objects,harmful_objects]

def objectDetection():
	sayMyText('Capturing the image')
	captureFrames(1)
	sayMyText('Processing it')
	mylist = predictObjects('0.png')
	print(mylist)
	if len(mylist[1]) == 0:
		sayMyText('There are no harmful objects')
	else:
		sayMyText('Harmful objects are ')
		for i in mylist[1]:
			sayMyText(i)
	if len(mylist[0]) == 0:
		sayMyText('There are no non harmful objects')
	else:
		sayMyText('Non-harmful objects are ')
		for i in mylist[0]:
			sayMyText(i)

######################## CURRENCY RECOGNITION ########################

def getLabel(file_path):
    processed_image = processImage(file_path)
    classes = model.predict(processed_image)
#    print(classes)
    predicted_labels = classes.tolist()[0]
    denomination=[200, 2000, 500]
    percentage_labels=[0.0 ,0.0 ,0.0]
    sum_total=sum(predicted_labels)
    for i in range(0,len(predicted_labels)):
        percentage_labels[i]=round((predicted_labels[i]/sum_total)*100,2)
    result_dict = dict(zip(denomination,percentage_labels))
    # print(result_dict)
    denomination.clear()
    percentage_labels.clear()
    ans = -1
    for i in sorted(result_dict):
        denomination.append(str(i))
        percentage_labels.append(result_dict[i])
        print((i, result_dict[i]), end =" ")
        if(result_dict[i] >= 50):
            ans = i
    print()
    return ans

def processImage(file_path):
    # img = load_img(file_path, target_size=(256, 256, 3))
    img = load_img(file_path, target_size=(128, 96, 3))
    img = img_to_array(img)
    # img = img.reshape(1,256,256,3).astype('float')
    img = img.reshape(1,128,96,3).astype('float')
    img /= 255
    return img

def recogniseDenomination():
    _200 = 0
    _500 = 0
    _2000 = 0
    sayMyText('Please place the currency note in front of the camera')
    sayMyText('Capturing the image')
    captureFrames(15)
    sayMyText('Processing it')
    for i in range(15):
        ans = getLabel(f'./{i}.png')
        if ans == 200:
            _200+=1
        elif ans == 500:
            _500+=1
        elif ans == 2000:
            _2000+=1
    if(_200 > _500 and _200 > _2000):
        sayMyText('The currency note is 200')
    elif(_2000 > _500 and _2000 > _200):
        sayMyText('The currency note is 2000')
    elif(_500 > _200 and _500 > _2000):
        sayMyText('The currency note is 500')
    else:
        sayMyText('No result')

######################## FACE DETECTION AND RECOGNITION ########################

# Importing Haarcascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Calculated Euclidean Distance between two 1-D vectors 
def distance(v1, v2):
    return np.sqrt(((v1-v2)**2).sum())

# Performing KNN on test image
def knn(train, test, k=5):
    dist = []
    
    for i in range(train.shape[0]):
        # Get the Vector and Label
        ix = train[i, :-1]
        iy = train[i, -1]
        # Compute the Distance from Test Point
        d = distance(test, ix)
        dist.append([d, iy])
    
    # Sort based on Distance and get Top K
    dk = sorted(dist, key = lambda x: x[0])[:k]
    # Retrieve only the Labels
    labels = np.array(dk)[:, -1]
    
    # Get Frequencies of Each Label
    output = np.unique(labels, return_counts = True)
    # Find Max Frequency and Corresponding Label
    index = np.argmax(output[1])
    return output[0][index]

# Path for storing all numpy vectors of facial features    
dataset_path = './faceData/'

# This function asks name for the person while storing a new face
def askForName():
	sayMyText('Say the name of the new person:')
	r = sr.Recognizer()
	while True:
	    with sr.Microphone() as source:
	        print('Ready for storing name...')
	        r.pause_threshold = 1
	        r.adjust_for_ambient_noise(source, duration=0.5)
	        audio = r.listen(source)

	    try:
		    command = r.recognize_google(audio).lower()
		    sayMyText('Capturing image for '+command)
		    return command
	    except:
	    	sayMyText('Sorry, your voice was unclear. Please repeat the name')

# Stroing a new face
def storeANewFace():
    skip = 0
    face_data = []
    
    #Init Camera
    cap = cv2.VideoCapture(0)
    file_name = askForName()
    start = time.time()
    no_face = 0
    while True:
        if time.time() - start > 10:
            no_face = 1
            break
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame, 1.1, 4)
        faces = sorted(faces, key = lambda f: f[2]*f[3])
    
        # Pick the Last Face (because it is the largest face acc. to Area(f[2]*f[3]))     
        for (x, y, w, h) in faces[-1:]:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extract (Crop out the Required Face) : Region Of Interest
            offset = 10
            face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            face_section = cv2.resize(face_section, (100, 100))
        
            skip += 1
            if skip%3==0:
                face_data.append(face_section)
                print(len(face_data))
            
            #cv2.imshow('img', frame)
            #cv2.imshow('Face Section', face_section)   
        
        exitKey = cv2.waitKey(30) & 0xFF
        if skip>42:
            break
#        if exitKey==27:
#            break
    if no_face == 1:
    	sayMyText("No face detected")
    else:
	    sayMyText("Face Data Collected Successfully")
	    # Convert our Face List Array into a Numpy Array
	    face_data = np.asarray(face_data)
	    face_data = face_data.reshape((face_data.shape[0], -1))
	    print(face_data.shape)

	    # Save this Data into File System
	    np.save(dataset_path + file_name + '.npy', face_data)
	    print("Data Successfully Saved at " + dataset_path + file_name + '.npy')
    
    cap.release()
    cv2.destroyAllWindows()

names = {}

def prepareData():
    data = []
    labels = []
    class_id = 0

    for fx in os.listdir(dataset_path):
        if fx.endswith('.npy'):
            # Create a Mapping btw class_id and Name
            names[class_id] = fx[:-4]
#            print("Loaded "+ fx)
            data_item = np.load(dataset_path + fx)
            data.append(data_item)
        
            # Create Labels for the Class
            target = class_id*np.ones((data_item.shape[0]))
            class_id += 1
            labels.append(target)
        
            face_dataset = np.concatenate(data, axis = 0)
            face_labels = np.concatenate(labels, axis = 0).reshape((-1, 1))

#        print(face_dataset.shape)
#        print(face_labels.shape)

        trainset = np.concatenate((face_dataset, face_labels), axis = 1)
#        print(trainset.shape)
        
        return trainset

def captureFace():
    cap = cv2.VideoCapture(0)
    start = time.time()
    no_face = 0
    while True:
        if time.time() - start > 4:
            no_face = 1
            break
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        faces = face_cascade.detectMultiScale(frame, 1.1, 4)
        if len(faces)>0:
            recognizeFace(frame)
            break
    if no_face == 1:
    	sayMyText("No face detected")
    cap.release()
    cv2.destroyAllWindows()
    
def recognizeFace(frame):
    set = prepareData()
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    
    for (x, y, w, h) in faces:
        
    # Get the Face ROI
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))
        
    # Predicted Label (out)
        out = knn(set, face_section.flatten())
        
    # Display on the Screen the Name and Rectangle around it
        pred_name = names[int(out)]
#    font = cv2.FONT_HERSHEY_SIMPLEX
#    cv2.putText(frame, pred_name, (x, y-10), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
#    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        sayMyText("The Person in Front is : " + pred_name)
#    cv2.imshow("Faces", frame)

######################## DRIVER FUNCTION ########################

def myCommand():
    r = sr.Recognizer()
    
    initialCommand = 'Please speak after the beep'
    sayMyText(initialCommand)
    os.system("mpg321 beep.mp3")
    print(initialCommand)
    
    with sr.Microphone() as source:
        print('Ready Sub Menu...')
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source)

    try:
        command = r.recognize_google(audio).lower()
        if command in currency_voice_commands:
    	    recogniseDenomination()
        elif command in object_voice_commands:
            objectDetection()
        elif command in store_person_commands:
        	storeANewFace()
        elif command in recognise_person_commands:
        	captureFace()
        else:
            print('Not able to understand',command)
            command = 'Sorry! Your last command was invalid'
            print(command)
            sayMyText(command)

    # Generate Error if Command is not heard
    except sr.UnknownValueError:
        command = 'Sorry! Your last command couldn\'t be heard'
        print(command)
        sayMyText(command)

    return command

#while True:
# os.chdir('/home/kunal/capstone-project')
while 1:
    # # listens for commands
    # r = sr.Recognizer()
    # # os.system("mpg321 beep.mp3")
    # with sr.Microphone() as source:
    #     print('Ready Main Menu...')
    #     r.pause_threshold = 1
    #     r.adjust_for_ambient_noise(source, duration=0.5)
    #     audio = r.listen(source)

    # try:
	   #  command = r.recognize_google(audio).lower()
	   #  if command in greetings:
		  #   myCommand()
	   #  else:
		  #   print(command)
    # except:
    # 	pass
    myCommand()
