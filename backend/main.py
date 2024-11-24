from fastapi import FastAPI, HTTPException,File, UploadFile, Form, BackgroundTasks,WebSocket,WebSocketDisconnect,Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import cv2 
import os
from typing import List
import shutil
from fastapi.responses import JSONResponse
import face_recognition
import pandas as pd
import numpy as np
from ultralytics import YOLO
import torch
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
import base64
import asyncio

#import torch
#print(torch.cuda.get_device_name())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
# MongoDB connection
client = AsyncIOMotorClient("PUT MONGODB URI HERE !!!")  
db = client['AIML']  #database name
collection = db['users']  #collection name
currDirectory = os.path.dirname(os.path.abspath(__file__))
usersFolder = os.path.join(currDirectory,"../","database","users")


@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    model = YOLO("yolo11n.pt")
    userID = await websocket.receive_text() 
    userFolder = os.path.join(usersFolder,userID)
    label_encoder_path = os.path.join(userFolder, "label_encoder.pkl")
    model_path = os.path.join(userFolder, "xgboost_model.pkl")
    with open(label_encoder_path, "rb") as file:
        label_encoder = pickle.load(file)
    with open(model_path, "rb") as file:
        xgboost_model = pickle.load(file)
    url = "CC CAMERA IP ADRESS HERE!!!/video"
    cap = cv2.VideoCapture(url)  # or use a video file path instead of 0
    try:
        frame_index = 0
        print("Starting real-time video processing.")
        myDict = {}
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            print(f"Processing frame {frame_index}.")
            results = model.track(frame, persist=True)
            trackingIDsInImage = []
            boundingBoxes = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    classID = int(box.cls)
                    label = model.names[classID]

                    if label == 'person':
                        # Ensure box.id is valid
                        track_id = int(box.id) if (hasattr(box, 'id') and box.id is not None) else None
                        if track_id is not None:
                            trackingIDsInImage.append(track_id)
                        boundingBoxes.append((x1, y1, x2, y2))
                        if track_id not in myDict:
                            myDict[track_id] = ["Unknown", 0]

                        # Display label and confidence on the frame
                        cv2.putText(frame, str(myDict[track_id][0]) + " Confidence: " + str(myDict[track_id][1]), 
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
            if frame_index % 10 == 0:
                for i, personID in enumerate(trackingIDsInImage):
                    x1, y1, x2, y2 = boundingBoxes[i]
                    faceRegion = frame[y1:y2, x1:x2]
                    faceRegionRGB = cv2.cvtColor(faceRegion, cv2.COLOR_BGR2RGB)
                    faceEncodings = face_recognition.face_encodings(faceRegionRGB)

                    if faceEncodings:
                        print("No of people: ", len(faceEncodings))
                        faceEncoding = faceEncodings[0].reshape(1, -1)
                        predictedLabel = xgboost_model.predict(faceEncoding)[0]
                        confidence_scores = xgboost_model.predict_proba(faceEncoding)
                        confidence = max(confidence_scores[0])
                        predictedRollNo = label_encoder.inverse_transform([predictedLabel])[0]
                        print("Predicted: ", predictedRollNo)

                        if confidence > myDict.get(personID, [None, 0])[1]:
                            myDict[personID] = [predictedRollNo, confidence]
            

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame_index += 1
            
            
            
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(frame_data)
            await asyncio.sleep(0.05)  # adjust delay for frame rate control
    except Exception as e:
        print("Connection closed:", e)
    finally:
        cap.release()
        await websocket.close()
# def runRealTime(userID):
#     try:
#         print("Started!")
#         userFolder = os.path.join(usersFolder, userID)
#         model = YOLO("yolo11n.pt")
#         label_encoder_path = os.path.join(userFolder, "label_encoder.pkl")
#         model_path = os.path.join(userFolder, "xgboost_model.pkl")
#         average_encodings_path = os.path.join(userFolder, "encodings_with_average.pkl")

#         # Load label encoder, XGBoost model, and face encodings
#         with open(label_encoder_path, "rb") as file:
#             label_encoder = pickle.load(file)
#         with open(model_path, "rb") as file:
#             xgboost_model = pickle.load(file)
#         with open(average_encodings_path, "rb") as file:
#             encodings = pickle.load(file)

#         # Open video stream
#         #url = "http://172.20.254.139:1234/video"
#         url = "http://192.168.137.251:1234/video"
#         cap = cv2.VideoCapture(url)
#         if not cap.isOpened():
#             print("Error: Unable to open video stream")
#             return {"error": "Unable to open video stream"}

#         detectedTrackingIDs = []
#         frame_index = 0
#         print("Starting real-time video processing.")
#         myDict = {}

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to read frame or end of video stream.")
#                 break

#             print(f"Processing frame {frame_index}.")
#             results = model.track(frame, persist=True)
#             trackingIDsInImage = []
#             boundingBoxes = []

#             # Process each detection result
#             for result in results:
#                 boxes = result.boxes
#                 for box in boxes:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     classID = int(box.cls)
#                     label = model.names[classID]

#                     if label == 'person':
#                         # Ensure box.id is valid
#                         track_id = int(box.id) if (hasattr(box, 'id') and box.id is not None) else None
#                         if track_id is not None:
#                             trackingIDsInImage.append(track_id)
#                         boundingBoxes.append((x1, y1, x2, y2))
#                         if track_id not in myDict:
#                             myDict[track_id] = ["Unknown", 0]

#                         # Display label and confidence on the frame
#                         cv2.putText(frame, str(myDict[track_id][0]) + " Confidence: " + str(myDict[track_id][1]), 
#                                     (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

#             # Run face recognition every 10 frames
#             if frame_index % 10 == 0:
#                 for i, personID in enumerate(trackingIDsInImage):
#                     x1, y1, x2, y2 = boundingBoxes[i]
#                     faceRegion = frame[y1:y2, x1:x2]
#                     faceRegionRGB = cv2.cvtColor(faceRegion, cv2.COLOR_BGR2RGB)
#                     faceEncodings = face_recognition.face_encodings(faceRegionRGB)

#                     if faceEncodings:
#                         print("No of people: ", len(faceEncodings))
#                         faceEncoding = faceEncodings[0].reshape(1, -1)
#                         predictedLabel = xgboost_model.predict(faceEncoding)[0]
#                         confidence_scores = xgboost_model.predict_proba(faceEncoding)
#                         confidence = max(confidence_scores[0])
#                         predictedRollNo = label_encoder.inverse_transform([predictedLabel])[0]
#                         print("Predicted: ", predictedRollNo)

#                         if confidence > myDict.get(personID, [None, 0])[1]:
#                             myDict[personID] = [predictedRollNo, confidence]

#                         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             frame_index += 1
#             cv2.imshow("Frame", frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#         print("Real-time video processing completed.")
#     except Exception as e:
#         print(f"Error in real-time tracking: {e}")
#         return {"status": "error", "message": str(e)}


    
# @app.post("/realTimeTrack")
# async def run_python_script(background_tasks: BackgroundTasks, userId: str = Form(...), username: str = Form(...), userEmail: str = Form(...)):
#     try:
#         print("called")
#         # Run the function in the background
#         background_tasks.add_task(runRealTime, userId)
#         return {"status": "success", "message": "Real-time tracking started successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
async def get_user(email: str):
    user_data = await collection.find_one({"email": email})
    if user_data:
        user_data['id'] = str(user_data['_id'])  # Convert ObjectId to string
        return user_data
@app.post("/register")
async def register(user: dict):
    existing_user = await collection.find_one({"email": user['email']})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")


    user_data = {
        "username": user["username"],
        "email": user["email"],
        "password": user["password"]  #NO ENCRYPTION CURRENTLY BEWARE!!!
    }
    
    result = await collection.insert_one(user_data)
    user_data["id"] = str(result.inserted_id) 
    os.makedirs(os.path.join(usersFolder,user_data['id']), exist_ok=True)
    os.makedirs(os.path.join(usersFolder,user_data['id'],"images"), exist_ok=True)

    return {"id": user_data["id"], "username": user_data["username"], "email": user_data["email"]}

@app.post("/login")
async def login(user: dict):
    db_user = await get_user(user['email'])
    if not db_user or db_user['password'] != user['password']:  # Check plain text password
        raise HTTPException(status_code=400, detail="Invalid credentials")
    os.makedirs(os.path.join(usersFolder,db_user['id']), exist_ok=True)
    os.makedirs(os.path.join(usersFolder,db_user['id'],"images"), exist_ok=True)
    
    return {"id": db_user["id"], "username": db_user["username"], "email": db_user["email"]}

# New route for adding samples
@app.post("/addsamples")
async def add_samples(
    userId: str = Form(...),
    username: str = Form(...),
    userEmail: str = Form(...),
    names: List[str] = Form(...),
    images: List[UploadFile] = File(...)
):
    user = await get_user(userEmail)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_folder = os.path.join(usersFolder, userId, "images")

    image_index = 0
    # Process each name and associated images
    for i, name in enumerate(names):
        person_folder = os.path.join(user_folder, name)
        os.makedirs(person_folder, exist_ok=True)
        
        # Save each image associated with this person
        while image_index < len(images):
            image = images[image_index]
            image_path = os.path.join(person_folder, image.filename)
            with open(image_path, "wb") as file:
                file.write(await image.read())
            image_index += 1  # Move to the next image

    return {"status": "Samples added successfully"}

@app.post("/processfootage")
async def process_footage(
    video: UploadFile = File(...),
    userId: str = Form(...),
    username: str = Form(...),
    userEmail: str = Form(...),
):
    try:
        userFolder = os.path.join(usersFolder, userId)        
        average_encodings_path = os.path.join(userFolder, "encodings_with_average.pkl")  
        
        with open(average_encodings_path, "rb") as file:
            encodings = pickle.load(file)  

        video_path = os.path.join(userFolder, "video", video.filename)
        os.makedirs(os.path.join(userFolder, "video"), exist_ok=True)
        with open(video_path, "wb") as video_file:
            shutil.copyfileobj(video.file, video_file)
        print(f"Video saved to: {video_path}")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        outputDirectory = os.path.join(userFolder, "output")
        os.makedirs(outputDirectory, exist_ok=True)
        print(f"Output directory created: {outputDirectory}")


        model = YOLO("yolo11x.pt")
        model.to(device)
        print("YOLO model loaded and moved to device")

        detectedTrackingIDs = []  
        imagePaths = []
        allBoundingBoxes = []
        frame_index = 0

        cap = cv2.VideoCapture(video_path)
        print("Starting video processing")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break

            results = model.track(frame, persist=True)
            outputFrame = results[0].plot()
            trackingIDsInImage = []
            boundingBoxes = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    classID = int(box.cls)
                    label = model.names[classID]
                    if label == 'person':
                        print(f"Person detected with bounding box: {x1}, {y1}, {x2}, {y2}")

                        track_id = int(box.id) if hasattr(box, 'id') else None
                        if track_id is not None:
                            trackingIDsInImage.append(track_id)
                        boundingBoxes.append(f"{x1},{y1},{x2},{y2}")

            outputImgPath = os.path.join(outputDirectory, f"frame{frame_index}.png")
            cv2.imwrite(outputImgPath, outputFrame)
            print(f"Frame {frame_index} processed and saved to: {outputImgPath}")

            detectedTrackingIDs.append(trackingIDsInImage if trackingIDsInImage else [])
            imagePaths.append(outputImgPath)
            allBoundingBoxes.append("; ".join(boundingBoxes) if boundingBoxes else "None")
            frame_index += 1

        cap.release()
        print("Video processing completed")

        # Save tracking results in CSV files
        df = pd.DataFrame({
            'Bounding Boxes': allBoundingBoxes,
            'Processed Image Path': imagePaths,
            'Detected Tracking IDs': detectedTrackingIDs
        })

        model_path = os.path.join(userFolder, "xgboost_model.pkl")  # Path to the pickled XGBoost model
        with open(model_path, "rb") as file:
            xgboost_model = pickle.load(file)

        print("XGBoost model loaded successfully!")

        # Load the label encoder
        label_encoder_path = os.path.join(userFolder, "label_encoder.pkl")  # Path to the pickled label encoder
        with open(label_encoder_path, "rb") as file:
            label_encoder = pickle.load(file)

        allHumansTracked = set().union(*df['Detected Tracking IDs'].apply(set))
        encodings = encodings.set_index('Name')['Average Encoding'].to_dict()
        myDict = {x : ["uknown","unknown","unknown","unknown"] for x in allHumansTracked}
        
        def processBoundingBoxes(boundingBoxesList):
            for i, x in enumerate(boundingBoxesList):
                boundingBoxesList[i] = x.split(',')
                
        for i, row in df.iterrows():
            people = row["Detected Tracking IDs"]  # List of tracking IDs of all people present in that frame
            
            if not people:
                continue  # No people in this frame, continue to next frame

            boundingBoxes = row["Bounding Boxes"].split(";")  
            processBoundingBoxes(boundingBoxes) 

            imgPath = row["Processed Image Path"]
            img = cv2.imread(imgPath)

            for j in range(len(people)):
                currPersonID = people[j]
                boundingBox = boundingBoxes[j] 
                x1, y1, x2, y2 = boundingBox  # Directly unpack the values

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                faceRegion = img[y1:y2, x1:x2]
                faceRegionRGB = cv2.cvtColor(faceRegion, cv2.COLOR_BGR2RGB)
                faceEncodings = face_recognition.face_encodings(faceRegionRGB)

                
                if faceEncodings:
                    faceEncoding = faceEncodings[0]
                    faceEncoding = faceEncoding.reshape(1, -1)

                    # Predicting using XGBoost model
                    predictedLabel = xgboost_model.predict(faceEncoding)[0]
                    confidence_scores = xgboost_model.predict_proba(faceEncoding)  
                    confidenceArray = confidence_scores[0]  
                    predictedIndex = np.argmax(confidenceArray)
                    confidence = confidenceArray[predictedIndex]
                    
          
                    predictedRollNo = label_encoder.inverse_transform([predictedLabel])[0]

                    # Comparing with stored encoding using Euclidean distance
                    storedEncoding = encodings.get(predictedRollNo, None)
                    if storedEncoding is not None:
                        storedEncoding = storedEncoding.reshape(1, -1)
                        euclideanDistance = face_recognition.face_distance(faceEncoding, storedEncoding)

                        # Check if confidence is higher or the person is unknown
                        if myDict[currPersonID][1] == "unknown" or confidence > myDict[currPersonID][1]:
                            myDict[currPersonID][0] = predictedRollNo
                            myDict[currPersonID][1] = confidence
                            myDict[currPersonID][2] = euclideanDistance
                            myDict[currPersonID][3] = imgPath

        # Prepare the final DataFrame
        temp0 = []  # tracking id
        temp1 = []  # roll no
        temp2 = []  # confidence
        temp3 = []  # euclidean distance
        temp4 = []  # path

        for x in myDict:
            temp0.append(x)
            temp1.append(myDict[x][0])
            temp2.append(float('-inf')) if myDict[x][1] == "unknown" else temp2.append(myDict[x][1])
            temp3.append(float('inf')) if myDict[x][2] == "unknown" else temp3.append(myDict[x][2][0])
            temp4.append(myDict[x][3])
        
        returnDF = pd.DataFrame({
            "trackingID": temp0,
            "rollNo": temp1,
            "confidence": temp2,
            "euclideanDistance": temp3,
            "imagePath": temp4
        }) 
        returnDF.to_csv(os.path.join(outputDirectory,"final.csv"))
        print(returnDF)
        df_json = returnDF.applymap(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)  # Convert any np.array to list
        df_json = returnDF.to_dict(orient='records')  # Convert the DataFrame to a list of dictionaries

        return JSONResponse(content={"data": df_json})
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return {"error": str(e)}


@app.post("/removeusers")
async def remove_users(data: dict):
    user_id = data.get("userId")
    names_to_remove = data.get("names", [])
    
    # Path to the base directory where user folders are stored
    user_folder_base = os.path.join(usersFolder, user_id, "images")
    
    removed_names = []
    failed_names = []
    
    for name in names_to_remove:
        # Check if the folder for the name exists
        person_folder = os.path.join(user_folder_base, name)
        
        if os.path.exists(person_folder) and os.path.isdir(person_folder):
            # Remove the user folder
            shutil.rmtree(person_folder)
            removed_names.append(name)
        else:
            # If the folder doesn't exist, add to failed list
            failed_names.append(name)
    
    # Return a message with details about what was removed
    return {
        "status": "Users removed successfully",
        "removed": removed_names,
        "failed": failed_names
    }
    
@app.post("/makemodel")
async def make_model(userId: str = Form(...), username: str = Form(...), userEmail: str = Form(...)):

    user = await get_user(userEmail)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    userFolder = os.path.join(usersFolder, userId)
    if not os.path.exists(userFolder):
        os.makedirs(userFolder)

    userImagesFolder = os.path.join(userFolder, "images")
    if not os.path.exists(userImagesFolder):
        raise HTTPException(status_code=404, detail="No images folder found for this user.")

    imagePaths = []
    peopleList = []

    people = os.listdir(userImagesFolder) 
    for person in people:
        personPath = os.path.join(userImagesFolder, person)
        if os.path.isdir(personPath): 
            images = os.listdir(personPath)
            for image in images:
                imagePaths.append(os.path.join(personPath, image))
                peopleList.append(person)

    df = pd.DataFrame({
        "Image path": imagePaths,
        "Name": peopleList
    })

    nameOrder = []
    encodingOrder = []
    imageOrder = []


    def createEncoding(filePath):
        img = face_recognition.load_image_file(filePath)
        boundingBoxes = face_recognition.face_locations(img)
        if len(boundingBoxes) == 1:
            encoding = face_recognition.face_encodings(img)[0]
        else:
            encoding = None
        return encoding


    for i, r in df.iterrows():
        currName = r['Name']
        currPath = r['Image path']
        currEncoding = createEncoding(currPath)
        if currEncoding is not None:
            nameOrder.append(currName)
            encodingOrder.append(currEncoding)
            imageOrder.append(currPath)

    label_encoder = LabelEncoder()
    numeric_labels = label_encoder.fit_transform(nameOrder)

 
    xgb_model = XGBClassifier()
    xgb_model.fit(encodingOrder, numeric_labels)

    sgd_model = SGDClassifier()
    sgd_model.fit(encodingOrder, numeric_labels)


    model_path_xgb = os.path.join(userFolder, "xgboost_model.pkl")
    model_path_sgd = os.path.join(userFolder, "sgd_model.pkl")
    label_encoder_path = os.path.join(userFolder, "label_encoder.pkl")
    
    with open(model_path_xgb, "wb") as model_file:
        pickle.dump(xgb_model, model_file)
    with open(model_path_sgd, "wb") as sgd_file:
        pickle.dump(sgd_model, sgd_file)
    with open(label_encoder_path, "wb") as encoder_file:
        pickle.dump(label_encoder, encoder_file)

    # Save the encodings CSV
    encoding_df = pd.DataFrame({
        "Name": nameOrder,
        "Image path": imageOrder,
        "Encoding": [enc.tolist() for enc in encodingOrder]  # Convert numpy arrays to lists for JSON serialization
    })
    encoding_df.to_csv(os.path.join(userFolder, "encodings.csv"), index=False)

    # Calculate the average encoding for each person
    countDict = {}
    encodingDict = {}
    for name, encoding in zip(nameOrder, encodingOrder):
        if name not in countDict:
            countDict[name] = 1
            encodingDict[name] = encoding
        else:
            countDict[name] += 1
            encodingDict[name] = np.add(encodingDict[name], encoding)
    
    for name in encodingDict:
        encodingDict[name] = encodingDict[name] / countDict[name]


    names = list(encodingDict.keys())
    averageEncoding = list(encodingDict.values())
    samples = [countDict[name] for name in names]
    encodings_df = pd.DataFrame({
        "Name": names,
        "Samples": samples,
        "Average Encoding": averageEncoding
    })
    encodings_df.to_csv(os.path.join(userFolder, "encodings_with_average.csv"), index=False)
    encodings_df_path = os.path.join(userFolder, "encodings_with_average.pkl")
    with open(encodings_df_path, "wb") as avg_encoding_file:
        pickle.dump(encodings_df, avg_encoding_file)

    return {"status": "Model created and saved successfully!"}