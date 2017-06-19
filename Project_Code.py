"""
Name: Stephen Curran
Student number: L00107244
E-mail: L00107244@student.lyit.ie
Date: 31-01-2017
"""
from __future__ import print_function

"""------------APIS---------------"""

from errno import errorcode
from io import BytesIO
import tkinter as tk
import PIL
import requests
import numpy as np
import speech_recognition as sr
import mysql.connector
import shutil
from os import path
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from PIL import ImageTk
from google.cloud import vision

"""" -----------------Methods-------------"""
"""variables used withing the code"""
ima = None
global img
global Results
"First connect to server through the following code"
try:
    cnx = mysql.connector.connect(user='root',
                                  database='4th_year_project')  # sets the username and password to access the database
    cursor = cnx.cursor()
    if cnx.is_connected():
        print("connected")  # prints connected if there was a successful connection to the database
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("info", "Wrong username or password entered")  # prints wrong password error
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("info", "Database does not exist")  # prints database doesnt not exist error
    else:
        print(err)
"Function to open a image"


def open_a_image():
    global tk_im
    length = 10  # sets the length for the for loop
    filename = askopenfilename(filetypes=(("image Files", "*.jpg"), ("All files", "*.*")))  # produces file browser
    img = PIL.Image.open(filename)  # opens that file name
    size = 250, 250  # used to set the size of the image shown
    img.thumbnail(size)  # method to set the size of the image using the variable size
    tk_im = ImageTk.PhotoImage(img)  # sets tk_im as image
    label1.configure(image=tk_im)  # places chosen image into label
    for i in range(length):
        i = 'C:\\Users\\User1\\scikit_learn_data\\lfw_home\\lfw_funneled\\test_images'  # file path for test images
        shutil.copy2(filename, i)  # saves image to file test images


"""Function to run speech recognition on a file"""


def RunSpeechRec0nFile():
    global audio  # variable
    fileName = askopenfilename(
        filetypes=(("Audio Files", "*.wav"), ("All files", "*.*")))  # allows for searching threw files for recordings
    audio = path.join(fileName)  # sets the audio variable variable as the file name
    rec = sr.Recognizer()  # sets rec as the speech recogniser
    with sr.AudioFile(audio) as source:
        audio = rec.record(source)  # find the source of the audio
    try:
        Prediction = rec.recognize_google(audio)  # method used to analyse what is being said in the recording
        Speach_Rec_Results = tk.Toplevel()  # This creates a window, but it won't show up
        Speach_Rec_Results.geometry('500x200')  # sets the size of the window
        Speach_Rec_Results.resizable(0, 0)  # restrict size of the gui to specified size
        Speach_Rec_Results.title('4th_year_project')  # set the title of the project
        Speach_Rec_Results.configure(bg='white')  # set the background color of the gui
        Label4 = tk.Label(Speach_Rec_Results, text=("Predicted what was said in speach: " + "\n" + "%s" % Prediction),
                          bg='white',
                          font=('Helvetica', 16, 'bold'))  # sets up the label to produce speech recognition results
        Label4.place(x=200, y=100, width=250, height=300)  # sets size and position of label
        Label4.pack()  # puts the label on the GUI
        OK = tk.Button(Speach_Rec_Results, text="OK", bg='blue', command= Speach_Rec_Results.withdraw, font=('Helvetica', 12, 'bold'),
                       fg='white')  # button to exit the application
        OK.pack(padx=10)  # displays the exit button
        OK.place(x=120, y=150, width=250)  # position and set size of exit button
        #Speach_Rec_Results.mainloop()  # displays the main window for the project
    except sr.UnknownValueError:
        messagebox.showinfo("info",
                            "Could not make out speech in recording")  # error produce if audio could not be used
    except sr.RequestError as e:
        messagebox.showinfo("info", "There is no internet connected")  # error if internet is not connected


"Exit Main Gui Function"


def Exit():
    root.destroy()  # used to exit the application


"Function to run the live recording and recognise what was being said"


def RunLiveRecording():
    reco = sr.Recognizer()  # brings in the speech recogniser from pythons liberary
    with sr.Microphone() as source:
        print("Speak into the microphone!!")
        live_recording = reco.listen(source)  # allows user to record their voice
        messagebox.showinfo("info", "Recording Complete")  # message box produced when the recording is complete
    with open("live_recoding.wav", "wb") as write_audio:
        write_audio.write(live_recording.get_wav_data())
    live_recording_results = path.join(path.dirname(path.realpath(__file__)),
                                       "live_recoding.wav")  # saves the recording to the program files
    with sr.AudioFile(live_recording_results) as source:
        live_recording_results = reco.record(source)  # opens the recording
    try:
        speech_pridiction = reco.recognize_google(
            live_recording_results)  # method used to analyse what is being said in the recording
        Live_Speach_Rec_Results = tk.Toplevel()  # This creates a window, but it won't show up
        Live_Speach_Rec_Results.geometry('500x200')  # sets the size of the window
        Live_Speach_Rec_Results.resizable(0, 0)  # restrict size of the gui to specified size
        Live_Speach_Rec_Results.title('4th_year_project')  # set the title of the project
        Live_Speach_Rec_Results.configure(bg='white')  # set the background color of the gui
        Label4 = tk.Label(Live_Speach_Rec_Results,
                          text=("Predicted what was said in speach: " + "\n" + "%s" % speech_pridiction),
                          bg='white',
                          font=('Helvetica', 16, 'bold'))  # sets up the label to produce speech recongiotn results
        Label4.place(x=200, y=100, width=250, height=300)  # sets size and postion of label
        Label4.pack()
        OK = tk.Button(Live_Speach_Rec_Results, text="OK", bg='blue', command= Live_Speach_Rec_Results.withdraw, font=('Helvetica', 12, 'bold'),
                       fg='white')  # button to exit the application
        OK.pack(padx=10)  # dispays the exit button
        OK.place(x=120, y=150, width=250)  # position and set size of exit button
    except sr.UnknownValueError:
        messagebox.showinfo("info", "Could not understand speech in audio")  # error produce if audio could not be used
    except sr.RequestError as e:
        messagebox.showinfo("info", "There is no internet connected")  # error if internet is not connected


"""Method used by facial recognition to get results"""


def find_nearest(array, value):
    idx = (np.abs(array <= value)).argmin()
    return array[idx]


"""Facial recognition function"""


def RunFaceRec():
    """Variable used for speech recongition"""
    global database_image
    global tk_im2
    global img
    """crate and array of the images in the dataset and sets them to a minimum faces of two to be called in and resizes it to 0.4"""
    lfw_people = fetch_lfw_people(min_faces_per_person=3, resize=0.4)  # stores file images into an array
    n_samples, h, j = lfw_people.images.shape  # finds the shapes for plotting the images
    X = lfw_people.data
    y = lfw_people.target
    """used for predicting the names of the people in the file dataset"""
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    n_components = 1  # set the amount of faces for face recogniton

    pca_method = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(
        X_train)  # PCA method to prepare the images for facial recognition
    pca_method.components_.reshape((n_components, h, j))
    X_train_pca = pca_method.transform(X_train)
    X_test_pca = pca_method.transform(X_test)
    """Folowing code is used to produce the results of facial recognition"""
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    F1_score = precision_recall_fscore_support(y_test, y_pred, average=None,
                                               labels=range(n_classes))  # produces the F1-score, recall and percesion
    array1 = F1_score[2]
    # Sets array to just f1-score
    array2 = array1[10]  # Sets array 2 variable to the f1-score of the test images
    rounded_score = np.around(array2, decimals=2)
    print("F1_score = ", rounded_score)
    cnx = mysql.connector.connect(user='root', database='4th_year_project')
    cursor = cnx.cursor()
    if cnx.is_connected():
        cursor.execute("select f1_score from person")
        database_f1_scores = cursor.fetchall()
        results = find_nearest(database_f1_scores, rounded_score)
        cursor.execute("select front_facing_image from person where f1_score =%s" % results)
        for row in cursor:
            database_image = row[0]
        cursor.execute("SELECT First_name FROM person where f1_score =%s" % results)
        for row2 in cursor:
            name = row2[0]
            # print(database_image)
            # This creates a window, but it won't show up
        Results = tk.Toplevel()
        Results.geometry('500x500')  # sets the size of the window
        Results.resizable(0, 0)  # restrict size of the gui to specified size
        Results.title('4th_year_project')  # set the title of the project
        Results.configure(bg='white')  # set the background color of the gui
        Label3 = tk.Label(Results, text=" ", bg='white')
        img = PIL.Image.open(BytesIO(database_image))  # opens that file name
        size = 850, 250
        img.thumbnail(size)
        tk_im2 = ImageTk.PhotoImage(img)
        Label3.configure(image=tk_im2)
        # sets tk_im as image
        Label3.pack()
        Label3.place(x=120, y=100, width=250, height=300)
        Label4 = tk.Label(Results, text=("Predicted %s" % name + " With a score of %s" % rounded_score), bg='white',
                          font=('Helvetica', 16, 'bold'))
        Label4.pack()
        OK = tk.Button(Results, text="OK", bg='blue', command=Results.withdraw, font=('Helvetica', 12, 'bold'),
                       fg='white')  # button to exit the application
        OK.pack(padx=10)  # dispays the exit button

        OK.place(x=120, y=450, width=250)  # position and set size of exit button
    cnx.close()  # closes the connection to the database


"Download an image GUI function"


class download_images(tk.Tk):
    def __init__(enter):
        tk.Tk.__init__(enter)
        enter.geometry('600x200')  # sets the size of the GUI
        enter.configure(bg='white')  # sets background color of the gui
        enter.Label2 = tk.Label(enter, text=("Enter URL into the grey box below"), bg='white',
                                font=('Helvetica', 16, 'bold')).pack()  # adds the top label
        enter.entryBox = tk.Entry(enter, width=160, bg='grey')  # sets up the text entry box
        enter.entryBox.pack(side='left')  # sets text entry to a side
        enter.Send_to_download = tk.Button(enter, text="Download!!", command=enter.download_image, bg='blue',
                                           font=('Helvetica', 12, 'bold'),
                                           fg='white')  # sets up the butoon
        enter.Send_to_download.pack()
        enter.Send_to_download.place(x=160, y=150, width=250)


    "Places the GUI into the patchs"

    def download_image(dl):
        o = 'C:\\Users\\User1\\Desktop\\4th year\\Semister 2\\Development project\\4th_year_project_code\\Download_test_image.jpg'
        i = 'C:\\Users\\User1\\scikit_learn_data\\lfw_home\\lfw_funneled\\test_images'
        url = dl.entryBox.get()  # gets the url given
        response = requests.get(url, stream=True)
        with open('Download_test_image.jpg', 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)  # saves URL to the program file
            shutil.copy2(o, i)  # copies the URL to the facial recognition dataset
        del response
        messagebox.showinfo("info", "Download Complete")#appears when image download is complete
        """THIS IS AN EXAMPLE OF THE CODE NEEDED TO RUN FACIAL RECOGNITION ON A IMAGE AND SEARCH THE WEB FOR A MATCH
           FROM GOOGLES CLOUD VISION API"""

        """vision_client = vision.Client()
        with io.open('Download_test_image.jpg', 'rb') as image_file:
             content = image_file.read()
        image = vision_client.image(content=content)
        notes = image.detect_web()
        if notes.web_entities:
         print ('\n{} Web entities found: '.format(len(notes.web_entities)))
        for entity in notes.web_entities:
            print('Score      : {}'.format(entity.score))
            print('Description: {}'.format(entity.description)) """


# --------------------------Main Window--------------#
root = tk.Tk()  # This creates a window, but it won't show up
root.geometry('700x500')  # sets the size of the window
root.resizable(0, 0)  # restrict size of the gui to specified size
root.title('4th_year_project')  # set the title of the project
root.configure(bg='white')  # set the background color of the gui
# ------------------------Buttons-------------------#
importImage = tk.Button(root, text="Import Image", command=open_a_image, bg='blue', font=('Helvetica', 12, 'bold'),
                        fg='white')  # Button to import a image
label1 = tk.Label(root, text="", bg='white')
label1.pack()
label1.place(x=40, y=30, width=250, height=300)
DownloadImage = tk.Button(root, text="Download a image", command=download_images, bg='blue',
                          font=('Helvetica', 12, 'bold'),
                          fg='white')  # Button to download a image
RunfaceRec = tk.Button(root, text='Run Facial Recognition', command=RunFaceRec, bg='blue',
                       font=('Helvetica', 12, 'bold'), fg='white')  # button to run face recognition
ImportAudio = tk.Button(root, text='Import file for speech recogniton', command=RunSpeechRec0nFile, bg='blue',
                        font=('Helvetica', 12, 'bold'),
                        fg='white')  # Button to import audio files
label2 = tk.Label(root, text="", bg='white')
label2.pack()
label2.place(x=410, y=300, width=250, height=30)
RunSpeechRec = tk.Button(root, text='Record yourself', command=RunLiveRecording, bg='blue',
                         font=('Helvetica', 12, 'bold'), fg='white')  # Button to run speech recognition
ExitButton = tk.Button(root, text="Exit", command=Exit, bg='blue', font=('Helvetica', 12, 'bold'),
                       fg='white')  # button to exit the application
# ------------------------Produce Buttons to Screen and set the size--------------#
importImage.pack(padx=10)  # displays the import image button
DownloadImage.pack(padx=10)
DownloadImage.place(x=40, y=362, width=250)
importImage.place(x=40, y=325, width=250)  # position and set size of import image button
RunfaceRec.pack(padx=10)  # displays the run facial recognition button
RunfaceRec.place(x=40, y=400, width=250)  # position and set size of import run face recognition button
ImportAudio.pack(padx=10)  # displays the import audio button
ImportAudio.place(x=410, y=350, width=250)  # position and set size of import audio button
RunSpeechRec.pack(padx=10)  # displays the run speech recognition button
RunSpeechRec.place(x=410, y=400, width=250)  # position and set size of run speech recognition button
ExitButton.pack(padx=10)  # dispays the exit button
ExitButton.place(x=225, y=450, width=250)  # position and set size of exit button
root.mainloop()  # displays the main window for the project
