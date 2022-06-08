# Load Packages
import pandas as pd
from sklearn.metrics import accuracy_score


# Avoid Warnings
import warnings


# Text Cleaning
import neattext.functions as nfx


# Load ML Pkgs
from sklearn.linear_model import LogisticRegression


# Transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# GUI imports
from tkinter import *
from tkinter import Button
from PIL import Image, ImageTk
import speech_recognition as sr
from tkinter import messagebox


# Initializing Global Variables
global output
global result


# Taking Voice Input from User
def voiceinput():
    global output
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)

        try:
            output = r.recognize_google(audio)
        except:
            output = "Sorry, I can't hear your voice. Try Again!"


# Detect Emotion
def detect_emotion():
    global result
    # Load Dataset
    df = pd.read_csv("emotion_dataset_raw.csv")

    # User handles
    df['Clean_Text'] = df['Text'].apply(nfx.remove_userhandles)

    # Stopwords
    df['Clean_Text'] = df['Clean_Text'].apply(nfx.remove_stopwords)

    # Features & Labels
    x_features = df['Clean_Text']
    y_labels = df['Emotion']

    #  Split Data
    x_train, x_test, y_train, y_test = train_test_split(x_features, y_labels, test_size=0.3, random_state=42)

    # Build Pipeline
    from sklearn.pipeline import Pipeline

    # LogisticRegression Pipeline
    pipe_lr = Pipeline(steps=[('cv', CountVectorizer()), ('lr', LogisticRegression())])
    warnings.filterwarnings('ignore')

    # Train and Fit Data
    pipe_lr.fit(x_train, y_train)

    # pipe_lr
    # Check Accuracy
    pipe_lr.score(x_test, y_test)

    # Make A Prediction
    ex1 = output
    result = (pipe_lr.predict([ex1]))


# Logic for Emotion Screen
def emotion_screen():
    emotion_root = Tk()
    emotion_root.attributes('-fullscreen', True)
    emotion_root.configure(bg='blue')

    f1 = Frame(emotion_root)
    f1.pack(pady=50)
    l1 = Label(f1, text="Text Emotion", bg='blue', fg='white', font='comicsansms 50 bold')
    l1.pack()

    f2 = Frame(emotion_root, bg='blue')
    f2.pack(pady=20)
    l2 = Label(f2, text=(result[0][0].upper() + result[0][1:]), bg='green', font="comicsansms 30 bold", fg='white', borderwidth=3, relief=RAISED)
    l2.pack()

    f3 = Frame(emotion_root, bg='blue')
    f3.pack(pady=20)
    b3 = Button(f3, text="Exit", font="comicsansms 15 bold", fg="white", bg="red", borderwidth=3, relief=RAISED,
                command=exit)
    b3.pack()

    emotion_root.mainloop()


# Logic for Output Screen
def outputscreen():
    outputroot = Tk()
    outputroot.attributes('-fullscreen', True)
    outputroot.configure(bg='blue')

    f1 = Frame(outputroot, bg='blue')
    f1.pack(pady=50)
    l1 = Label(f1, text="Speech To Text", font='comicsansms 50 bold', bg='blue', fg='white')
    l1.pack()

    f2 = Frame(outputroot, bg='blue')
    f2.pack(pady=50)
    l2 = Label(f2, text=((output[0][0]).upper() + output[0][1:] + output[1:]), bg='green', fg='white',
               font='comicsansms 30 bold', borderwidth=3, relief=RAISED)
    l2.pack()

    f3 = Frame(outputroot)
    f3.pack(pady=40)
    b1 = Button(f3, text="Check Emotion", bg="brown", font="comicsansms 25 bold", fg='white', borderwidth=3, relief=RAISED, command=detect_emotion)
    b1.pack()

    f4 = Frame(outputroot, bg='blue')
    f4.pack()
    b2 = Button(f4, text="Proceed", bg='green', fg='white', font='comicsansms 15 bold', borderwidth=3, relief=RAISED, command=emotion_screen)
    b2.pack(anchor=CENTER, pady=10)

    f5 = Frame(outputroot, bg='blue')
    f5.pack(anchor='se')
    b3 = Button(f5, text="Back", bg='orange', fg='white', font='comicsansms 15 bold', borderwidth=3, relief=RAISED,
                command=inputscreen)
    b3.pack(side=LEFT, padx=15)
    b4 = Button(f5, text="Exit", bg='red', fg='white', font='comicsansms 15 bold', borderwidth=3, relief=RAISED,
                command=exit)
    b4.pack(side=RIGHT, padx=15)

    outputroot.mainloop()


# Logic for Proceed Button of Input Screen
def proceed_button():
    global output
    if output == "":
        messagebox.showinfo('Input', 'Please provide input voice.')
        inputscreen()
    else:
        outputscreen()


# Logic for Input Screen
def inputscreen():
    inputroot = Tk()
    inputroot.attributes('-fullscreen', True)
    inputroot.title("User Input")
    inputroot.configure(bg='blue')

    f1 = Frame(inputroot, bg="blue")
    f1.pack(side=TOP)
    l1 = Label(f1, text="User Input", font="comicsansms 50 bold", bg='blue', fg='white')
    l1.pack(pady=100)
    f2 = Frame(inputroot, bg='blue')
    f2.pack(pady=50)
    l2 = Label(f2, text="Click the below button to Record", font='comicsansms 20 bold')
    l2.pack()

    f3 = Frame(inputroot, bg='blue')
    f3.pack(pady=50)
    button1 = Button(f3, text='Start Listening', font="comicsansms 20 bold", fg='white', bg='brown', borderwidth=3,
                     relief=RAISED, command=voiceinput)
    button1.pack(side=LEFT, padx=100)

    f4 = Frame(inputroot, bg='blue')
    f4.pack(anchor='se')

    b3 = Button(f4, text="Proceed", font='comicsansms 15 bold', fg='white', bg='green', borderwidth=3, relief=RAISED,
                command=proceed_button)
    b3.pack(side=LEFT)
    b4 = Button(f4, text="Exit", font="comicsansms 15 bold", fg="white", bg="red", borderwidth=3, relief=RAISED,
                command=exit)
    b4.pack(side=RIGHT, padx=25)

    inputroot.mainloop()


# Main Function
if __name__ == "__main__":
    result = ""
    output = ""
    welcomeroot = Tk()
    welcomeroot.attributes('-fullscreen', True)
    welcomeroot.title("Welcome Screen")
    welcomeroot.configure(background="blue")

    f1 = Frame(welcomeroot, bg='red')
    f1.pack(side=TOP, pady=75)
    label1 = Label(f1, text="Welcome To My Project", font="comicsansms 50 bold", bg='blue', fg='white')
    label1.pack()

    photo = Image.open("Login Pic.jpeg")
    resize_image = photo.resize((250,250))
    img = ImageTk.PhotoImage(resize_image)
    myimage = Label(welcomeroot, image=img, bg='white')
    myimage.pack(anchor=CENTER)

    f2 = Frame(welcomeroot, bg='blue')
    f2.pack()
    button1 = Button(f2, text="Get Started", font="comicsansms 20 bold", bg="brown", fg='white', borderwidth=3,
                     relief=RAISED, command=inputscreen)
    button1.pack(pady=75)
    welcomeroot.mainloop()