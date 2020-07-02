from gtts import gTTS
import os
import speech_recognition as sr

def sayMyText(mytext):
    # Passing the text and language to the engine, here we have marked slow=False, which tells the module that the converted audio should have a high speed 
    myobj = gTTS(text=mytext, lang='en', slow=False)
    
    # Saving the converted audio in a mp3 file named
    myobj.save("result.mp3")
    
    # Playing the converted file 
    os.system("mpg321 result.mp3")

def myCommand():
    # listens for commands
    r = sr.Recognizer()
    
    initialCommand = 'Please speak after the beep'
    sayMyText(initialCommand)
    os.system("mpg321 beep.mp3")
    print(initialCommand)
    
    with sr.Microphone() as source:
        print('Ready...')
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=0)
        audio = r.listen(source)

    try:
        command = r.recognize_google(audio).lower()
        print('Did you say: ' + command + '\n')
        sayMyText('Did you say ' + command);

    # generate error if command is not heard
    except sr.UnknownValueError:
        command = 'Sorry! Your last command couldn\'t be heard'
        print(command)
        sayMyText(command);

    return command

#while True:
os.chdir('/home/kunal/capstone-project')
myCommand()
    

