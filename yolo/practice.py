import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voices", voices[1].id)
engine.setProperty("rate", 150)
engine.say("Hello Anton,how are you")
engine.runAndWait()
