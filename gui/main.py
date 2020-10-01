from os import system

windows = False

if(not windows):
    system('python3 text_box.py &')
    system('python3 cam.py &')
else:
    system('python text_box.py &')
    system('python cam.py &')
