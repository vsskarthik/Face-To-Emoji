# This files runs the whole application

import os
from os import system

use_qt = False

if os.name == "nt":
    print('Windows')
    if(use_qt):
        system('start python qt_gui.py')
    else:
        system('start python text_box.py')
    system('start python cam.py')
else:
    print('Linux/Unix')
    if(use_qt):
        system('python3 qt_gui.py &')
    else:
        system('python3 text_box.py &')
    system('python3 cam.py &')
