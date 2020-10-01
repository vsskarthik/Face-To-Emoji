import sys
sys.path.append('.')
import emoji
from tkinter import *
import pyperclip as cb

#from cam import get_curr_emoji

emotion = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Suprise']
#emojis = [😡,🤢,😨,😀,😐,😥,😲]
emojis = ['\U0001F621','\U0001F922','\U0001F628','\U0001F600','\U0001F610','\U0001F625','\U0001F632']
emoji_text = [' <angry> ',' <disgust> ',' <fear> ',' <happy> ',' <neutral> ',' <sad> ',' <suprised> ']

def insert_emoji(text):
    l = text.split()
    for idx,i in enumerate(l):
        i=i.strip()
        if(i[0] == '<' and i[-1] == ">"):
            try:
                l[idx] = emoji.emojize(emojis[emoji_text.index(" "+i+" ")])
            except:
                pass
    text = ' '.join(l)
    return text

def copy_to_clip(root,text):
    text = insert_emoji(text)
    cb.copy(text)

def capture_emoji(text):
    curr_emoji = None
    with open('curr_emoji.txt', 'r') as f:
        curr_emoji = f.read()
    #print(emojis[int(curr_emoji)])
    text.insert("end",emoji_text[int(curr_emoji)])





root = Tk()
text = Text(root)
text.grid(column=0, row=7, padx=10, pady=20)

button = Button(root,text="Capture Emoji",command=lambda:capture_emoji(text))
button.grid(column=0, row=10, padx=10, pady=20)

button = Button(root,text="Copy To Clipboard",command=lambda:copy_to_clip(root,text.get("1.0","end")))
button.grid(column=0, row=11, padx=10, pady=20)

root.mainloop()
