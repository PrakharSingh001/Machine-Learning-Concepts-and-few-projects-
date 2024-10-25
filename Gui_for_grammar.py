from tkinter import*
t=Tk()
import re
import pickle
with open('grammar_correction','rb') as file:
    model=pickle.load(file)
box=Text(font=('calibri',15,'bold'),wrap=WORD)
box.place(x=100,y=200)
def code():
    get=box.get('1.0','end')
    txt=get.split('.')
    
    box.delete('1.0','end')
    for i in txt:
        i=i.rstrip()
        i=i.replace('\n','')
        predicted=model.predict([i])
        print(predicted)
        print(len(txt))
        if predicted==3:
            box.tag_configure("new",foreground='black',font=('calibri',15,'bold'))
            new_text=f'{i}.'
            print(i)
            box.insert('end',new_text,"new")
            print('correct')
        else:
            box.tag_configure("bold",foreground='red',font=('calibri',15,'bold','underline'))
            new_text=f'{i}.'
            box.insert('end',new_text,"bold")
            print(i)
    '''
    if predicted==0:
        print('Incorrect Tense Usage ')
    if predicted==1:
        print('Preposition Error  ')
    if predicted==2:
        print(' Adverb Error ')
    if predicted==3:
        print(' Correct ')
    if predicted==4:
        print(' Subject-Verb Agreement ')'''

sub=Button(text='sub',command=code)
sub.place(x=100,y=0)

t.mainloop()