from tkinter import*
t=Tk()
import pickle
with open('code_explain','rb') as file:
    model=pickle.load(file)

code=Text(font=('arial',14,'bold'))
code.place(x=100,y=100)

explain=Text(font=('arial',14,'bold'))
explain.place(x=400,y=100)

def submit():
    explain.delete('1.0','end')
    text=code.get('1.0','end')
    new=text.split('\n')
    length=len(new)
    print(new)
    print(length)
    #new.pop(length)
    for i in new:
        if i=='':
            pass
        else:
            now=model.predict([i])
            now=f"{now}\n"
            explain.insert('end',now)
sub=Button(text='submit',command=submit)
sub.place(x=100,y=300)
t.mainloop()