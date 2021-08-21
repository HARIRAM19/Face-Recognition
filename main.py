import os.path
from tkinter import *
from tkinter import messagebox
import cv2
import numpy
import os

class FaceRec:
    def __init__(self,root):
        self.root=root
        self.haar_file='haarcascade_frontalface_default.xml'

        Label(self.root,height=2,bg='White').pack(fill=BOTH)
        Label(self.root,bg='Light Green',text='FACE RECOGNITION APP',font=('algerian',18,'bold'),height=3,bd=2,relief=GROOVE).pack(fill=BOTH)

        control_frame=Frame(self.root,height=200,bg='White',bd=4,relief=RIDGE)
        control_frame.pack(fill=BOTH,pady=20,padx=10)

        Button(control_frame,text='Train-Model',bg='Yellow',fg='Red',bd=2,height=3,relief=GROOVE,font=('arial',13,'bold'),width=12,command=self.get_data).place(x=50,y=50)
        Button(control_frame,text='Test-Model',bg='Yellow',fg='Red',bd=2,height=3,relief=GROOVE,font=('arial',13,'bold'),width=12,command=self.test_model).place(x=200,y=50)
        Button(control_frame,text='Exit App',bg='Yellow',fg='Red',bd=2,height=3,relief=GROOVE,font=('arial',13,'bold'),width=12,command=self.exit_app).place(x=350,y=50)

    def get_data(self):
        self.top=Toplevel()
        self.top.geometry('300x200+240+200')
        self.top.configure(bg='Light Green')
        self.top.resizable(0,0)

        name_lbl=Label(self.top,text="Name:",bg='Light Green',width=10,font=('arial',13,'bold')).place(x=10,y=20)
        self.name=Entry(self.top,width=15,font=('arial',13))
        self.name.place(x=120,y=20)

        id_lbl=Label(self.top,text="ID:",bg='Light Green',width=10,font=('arial',13,'bold')).place(x=10,y=60)
        self.id=Entry(self.top,width=15,font=('arial',13))
        self.id.place(x=120,y=60)

        btn=Button(self.top,text="Train-Model",font=('arial',13,'bold'),command=self.train_model)
        btn.place(x=100,y=120)

    def train_model(self):
        name=self.name.get()
        id_=self.id.get()
        if name!='' and id_!='':
            print(name,id_)
            self.top.destroy()
            self.take_images(name,id_)
        else:
            messagebox.showwarning('Warning','Please Fill all Required Fields')

    def take_images(self,name_,id_):
        datasets='dataset'
        subdata=str(name_)+'-'+str(id_)
        path=os.path.join(datasets,subdata)
        if not os.path.isdir(path):
            os.mkdir(path)
        face_cascade=cv2.CascadeClassifier(self.haar_file)
        webcam=cv2.VideoCapture(0)
        count=1
        width,height=130,100
        while count<=30:
            _,im=webcam.read()
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray,1.3,4)
            for(x,y,w,h) in faces:
                cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
                face=gray[y:y+h,x:x+w]
                face_resize=cv2.resize(face,(width,height))
                cv2.imwrite('%s/%s.png'%(path,count),face_resize)
            count+=1
            cv2.imshow('Image',im)
            key=cv2.waitKey(10)
            if key==27:
                break
        cv2.destroyAllWindows()
        webcam.release()
        messagebox.showinfo('Train-Model','Data is Saved')

    def test_model(self):
        datasets='dataset'
        (images,labels,names,id_)=([],[],{},0)
        for (subdirs,dirs,files) in os.walk(datasets):
            for subdir in dirs:
                names[id_]=subdir
                subjectpath=objectpath=os.path.join(datasets,subdir)
                for filename in os.listdir(subjectpath):
                    path=subjectpath+'/'+filename
                    label=id_
                    images.append(cv2.imread(path,0))
                    labels.append(int(label))
                id_+=1
        width,height=130,100
        images,labels=[numpy.array(lis) for lis in [images,labels]]
        model=cv2.face.LBPHFaceRecognizer_create()
        model.train(images,labels)
        face_cascade=cv2.CascadeClassifier(self.haar_file)
        webcam=cv2.VideoCapture(0)
        while True:
            (_,im)=webcam.read()
            gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray,1.3,5)
            for(x,y,w,h) in faces:
                cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
                face=gray[y:y+h,x:x+w]
                face_resize=cv2.resize(face,(width,height))
                prediction=model.predict(face_resize)
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
                if prediction[1]<500:
                    cv2.putText(im,'%s-%.0f'%(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255))
                else:
                    cv2.putText(im,'Not Recognized',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            cv2.imshow('Face Recognizer',im)
            key=cv2.waitKey(10)
            if key==27:
                break
        webcam.release()
        cv2.destroyAllWindows()

    def exit_app(self):
        root.quit()


root=Tk()
FaceRec(root)
root.geometry('550x330+240+200')
root.title("Face Recognition")
root.resizable(0,0)
root.configure(bg='Light Green')
root.mainloop()