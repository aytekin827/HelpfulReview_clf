import tkinter
import utils
import tkinter.messagebox as msgbox

def value_check(self):
    valid = False
    if self.isdigit():
        if (int(self) <= 50 and int(self) >= 0):
            valid = True
    elif self == '':
        valid = True
    return valid

def value_error(self):
    tkinter.messagebox.showwarning("입력값 오류", "0~50 사이의 숫자를 입력해주세요 :)")

def mail_alt():
    msgbox.showinfo('엇! 이봐요!', '문제가 있나요?ㅎㅎ\nminjae.dev@gmail.com\n메일을 보내주세요 :)')
    return

def show_model_info():
    msgbox.showinfo('model info',
     '''
     data : coupang review
     pretrained model : KcELECTRA

     [후드집업]
     validation acc score : 
     test acc score : 

     [스포츠의류]
     validation acc score : 
     test acc score : 

     [바지레깅스]
     validation acc score : 
     test acc score : 

     ''')
    return

def show_creator():
    msgbox.showinfo('about creator',
        """만든이\n
name : 정민재\nmail : minjae.dev@gmail.com\nblog : https://aytekin.tistory.com\ngithub : https://github.com/aytekin827\nportfolio : notion\n
데이터를 통해 세상을 바라보고 선한 영향력을 꿈꾸는 AI개발자 입니다 :)"""
        )