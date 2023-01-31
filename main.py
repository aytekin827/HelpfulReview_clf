import tkinter
import tkinter.messagebox
from tkinter import *
from tkinter.ttk import *
from glob import glob
import utils

root = Tk()

root.title("helpful review")
root.resizable(False, False)

# 상품 카테고리
category_label = Label(root, text='상품 카테고리')
values=['후드집업','스포츠의류','바지레깅스'] 
category_combobox=Combobox(root, height=5, values=values)
category_combobox.set("카테고리 선택")


# 상품명
prod_label = Label(root, text='상품명')
input_text = Entry(root, width=30)
input_text.grid(column=0, row=2)

# 첨부파일개수
attach_label = Label(root, text='첨부파일 개수')
def value_check(self):
    valid = False
    if self.isdigit():
        if (int(self) <= 50 and int(self) >= 0):
            valid = True
    elif self == '':
        valid = True
    return valid

def value_error(self):
    tkinter.messagebox.showinfo("입력값 오류", "0~50 사이의 숫자를 입력해주세요 :)")

validate_command=(root.register(value_check), '%P')
invalid_command=(root.register(value_error), '%P')
spinbox=Spinbox(root, from_ = 0, to = 50, validate = 'all', validatecommand = validate_command, invalidcommand=invalid_command)

# 리뷰
review_label = Label(root, text='리뷰')
text_review = Text(root, width=50, height=10)

# 분석시작 버튼
def btncmd():
    category = category_combobox.get()
    if category == '카테고리 선택':
        tkinter.messagebox.showinfo("알림!", "상품 카테고리를 선택하세요 :)")

    prod_name = input_text.get()
    review = text_review.get('1.0','end')

    model, tokenizer = utils.load_cate_model(category)

    res, Preds_percentage = utils.analyze_Bert(prod_name=prod_name, review=review, model=model, tokenizer=tokenizer)

    if res == 1:
        print('helpful review')
        text_result.delete(0, END)
        text_result.insert(0, 'helpful review')

    elif res == 0:
        print('helpless review')
        text_result.delete(0, END)
        text_result.insert(0, 'helpless review')


button = tkinter.Button(root, overrelief="solid", width=15, command=btncmd, repeatdelay=1000, repeatinterval=100,text='분석시작')

# 결과
result_label = Label(root, text='결과')
text_result = Entry(root, width=30)

# 끄기버튼
button_quit = tkinter.Button(root, text='종료', command=quit)

category_label.grid(row=0, column=0, sticky='ew')
category_combobox.grid(row=0, column=1, sticky='ew')
prod_label.grid(row=1, column=0, sticky='ew')
input_text.grid(row=1, column=1, sticky='ew')
attach_label.grid(row=2, column=0, sticky='ew')
spinbox.grid(row=2, column=1, sticky='ew')
review_label.grid(row=3, column=0, sticky='ew')
text_review.grid(row=3, column=1, sticky='ew')
button.grid(row=4,column=1)
result_label.grid(row=5, column=0, sticky='ew')
text_result.grid(row=5, column=1, sticky='ew')
button_quit.grid(row=6, column=1)

root.mainloop()