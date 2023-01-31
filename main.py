import tkinter
import tkinter.messagebox
from tkinter import *
from tkinter.ttk import *
from glob import glob
import utils
from funcs import *

root = Tk()

def close():
    root.quit()
    root.destroy()

menubar=tkinter.Menu(root)

menu_file=tkinter.Menu(menubar, tearoff=0)
menu_file.add_command(label="save")
menu_file.add_separator()
menu_file.add_command(label="quit", command=close)
menubar.add_cascade(label="File", menu=menu_file)

menu_help=tkinter.Menu(menubar, tearoff=0)
menu_help.add_command(label="About Model", command=show_model_info)
menu_help.add_command(label="버그 및 문의 메일", command=mail_alt)
menu_help.add_command(label="About Creator", command=show_creator)
menu_help.add_separator()
menu_help.add_command(label="Version_1.0")
menubar.add_cascade(label="Help", menu=menu_help)

root.title("내 리뷰가 도움이 되니?")
root.resizable(False, False)

# 상품 카테고리
frame_select_cate = LabelFrame(root, text='제품 카테고리 선택', borderwidth=4)
frame_select_cate.grid(row=0, column=0, padx=5, pady=5, sticky='w')

category_combobox=Combobox(frame_select_cate,width=35 ,height=3, values=['후드집업','스포츠의류','바지레깅스'] )
category_combobox.set("")
category_combobox.pack(side='right',fill='x',padx=5, pady=5, expand=True, ipady=5)

# 상품정보 입력
frame_select_prodinfo = LabelFrame(root, text='제품 정보 입력', borderwidth=4)
frame_select_prodinfo.grid(row=1, column=0, padx=5, pady=5, sticky='w')

label_prodname = Label(frame_select_prodinfo, text='상품명 :')
entry_prodname = Entry(frame_select_prodinfo, width=25)
label_prodname.grid(row=0, column=0, padx=3, pady=5)
entry_prodname.grid(row=0, column=1, padx=3, pady=5)

label_attach_count = Label(frame_select_prodinfo, text='첨부파일개수 :')
validate_command=(frame_select_prodinfo.register(value_check), '%P')
invalid_command=(frame_select_prodinfo.register(value_error), '%P')
spinbox=Spinbox(frame_select_prodinfo, from_ = 0, to = 50, validate = 'all', validatecommand = validate_command, invalidcommand=invalid_command)
label_attach_count.grid(row=1, column=0, padx=3, pady=5)
spinbox.grid(row=1, column=1, padx=3, pady=5)


# 리뷰
frame_review = LabelFrame(root, text='리뷰')
frame_review.grid(row=2, column=0, padx=5, pady=5)

text_review = Text(frame_review, width=65, height=15)
text_review.pack(fill='x',padx=5, pady=5, expand=True, ipady=5)

def btncmd():
    category = category_combobox.get()
    if category == '':
        tkinter.messagebox.showwarning("알림!", "상품 카테고리를 선택하세요 :)")

    prod_name = entry_prodname.get()
    review = text_review.get('1.0','end')

    model, tokenizer = utils.load_cate_model(category)

    res, Preds_percentage = utils.analyze_Bert(prod_name=prod_name, review=review, model=model, tokenizer=tokenizer)

    print(model)
    
    if res == 1:
        # print('helpful review')
        entry_result.delete(0, END)
        entry_result.insert(0, f'{Preds_percentage[0][0]*100:0.1f}%의 확률로 도움이 되는 리뷰(Helpful Review)입니다.')

    elif res == 0:
        # print('helpless review')
        entry_result.delete(0, END)
        entry_result.insert(0, f'{Preds_percentage[0][1]*100:0.1f}%의 확률로 도움이 안되는 리뷰(Helpless Review)입니다.')

# 분석시작 버튼
button_analyze = tkinter.Button(root, overrelief="solid", bd=2, relief="raised", command=btncmd, text='리뷰 분석시작')
button_analyze.grid(row=3, column=0, padx=5, pady=5, sticky=N+E+W+S)

# 결과
frame_result = LabelFrame(root, text='결과')
frame_result.grid(row=4, column=0, padx=5, pady=5, sticky='w')

label_result = Label(frame_result, text='도움이 되는 리뷰인가요?')
entry_result = Entry(frame_result, width=50)
label_result.pack(side='left', fill='x',padx=5, pady=5, expand=True, ipady=5)
entry_result.pack(side='right',fill='x',padx=5, pady=5, expand=True, ipady=5)


# # 끄기버튼
# button_quit = tkinter.Button(root, text='종료', command=quit, relief="groove", overrelief="solid")
# button_quit.grid(row=4, column=0, sticky='e',padx=10, pady=5)

root.config(menu=menubar)
root.mainloop()