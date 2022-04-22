import tkinter as tk
import random
from tkinter import ttk


from tkinter.filedialog import askopenfilename


def open_file():
    """Открываем файл для редактирования"""
    filepath = askopenfilename(
        filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
    )
    if not filepath:
        return
    txt_edit.delete("1.0", tk.END)
    with open(filepath, "r") as input_file:
        text = input_file.read()
        txt_edit.insert(tk.END, text)
    window.title(f"Простой текстовый редактор - {filepath}")


window = tk.Tk()
window.title("marking of minerals")

window.rowconfigure(1, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)

# здесь будет изображение
txt_edit = tk.Text(window)

# боковая менюшка
fr_left_side = tk.Frame(window)


# кнопки боковой менюшки
btn_open = tk.Button(fr_left_side, text="Открыть", command=open_file)
btn_save = tk.Button(fr_left_side, text="Сохранить как...")

fontExample = ("Courier", 16, "bold")
cmb_markers = ttk.Combobox(master=fr_left_side,
                           values=[
                                    "background",
                                    "chalcopyrite",
                                    "galena",
                                    "magnetite",
                                    "bornite",
                                    "pyrrhotite",
                                    "pyrite/marcasite",
                                    "pentlandite",
                                    "sphalerite",
                                    "arsenopyrite",
                                    "hematite",
                                    "tenantite-tetrahedrite group",
                                    "covelline"],
                           font=fontExample)
# размещаем кнопки на боковой панели
btn_open.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
btn_save.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
cmb_markers.grid(row=2, column=0, sticky="ew", padx=5)


# верхняя менюшка
fr_up_side = tk.Frame(window, relief=tk.RAISED, borderwidth=1)
flag = tk.BooleanVar()
bt_show_mask = tk.Checkbutton(master=fr_up_side, text="Show mask", variable=flag)
bt_show_mask.pack()
fr_up_side.grid(row=0, column=0, sticky="ew")
#fr_up_side.pack(fill=tk.X, expand=True)


fr_left_side.grid(row=1, column=0, sticky="ns")
txt_edit.grid(row=1, column=1, sticky="nsew")


window.mainloop()


"""scale = 1.25
width = int(self.master.winfo_width() / scale)  # задаем ширину холста
height = int(self.master.winfo_height() / scale)  # задаем высоту холста
"""

"""self.image = Image.open("01.jpg")
self.image = self.image.resize((width, height))  # Изменяем размер картинки
self.photo = ImageTk.PhotoImage(self.image)"""


