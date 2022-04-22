import tkinter as tk
from collections import OrderedDict
from tkinter import ttk, Frame, Button, Label
from tkinter.filedialog import askopenfilename

import numpy as np
from PIL import Image, ImageTk

from segmenter import Segmenter


class Application(Frame):

    def __init__(self, parent):
        super().__init__(master=parent)
        self.brush_size = 2
        self.sens_val_scale = 0
        # цвета маркеров
        self.markers = OrderedDict({
               "chalcopyrite": '#ff0000',
               "galena": '#cbff00',
               "magnetite": '#00ff66',
               "bornite": '#0065ff',
               "pyrrhotite": '#cc00ff',
               "pyrite/marcasite": '#ff4c4c',
               "pentlandite": '#dbff4c',
               "sphalerite": '#4cff93',
               "arsenopyrite": '#4c93ff',
               "hematite": '#db4cff',
               "tenantite-tetrahedrite group": '#ff9999',
               "covelline": '#eaff99'
        })
        self.curr_marker, self.color = list(self.markers.items())[0]

        # dict["marker_name": (marker_idx, marker_hex_color)]
        self.markers = {item[0]: (i, item[1]) for i, item in enumerate(self.markers.items(), start=1)}
        self.previous_masks_stack = []

        # маска последнего штришка
        self.marker_mask = None
        # итоговая маска
        self.mask = None
        # отображаемая картинка
        self.photo = None

        self.initUI()

    # создать интерфейс
    def initUI(self):
        self.pack(fill=tk.BOTH, expand=True)
        self.master.bind('<Control-z>', self.ctrl_z)
        # для растяжения боковой панели по вертикали
        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)

    # <--БОКОВАЯ ПАНЕЛЬ-->
        self.frame_left_side = Frame(master=self, relief=tk.RAISED, borderwidth=1)
        self.frame_left_side.grid(row=0, column=0, sticky=tk.NS, rowspan=2)

        # кнопка открыть файл
        bt_open = Button(master=self.frame_left_side, text="Open file",
                         width=10, command=self.open_file)
        bt_open.grid(row=0, column=0, sticky=tk.EW, padx=5, pady=2)

        # кнопка сохранить файл
        bt_save = Button(master=self.frame_left_side, text="Save file", width=10)
        bt_save.grid(row=1, column=0, sticky=tk.EW, padx=5, pady=2)
    # <--БОКОВАЯ ПАНЕЛЬ--/>


    # <--ВЕРХНЯЯ ПАНЕЛЬ-->
        self.frame_bar = Frame(self)
        self.frame_bar.grid(row=0, column=1, sticky=tk.EW)

        # выбор маркера
        frame_markers = Frame(master=self.frame_bar)

        # список маркеров
        lbl_markers = Label(master=frame_markers, text="Marker:")
        lbl_markers.pack(side=tk.LEFT)

        font = ("Courier", 16, "bold")
        cmb_markers = ttk.Combobox(master=frame_markers,
                                   values=list(self.markers.keys()),
                                   font=font)
        cmb_markers.bind("<<ComboboxSelected>>", self.marker_changed)
        cmb_markers.current(0)  # ///////////////////////////

        cmb_markers.pack(side=tk.LEFT, padx=5, pady=2)
        frame_markers.pack(side=tk.LEFT, padx=5, pady=2)

        # ползунок чувствительности
        frame_sens = Frame(master=self.frame_bar)

        lbl_sens = Label(master=frame_sens, text="Sensitivity:")
        lbl_sens.pack(side=tk.LEFT, padx=5, pady=2)

        scale = tk.Scale(master=frame_sens, from_=0, to=1, resolution=0.01, sliderlength=25,
                         orient="horizontal", length=200, command=self.sens_changed)
        scale.pack(side=tk.LEFT, padx=5, pady=2)

        frame_sens.pack(side=tk.LEFT)
    # <--ВЕРХНЯЯ ПАНЕЛЬ--/>


    # выбор маркера
    def marker_changed(self, event):
        self.curr_marker = event.widget.get()
        self.color = self.markers[self.curr_marker][1]
        return


    # Открываем файл для редактирования
    def open_file(self):

        filepath = askopenfilename()
        if not filepath:
            return

        # открываем картинку
        image = Image.open(filepath).convert("RGB")
        self.mask = np.zeros((image.height, image.width), dtype='uint8')
        self.marker_mask = np.zeros((image.height, image.width), dtype='uint8')

        # создаем сегментатор
        self.segmentator = Segmenter(image, self.markers)
        # выбираем картинку
        self.photo = ImageTk.PhotoImage(self.segmentator.rgb_marked_image)

        # полотно
        self.canv = tk.Canvas(master=self, bg="white", width=image.width, height=image.height)
        self.canv.grid(row=1, column=1)
        self.canv.bind("<B1-Motion>", self.draw)

        # маска для отправки меток в сегментатор
        self.canv.bind("<ButtonRelease>", self.end_draw)
        self.canv.create_image(0, 0, anchor='nw', image=self.photo)
        return


    # рисование (полотно появляется после загрузки картинки)
    def draw(self, event):
        self.canv.create_oval(event.x - self.brush_size, event.y - self.brush_size,
                              event.x + self.brush_size, event.y + self.brush_size,
                              fill=self.color, outline=self.color)
        # заполняем маску значением: индекс маркера
        self.marker_mask[event.y, event.x] = self.markers[self.curr_marker][0]
        return


    # функция отправки маски в сегментатор
    def end_draw(self, event):
        self.previous_masks_stack.append((self.mask.copy(), self.marker_mask.copy(), self.curr_marker))
        self.mask = self.segmentator.draw_regions(self.mask, self.marker_mask, self.curr_marker, self.sens_val_scale)
        self.marker_mask = np.zeros(self.mask.shape, dtype=self.mask.dtype)

        # меняем картинку с добавленными изменениями от штришка
        self.photo = ImageTk.PhotoImage(self.segmentator.rgb_marked_image)
        self.canv.create_image(0, 0, anchor='nw', image=self.photo)


    # установка значения ползунка чувствительности [ ]
    # mask, marker_mask, marker = mask0, mask_marker, marker,
    def sens_changed(self, val):
        self.sens_val_scale = float(val)
        if len(self.previous_masks_stack) == 0:
            return

        mask, marker_mask, marker = self.previous_masks_stack[-1]
        self.mask = self.segmentator.draw_regions(mask, marker_mask, marker, self.sens_val_scale)

        # меняем картинку с добавленными изменениями от штришка
        self.photo = ImageTk.PhotoImage(self.segmentator.rgb_marked_image)
        self.canv.create_image(0, 0, anchor='nw', image=self.photo)
        return


    def ctrl_z(self, event):
        if len(self.previous_masks_stack) == 0:
            print("kek")
            return

        self.mask = self.previous_masks_stack.pop()[0]
        self.segmentator.draw_regions(self.mask)
        self.photo = ImageTk.PhotoImage(self.segmentator.rgb_marked_image)
        self.canv.create_image(0, 0, anchor='nw', image=self.photo)







def main():
    root = tk.Tk()
    root.title("marking of minerals")
    root.attributes('-fullscreen', True)
    root.geometry("300x300")
    app = Application(root)
    root.mainloop()


if __name__ == '__main__':
    main()