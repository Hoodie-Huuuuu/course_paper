# from ..segmenter.segmenter_service import Server

import tkinter as tk
from collections import OrderedDict
from tkinter import ttk, Frame, Button, Label
from tkinter.filedialog import askopenfilename, asksaveasfilename
from typing import *

import numpy as np
from PIL import Image, ImageTk, ImageColor
from icecream import ic
import app_utils
import config

import grpc
import segmenter_service_pb2 as pb2
import segmenter_service_pb2_grpc as pb2_grpc


class Application(Frame):
    def __init__(self, parent):
        super().__init__(master=parent)

        self.channel = grpc.insecure_channel(
            f"{config.SERVER_IP}:{config.SEGMENTER_PORT}"
        )
        self.stub = pb2_grpc.SegmenterStub(self.channel)

        self.brush_size = 2
        self.sens_val_scale = float(0)
        self.transparency_val = 0.5

        # цвета маркеров
        self.markers = OrderedDict(
            {
                "background": "#1c1818",
                "chalcopyrite": "#ff0000",
                "galena": "#cbff00",
                "magnetite": "#00ff66",
                "bornite": "#0065ff",
                "pyrrhotite": "#cc00ff",
                "pyrite/marcasite": "#ff4c4c",
                "pentlandite": "#dbff4c",
                "sphalerite": "#4cff93",
                "arsenopyrite": "#4c93ff",
                "hematite": "#db4cff",
                "tenantite-tetrahedrite group": "#ff9999",
                "covelline": "#eaff99",
            }
        )

        self.curr_marker, self._color = list(self.markers.items())[0]

        # dict["marker_name": (marker_idx, marker_hex_color)]
        self.markers = {
            item[0]: (i, item[1])
            for i, item in enumerate(self.markers.items(), start=1)
        }

        # список цветов в rgb
        hex2rgb = lambda hex_colour: ImageColor.getcolor(hex_colour, "RGB")
        self._colours_rgb = {
            i: hex2rgb(hex_color) for i, hex_color in self.markers.values()
        }

        self.methods = OrderedDict(
            {
                "SLIC": "slic",
                "Watershed": "watershed",
                "Quick shift": "quick_shift",
                "Felzenszwalb": "fwb",
            }
        )
        # словарь параметров для каждого метода, заполняется в интерфейсе
        self.params = {}

        self.curr_method = list(self.methods.values())[0]

        # маска последнего штришка
        self._marker_mask = None

        # итоговая маска
        self.mask = None

        # widgets
        self.method_params_widget = None
        self.button_segment = None

        self._curr_image = None
        self._curr_image_as_array = None
        self.draw_borders_var = tk.BooleanVar()

        self._photo = None

        self.initUI()

    # создать интерфейс
    def initUI(self):
        self.pack(fill=tk.BOTH, expand=True)
        self.master.bind("<Control-z>", self.ctrl_z)

        # для растяжения боковой панели по вертикали
        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)

        # <--БОКОВАЯ ПАНЕЛЬ-->
        self.frame_left_side = Frame(master=self, relief=tk.RAISED, borderwidth=1)
        self.frame_left_side.grid(row=0, column=0, sticky=tk.NS, rowspan=2)

        # кнопка открыть файл
        bt_open = Button(
            master=self.frame_left_side,
            text="Open file",
            width=10,
            command=self.open_file,
        )
        bt_open.grid(row=0, column=0, sticky=tk.EW, padx=5, pady=2)

        # кнопка сохранить файл
        self.bt_save = Button(
            master=self.frame_left_side,
            text="Save file",
            width=10,
            command=self.save_file,
            state="disabled",
        )
        self.bt_save.grid(row=1, column=0, sticky=tk.EW, padx=5, pady=2)

        # выбор метода сегментации
        self.method_frame = Frame(master=self.frame_left_side)
        lbl_methods = Label(master=self.method_frame, text="Segmentation method:")
        lbl_methods.pack(pady=10)

        cmb_methods = ttk.Combobox(
            master=self.method_frame,
            values=list(self.methods.keys()),
            font=("Courier", 16, "bold"),
        )
        cmb_methods.bind("<<ComboboxSelected>>", self.method_changed)
        cmb_methods.current(0)  # ///////////////////////////
        cmb_methods.pack(padx=5, pady=2)

        # рисует ползунки для параметров c кнопкой сегментацией в зависимости от метода
        self.set_params_widget(self.curr_method, self.method_frame)
        self.method_frame.grid(row=2, column=0, sticky=tk.EW, padx=5, pady=30)

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
        cmb_markers = ttk.Combobox(
            master=frame_markers, values=list(self.markers.keys()), font=font
        )
        cmb_markers.bind("<<ComboboxSelected>>", self.marker_changed)
        cmb_markers.current(0)  # ///////////////////////////

        cmb_markers.pack(side=tk.LEFT, padx=5, pady=2)
        frame_markers.pack(side=tk.LEFT, padx=5, pady=2)

        # ползунок чувствительности
        frame_sens = Frame(master=self.frame_bar)

        lbl_sens = Label(master=frame_sens, text="Sensitivity:")
        lbl_sens.pack(side=tk.LEFT, padx=5, pady=2)

        scale = tk.Scale(
            master=frame_sens,
            from_=0,
            to=1,
            resolution=0.01,
            sliderlength=25,
            orient="horizontal",
            length=200,
            command=self.sens_changed,
        )
        scale.pack(side=tk.LEFT, padx=5, pady=2)

        frame_sens.pack(side=tk.LEFT)

        # прозрачность маски
        frame_transparency = Frame(master=self.frame_bar)

        lbl_transparency = Label(master=frame_transparency, text="Transparency:")
        lbl_transparency.pack(side=tk.LEFT, padx=5, pady=2)

        scale_tr = tk.Scale(
            master=frame_transparency,
            from_=0,
            to=1,
            resolution=0.01,
            sliderlength=25,
            orient="horizontal",
            length=200,
            command=self.transparency_changed,
        )
        scale_tr.set(self.transparency_val)  # default value
        scale_tr.pack(side=tk.LEFT, padx=5, pady=2)

        frame_transparency.pack(side=tk.LEFT)

        # отрисовать границы для отладки
        self.check_button = tk.Checkbutton(
            master=self.frame_bar,
            text="draw borders",
            variable=self.draw_borders_var,
            offvalue=False,
            onvalue=True,
            command=self.draw_borders_button_changed,
        )
        self.check_button.pack(side=tk.LEFT, padx=5, pady=2)

    # <--ВЕРХНЯЯ ПАНЕЛЬ--/>

    def save_file(self):
        ic("кнопка нажимается")
        if self._curr_image is None:
            return
        # диалог
        outpath = asksaveasfilename()
        if outpath is None:  # asksaveasfile return `None` if dialog closed with "cancel".
            return
        
        np.savez(outpath, mask=self.mask)
        return

    # Открываем файл для редактирования
    def open_file(self):

        filepath = askopenfilename()
        if not filepath:
            return

        # открываем картинку
        image = Image.open(filepath).convert("RGB")
        self._curr_image = image
        self._curr_image_as_array = np.asarray(image)

        # маска для штрихов
        self._marker_mask = np.zeros((image.height, image.width), dtype="uint8")

        # создаем сегментатор
        self.stub.loadImage(
            pb2.LoadImageRequest(
                path=filepath,
                image_array=pb2.NdArray(
                    data=self._curr_image_as_array.tobytes(),
                    dtype=str(self._curr_image_as_array.dtype),
                    shape=self._curr_image_as_array.shape,
                ),
                markers={
                    m_name: idx_and_color[0]
                    for m_name, idx_and_color in self.markers.items()
                },
                method_params=pb2.MethodParams(
                    method_name=self.curr_method, params=self.params
                ),
            )
        )
        # self.server.MakeSegmenter(image, self.markers, self.curr_method, **self.params)

        self.mask = np.zeros((image.height, image.width), dtype="uint8")

        # полотно
        self.canv = tk.Canvas(
            master=self, bg="white", width=image.width, height=image.height
        )
        self.canv.grid(row=1, column=1)
        self.canv.bind("<B1-Motion>", self.draw)
        self.canv.bind("<ButtonRelease>", self.end_draw)

        # создаем картинку
        self.render_image(self.get_rgb_marked_image())

        # делаем доступной кнопку сохранить
        self.bt_save["state"] = "normal"
        return

    # рисование (полотно появляется после загрузки картинки)
    def draw(self, event):
        self.canv.create_oval(
            event.x - self.brush_size,
            event.y - self.brush_size,
            event.x + self.brush_size,
            event.y + self.brush_size,
            fill=self._color,
            outline=self._color,
        )
        # todo заполнять все пиксели под маркером
        # заполняем маску значением - индекс маркера
        self._marker_mask[event.y, event.x] = self.markers[self.curr_marker][0]
        return

    # функция отправки маски в сегментатор
    def end_draw(self, event):
        height, width = self._marker_mask.shape
        mask2D = pb2.Mask2D(
            data=self._marker_mask.tobytes(),
            dtype=str(self._marker_mask.dtype),
            height=height,
            width=width,
        )

        self.stub.sendMarkerMask(
            pb2.MarkerMask(
                mask=mask2D,
                marker=self.curr_marker,
                sens=pb2.SensitivityValue(value=np.float32(0)),
            )
        )

        self.mask = app_utils.Mask2D_to_array(
            self.stub.getMarkedMask2D(pb2.google_dot_protobuf_dot_empty__pb2.Empty())
        )

        # обновить отправляемую маску
        self._marker_mask = np.zeros(self.mask.shape, dtype=self.mask.dtype)

        # меняем картинку с добавленными изменениями от штришка
        self.render_image(self.get_rgb_marked_image())

    def render_image(self, image):
        self._photo = ImageTk.PhotoImage(image)
        self.canv.create_image(0, 0, anchor="nw", image=self._photo)
        return

    def get_rgb_marked_image(self):
        regions = None
        if self.draw_borders_var.get():
            regions = app_utils.Mask2D_to_array(
                self.stub.getRegions(pb2.google_dot_protobuf_dot_empty__pb2.Empty())
            )

        # создать изображение с маской
        return app_utils.get_image(
            mask=self.mask,
            source_image=self._curr_image_as_array,
            colors_rgb=self._colours_rgb,
            alpha=self.transparency_val,
            draw_borders=self.draw_borders_var.get(),
            regions=regions,
        )

    def transparency_changed(self, val):
        self.transparency_val = float(val)
        if self._curr_image is None:
            return

        self.render_image(self.get_rgb_marked_image())

    # установка значения ползунка чувствительности [ ]
    def sens_changed(self, val):
        nothing_is_drawn = not np.any(self.mask)
        if nothing_is_drawn:
            return

        self.sens_val_scale = float(val)

        self.stub.sendNewSens(pb2.SensitivityValue(value=self.sens_val_scale))
        self.mask = app_utils.Mask2D_to_array(
            self.stub.getMarkedMask2D(pb2.google_dot_protobuf_dot_empty__pb2.Empty())
        )
        self.render_image(self.get_rgb_marked_image())
        return

    def ctrl_z(self, event):
        print("ctrl+z")
        nothing_is_drawn = not np.any(self.mask)
        if nothing_is_drawn:
            return

        old_mask = app_utils.Mask2D_to_array(
            self.stub.popState(pb2.google_dot_protobuf_dot_empty__pb2.Empty())
        )
        self.mask = old_mask

        self.render_image(self.get_rgb_marked_image())

    def draw_borders_button_changed(self):
        if self._curr_image is None:
            return
        # меняем картинку с добавленными изменениями от штришка
        self.render_image(self.get_rgb_marked_image())

    # выбор маркера
    def marker_changed(self, event):
        self.curr_marker = event.widget.get()
        self._color = self.markers[self.curr_marker][1]
        return

    def method_changed(self, event):
        print("method selected")
        self.curr_method = self.methods[event.widget.get()]
        self.set_params_widget(self.curr_method, self.method_frame)  # c кнопкой

    # todo ДОБАВИТЬ ФИКСАЦИЮ в СЕГМЕНТАТОР
    def handler_segment_button(self):
        print("start segmentation for " + self.curr_method)

        # создаем сегментатор
        self.stub.newMethod(
            pb2.NewMethodRequest(
                method_params=pb2.MethodParams(
                    method_name=self.curr_method, params=self.params
                ),
                save_current_mask=True,
            )
        )
        self.mask = app_utils.Mask2D_to_array(
            self.stub.getMarkedMask2D(pb2.google_dot_protobuf_dot_empty__pb2.Empty())
        )

        # создаем картинку
        self.render_image(self.get_rgb_marked_image())
        return

    def set_params_widget(
        self, method: Literal["slic", "watershed", "quick_shift", "fwb"], parent_widget
    ):
        """
        :param method:
        :param parent_widget:
        :return:

        по выбранному методу создает виджет с парметрами, и кнопку
        """
        if self.method_params_widget is not None:
            self.method_params_widget.destroy()
        if self.button_segment is not None:
            self.button_segment.destroy()

        self.method_params_widget = Frame(
            master=parent_widget, relief=tk.RAISED, borderwidth=1
        )

        if method == "slic" or method == "watershed":
            # num of superpixels
            lbl_n_segments = Label(
                master=self.method_params_widget, text="num of superpixels"
            )
            lbl_n_segments.pack(padx=5, pady=10)

            def n_segments_handler(val):
                self.params["n_segments"] = int(val)

            scale = tk.Scale(
                master=self.method_params_widget,
                from_=0,
                to=1000,
                resolution=1,
                sliderlength=30,
                orient="horizontal",
                length=200,
                command=n_segments_handler,
            )
            scale.pack(padx=5, pady=2)

            if method == "slic":
                # compactness
                lbl_n_segments = Label(
                    master=self.method_params_widget, text="compactness"
                )
                lbl_n_segments.pack(padx=5, pady=10)

                def compactness_handler(val):
                    self.params["compactness"] = int(val)

                scale_c = tk.Scale(
                    master=self.method_params_widget,
                    from_=1,
                    to=40,
                    resolution=1,
                    sliderlength=30,
                    orient="horizontal",
                    length=200,
                    command=compactness_handler,
                )
                scale_c.pack(padx=5, pady=2)

                self.params = {
                    "n_segments": 700,
                    "compactness": 5,
                    "sigma": 0,
                    "start_label": 1,
                }

            elif method == "watershed":
                # compactness
                lbl_n_segments = Label(
                    master=self.method_params_widget, text="compactness"
                )
                lbl_n_segments.pack(padx=5, pady=10)

                start = 0
                end = 100
                max_compactness = 0.005

                def compactness_handler(val):
                    self.params["compactness"] = (
                        max_compactness * (float(val) - start) / (end - start)
                    )

                scale_c = tk.Scale(
                    master=self.method_params_widget,
                    from_=start,
                    to=end,
                    resolution=1,
                    sliderlength=30,
                    orient="horizontal",
                    length=200,
                    command=compactness_handler,
                )
                scale_c.pack(padx=5, pady=2)

                self.params = {
                    "n_segments": 700,
                    "compactness": 0.001,
                }

        elif method == "fwb":
            # scale
            lbl_n_scale = Label(master=self.method_params_widget, text="scale")
            lbl_n_scale.pack(padx=5, pady=10)

            def scale_handler(val):
                self.params["scale"] = int(val)

            scale = tk.Scale(
                master=self.method_params_widget,
                from_=0,
                to=1000,
                resolution=1,
                sliderlength=30,
                orient="horizontal",
                length=200,
                command=scale_handler,
            )
            scale.pack(padx=5, pady=2)

            self.params = {"scale": 400, "sigma": 1, "min_size": 50}

        elif method == "quick_shift":

            # compactness
            lbl_compactness = Label(
                master=self.method_params_widget, text="compactness"
            )
            lbl_compactness.pack(padx=5, pady=10)

            start = 0
            end = 100
            max_compactness = 1

            def compactness_handler(val):
                self.params["ratio"] = (
                    max_compactness * (float(val) - start) / (end - start)
                )

            scale_c = tk.Scale(
                master=self.method_params_widget,
                from_=start,
                to=end,
                resolution=1,
                sliderlength=30,
                orient="horizontal",
                length=200,
                command=compactness_handler,
            )
            scale_c.pack(padx=5, pady=2)

            self.params = {"sigma": 1, "max_dist": 1000, "ratio": 0.5}

        self.method_params_widget.pack()

        self.button_segment = Button(
            master=self.method_frame,
            text="do segmentation",
            width=10,
            command=self.handler_segment_button,
        )
        self.button_segment.pack()


def main():
    root = tk.Tk()
    root.title("marking of minerals")
    root.attributes("-fullscreen", True)
    root.geometry("300x300")
    app = Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
