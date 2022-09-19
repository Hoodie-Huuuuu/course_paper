Интерфейс приложения реализован при помощи библиотеки tkinter в файле app.py
# app.py:
1) внутри класс Application, который отрисовывает интерфейс при создании экземпляра
2) у класса есть метод open_file, который загружает в сегментатор картинку, а Application ее отрисовывает
3) после разметки используется метод save_file класса Application, который открывает диалоговое окно; после выбора пути для сохранения, будет загружен файл .npz
4) для доступа к размеченной маске:
npzfile = np.load(outpath + '.npz')
npzfile['mask']

5) для доступа к штрихам пользователя
npzfile['user_marks']