### [Интерфейс приложения](./app_segmentation)

### [Сервер](./segmenter_service)


## Запустить приложение с помощью python
После клонирования все команды выполняются из корневой директории проекта

1. Склонировать репозиторий 
    ```shell
    git clone https://github.com/Hoodie-Huuuuu/course_paper.git
    ```
3. Создать venv и активировать его
    ```shell
    python3 -m venv venv
    source ./venv/bin/activate
    ```

4. Скачать зависимости
    ```shell
    pip3 install -r requirements.txt
    ```

5. В отдельном терминале запустить сервер
   
   (Вместо 5005 можно указать любой свободный порт, но тогда в файле [config.py](app_segmentation/config.py) нужно поменять переменную SEGMENTER_PORT = "ваш_порт"
    ```shell
    python3 ./segmenter_service/segmenter_service.py -p 5005
    ```

6. Запустить клиент
   
   ```shell
   python3 ./app_segmentation/app.py
   ```

7. Tkinter по умолчанию установлен в python3, но на всякий можно написать так
   ```shell
   pip3 install tk
   ```



 ## Для запуска сервера в docker
1.  ```shell
    docker compose build
    ```
2. ```shell
   docker compose up
   ```
3. Для запуска клиента нужно проделать шаги 2,3 и 5 вот [тут](#запустить-приложение-с-помощью-python)







## Упаковка приложения

### Подготовка

Устанавливать лучше в виртуальное окружение, так как запаковщик стягивает все пакеты из окружения, а не только те, что используются в проекте. (Возможно есть опция командной строки, но я такой не обнаружил)

Виртуальное окружение можно поставить так

```bash
python -m venv <name_of_venv>
```

Затем его надо активировать

```bash
# for unix and macOS
source <name_of_venv>/bin/activate

# for windows
<name_of_venv>\Scripts\activate.bat
```

Теперь можно проверить какой python и pip у вас сейчас

```bash
# for MacOS
which python
which pip
```
*Должен быть путь вида* 

venv_name/bin/python

venv_name/bin/pip

Теперь необходимо установить все пакеты из файла requirements.txt (убедитесь, что активирована venv)

```bash
pip install -r requirements.txt
```

### Установка

Теперь из виртуального окружения нужно установить пакет-запаковщик

```bash
pip install pyinstaller
```

#### Запаковка для macOS

Эта команда сработала, но с кучей Warnings

Для windows должно быть то же самое
```bash
pyinstaller ./app_segmentation/app.py -F -y --collect-all skimage
```

Флаг -F запаковывает все в один бинарный файл

Флаг --collect-all ищет все подмодули и бинарные файлы указанного пакета




