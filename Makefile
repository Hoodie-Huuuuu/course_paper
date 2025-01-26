

all:
	start-server start-app

start-app:
	python ./app_segmentation/app.py

start-server:
	python ./segmenter_service/segmenter_service.py -p 5005