'''
To record video from drone by each frame:
run example: python main.py --save_image True --vision_attributes True
To record video from drone in mp4:
run example: python main.py --save_video True --vision_attributes True
To record video from GUI in mp4:
run example: python main.py --record_video True
'''

from src.pipeline import Pipeline


if __name__ == '__main__':
    runer = Pipeline()
    runer.run()
