
import os
from PIL import Image
from ffmpy3 import FFmpeg
 
inputPath = 'projects/pilot/pack_tools/data/test/pilot_test/data'
outputYUVPath = 'projects/pilot/pack_tools/data/test/pilot_test/data_out'
 
piclist = os.listdir(inputPath)
for pic in piclist:
    picpath = os.path.join(inputPath,pic)
    img = cv2.imread(picpath)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
    # img = Image.open(picpath)
    # in_wid,in_hei = img.size
    # out_wid = in_wid//2*2
    # out_hei = in_hei//2*2
    # size = '{}x{}'.format(out_wid,out_hei)  #输出文件会缩放成这个大小
    # purename = os.path.splitext(pic)[0]
    # outname = outputYUVPath + '/' + purename + '_' + size+ '.yuv'
    
    # ff = FFmpeg(inputs={picpath:None},
    #             outputs={outname:'-s {} -pix_fmt yuv420p'.format(size)})
    # print(ff.cmd)
    # ff.run()