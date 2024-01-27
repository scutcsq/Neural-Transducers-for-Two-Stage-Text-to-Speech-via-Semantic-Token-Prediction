import os
import subprocess
place = r'VCTKdata/wav48_silence_trimmed/'
outplace = r'VCTKdata/totalwav/'
dir1 = os.listdir(place)
for i in range(len(dir1)):
    # os.mkdir(os.path.join(outplace, dir1[i]))
    dir2 = os.listdir(os.path.join(place, dir1[i]))
    for j in range(len(dir2)):
        name = dir2[j]
        newname = name[:-5] + '.wav'
        path2 = os.path.join(place, dir1[i], name)
        # savepath = os.path.join(outplace, dir1[i], newname)
        savepath = os.path.join(outplace, newname)
        strings = "ffmpeg -i " + path2 + " -ac 1 -ar 16000 " + savepath
        subprocess.call(strings, shell=True)
