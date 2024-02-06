import os
import textgrid
spk = 0
train_result = []
valid_result = []
inplace = r'LibriTTS/TextGrid/'
dir1 = os.listdir(inplace)
valid_dir1 = os.listdir(r'LibriTTS/test-clean')
valid_dir2 = os.listdir(r'LibriTTS/LibriTTS/test-other')
valid_dir1.extend(valid_dir2)
print('valid_speakers: ', valid_dir1)
for i in range(len(dir1)):
    if(dir1[i] not in valid_dir1):
    # if('clean' in dir1[i]):
        dir2 = os.listdir(os.path.join(inplace, dir1[i]))
        for j in range(len(dir2)):
            dir3 = os.listdir(os.path.join(inplace, dir1[i], dir2[j]))
            for k in range(len(dir3)):
                # if(dir3[k][-4:] != '.txt'):
                #     dir4 = os.listdir(os.path.join(inplace, dir1[i], dir2[j], dir3[k]))
                    # for l in range(len(dir4)):
                        # if dir3[k][:-9] in eihey:
                
                txt1 = os.path.join(inplace, dir1[i], dir2[j], dir3[k])
                tg = textgrid.TextGrid.fromFile(txt1)
                length = len(tg[1])
                txt2 = dir3[k][:-9] + "/Libritts/" + "{:05d}".format(spk) + '/'
                
                
                
                for m in range(length):
                    if(m == 0):
                        if(tg[1][m].mark != "" and tg[1][m].mark!="sil" and tg[1][m].mark!='sp'):
                            txt3 = "sil|0\t"
                            txt3 += tg[1][m].mark + '|' + str(tg[1][m].maxTime - tg[1][m].minTime) + "\t"
                        else:
                            txt3 = "sil|"
                            txt3 += str(tg[1][m].maxTime - tg[1][m].minTime) + "\t"
                    else:
                        temp_txt = tg[1][m].mark
                        if(temp_txt == ""):
                            temp_txt = "sp"
                        # print(temp_txt)
                        temp_txt += "|" + str(tg[1][m].maxTime - tg[1][m].minTime) + "\t"
                        txt3 += temp_txt

                        if(m == length -1 and (tg[1][m].mark != "" and tg[1][m].mark != "sil" and tg[1][m].mark != 'sp')):
                            txt3 += "sp|0\t"
                txt2 += txt3
                train_result.append(txt2)
    else:
        dir2 = os.listdir(os.path.join(inplace, dir1[i]))
        for j in range(len(dir2)):
            dir3 = os.listdir(os.path.join(inplace, dir1[i], dir2[j]))
            for k in range(len(dir3)):
                # if(dir3[k][-4:] != '.txt'):
                #     dir4 = os.listdir(os.path.join(inplace, dir1[i], dir2[j], dir3[k]))
                    # for l in range(len(dir4)):
                        # if dir3[k][:-9] in eihey:
                
                txt1 = os.path.join(inplace, dir1[i], dir2[j], dir3[k])
                tg = textgrid.TextGrid.fromFile(txt1)
                length = len(tg[1])
                txt2 = dir3[k][:-9] + "/Libritts/" + "{:05d}".format(spk) + '/'
                
                
                
                for m in range(length):
                    if(m == 0):
                        if(tg[1][m].mark != "" and tg[1][m].mark!="sil" and tg[1][m].mark!='sp'):
                            txt3 = "sil|0\t"
                            txt3 += tg[1][m].mark + '|' + str(tg[1][m].maxTime - tg[1][m].minTime) + "\t"
                        else:
                            txt3 = "sil|"
                            txt3 += str(tg[1][m].maxTime - tg[1][m].minTime) + "\t"
                    else:
                        temp_txt = tg[1][m].mark
                        if(temp_txt == ""):
                            temp_txt = "sp"
                        # print(temp_txt)
                        temp_txt += "|" + str(tg[1][m].maxTime - tg[1][m].minTime) + "\t"
                        txt3 += temp_txt

                        if(m == length -1 and (tg[1][m].mark != "" and tg[1][m].mark != "sil" and tg[1][m].mark != 'sp')):
                            txt3 += "sp|0\t"
                txt2 += txt3
                valid_result.append(txt2)
    spk += 1
print('libitts finish!!')
with open('libritts_train.txt', 'w') as f:
    for line in train_result:
        f.write(line + '\n') 

with open('libritts_valid.txt', 'w') as f:
    for line in valid_result:
        f.write(line + '\n')
