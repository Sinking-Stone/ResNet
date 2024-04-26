import numpy as np
import matplotlib.pyplot as plt

# 假设 byte_data 是您提供的字节串数据
data_file = "../data/example.txt"


# 存放所有数据的频率
alldata = []

with open(data_file, 'r') as f:
    lines = f.readlines()


    for line in lines:
        
        
        linelist=[]
        for i in range(25):
            linelist.append(0)

        # 将字节串解码为字符串
        data_str = line
        pos=data_str.find('b')
        print(pos)

        # 去掉字符串开头的 'b' 和结尾的逗号
        data_str = data_str[pos+2:-1]

        # 用逗号替换字符串中的空格，以便能够分割数据
        data_str = data_str.replace('[', ' ')
        data_str = data_str.replace(']', ' ')
        data_str = data_str.replace('\\n', ' ')
        data_str = data_str.replace('\'', ' ')


        # 将字符串分割为单独的浮点数表示
        data_list = data_str.split(',')

        print(len(data_list))
        # 将字符串列表转换为浮点数列表
        for i in range(len(data_list)):
            data_list[i]=data_list[i].strip()

        data_float = [float(x) for x in data_list if x]

        for num in data_float:
            if num < -10:
                linelist[0]+=1
            elif num < -9:
                linelist[1]+=1
            elif num < -8:
                linelist[2]+=1
            elif num < -7:
                linelist[3]+=1
            elif num < -6:
                linelist[4]+=1
            elif num < -5:
                linelist[5]+=1
            elif num < -4:
                linelist[6]+=1
            elif num < -3:
                linelist[7]+=1
            elif num < -2:
                linelist[8]+=1
            elif num < -1:
                linelist[9]+=1
            elif num < 0:
                linelist[10]+=1
            elif num < 1:
                linelist[11]+=1
            elif num < 2:
                linelist[12]+=1
            elif num < 3:
                linelist[13]+=1
            elif num < 4:
                linelist[14]+=1
            elif num < 5:
                linelist[15]+=1
            elif num < 6:
                linelist[16]+=1
            elif num < 7:
                linelist[17]+=1
            elif num < 8:
                linelist[18]+=1
            elif num < 9:
                linelist[19]+=1
            elif num < 10:
                linelist[20]+=1
            elif num < 11:
                linelist[21]+=1
            elif num < 12:
                linelist[22]+=1
            elif num < 13:
                linelist[23]+=1
            else:
                linelist[24]+=1
        alldata.append(linelist)

# print(alldata)
pos=0
for num in alldata:
    print(pos)
    pos+=1
    print(num)

# plt.hist(alldata[0])
# plt.title('ReLU data distribution')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.xticks(["<-10","-10~-9","-9~-8","-8~-7","-7~-6","-6~-5","-5~-4","-4~-3","-3~-2","-2~-1","-1~0","0~1","1~2","2~3","3~4","4~5","5~6","6~7","7~8","8~9","9~10","10~11","11~12","12~13",">13"])
# plt.show()