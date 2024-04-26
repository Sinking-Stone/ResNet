import numpy as np

# 假设 byte_data 是您提供的字节串数据
data_file = "../data/example1.txt"



with open(data_file, 'r') as f:
    lines = f.readlines()
    # print(len(lines))
    for line in lines:
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

        # for num in data_float:
        #     if num < -1 or num > 1:
        #         print(num)
