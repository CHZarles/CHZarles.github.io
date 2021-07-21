import os



# 增加<!--more-->以及将处理post_modify，处理copyright
def delt(filepath):
    with open(filepath ,encoding = 'utf-8' , mode = 'r') as f:
        article = ""
        # 数行数
       
        for line in f:
            if line.find('details>') == -1 and line.find('summary>') == -1 :
                article+=line

            #检测到第一个
            

        # 重新写入文件
        with open(filepath ,encoding = 'utf-8' , mode = 'w') as fw:
            fw.write(article)
            #print(article)








for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        #print(os.path.join(root, name))
        if os.path.join(root, name)[-3:] == ".md" :
            print("work")
            print(os.path.join(root, name))
            delt(os.path.join(root, name))

    for name in dirs:
        #print(os.path.join(root, name))
        #print(os.path.join(root, name)[-4:-1])
        if os.path.join(root, name)[-3:] == ".md":
            print("work")
            print(os.path.join(root, name))
            delt(os.path.join(root, name))





