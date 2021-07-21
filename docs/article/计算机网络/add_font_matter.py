import os



# 增加<!--more-->以及将处理post_modify，处理copyright
def modify(filepath):

    with open(filepath ,encoding = 'utf-8' , mode = 'r') as f:
        article = ""
        # 数行数
        tmp = 0
        #两个用来检测代码块的标记
        flag_1 = False
        flag_2 = False
        checked = False
        while True:
            tmp +=1
            checked = False
            line = f.readline()

            #检测到第一个
            if "```" in line and flag_2 == False and flag_1 == False :
                flag_1 = True
                print("检测到第一个,行"+str(tmp)+" :"+line)
                checked= True

            #检查到第二个
            if "```" in line and flag_1 == True and flag_2 == False and checked == False:
                flag_2 = True
                flag_1 = False
                print("检测到第二个,行"+str(tmp)+" :"+line)

            #经过了第二个标记，清空标记
            if not ("```" in line) and flag_1 == False and flag_2 == True :
                flag_2 = False
                print("清空标记，行"+str(tmp)+" :" + line)

            if len(line) == 0:
                break
            article += line

            if tmp > 25 and flag_1 == False and flag_2 == False:
                article += "<!--more-->\n"
                #锁住
                flag_2 = True
                flag_1 = True


        # 获取要修改的位置
        idx1 = article.find('''Post modified:''')
        idx2 = article.find('''copyright: true''')
        # 处理post_modify
        if idx1 != -1:
            new_article = ""
            new_article = article[0:idx1]
            article = new_article

        # 处理copyright
        if idx2 != -1:
            article = article.replace('''copyright: true''' ,'''copyright: false''' ,1)

        # 重新写入文件
        with open(filepath ,encoding = 'utf-8' , mode = 'w') as fw:
            fw.write(article)
            #print(article)


#增加标签
def addtag(filepath,tagname,word):

    
    with open(filepath ,encoding = 'utf-8' , mode = 'r') as f:
        article = ""
        while True:
            line = f.readline()
            if len(line)==0:
                break
            else:
                article+=line
        
        #定位,因为有两个--- 定位下面那个
        idx = article.find('''---''',3)

        #构造字符串co
        newtag = tagname + ": " + word + "\n"
        new_article = article[0:idx] + newtag + article[idx:]
        
    # 重新写入文件
    with open(filepath ,encoding = 'utf-8' , mode = 'w') as fw:
        fw.write(new_article)
        #print(article)



tagname=input("输入tagname:")
word=input("输入参数:")

for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        #print(os.path.join(root, name))
        if os.path.join(root, name)[-3:] == ".md" :
            print("work")
            print(os.path.join(root, name))
            addtag(os.path.join(root, name),tagname,word)

    for name in dirs:
        #print(os.path.join(root, name))
        #print(os.path.join(root, name)[-4:-1])
        if os.path.join(root, name)[-3:] == ".md":
            print("work")
            addtag(os.path.join(root, name),tagname,word)





