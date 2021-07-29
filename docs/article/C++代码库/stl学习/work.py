import os

def work(path):
    with open(path,mode='r') as f:
        res=f.read() # 会将文件的内容由硬盘全部读入内存，赋值给res
        res='```C++\n' + res +  '\n```'
        with open(path[:-4]+'.md','w') as w:
            w.write(res)




for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        #print(os.path.join(root, name))
        if os.path.join(root, name)[-4:] == ".cpp" :
            work(os.path.join(root, name))
      

    for name in dirs:
        #print(os.path.join(root, name))
        #print(os.path.join(root, name)[-4:-1])
        if os.path.join(root, name)[-4:] == ".cpp":
             work(os.path.join(root, name))
           
