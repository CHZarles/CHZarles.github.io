import os
import shutil


for root, dirs, files in os.walk(".", topdown=False):
    for name in files:
        if os.path.join(root, name)[-3:] == ".md" :
            
            src = name[:-3]
            os.system('cd %s' %os.getcwd())
            print('cd %s' %os.getcwd())
            os.system("hexo n %s" %src)
            print("hexo n %s" %src)
            shutil.copyfileobj(open(os.path.join(root, name),'rb'), open(r"C:\Users\Charles\blog\source\_posts\%s" %name, 'wb'))
   

    for name in dirs:
        if os.path.join(root, name)[-3:] == ".md":
           src = name[:-3]
           os.system('cd %s' %os.getcwd())
           print('cd %s' %os.getcwd())
           os.system("hexo n %s" %src)
           print("hexo n %s" %src)
           shutil.copyfileobj(open(os.path.join(root, name),'rb'), open(r"C:\Users\Charles\blog\source\_posts\%s" %name, 'wb'))
        



