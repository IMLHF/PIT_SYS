import os

def print_(msg):
  print(msg)
  if not os.path.exists("log"):
    os.mkdir("log")
  with open('log/report.txt','a+') as f:
    f.writelines(str(msg)+'\n')

def print_f(msg,file):
  print(msg)
  if not os.path.exists(file[:file.rfind('/')]):
    os.makedirs(file[:file.rfind('/')])
  with open(file,'a+') as f:
    f.writelines(str(msg)+'\n')
