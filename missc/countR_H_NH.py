import glob

rider = 0
helmet = 0
no_helmet = 0

for i in glob.glob(r'C:\Users\Dev Agarwal\Desktop\Linear Regression\train\train/*.txt'):
    with open(i) as f:
        lines = f.readlines() # list containing lines of file
        columns = [] # To store column names

    for i in range(len(lines)):
        x = lines[i][0]
        if int(x) == 0:
            rider+=1
        if int(x) == 1:
            helmet+=1
        if int(x) == 2:
            no_helmet+=1

print('rider:',rider)
print('helmet:',helmet)
print('no_helmet:',no_helmet)
