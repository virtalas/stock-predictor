from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
f = open("myfile.txt", "w")
f.write(dt_string)
f.close()
