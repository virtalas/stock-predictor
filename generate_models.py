from datetime import datetime
import db
from db import Model

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

model = Model('AAPL', dt_string) # Demo
db.save(model)
model2 = Model('AAPL2', 'fjseifj39jfsfk') # Demo
db.save(model2)
