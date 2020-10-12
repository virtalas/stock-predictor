import os
import psycopg2

DATABASE_URL = os.environ['DATABASE_URL']

class Model:
  def __init__(self, name, data, model_id=None):
    self.model_id = model_id
    self.name = name
    self.data = data

def connect():
  connection = psycopg2.connect(DATABASE_URL)
  cursor = connection.cursor()
  return connection, cursor

# Init DB and connection.
conn, cur = connect()
create_table_query = '''
  CREATE TABLE IF NOT EXISTS Model (
    id serial PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    data TEXT
  );
'''
cur.execute(create_table_query)
conn.commit()
conn.close()

def save(model):
  connection, cursor = connect()
  cursor.execute('SELECT id FROM Model WHERE name = %s;', [model.name, ])
  found = cursor.fetchone()
  if found:
    # Update
    cursor.execute('UPDATE Model SET data = %s WHERE id = %s;', [model.data, found[0]])
    connection.commit()
    print('Updated ', model.name)
  else:
    # Insert
    cursor.execute('INSERT INTO Model (name, data) VALUES (%s, %s);', [model.name, model.data])
    connection.commit()
    print('Inserted ', model.name)
  connection.close()

def fetch_all():
  connection, cursor = connect()
  cursor.execute('SELECT * FROM Model;')
  models = []
  for row in cursor.fetchall():
    models.append(Model(row[1], row[2]))
  connection.close()
  return models
