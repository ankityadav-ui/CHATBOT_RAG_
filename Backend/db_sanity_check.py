from dotenv import load_dotenv
import os
import psycopg2

load_dotenv('Backend/.env')
DB_URL=os.getenv('DATABASE_URL')
conn=psycopg2.connect(DB_URL)
cur=conn.cursor()
# create/find user
username='sanity_user'
cur.execute('SELECT id FROM users WHERE username=%s',(username,))
row=cur.fetchone()
if row:
    user_id=row[0]
else:
    cur.execute('INSERT INTO users (username) VALUES (%s) RETURNING id',(username,))
    user_id=cur.fetchone()[0]
    conn.commit()
# insert file
cur.execute('INSERT INTO files (user_id, filename, filepath) VALUES (%s,%s,%s) RETURNING id', (user_id, 'sanity.txt', 'uploads/sanity.txt'))
file_id=cur.fetchone()[0]
conn.commit()
# fetch back
cur.execute('SELECT u.username, f.filename, f.filepath FROM files f JOIN users u ON f.user_id=u.id WHERE f.id=%s',(file_id,))
print(cur.fetchone())
cur.close()
conn.close()
