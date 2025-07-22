import sqlite3

conn = sqlite3.connect('instance/driving_behavior.db')
cur = conn.cursor()

cur.execute("SELECT * FROM driving_log ORDER BY timestamp DESC LIMIT 20")
rows = cur.fetchall()

print("Last 20 predictions:")
for r in rows:
    print(r)