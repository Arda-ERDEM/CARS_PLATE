import datetime
from time import time

def write(message):
    try:
        f = open("data.txt", "a+")
        f.seek(0)
        content = f.read()
        if message not in content:
            now = datetime.datetime.now()
            f.write(f"{message} {now}\n")
        f.close()
    except Exception as e:
        print(f"Kaydetme hatası: {e}")