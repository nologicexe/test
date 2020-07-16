import sqlite3
import os

import random
from datetime import datetime





class Database():
    
    def __init__(self, dbname, images_dir):
        
        self.dbname = dbname
        self.images_dir = images_dir
#        self.conn = sqlite3.connect(dbname)
        with sqlite3.connect(self.dbname) as conn:

            conn.execute("""CREATE TABLE IF NOT EXISTS votes(
                                   image_id  TEXT, 
                                   user_id   INTEGER, 
                                   type_id   INTEGER, 
                                   answer    INTEGER, 
                                   created   TEXT,
                            PRIMARY KEY (image_id, user_id, type_id))
                            WITHOUT ROWID""")

            
            
    def select(self, user_id, type_id, n_images):
        
        with sqlite3.connect(self.dbname) as conn:

            query = conn.execute("""SELECT image_id
                                          FROM votes
                                         WHERE image_id > ''
                                           AND user_id > 0
                                           AND type_id = (?)""", (type_id,))

            voted = set(q[0] for q in query)

#            query = conn.execute("""SELECT image_id
#                                          FROM votes
#                                         WHERE image_id > ''
#                                           AND user_id = (?)
#                                           AND type_id = (?)""", (user_id, type_id))
#
#            voted = set(q[0] for q in query)

        all_images = (os.path.basename(f) for f in os.listdir(self.images_dir))
        all_images = set(f[:-4] for f in all_images)
        all_images.difference_update(voted)
        
        try:
            images = random.sample(all_images, n_images)
        except ValueError:
            images = all_images

        return list(images)
        
        
        
    def insert(self, user_id, type_id, images, votes):
        
        now = str(datetime.now())
        created, *_ = now.partition('.')

        try:
            with sqlite3.connect(self.dbname) as conn:
                
                [conn.execute("""INSERT INTO votes 
                    VALUES (?, ?, ?, ?, ?)""", (img, user_id, type_id, vote, created))
                                    for img, vote in zip(images, votes) if img]

        except sqlite3.IntegrityError:
            pass
#        return aaa
    


if __name__ == '__main__':

    images_dir = 'images/'
    
    if os.path.exists(images_dir):
        
        db = Database('test.db', images_dir)
        
        sel = db.select(1, 1, 4) # := ))
        print(sel)

        vot = list(range(len(sel)))
        db.insert(1, 1, sel, vot)

        sel = db.select(1, 1, 4) 
        print(sel)

        sel = db.select(0, 1, 4) 
        print(sel)

#        sel = db.select(1, 0, 4) 
#        print(sel)
#































