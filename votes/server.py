from aiohttp import web

from random import randint

#import numpy as np
#import os

#from base64 import b64decode, b64encode
#
#from skimage.io import imsave
#from json import dumps

from database import Database
from pages import index, votes






class Handler:

    solvers = [] #solvers # select solver at the time of solve request


    ready = True
    types = 8
    images_on_page = 8
    images_per_line = 4
    images_dir = 'images/'
    dbname = 'votes.db'
    DB = Database(dbname, images_dir)



#    def __init__(self):
#        self.images = []



    async def index(self, request):
        
        return web.Response(body=index(),
                            headers={'Content-Type': 'text/html'}) #, 'Access-Control-Allow-Origin': '*'})




    async def save_votes(self, request):
        
        data = await request.post()

        user_id = data.get('user_id', -1)
        type_id = data.get('type_id', -1)

        images = [data.get(f'image_{i}') for i in range(Handler.images_on_page)]
        answers = [data.get(f'answer_{i}') for i in range(Handler.images_on_page)]
        
        Handler.DB.insert(user_id, type_id, images, answers)

        return await self.new_votes(request, user_id)



    async def new_votes(self, request, user_id=None):

        if user_id is None:
            user_id = request.query.get('user_id')

        type_id = randint(0, Handler.types - 1)

        images = Handler.DB.select(user_id, type_id, Handler.images_on_page)
        
        if images:
            body = votes(user_id, type_id, images, per_line=Handler.images_per_line)
    
            return web.Response(body=body,
                                headers={'Content-Type': 'text/html'}) #, 'Access-Control-Allow-Origin': '*'})
        else:
            return web.Response(text='почти всё размечено, обновите страницу',
                                headers={'Content-Type': 'text/html'})



handler = Handler()

app = web.Application()
app.add_routes([web.get('/', handler.index),
                web.get('/vote', handler.new_votes), 
                web.post('/vote', handler.save_votes),
                web.static('/images', handler.images_dir)])
    

web.run_app(app, port=8888)








