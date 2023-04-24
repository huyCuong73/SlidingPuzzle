from flask_restful import Api,Resource
import test

from flask import request, jsonify
import json

class Controller(Resource):

    def post(self):
        board = request.json
        data = test.solvePuzzle(board["dimensions"], board["board"], board["h"])
        return jsonify(data)
        

def router(app):
    route = Api(app)
    route.add_resource(Controller, '/solve', endpoint="solve")
