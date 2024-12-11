# import os
# import logging
import threading



from flask import Flask, request, jsonify
from sd_utils import ServerConfig,remove_before_first_underscore
# import json
import yaml
from backend.server import *
from munch import Munch
app = Flask(__name__)
servers = {}
clients = {}
lock = threading.Lock()
def construct_munch(loader, node):
    # 构造一个 Munch 对象
    data = loader.construct_mapping(node)
    return Munch(data)
yaml.SafeLoader.add_constructor('!munch.Munch', construct_munch)
@app.route("/")
@app.route("/index")
def index():
    return "Much to do!"

@app.route("/init", methods=["POST"])
def initialize_server():
    args = yaml.safe_load(request.data)
    client_id = args["client_id"]
    server_id = remove_before_first_underscore(args['video_name'])
    if client_id not in clients:
        clients[client_id]=(args,server_id)
        os.makedirs(f"server_temp_{client_id}", exist_ok=True)
        os.makedirs(f"server_temp_{client_id}-cropped", exist_ok=True)
    global servers
    with lock:
        if server_id not in servers:
            logging.basicConfig(
                format="%(name)s -- %(levelname)s -- %(lineno)s -- %(message)s",
                level="INFO")
            servers[server_id] = Server(args, args["nframes"])
            # os.makedirs(f"server_temp_{client_id}", exist_ok=True)
            # os.makedirs(f"server_temp_{client_id}-cropped", exist_ok=True)
            return jsonify({"status": "New Init", "client_id": client_id, "server_id": server_id})
        else:
            servers[server_id].add_client(args)
            # servers[server_id].reset_state(int(args["nframes"]), client_id)
            return jsonify({"status": "Reset", "client_id": client_id, "server_id": server_id})

@app.route("/low/<client_id>", methods=["POST"])
def low_query(client_id):
    file_data = request.files["media"]
    args, server_id = clients[client_id]
    results = servers[server_id].perform_low_query(file_data,client_id)
    return jsonify(results)

@app.route("/high/<client_id>", methods=["POST"])
def high_query(client_id):
    file_data = request.files["media"]
    args, server_id = clients[client_id]
    results = servers[server_id].perform_high_query(file_data,client_id)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)