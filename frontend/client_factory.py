from frontend.client import Client
from frontend.roi_client import RoIClient
class ClientFactory:
    @staticmethod
    def get_client(config, client_id, server=None, hname=None):
        if hname:

            return Client(hname,config, client_id, server_handle=hname)
        else:

            return Client( hname,config,client_id, server_handle=server)

    @staticmethod
    def get_roi_client(config, client_id, server=None, hname=None):
        if hname:

            return RoIClient(hname,config ,client_id, server_handle=hname)
        else:

            return RoIClient(hname,config ,client_id, server_handle=server)