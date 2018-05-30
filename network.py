import zmq


class Network():

    def __init__(self):
        self.socket = None
    

    def create_connection(self, port=55501):
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://*:%s" % port)
        

    def publish_vector(self, topic, le_vec, re_vec):
        le_eye = self.convert_to_str(le_vec)
        re_eye = self.convert_to_str(re_vec)
        msg = topic + " " + le_eye + " " + re_eye
        self.socket.send(msg)


    def publish_coord(self, topic, coord):
        ncoord = self.__convert_to_str(coord)
        msg = topic + " " + ncoord
        self.socket.send(msg)


    def __convert_to_str(self, eye):
        e0 = "{:.8f}".format(eye[0])
        e1 = "{:.8f}".format(eye[1])
        if len(eye) == 3:
            e2 = "{:.8f}".format(eye[2])
            return e0 + ';' + e1 + ';' + e2
        return e0 + ';' + e1
