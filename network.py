import socket


class Network():

    def __init__(self):
        self.socket = None
        self.address = None
    

    def create_connection(self, ip, port=55502):
        if ip == "":
            context = zmq.Context()
            self.socket = context.socket(zmq.PUB)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect('tcp://*:' + port)
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.address = (ip, port)
        

    def publish_vector(self, topic, le_vec, re_vec):
        le_eye = self.__convert_to_str(le_vec)
        re_eye = self.__convert_to_str(re_vec)
        msg = topic + " " + le_eye + " " + re_eye
        self.__send_msg(msg)


    def publish_coord(self, topic, coord):
        ncoord = self.__convert_to_str(coord)
        msg = topic + " " + ncoord
        self.__send_msg(msg)

    
    def __send_msg(self, msg):
        if self.address is None:
            self.socket.send_string(msg)
        else:
            self.socket.sendto(msg.encode(), self.address)


    def __convert_to_str(self, eye):
        e0 = "{:.8f}".format(eye[0])
        e1 = "{:.8f}".format(eye[1])
        if len(eye) == 3:
            e2 = "{:.8f}".format(eye[2])
            return e0 + ';' + e1 + ';' + e2
        return e0 + ';' + e1


    def close(self):
        self.socket.close()
