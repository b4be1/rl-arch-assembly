import logging
import time
from xmlrpc.client import ServerProxy


class SchunkGripper:
    def __init__(self, robot_ip="192.168.1.102", robot_xmlrpc_port="40408", gripper_ip="192.168.1.253"):
        self._logger = logging.Logger("schunk gripper interface")
        self._proxy = ServerProxy("http://{}:{}/".format(robot_ip, robot_xmlrpc_port))
        self._proxy.set_Ip(gripper_ip)
        connection_id = self._proxy.connect(True)
        if connection_id == 0:
            self._logger.log(logging.ERROR, "Failed to connect to the gripper")
        time.sleep(1.0)
        self._proxy.startModbusTCP()
        time.sleep(3.0)
        self._proxy.cmdAcknowledge()
        self._proxy.cmdAcknowledge()
        time.sleep(2.0)
        referencing_sucessful = self._proxy.cmdReferencing()
        if not referencing_sucessful:
            self._logger.log(logging.ERROR, "Referencing failed")

    def grip(self):
        return self._proxy.cmdGrip()

    def release(self):
        return self._proxy.cmdRelease()
