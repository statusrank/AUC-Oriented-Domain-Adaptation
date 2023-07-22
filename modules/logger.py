# -*- coding: utf-8 -*-
'''
CopyRight: Shilong Bao
Email: baoshilong@iie.ac.cn
'''
import logging
import getpass
import sys
import os 

class CompleteLogger(object):
    def __init__(self, init_file=None):
        user=getpass.getuser()
        self.logger=logging.getLogger(user)
        self.logger.setLevel(logging.DEBUG)
        if init_file is None:
            logFile=sys.argv[0][0:-3] + '.log'
        else:
            logFile = init_file + '.log'
        formatter=logging.Formatter('%(asctime)-12s %(levelname)-8s %(name)-10s %(message)-12s')

        logHand=logging.FileHandler(logFile,encoding="utf8")
        logHand.setFormatter(formatter)
        logHand.setLevel(logging.INFO)#只记录错误

        logHandSt=logging.StreamHandler()
        logHandSt.setFormatter(formatter)

        self.logger.addHandler(logHand)
        self.logger.addHandler(logHandSt)

        self.checkpoint_directory = 'default_save'
        if init_file is not None:
            self.checkpoint_directory = init_file
        
        if not os.path.exists(self.checkpoint_directory):
            os.makedirs(self.checkpoint_directory)
        
        # self.checkpoint_name = os.path.splitext(logFile)[-2]

    def debug(self,msg):
        self.logger.debug(msg)
    def info(self,msg):
        self.logger.info(msg)
    def warn(self,msg):
        self.logger.warning(msg)
    def error(self,msg):
        self.logger.error(msg)
    def critical(self,msg):
        self.logger.critical(msg)

    def get_checkpoint_path(self, name=None):
        """
        Get the full checkpoint path.
        Args:
            name (optional): the filename (without file extension) to save checkpoint.
                If None, when the phase is ``train``, checkpoint will be saved to ``{epoch}.pth``.
                Otherwise, will be saved to ``{phase}.pth``.
        """
        if name is None:
            name = self.checkpoint_name
        else:
            # name = self.checkpoint_name + '_' + name
            name = 'model' + '_' + name
        name = str(name)
        return os.path.join(self.checkpoint_directory, name + ".pth")

if __name__=='__main__':
    mylog=MyLog()
    mylog.debug("I'm debug")
    mylog.info("I'm info")
    mylog.warn("I'm warning")
    mylog.error("I'm error")
    mylog.critical("I'm critical")