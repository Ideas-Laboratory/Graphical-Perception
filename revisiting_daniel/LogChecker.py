import os
import time
import random
import util.pyutils as upy
from util import Config
import util.shareCode
from util.shareCode import programInit, globalCatch
from . import TaskScheduler
from TaskScheduler import TaskInfo



def main():
    config = programInit()
    tasks = config.tasks
    self.tasks={}
    for x in config.tasks:
        self.tasks[x.id] = TaskInfo(**Config.obj2dic(x))



if __name__=="__main__":
    globalCatch(main)