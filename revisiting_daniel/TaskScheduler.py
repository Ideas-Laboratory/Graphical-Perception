from multiprocessing import Process
import subprocess
import os
import logging
import util
import time
import random
import util.pyutils as upy
from util import Config
import util.shareCode
from util.shareCode import programInit, globalCatch

# A simple Task Scheduler (support piority and DAG)

class TaskInfo:
    def __init__(self,cmd,id,res,waitID=[],rerun=False,weight=0):
        self.cmd=cmd
        self.id=id
        self.res=res
        self.waitID=waitID
        self.done=False
        self.rerun=rerun # run this task again
        self.notifyID=[] # notify these tasks after complete
        self.weight=weight

    def checkFinish(self,tagFolder):
        return os.path.exists(os.path.join(tagFolder,str(self.id))) and not self.rerun

    def setFinish(self,tagFolder,info=""):
        if not self.checkFinish(tagFolder):
            pth=os.path.join(tagFolder,str(self.id))
            f = open(pth,"w")
            f.write(self.cmd+"\n")
            f.write(info+"\n")
            f.close()

class TaskProcess(Process):
    def __init__(self,command,id,gpuIndex):
        super(TaskProcess,self).__init__()
        self.command=command
        self.id=id
        self.gpuIndex=gpuIndex

    def run(self):
        ret = subprocess.run(self.command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        #ret = subprocess.run(self.command,shell=True)
        if "<CRITICAL> 	Exception Occurs!" in str(ret.stdout) or "<CRITICAL> 	Exception Occurs!" in str(ret.stderr):
            print("[                   ]<     ><CRITICAL> -"*10)
            print("[                   ]<     ><CRITICAL> Detect Errors in the process")
            print("[                   ]<     ><CRITICAL> ID : "+self.id)
            print("[                   ]<     ><CRITICAL> Cmd: "+self.cmd)
            print("[                   ]<     ><CRITICAL> -"*10)
        
        #print(ret)

class TaskScheduler:
    def __init__(self,config):

        #process tasks
        self.config=config
        os.makedirs(config.tagFolder,exist_ok=True)
        self.tasks={}
        for x in config.tasks:
            self.tasks[x.id] = TaskInfo(**Config.obj2dic(x))
        for k,v in self.tasks.items():
            for waitID in v.waitID:
                self.tasks[waitID].notifyID.append(k)

        self.canRun = []
        for k,v in self.tasks.items():
            if len(v.waitID)==0:
                self.canRun.append((v.weight,k))

        self._lastTime=time.time()
        self._updatedStat=False


    def run(self):
        complete=0
        totalCount=len(self.tasks)
        process=[]
        #gpuUsage=[0]*len(self.config.gpuComputeCapbility)
        gpuLeftCap = [x for x in self.config.gpuComputeCapbility]
        gpuid = self.config.useGpu
        diskLeftCap = self.config.diskCapbility
        changeCanRun=False
        while complete<totalCount:
            time.sleep(0.05)
            if time.time()-self._lastTime>30 and not self._updatedStat:
                logging.info("--- Current State ---")
                s="GPU (%d) Device | Left >"%len(gpuLeftCap)
                for i,v in enumerate(gpuLeftCap):
                    s+= "%3d (%3d) |"%(v,self.config.gpuComputeCapbility[i])
                logging.info(s)
                logging.info("Disk Left %d"%diskLeftCap)
                logging.info("Total Tasks %4d | Complete %4d | Running %4d | Avaliable %4d"%(totalCount,complete,len(process),len(self.canRun)))
                logging.info("Avaliable Tasks (Show Top 5):")
                for i in range(min(5,len(self.canRun))):
                    logging.info("  > %4d | %s"%(self.canRun[i][0],self.canRun[i][1]))
                logging.info("")
                self._updatedStat=True
            # check if task is completed
            processTemp=[]
            for proc in process:
                if not proc.is_alive():
                    id = proc.id
                    task = self.tasks[id]
                    useGpuIndex=proc.gpuIndex
                    gpuLeftCap[useGpuIndex]+=task.res["gpu"]
                    diskLeftCap+=task.res["disk"]
                    task.setFinish(self.config.tagFolder)
                    complete+=1
                    logging.info("End Task %s, Left Tasks %d"%(str(task.id),totalCount-complete))
                    # remove related dependency information
                    for noteID in task.notifyID:
                        taskDep = self.tasks[noteID]
                        taskDep.waitID.remove(id)
                        if len(taskDep.waitID)==0: # previous tasks are all done
                            self.canRun.append((taskDep.weight,taskDep.id))
                            logging.info("Task %s is enabled"%str(taskDep.id))
                            changeCanRun=True
                else:
                    processTemp.append(proc)

            process = processTemp

            # execute task
            if len(self.canRun)==0:
                continue
            #id = self.canRun[random.randint(0,len(self.canRun)-1)][1]
            id = self.canRun[0][1]
            task = self.tasks[id]
            if task.checkFinish(self.config.tagFolder): # if already finished
                task.done=True
                complete+=1
                logging.info("Already Completed Task %s, Left Tasks %d"%(str(task.id),totalCount-complete))
                for noteID in task.notifyID:
                    taskDep = self.tasks[noteID]
                    taskDep.waitID.remove(id)
                    if len(taskDep.waitID)==0: # previous tasks are all done
                        self.canRun.append((taskDep.weight,taskDep.id))
                        logging.info("Task %s is enabled"%str(taskDep.id))
                self.canRun.remove((task.weight,id))
                changeCanRun=True
                self._updatedStat=False
                self._lastTime=time.time()
            else:
                # check resources
                if diskLeftCap<task.res["disk"]:
                    continue
                useGpuIndex=None
                for i,gpu in enumerate(gpuLeftCap):
                    if gpu>=task.res["gpu"]:
                        useGpuIndex=i
                        break
                if useGpuIndex is None:
                    continue
                # deliver gpu resources
                gpuCmd = " --gpu %d"%gpuid[useGpuIndex]
                gpuLeftCap[useGpuIndex]-=task.res["gpu"]
                diskLeftCap-=task.res["disk"]
                proc=None
                if " --gpu %d" in task.cmd:
                    proc = TaskProcess(task.cmd.strip()%gpuid[useGpuIndex],task.id,useGpuIndex)
                else:
                    proc = TaskProcess(task.cmd.strip()+gpuCmd,task.id,useGpuIndex)
                logging.info("Start Task %s, Left Tasks %d"%(str(task.id),totalCount-complete))
                logging.info("Assign GPU %d, Usage %4d, Compute Resource [%4d -> %4d]"%(gpuid[useGpuIndex],task.res["gpu"],gpuLeftCap[useGpuIndex]+task.res["gpu"],gpuLeftCap[useGpuIndex]))
                logging.info("Disk Usage %4d, Compute Resource [%4d -> %4d]"%(task.res["disk"],diskLeftCap+task.res["disk"],diskLeftCap))
                logging.info("Run %s"%str(task.cmd))
                proc.start()
                process.append(proc)
                self.canRun.remove((task.weight,id))
                changeCanRun=True
                self._updatedStat=False
                self._lastTime=time.time()
            if changeCanRun:
                self.canRun = sorted(self.canRun,reverse=True)
                changeCanRun=False
                
        pass






def main():
    config = programInit()

    scheduler = TaskScheduler(config)
    scheduler.run()

if __name__=="__main__":
    globalCatch(main)

