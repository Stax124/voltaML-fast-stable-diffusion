import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from typing import List, Tuple

from PIL.Image import Image

from core.errors import DimensionError, ModelFailedError
from core.models import ModelHandler
from core.types import Txt2ImgQueueEntry


class ThreadWithReturnValue(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}):
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None: # type: ignore
            self._return = self._target(*self._args,**self._kwargs) # type: ignore
                                                
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

def run(model_handler: ModelHandler, job: Txt2ImgQueueEntry) -> List[Image]:
        return model_handler.generate(job)
    

class Queue:
    def __init__(self) -> None:
        self.jobs: List[Txt2ImgQueueEntry] = list()
        self.running = False
        self.model_handler: ModelHandler = ModelHandler()
        self.thread_pool = ThreadPoolExecutor(max_workers=1)
    
    async def add_job(self, job: Txt2ImgQueueEntry) -> Tuple[List[Image], float]:
        try:
            logging.info(f"Adding job {job.data.id} to queue")
            
            if job.data.width % 8 != 0 or job.data.height % 8 != 0:
                raise DimensionError("Width and height must be divisible by 8")
            
            self.jobs.append(job)
            
            # Wait for the job to be at the front of the queue
            while self.jobs[0] != job:
                await asyncio.sleep(0.1)
            

            start_time = time.time()
            
            # create a new thread
            thread = ThreadWithReturnValue(target=run, args=(self.model_handler, job))

            # start the thread
            thread.start()

            # wait for the thread to finish
            logging.info(f"Job {job} finished")
            
            # get the value returned from the thread
            while thread.is_alive():
                await asyncio.sleep(0.1)
            
            images = thread.join()
            
            self.jobs.pop(0)
            
            if images is None:
                raise ModelFailedError("Model failed to generate image")
        
            deltatime = time.time() - start_time

            return (images, deltatime)

        except Exception as e:
            # Clean up the queue
            if job in self.jobs:
                self.jobs.remove(job)
            
            raise e






