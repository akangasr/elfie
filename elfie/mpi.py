import sys
import time
import itertools
from enum import IntEnum
from collections import deque

import numpy as np
from mpi4py import MPI
from elfi.client import set_client, ClientBase

import logging
logger = logging.getLogger(__name__)


class Tag(IntEnum):
    """ MPI communication tags """
    FREE  = 0  # Worker indicating that it is free to do work
    RECV  = 1  # Worker will next receive task object
    TASK  = 2  # Task object to be run
    READY = 3  # Returned ready task object
    END   = 4  # Worker will next terminate


def mpi_main(master_main, *args, **kwargs):
    """ Every MPI process should start by calling this function.

    For Master:
    - Sets MPIClient as default for ELFI
    - Runs master_main(*args, **kwargs)
    - Terminates workers

    For Workers:
    - Starts worker busy loop
    - Terminates process after loop ends

    Parameters
    ----------
    master_main : callable(*args, **kwargs)
        The main function of the master process
    """
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        status = MPI.Status()
    except Exception as e:
        logger.critical("MPI initialization error!")
        tb = traceback.format_exc()
        logger.critical(tb)
        sys.exit()

    try:
        dummy = np.zeros(1)
        if rank == 0:
            client = MPIClient(comm, size, status, dummy)
            set_client(client)
            master_main(*args, **kwargs)
            client.end()
        else:
            _worker_loop(comm, rank, size, status, dummy)
    except WorkerFinishedException as e:
        pass
    except:
        tb = traceback.format_exc()
        logger.critical(tb)
        comm.Abort()
        sys.exit()


class WorkerFinishedException(Exception):
    def __init__(self, worker_id):
        self.worker_id = worker_id

    def __str__(self):
        return "Worker {} finished".format(self.worker_id)


def _worker_loop(comm, rank, size, status, dummy):
    """ MPI worker busy loop """
    logger.debug("MPI WORKER {}: Setup done".format(rank))
    while True:
        logger.debug("MPI WORKER {}: Free..".format(rank))
        comm.Send(dummy, dest=0, tag=Tag.FREE)
        comm.Recv(dummy, source=0, tag=MPI.ANY_SOURCE, status=status)
        tag = status.Get_tag()

        if tag == Tag.RECV:
            task = comm.recv(None, source=0, tag=Tag.TASK)
            logger.debug("MPI WORKER {}: Executing task {}..".format(rank, task))
            task.run()
            comm.send(task, dest=0, tag=Tag.READY)
        elif tag == Tag.END:
            break
        else:
            logger.debug("MPI WORKER {}: Received unexpected command: '{}'!".format(rank, tag))
    logger.debug("MPI WORKER {}: Terminating.".format(rank))
    raise WorkerFinishedException(rank)


class MPITask():

    def __init__(self, idx, kallable, args, kwargs):
        self.idx = idx
        self.kallable = kallable
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.created_time = time.time()
        self.start_time = None
        self.end_time = None
        self.run_duration = None

    def run(self):
        self.start_time = time.time()
        self.result = self.kallable(*self.args, **self.kwargs)
        self.end_time = time.time()
        self.run_duration = self.end_time - self.start_time

    def __str__(self):
        return "{}".format(self.idx)


class MPIClient(ClientBase):
    """ Client for parallelizing ELFI computation over MPI.
        Requires that the process in run in an MPI environment.

        Parameters
        ----------
        comm : MPI communicator object
        size : size of MPI process group
        status : MPI status object
        dummy : data to use as dummy message payload
        wait_delay: polling time when waiting for tasks to complete (in seconds)
    """
    # TODO: should ready_tasks be emptied at some point?

    def __init__(self, comm, size, status, dummy, wait_delay=2.0):
        self.comm = comm
        self.size = size
        self.status = status
        self.dummy = dummy
        self.wait_delay = wait_delay
        self.waiting_tasks = deque()
        self.pending_tasks = {}  # {worker (int) : task (obj)}
        self.ready_tasks = {}  # {idx (int) : task (obj)}
        self._idx = itertools.count()
        self.pool = set()
        self._find_workers()
        logger.debug("MPI MASTER: Setup done")

    def end(self):
        """ instruct workers to terminate """
        logger.debug("MPI MASTER: Terminating workers")
        for j in range(1,self.size):
            self.comm.Send(self.dummy, dest=j, tag=Tag.END)

    @property
    def num_cores(self):
        """ number of MPI workers """
        return self.size

    @property
    def is_full(self):
        """ false if there are less tasks than workers """
        return len(self.pending_tasks) + len(self.waiting_tasks) >= self.num_cores

    def _run_tasks(self):
        """ distribute tasks (FIFO order) to free workers """
        self._find_workers()
        while len(self.pool) > 0 and len(self.waiting_tasks) > 0:
            worker = self.pool.pop()
            task = self.waiting_tasks.popleft()
            self.comm.Send(self.dummy, dest=worker, tag=Tag.RECV)
            self.comm.send(task, dest=worker, tag=Tag.TASK)
            logger.debug("MPI MASTER: Executing task {} at Worker {}".format(task, worker))
            self.pending_tasks[worker] = task

    def _wait_for_next_ready(self):
        """ wait until next pending task completes """
        self._run_tasks()
        while len(self.pending_tasks) > 0:
            if self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=Tag.READY, status=self.status):
                worker = self.status.Get_source()
                del self.pending_tasks[worker]
                task = self.comm.recv(None, source=worker, tag=Tag.READY)
                logger.debug("MPI MASTER: Received task {} from worker {}".format(task.idx, worker))
                self.ready_tasks[task.idx] = task
                self._run_tasks()
                break
            time.sleep(self.wait_delay)
            logger.debug("MPI MASTER: Waiting for any task to complete")

    def _find_workers(self):
        """ gather free workers to pool """
        while self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=Tag.FREE, status=self.status):
            worker = self.status.Get_source()
            self.pool.add(worker)
            self.comm.Recv(self.dummy, source=worker, tag=Tag.FREE)
            logger.debug("MPI MASTER: Added Worker {} to pool".format(worker))

    def apply(self, kallable, *args, **kwargs):
        """ add job to list and return job index immedately """
        idx = self._idx.__next__()
        task = MPITask(idx, kallable, args, kwargs)
        self.waiting_tasks.append(task)
        logger.debug("MPI MASTER: Added task {} to waitlist".format(idx))
        self._run_tasks()
        return idx

    def apply_sync(self, kallable, *args, **kwargs):
        """ add job to list and wait until finished, return result """
        idx = self.apply(kallable, *args, **kwargs)
        return self.get(idx)

    def get(self, idx):
        """ wait until this job is finished, return result """
        idx, result = self.wait_next([idx])
        return result

    def wait_next(self, idx_list):
        """ wait until one job in list finishes, return index and result """
        logger.debug("MPI MASTER: Waiting for any task in {}".format(idx_list))
        # TODO: may get into deadlock with badly chosen idx_list
        while True:
            for idx in idx_list:
                if self.is_ready(idx):
                    return idx, self.ready_tasks[idx].result
            self._wait_for_next_ready()

    def is_ready(self, idx):
        """ true if task is ready """
        return idx in self.ready_tasks.keys()

    def remove_task(self, idx):
        """ removes task from waiting tasks (no effect to running or ready tasks) """
        # TODO: should effect running and ready as well?
        for i in range(len(self.waiting_tasks)):
            if self.waiting_tasks[i].idx == idx:
                del self.waiting_tasks[i]
                return True
        return False

    def reset(self):
        """ clears all waiting and ready tasks (no effect to running tasks) """
        # TODO: should effect running as well?
        self.waiting_tasks = deque()
        self.ready_tasks = dict()

