"""
Please see the https://github.com/george-azzari/eetc repository for updated versions of this code.
Author of this file: Anthony Perez
"""
import time
import itertools
from collections import deque as LinkedList
from multiprocessing import Pool
import subprocess
import logging

import pandas as pd
import ee


logger = logging.getLogger(__name__)


class TaskSchedulerError(RuntimeError):
    pass


class Job(object):

    def __init__(self, task, jid, dependencies):
        self.task = task
        self.jid = jid
        self.dependencies = dependencies
        self._failed = False
        self._success = False
        self._finished = False

    def _update_state(self):
        if self._finished:
            return
        status = self.task.status()['state']
        self._finished = status not in [self.task.State.UNSUBMITTED, self.task.State.READY, self.task.State.RUNNING]
        self._failed = status in [self.task.State.FAILED, self.task.State.CANCEL_REQUESTED, self.task.State.CANCELLED]
        self._success = status == self.task.State.COMPLETED

    def mark_completed(self):
        self._finished = True
        self._failed = False
        self._success = True

    def mark_failed(self):
        self._finished = True
        self._failed = True
        self._success = False

    def queued(self):
        self._update_state()
        return not self._finished

    def finished(self):
        return not self.queued()

    def start(self):
        self.task.start()

    def failed(self):
        self._update_state()
        return self._failed

    def depends_on(self, job):
        return job.jid in self.dependencies

    def __repr__(self):
        return 'Job({}, {}, {})'.format(self.task, self.jid, self.dependencies)

# pylint: disable=E1601

class TaskScheduler(object):

    @staticmethod
    def wait_for_existing_tasks(sleep_time=120.0, verbose=0):
        """
        Block until all tasks have finished in the currently
        authenticated GEE account.  This involves looking up the
        task list from GEE.
        """
        def task_is_queued(task):
            status = task.config.get('state', None)
            queued_states = [task.State.READY, task.State.RUNNING]  # task.State.UNSUBMITTED
            return status in queued_states

        task_list = [
            Job(task=task, jid=i, dependencies=[])
            for i, task in enumerate(ee.batch.Task.list())
            if task_is_queued(task)
        ]
        TaskScheduler._wait_all_tasks(task_list, sleep_time=sleep_time, verbose=verbose)

    @staticmethod
    def _wait_all_tasks(jobs, sleep_time=120.0, verbose=0):
        # TODO - Potential infinite loop for unstarted jobs
        jobs = list(jobs)
        while len(jobs) > 0:
            job = jobs.pop()
            if job.queued():
                jobs.append(job)
                if verbose > 0:
                    print('Tasks remaining: {}'.format(len(jobs)))
                time.sleep(sleep_time)   # TODO, find a better way to do this.

    @staticmethod
    def _wait_any_tasks(jobs, sleep_time=120.0):
        # TODO - Potential infinite loop for unstarted jobs
        if len(jobs) == 0:
            return [], []
        while True:
            finished_tasks = []
            running_tasks = []
            for job in jobs:
                if job.queued():
                    running_tasks.append(job)
                else:
                    finished_tasks.append(job)
            if len(finished_tasks) > 0:
                return running_tasks, finished_tasks

            time.sleep(sleep_time)   # TODO, find a better way to do this.

    def __init__(self):
        self.tasks = dict()

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        """
        Iterates over all tasks, including finished tasks.
        Returns:
            (Iterable[ee.batch.Task]): 
        """
        for job in self.tasks.values():
            yield job.task

    def add_task(self, task, jid, dependencies=None):
        """
        Add a new task to the Task Scheduler.
        :param task: Task from ee.batch.Export
        :param jid: immutable type to be used as a job id
        :param dependencies: Any iterable containing job ids of tasks that should be completed before task is started.
        """
        if jid in self.tasks:
            raise TaskSchedulerError('jid already exists.')
        if dependencies is None:
            dependencies = set()
        self.tasks[jid] = Job(task, jid, set(dependencies))

    def mark_all_completed(self):
        for jid in list(self.tasks.keys()):
            self.mark_completed(jid)

    def mark_completed(self, jid):
        """Marks a task as already being completed sucessfully (convinence method for resuming a sequence of tasks)"""
        self.tasks[jid].mark_completed()

    def merge(self, task_scheduler):
        """
        Return a TaskScheduler whose tasks are the union of this and task_scheduler's tasks.
        Note that this is a shallow copy.  Marking a job as completed in one task scheduler will
        mark it as completed in the combined task scheduler.
        """
        new_scheduler = TaskScheduler()
        for jid, job in itertools.chain(self.tasks.items(), task_scheduler.tasks.items()):
            if jid in new_scheduler.tasks:
                raise TaskSchedulerError('Duplicate jid "{}" during merge.'.format(jid))
            new_scheduler.tasks[jid] = job
        return new_scheduler

    def start_and_remove_all(self):
        """
        Starts all tasks in the scheduler
        Returns:
            (List[tasks]):  The list of all tasks started
                and removed from the schulder.
        """
        for job in self.tasks.values():
            job.start()
        tasks = [job.task for job in self.tasks.values()]
        self.tasks = dict()
        return tasks

    def cancel_all(self):
        """
        Attempts to cancel all running tasks.  Will ignore
        errors as a result of attempting to cancel tasks. 
        """
        for task in self:
            try:
                task.cancel()
            except Exception:
                pass

    def run(self, max_processes=8, sleep_time=120.0, verbose=0,
            error_on_fail=False):
        """
        Run all tasks in order of dependency.
        Blocks until all tasks have finished.

        Will run max_processes tasks at a time.
        If a task fails, its dependents will not be run.

        TODO:  Detect dependency loops.
        Currently dependency loops will be detected after some tasks may have already been run.

        Args:
            max_processes (int):  The maximum number of tasks that are allowed to run simultaneously.
                Defaults to 8.  8 is a good choice for most applications since it allows the next set of tasks
                to begin immediately after the first batch of 4 ends.
            sleep_time (float):  The time to wait between checking if a task has finished, in seconds.
            verbose (int):  An integer affecting verbosity.  Uses print.
                > 0  to alter on starting tasks
                > 2  to alter on job finished
                > 6 for details
                > 9 for a dump of info on if a circular dependency is detected.
            error_on_fail (bool): raises an error if any task fails.
        """
        if len(self.tasks) == 0:
            raise TaskSchedulerError('Tried to run tasks with no tasks')

        jid_order = [jid for jid, job in self.tasks.items() if job.queued()]
        running = []

        def job_run(jid):
            """Return True if the job started or was marked as failed"""
            if len(running) >= max_processes:
                return False
            job = self.tasks[jid]

            for dep_jid in job.dependencies:
                dep_job = self.tasks[dep_jid]
                if dep_job.finished():
                    if dep_job.failed():
                        job.mark_failed()
                        return True
                else:
                    return False

            job.start()
            running.append(job)
            if verbose > 0:
                print('Running task: {}'.format(job.jid))
            return True

        def wait_tasks_helper():
            if verbose > 6:
                print('Waiting for a task to finish')
            _running, finished = TaskScheduler._wait_any_tasks(running, sleep_time=sleep_time)
            for job in finished:
                if verbose > 2:
                    print('Job {} {}'.format(job.jid, 'failed' if job.failed() else 'sucessful'))
                if error_on_fail and job.failed():
                    raise TaskSchedulerError('Job {} failed'.format(job.jid))
            return _running

        while len(jid_order) > 0:
            old_size = len(jid_order)
            old_running_size = len(running)
            jid_order = [jid for jid in jid_order if not job_run(jid)]
            if len(jid_order) >= old_size and old_running_size >= len(running):
                if len(running) == 0:
                    err_msg = 'Possible circular dependency'
                    if verbose > 9:
                        finished_tasks = [job for jid, job in self.tasks.items() if job.finished()]
                        remaining_tasks = [self.tasks[jid] for jid in jid_order]
                        err_msg = '{}\nRunning tasks:\n{}\nFinished tasks:\n{}\nRemaining tasks:\n{}\nAll task ids:\n{}' \
                                  .format(err_msg, running, finished_tasks, remaining_tasks, self.tasks.keys())
                    raise TaskSchedulerError(err_msg)
                running = wait_tasks_helper()

        if verbose > 6:
            print('All tasks have been started')

        while len(running) > 0:
            running = wait_tasks_helper()

        return [job.task for job in self.tasks.values()]


def asset_exists(asset_id):
    return ee.data.getInfo(asset_id) is not None


def rm_asset(asset_path):
    """
    Invoke earthengine rm asset_path
    Args:
        asset_dir_path (str):  The path to a directory of assets.
    """
    command = ' '.join(['earthengine', 'rm', asset_path])
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    tmp = proc.stdout.read()

    invalid_path_message = 'Asset does not exist or is not accessible:'
    bad_formatting = 'Path for the asset is invalid. '
    failure = 'Failed to delete '
    error_messages = [invalid_path_message, bad_formatting, failure]
    for err_msg in error_messages:
        if len(tmp) >= len(err_msg) and tmp[:len(err_msg)] == err_msg:
            raise ValueError(tmp)


def rm_assets(asset_path_list, num_parallel=1):
        if num_parallel is None or num_parallel < 2:
            for asset_path in asset_path_list:
                logger.info('Removing asset {}'.format(asset_path))
                rm_asset(asset_path)
        else:
            pool = Pool(processes=num_parallel)
            asset_path_list = list(asset_path_list)
            for asset_path in asset_path_list:
                logger.info('Queuing removal of asset {}'.format(asset_path))
            pool.map(rm_asset, asset_path_list)
            pool.close()
            pool.join()


def ls_asset_dir(asset_dir_path):
    """
    Invoke earthengine ls <asset_dir_path>
    Args:
        asset_dir_path (str):  The path to a directory of assets.
    Returns:
        (List[str]): Returns a list of full asset paths, e.g. [asset_dir_path/asset_name]
    Raises:
        ValueError if asset_dir_path is not a directory (including if
        it is a file or does not exist), or the path was invalid.
    """
    if not asset_exists(asset_dir_path):
        raise ValueError('Asset {} does not exist.'.format(asset_dir_path))

    children = ee.data.getList({'id': asset_dir_path})

    if children is None:
        return []
    else:
        result = [asset['id'] for asset in children]
        return result


def create_folder(folder_id):
    if ee.data._use_cloud_api:
        ee.data.createAsset(
            value={'type': ee.data.ASSET_TYPE_FOLDER_CLOUD},
            opt_path=folder_id
        )
    else:
        ee.data.createAsset(
            value={'type': ee.data.ASSET_TYPE_FOLDER},
            opt_path=folder_id
        )


def upload_geojson_to_gee(csv_path, asset_id, chunk_size=50,
                          lat_name='lat', lon_name='lon',
                          verbose=0):
    """
    Uploads a CSV as a feature collection.
    Blocks until the upload is complete.
    Args:
        csv_path (str):  Path to CSV.
        asset_id (str):  The Asset ID to upload the data frame to.
            Will temporarily create a folder named "asset_id + '_upload_folder'".
        chunk_size (Optional[int]):  The number of features to upload at once.  Lowering this may reduce
        API errors, but will reduce performance.  Defaults to 50.
        verbose (Optional[int]):  An integer affecting verbosity.  Uses print, passed to the TaskScheduler.
                > 0  to alter on starting tasks
                > 2  to alter on job finished
                > 6 for details
                > 9 for a dump of info on if a circular dependency is detected.
            Defaults to 0.
    Raises:
        (gee_tools.exports.task_scheduler.TaskSchedulerError):  On upload failure.
    """
    def divide_chunks(l, n):
        for i in range(0, len(l), n):  
            yield l[i:i + n] 

    df = pd.read_csv(csv_path, index_col=False)
    num_rows = df.shape[0]

    ee_features = []
    for i in range(num_rows):
        props = {
            c_name: df[c_name].iloc[i]
            for c_name in df.columns
        }
        _geometry = ee.Geometry.Point([
            float(df[lon_name].iloc[i]),
            float(df[lat_name].iloc[i]),
        ])
        ee_feat = ee.Feature(_geometry, props)
        ee_features.append(ee_feat)

    ee_colls = []
    for feature_subset in divide_chunks(ee_features, chunk_size):
        coll = ee.FeatureCollection(feature_subset)
        ee_colls.append(coll)

    ee_colls_and_asset_ids = [(coll, 'csv_upload_{}'.format(i)) for i, coll in enumerate(ee_colls)]

    upload_folder = asset_id + '_upload_folder'
    if asset_exists(upload_folder):
        raise ValueError('{} already exists, blocking the upload to {}'.format(upload_folder, asset_id))
    create_folder(upload_folder)
    if not asset_exists(upload_folder):
        raise ValueError('Failed to create {}'.format(upload_folder))

    scheduler = TaskScheduler()
    for coll, coll_asset_id in ee_colls_and_asset_ids:
        task = ee.batch.Export.table.toAsset(
            collection=coll,
            description=coll_asset_id,
            assetId=upload_folder + '/' + coll_asset_id,
        )
        scheduler.add_task(task, coll_asset_id)

    try:

        try:
            scheduler.run(max_processes=16, verbose=verbose, error_on_fail=True)
        except Exception:
            scheduler.cancel_all()
            raise
        except KeyboardInterrupt:
            scheduler.cancel_all()
            raise

        # The recursion depth is O(log(n)) using this method.
        # A linear method will have O(n) recursion which will occasionally cause an error.
        merged_fc = LinkedList([
            ee.FeatureCollection(upload_folder + '/' + coll_asset_id)
            for _, coll_asset_id in ee_colls_and_asset_ids
        ])
        while len(merged_fc) > 1:
            fc1 = merged_fc.pop()
            fc2 = merged_fc.pop()
            merged_fc.appendleft(fc1.merge(fc2))
        merged_fc = merged_fc.pop()

        task = ee.batch.Export.table.toAsset(
            collection=merged_fc,
            description='geojson_upload',
            assetId=asset_id,
        )
        merged_scheduler = TaskScheduler()
        merged_scheduler.add_task(task, 1)
        try:
            merged_scheduler.run(verbose=verbose, error_on_fail=True)
        except KeyboardInterrupt:
            merged_scheduler.cancel_all()
            raise

    finally:
        rm_assets(ls_asset_dir(upload_folder), num_parallel=5)
        rm_assets([upload_folder])
