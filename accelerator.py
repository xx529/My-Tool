import multiprocessing
import threading
import traceback
from functools import reduce
from itertools import product
from queue import Queue


class MultiAccelerator:


    def __init__(self, mode='process'):
        self.mode = mode


    def __call__(self, tasks, keep_order=True):
        task_param_list = []
        for task in tasks:
            fun, parallel_kwargs, static_kwargs = task
            parallel_kwargs_list = [dict(zip(list(parallel_kwargs.keys()), _zip)) for _zip in
                                    zip(*parallel_kwargs.values())]
            task_param_list.extend([(fun, {**parallel_kwargs, **static_kwargs}) for fun, parallel_kwargs in
                                    list(product([fun], parallel_kwargs_list))])

        results = self._accelerate(task_param_list, keep_order=keep_order)

        return results

    def _accelerate(self, task_param_list, keep_order):

        queue = multiprocessing.Queue() if self.mode == 'process' else Queue()
        accelerate_mode = multiprocessing.Process if self.mode == 'process' else threading.Thread

        def _worker(_queue, _idx, _fun, *args, **kwargs):
            try:
                _result = _fun(*args, **kwargs)
                _queue.put({_idx: _result, 'state': 'success', 'error': ''})
            except Exception as e:
                _queue.put({_idx: None, 'state': 'fail', 'error': traceback.format_exc() + '\n' + str(e)})

        worker_list = []

        for idx, (fun, param) in enumerate(task_param_list):
            worker_list.append(
                accelerate_mode(target=_worker, kwargs={'_queue': queue, '_idx': idx, '_fun': fun, **param}))

        for w in worker_list:
            w.start()

        result = [queue.get() for _ in worker_list]

        for w in worker_list:
            w.join()

        for i in result:
            if i['state'] == 'fail':
                raise Exception(i['error'])

        result = reduce(lambda x, y: {**x, **y}, result)

        del result['state'], result['error']

        if keep_order:
            result = list(dict(sorted(result.items(), key=lambda x: x[0])).values())

        else:
            result = list(result.values())

        return result
