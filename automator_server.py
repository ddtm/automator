from __future__ import print_function

import os
import argparse
import yaml
import Pyro4
from multiprocessing import Semaphore

import util
import preprocessing
from worker import Worker

def freeze(o):
  if isinstance(o, dict):
    return frozenset({k: freeze(v) for k, v in o.items()}.items())
  if isinstance(o, list):
    return tuple([freeze(v) for v in o])
  return o

def make_hash(o):
    """
    Makes a hash out of anything that contains only list,dict and hashable types including string and numeric types
    """
    return hash(freeze(o))  

class AutomatorServer(object):
    def __init__(self, caffe_root, limit, pyro_daemon):
        self.pyro_daemon = pyro_daemon
        self.caffe_root = caffe_root
        self.workers = []
        if limit <= 0:
            self.limit_sem = None
        else:
            print('[*] Simultaneous experiments limit set to %d.' % limit)
            self.limit_sem = Semaphore(limit)

    def push_experiments(self, path, replace=False, no_run=False):
        data = yaml.load(file(path, 'r'))
        root_path = data['root_path']
        experiments = preprocessing.preprocess_experiments(data)

        # Inject additional fields.
        for e in experiments:
            e['caffe_root'] = self.caffe_root
            h = make_hash({'model': e['model'], 'solver': e['solver']})
            e['hash'] = h
            e['no_run'] = no_run
            e['replace'] = replace

        self.cleanup()

        existing_hashes = [w.experiment['hash'] for w in self.workers]
        
        for e in experiments:
            if e['hash'] in existing_hashes:
                continue
            self.workers.append(Worker(root_path, e, self.limit_sem))
            self.workers[-1].start()

    def terminate(self):
        print('[*] Terminating server...')
        self.kill_all()
        print('    Done')
        self.pyro_daemon.shutdown()

    def kill(self, worker_idx):
        self.workers[worker_idx].shutdown()
        self.workers[worker_idx].join()
        self.cleanup()

    def kill_all(self):
        for w in self.workers:
            w.shutdown()
        self.join_workers()
        self.cleanup()

    def join_workers(self):
        for w in self.workers:
            w.join()

    def cleanup(self):
        self.workers = [w for w in self.workers if w.is_alive()]

    def get_workers_info(self):
        self.cleanup()
        
        info = []
        for w in self.workers:
            info.append({
                'state': w.get_state(),
                'experiment': w.experiment
            })
        return info

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The automator server.')
    parser.add_argument('-r', '--caffe-root', required=True)
    parser.add_argument('-l', '--limit', type=int, default=0)

    args = parser.parse_args()

    os.chdir(args.caffe_root)

    daemon = Pyro4.Daemon(host='localhost')

    automator_server = AutomatorServer(args.caffe_root, args.limit, daemon)
    uri = daemon.register(automator_server, 'automator_server')

    tmp_dir = os.path.expanduser('~/.automator')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    file(os.path.join(tmp_dir, 'uri'), 'w').write(str(uri))

    print('[*] Automator server is running.')

    daemon.requestLoop()
    