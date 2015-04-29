import os
import stat 
import sys
import re
import time

from string import Template
import multiprocessing as mp
import subprocess
import signal

import default_scripts
from util import clear_dir, get_latest

GPU_ID = 0

re_snapshot_prefix = re.compile('snapshot_prefix:\s*"([^"]+)"')
re_max_iter = re.compile('^max_iter: (\d+)')
re_iteration = re.compile('Iteration (\d+)')
re_top_output = re.compile('Iteration \d+, (\w+) = ([\.\d]+(e[+-][\d]+)*)')
re_output = re.compile('(Test|Train) net output #\d+: '
                       '(\w+) = ([\.\d]+(e[+-][\d]+)*)')

class Worker(mp.Process):
    def __init__(self, root_path, experiment, limit_sem=None):
        mp.Process.__init__(self)
        self.root_path = os.path.abspath(root_path)
        self.experiment = experiment
        self.caffe_root = experiment['caffe_root']
        self.custom_command = experiment['command']
        self.no_run = experiment['no_run']
        self.replace_mode = experiment['replace_mode']

        num_watched = len(experiment['watch'])
        self.watched_dict = dict(zip(experiment['watch'], range(num_watched))) 

        self.state_lock = mp.Lock()
        self.state = {
            'status': mp.Value('i', 0),
            'iter': mp.Value('f', 0.0),
            'max_iter': mp.Value('f', 0.0),
            'watched_values': mp.Array('f', [0] * num_watched)
        }

        self.control_cond = mp.Condition()
        self.is_terminated = mp.Value('i', 0)

        self.limit_sem = limit_sem

    def run(self):
        self.prepare_directories()

        if self.no_run:
            return

        if self.limit_sem:
            self.limit_sem.acquire()

        with self.state_lock:
            self.state['status'].value = 1

        self.run_training()
        self.main_loop()

        with self.state_lock:
            self.state['status'].value = 2
        
        if self.limit_sem:
            self.limit_sem.release()

    def shutdown(self):
        with self.control_cond:
            self.is_terminated.value = 1
            self.control_cond.notify()

    def get_state(self):
        state = {}
        with self.state_lock:
            for k, v in self.state.items():
                try:
                    state[k] = v.value
                except:
                    state[k] = v[:]
        return state

    def prepare_directories(self):
        self.path = os.path.join(self.root_path, self.experiment['path'])

        self.protos_path = os.path.join(self.path, 'protos')
        self.solver_path = os.path.join(self.protos_path, 'solver.prototxt')
        self.model_path = os.path.join(self.protos_path, 'train_val.prototxt')
        self.logs_path = os.path.join(self.path, 'logs')
        self.scripts_path = os.path.join(self.path, 'scripts')

        if self.replace_mode == 1 and os.path.exists(self.protos_path):
            self.prepare_protos()

        if self.replace_mode in [0, 1] and os.path.exists(self.path):
            self.snapshots_path = self.get_snapshots_directory()
            return

        for d in [self.protos_path, self.logs_path, self.scripts_path]:
            clear_dir(d)

        self.prepare_protos()
        self.prepare_scripts()
        
        self.snapshots_path = self.get_snapshots_directory()
        clear_dir(self.snapshots_path)

    def get_snapshots_directory(self):
        snapshot_prefix_match = re_snapshot_prefix.search(
            file(self.solver_path, 'r').read())
        return os.path.dirname(snapshot_prefix_match.group(1))
        
    def prepare_protos(self):
        model_template = file(self.experiment['model']['template'], 'r').read()
        solver_template = file(self.experiment['solver']['template'], 'r').read()

        model_template = Template(model_template)
        solver_template = Template(solver_template)

        model_values = self.experiment['model']['values']
        solver_values = self.experiment['solver']['values']

        solver_values.update({
            'net': self.model_path,
            'root': self.root_path,
            'path': self.experiment['path']
        })
        
        file(self.model_path, 'w').write(model_template.substitute(model_values))
        file(self.solver_path, 'w').write(solver_template.substitute(solver_values))  

    def prepare_scripts(self):
        train_script = Template(
            default_scripts.TRAIN_TEMPLATE).substitute({'solver_path': self.solver_path})
        plot_script = Template(
            default_scripts.PLOT_TEMPLATE).substitute({'logs_path': self.logs_path})

        file(os.path.join(self.scripts_path, 'train.sh'), 'w').write(train_script)
        file(os.path.join(self.scripts_path, 'plot.sh'), 'w').write(plot_script)

        for name in ['train.sh', 'plot.sh']:
            path = os.path.join(self.scripts_path, name)
            os.chmod(path, 0744)

    def run_training(self):
        self.log_path = os.path.join(self.logs_path, 'log.txt')
        binary_path = os.path.join(self.caffe_root, 'build/tools/caffe')

        # Check if there are any available snapshots and pick the
        # latest one for training continuation
        latest_snapshot = get_latest(self.snapshots_path, '*solverstate')

        if self.custom_command:
            training_string = 'srun --gres=gpu:1 {} 2>> {}'.format(
                self.custom_command, self.log_path)
        elif latest_snapshot:
            latest_snapshot = os.path.join(self.snapshots_path, latest_snapshot)

            training_string = (
                'srun --gres=gpu:1 '
                '{} train '
                '--solver={} '
                '--snapshot={} '
                '--gpu={} 2>> {}').format(
                    binary_path, self.solver_path, latest_snapshot, GPU_ID, 
                    self.log_path)
        else:
            if self.experiment['weights']:
                weights_string = '--weights=' + self.experiment['weights']
            else:
                weights_string = ''
                
            training_string = (
                'srun --gres=gpu:1 '
                '{} train '
                '--solver={} {} '
                '--gpu={} 2>> {}').format(
                    binary_path, self.solver_path, weights_string, GPU_ID, 
                    self.log_path)

        file(os.path.join(self.path, 'training_string.txt'), 'w').write(training_string)

        self.training_process = subprocess.Popen(training_string, 
            stdin=subprocess.PIPE, shell=True, preexec_fn=os.setsid)

    def main_loop(self):
        # Wait till log appears.
        while True:
            try:
                with open(self.log_path, 'r') as _:
                    break
            except IOError:
                time.sleep(1)

        iteration = 0.0
        max_iteration = 0.0
        watched_values = [0.0] * len(self.state['watched_values'])

        with open(self.log_path, 'r') as log_file:
            while True:
                with self.state_lock:
                    self.state['iter'].value = iteration
                    self.state['max_iter'].value = max_iteration
                    self.state['watched_values'][:] = watched_values[:]
                    
                with self.control_cond:
                    if self.is_terminated.value:
                        os.killpg(self.training_process.pid, signal.SIGTERM)
                        return

                where = log_file.tell()
                lines = log_file.readlines()
                if len(lines) == 0:
                    log_file.seek(where)

                    with self.control_cond:
                        self.control_cond.wait(5)
                else:
                    # We got non-empty log data. It's time to parse it.
                    for line in lines:
                        iteration_match = re_iteration.search(line)
                        if iteration_match:
                            iteration = float(iteration_match.group(1))

                        max_iter_match = re_max_iter.search(line)
                        if max_iter_match:
                            max_iteration = float(max_iter_match.group(1))

                        top_output_match = re_top_output.search(line)
                        if top_output_match:
                            value_name = top_output_match.group(1)
                            value = float(top_output_match.group(2))
                            if value_name in self.watched_dict.keys():
                                watched_values[self.watched_dict[value_name]] = value

                        output_match = re_output.search(line)
                        if output_match:
                            value_name = output_match.group(1).lower()
                            value_name += '_' + output_match.group(2)
                            value = float(output_match.group(3))
                            if value_name in self.watched_dict.keys():
                                watched_values[self.watched_dict[value_name]] = value

                self.training_process.poll()
                if self.training_process.returncode is not None:
                    return
    