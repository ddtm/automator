from __future__ import print_function

import os
import time
import curses
import argparse
import Pyro4

def get_fields(w, idx):
    value_names = w['experiment']['watch']
    values = w['state']['watched_values']

    ds = []

    num_values = len(value_names)

    for line_idx in xrange(max(num_values, 1)):
        if num_values == 0:
            value_name = ''
            value = ''
        else:
            value_name = value_names[line_idx]
            value = str(values[line_idx])

        d = {
            'idx': '',
            'status': '',
            'path': '',
            'desc': '',
            'progress': '',
            'value_name': value_name,
            'value': value
        }

        if line_idx > 0:
            ds.append(d)
            continue

        iteration = int(w['state']['iter'])
        max_iteration = int(w['state']['max_iter'])

        status = w['state']['status']
        if status == 0:
            status = 'WAITING'
        elif status == 1:
            status = 'RUNNING'
        else:
            status = 'FINISHED'

        d.update({
            'idx': str(idx),
            'status': status,
            'path': w['experiment']['path'],
            'desc': w['experiment']['description'],
            'progress': '{:d} / {:d}'.format(iteration, max_iteration)
        })

        ds.append(d)

    return ds

def display_info(stdscr, automator_server, interval):
    curses.curs_set(0)
    curses.start_color()

    curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)

    height, width = stdscr.getmaxyx()

    if interval < 0:
        interval = 0

    if interval == 0:
        stdscr.timeout(-1)
    else:
        stdscr.timeout(interval * 1000)

    header = {
        'idx': 'ID',
        'status': 'STATUS',
        'path': 'PATH',
        'desc': 'DESCRIPTION',
        'progress': 'ITER / MAX_ITER',
        'value_name': 'NAME',
        'value': 'VALUE'
    }

    fields_order = ['idx', 'status', 'path', 'desc', 'progress', 'value_name', 'value']
    
    format_s = []
    for k in fields_order:
        format_s.append('{{{}:<{{{}}}}}'.format(k, k + '_width'))
    format_s = '  '.join(format_s)

    while True:
        stdscr.clear()
        stdscr.border()

        stdscr.addstr(1, 2, 'List of running experiments:')

        workers_info = automator_server.get_workers_info()

        # Get all data as strings.
        entries = []
        for i, w in enumerate(workers_info):
            entries.append(get_fields(w, i))

        # Calculate widths.
        widths = {}
        for k, v in header.items():
            widths[k] = len(v)
        for e in entries:
            for l in e:
                for k, v in l.items():
                    widths[k] = max(widths[k], len(v))

        widths = dict(zip([k + '_width' for k in widths.keys()], widths.values()))

        # for k in widths.keys():
        #     widths[k] = widths.keys() + 1

        # Output data
        header_format_d = {}
        header_format_d.update(header)
        header_format_d.update(widths)
        s = format_s.format(**header_format_d)      
        stdscr.addstr(3, 2, s, curses.A_BOLD)

        line_idx = 4
        for i, e in enumerate(entries):
            for l in e:
                l.update(widths)
                s = format_s.format(**l)  
                stdscr.addstr(line_idx, 2, s, curses.color_pair(i % 2))
                line_idx += 1

        stdscr.refresh()

        if interval == 0:
            break

        c = stdscr.getch()
        if c != -1:
            break

    if interval == 0: 
        stdscr.getch()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Client for the automator.')
    parser.add_argument('-e', '--experiments', default='')
    parser.add_argument('-n', '--no-run', action='store_true', default=False)
    parser.add_argument('-l', '--list', action='store_true', default=False)
    parser.add_argument('-i', '--interval', type=int, default=0)
    parser.add_argument('-k', '--kill', type=int, default=-1)
    parser.add_argument('-p', '--port', type=int, default=-1)
    parser.add_argument('-t', '--terminate-server', action='store_true', default=False)

    args = parser.parse_args()

    tmp_dir = os.path.expanduser('~/.automator')
    if not os.path.exists(tmp_dir):
        if args.port < 0:
            raise RuntimeError('Unable to locate server. Please specify port.')
        uri = 'PYRO:automator_server@localhost:%d' % args.port
    else:
        uri = file(os.path.join(tmp_dir, 'uri'), 'r').read()

    automator_server = Pyro4.Proxy(uri)

    if args.experiments:
        experiments_path = os.path.abspath(os.path.expanduser(args.experiments))
        automator_server.push_experiments(experiments_path, args.no_run)

    if args.list:
        curses.wrapper(display_info, automator_server, args.interval)

    if args.kill >= 0:
        automator_server.kill(args.kill)

    if args.terminate_server:
        automator_server.terminate()      
