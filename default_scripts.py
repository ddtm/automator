TRAIN_TEMPLATE = """\
#!/usr/bin/env sh

TOOLS=./build/tools

# cuda-gdb --args \\
$$TOOLS/caffe train \\
    --solver=${solver_path} \\
    --gpu 0
"""

PLOT_TEMPLATE = """\
#!/usr/bin/env sh

python ./tools/extra/parse_log.py \\
    ${logs_path}/log.txt \\
    ${logs_path}

python ./tools/extra/plot_log.py \\
    ${logs_path}/log.txt.$$1 -f $$2 -r $$3
"""
