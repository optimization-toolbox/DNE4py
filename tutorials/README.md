## `run.py`
* It changes results folder
```console
foo@bar:~$ mpiexec -n 2 python3 run.py 3
```

## `pp_run.py`
* It changes pp_results folder
* To Optimize: (Use Optimize Transparency 10%) https://ezgif.com/optimize/

```console
foo@bar:~$ python3 pp_run.py 3
foo@bar:~$ cd pp_results/RandomSearch/
foo@bar:~$ convert -layers OptimizeTransparency -delay 20 -loop 0 `ls -v` randomsearch.gif
```

## `render.py`
* It defines the behaviour to render the optimization procedure

