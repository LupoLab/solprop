Keyword routes built on two threads at once no longer leave the process's warning
filters changed: each route's capture of the material warnings it names a keyword for
now waits for another thread's to end, where two overlapping captures restored each
other's filters and left every later warning recorded instead of shown.
