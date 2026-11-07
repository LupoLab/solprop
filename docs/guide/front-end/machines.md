# Running on other machines

A run, or a scan, can go to another machine: a workstation, a cluster with Slurm, or a
GPU machine you have rented. The window reaches it with SSH, the way you do from a
terminal, and keeps nothing between you and the run: each run keeps its own record in
its folder on that machine. You can close the window, lose the network or shut the
laptop, and the run goes on. When you come back, the window finds it again.

## Adding a machine

Choose **Run → Machines…**, or **Machines…** at the end of the list on the setup's
**Runs on** line, then **Add**:

- **From the SSH configuration** offers the hosts named in your `~/.ssh/config`, with
  their user names, keys, ports and jump hosts.
- **Type a host…** takes `user@host`, with a **Port** if it is not 22.

For each machine, say:

- **solprop's Python there**: the Python that has solprop, as the machine's shell
  finds it (`~/env/bin/python`, say). **Set up solprop there…** installs one for you
  (below).
- **Runs go in**: the folder of its runs on that machine (`~/solprop-runs` by default).
  On a cluster, choose a folder the compute nodes can reach.
- **Runs start** **directly**, as a process of their own, or **as Slurm jobs**. For
  Slurm, give what each job asks for: the **partition** (**Ask the machine** lists what
  its `sinfo` offers), the **time limit**, **CPUs**, **memory**, **generic resources**
  such as `gpu:1`, any further `#SBATCH` options, and any lines a job needs first, such
  as `module load …`.

**Test the connection** says which machine answered, its solprop version and whether
it has Slurm. The window never stores a password: it uses your keys and agent. A machine
that asks for a password or a one-time code asks in a small dialog, once per connection.

## Setting a machine up

**Set up solprop there…** shows the script it will run before it runs anything. The
script installs `uv` in your home folder if the machine has none, makes an environment
for the window's solprop version under **Environments go in** (`~/.solprop/envs` by
default; choose another folder where your home folder has a small quota), and installs
solprop there from PyPI. Nothing needs root. When it finishes, its Python becomes the
machine's. A machine that cannot reach PyPI, or a development build that PyPI does not
have, installs from wheels: **Add wheels…** copies them over first.

The machine's solprop must be the window's version, or the window refuses it and says
so. A development build on either side is allowed, with a note on the Runs on line.

## Runs on this computer, too

Every run works this way, on your own computer as well: it is a process of its own
with its record in its folder, and the window follows it. So a run goes on when the
window closes, or crashes, and a new window takes it up again, as **Runs** finds it.
Closing the window over a run asks whether to **keep it running**, **stop it** (it can
be resumed) or **kill it**. **Run → Tie runs to this window** makes runs on this
computer stop by themselves, resumably, if the window goes for any reason, a crash
included.

## Running there

Pick the machine on the **Runs on** line, then **Run**, as on your own computer. The run
goes into a new folder of the machine's, named for the date, its title and the time.
On a Slurm machine the line shows what the job will ask for, with **Change…**.

The run's progress, spectrum and pulse appear as they do for a run on your computer.
They come from the run's record on the machine, so nothing is lost while you are away:

- A Slurm job waiting in the queue shows **queued**, with Slurm's reason (`Priority`,
  `Resources`).
- **Stop** asks the run to stop at its next step. Its file can be resumed, as for a run
  on your computer.
- A Slurm job is sent `SIGTERM` two minutes before its time limit, so solprop writes
  where the run stood. **Resume** then continues it as a new job.

A data file the setup reads (a measured spectrum, say) is copied beside the run on the
machine before it starts, and the run reads it from there. It has to be in the setup's
own folder, as a run on your computer requires.

Setting up, checking and the sheet's numbers stay on your computer. A results file on
the machine is read there: the Analyse pages ask the machine and get back only what
they draw, so a large file is never copied to look at it. **Export → Publication
figure…** draws the figure on the machine and saves it on your computer, and **Export
→ A copy of the results file…** copies the file itself to your computer, as it does a
run of your own.

A scan run on a machine runs its points in a pool of processes, sized to the CPUs and
memory the machine or the Slurm job gives it. One job on a cluster node of 64 cores
runs many points at once.

## The Runs window

**Run → Runs…** lists every run on every machine, with its state, how far it has got and
when it started:

| state | what it means |
|---|---|
| **queued** | Submitted to Slurm and waiting, with why |
| **starting** | Started, and nothing recorded yet |
| **running** | Going; *stopping* once asked to stop |
| **finished**, **stopped**, **failed** | Ended, as the run recorded |
| **lost** | Ended without saying how: its process went, or its job ended or was cancelled first. Its file may still be resumable: **Resume** asks solprop |

**Open** follows a run that is going, or reads the results of one that has ended.
**Resume** continues a run that stopped, or ended early, from where solprop says it
can, on its machine, and follows it.
**Stop** asks a run to stop. **Cancel job** ends its Slurm job or its process at once;
solprop still writes where the run stood. **Fetch a copy** copies a run's results file
to your computer, under a folder named for the machine in your workspace, and opens
it. **Remove** deletes a run that has ended, after asking. A machine that cannot be
reached says so, and is tried again at the next refresh.

## What is on the machine

Each run's folder holds its recipe (`recipe.toml`), its results (`run.h5`, or
`scan.h5`) and its log (`run.log`), as on your computer, and its record:

- `job.json` says how it was started;
- `events.bin` holds its rows and events;
- `status.json` summarises it;
- `submitted.json` says what started it;
- a file named `stop` asks it to stop.

A Slurm run also keeps its batch script (`job.slurm`) and its job's own output
(`job.out`). The folder is the truth: anyone who can read it can follow the run, and
`python -m solprop.service.job FOLDER` runs a prepared folder by hand.
