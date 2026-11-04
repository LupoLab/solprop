Reads of the transverse edge on a free-space run are cheaper: a read now leaves out the
frequencies whose energy together is at most a thousandth of the warning, so a face's share
can read low by at most `1e-9` of the launch energy against a warning at `1e-6`. Runs with a
requested collar still read every frequency. A Kerr beam's complete run on 16 cores of an AMD
EPYC 7763, 384² with four steps, took 29.9 s and now takes 28.8 s; at one thread it fell from
117.1 s to 106.5 s (#453).
