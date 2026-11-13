Dropped optmat's internal tracker ids, such as `(TC-688)`, from the optmat warnings
solprop passes on as `MaterialRangeWarning` and from the run's material record; the
rest of each message is unchanged. A Raman response used outside its density or
temperature range, hydrogen at 0.5 bar for example, no longer ends its warning with a
reference users cannot look up ([#166](https://github.com/LupoLab/solprop/issues/166)).
