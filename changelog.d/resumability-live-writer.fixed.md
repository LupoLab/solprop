Fixed `solprop.output.resumability` reporting that a writer still holds a file when the
writer lock could not be taken for another reason and the file's path contained the
words "another writer"; it now tells a held lock by its error type, not its message.
