# Tao.jl
Interface to the Tao program (from the Bmad ecosystem) for simulating high energy charged particle beams and X-rays.

## Command line interface to Tao
There are currently two ways of running Tao from a Julia program. The first, and recommended way, is to have the program create a separate Tao command file (e.g. tao.cmd), which is then fed into Tao in the command line interface. For example, in the Julia program:
```
run(`tao -lat esr.bmad -command "call tao.cmd"`)
```
This method of calling Tao from Julia does not lead to race conditions with the commands run in Tao; the Julia program will wait until the entire sequence of commands has been run and completed before continuing to the next lines in the program. So, files can be written from Tao in the tao.cmd, which are then later read into the Julia program without any issues.

The second way of running Tao from a Julia program is more interactive, however race conditions are present. With this method, to run Tao in a Julia program, use:
```
tao_cmd = Cmd(`tao -lat esr-main-unique.bmad -noplot -no_rad_int -quiet`)
const global tao = open(tao_cmd, "r+")
```
Commands can then be be run in Tao using ```println(tao, ...)```. For example,
```
println(tao, "set ele * kick = 0")
println(tao, "show -write uni.txt uni")
```
With the above example, however, Julia will not wait until the `uni.txt` file is written before continuing to the next line in the program. Therefore, if the Julia program attempts to read `uni.txt` shortly after, a race condition problem is highly likely.
