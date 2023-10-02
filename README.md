# Tao.jl
Interface to the Tao program (from the Bmad ecosystem) for simulating high energy charged particle beams and X-rays.

## Command line interface to Tao
To run Tao in a Julia program, use
```
tao_cmd = Cmd(`tao -lat esr-main-unique.bmad -noplot -no_rad_int -quiet`)
const global tao = open(tao_cmd, "r+")
```
Commands can then be be run in Tao using ```println(tao, ...)```. For example,
```
println(tao, "set ele * kick = 0")
println(tao, "show -write uni.txt uni")
```
Output from Tao can then be read from files written using the ```-write``` flag in a ```show``` command.
