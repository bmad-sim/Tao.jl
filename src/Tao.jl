module Tao
using DelimitedFiles
using LinearAlgebra
using Printf

export BAGELS_1, metadata_path

function metadata_path(lat)
  path = homedir() * "/.tao.jl" * pwd()[homedir()+1:end] * "/" * lat
  if !ispath(path)
    mkpath(path)
  end
  return path
end

# Polarization routines ======================================

"""
    BAGELS_1(lat, phi_diff, phi_start, sgn)

Calculates the response of dn/ddelta with all combinations of closed orbit 2-bumps.
Coil pairs with phase separation of dPhi are included when 
rem(dPhi - phi_start, phi_step, RoundNearest) < tol. sgn specifies the sign of the first
kick times the sign of the second kick. E.g., for only pi-bumps, use phi_start = pi, 
phi_step = (something large), and sgn = -1. For all orthogonal kicks (equal kicks 
2*pi*N apart), use phi_start = 0, phi_step = 2*pi, sgn = 1.

### Input
- `lat`       -- Lattice file name
- `phi_start` -- phi_start as described above
- `phi_step`  -- phi_step as described above
- `sgn`       -- sgn as described above
- `kick`      -- (Optional) Coil kick, default is 1e-5
- `tol`       -- (Optional) Tolerance for difference in phase, default is 1e-8
"""
function BAGELS_1(lat, phi_diff, phi_start, sgn, kick=1e-5, tol=1e-8)
  path = metadata_path(lat)
  str_phi = @sprintf("%1.2e", phi_start) * "_" * @sprintf("%1.2e", phi_step)
  str_kick = @sprintf("1.2e", kick)
  # First, obtain all combinations of bumps with desired phase advance
  if !isfile("$(path)/bumps$(str_phi).txt")
    if !isfile("$(path)/vkickers.txt")
      run(`tao -lat $lattice -noplot -command "set ele * kick = 0; show -write $(path)/vkickers.txt lat vkicker::* -at phi_b@f20.16; exit"`)
    end
    if !isfile("$(path)/kicks.txt")
      run(`tao -lat $lattice -noplot -command "show -write $(path)/kicks.txt lat vkicker::* -at phi_b@f20.16 -at kick@f20.16; exit"`)
    end

  # Read coil data and extract valid pairs
  coil_data = readdlm("$(path)/vkickers.txt", skipstart=2)
  coil_names = coil_data[1:end-2, 2]
  coil_phis = coil_data[1:end-2, 6]
  coil_kicks = readdlm("$(path)/kicks.txt", skipstart=2)[1:end-2,7]

  coil_pairs_list = []
  coil_kicks_list = []

  for i=1:length(coil_names)-1
    for j=i+1:length(coil_names)
      if abs(rem(coil_phis[j] - coil_phis[i] - phi_start, phi_step, RoundNearest)) < 1e-8
        push!(coil_pairs_list, coil_names[i])
        push!(coil_pairs_list, coil_names[j])
        push!(coil_kicks_list, coil_kicks[i])
        push!(coil_kicks_list, coil_kicks[j])
      end
    end
  end

  coil_pairs = permutedims(reshape(coil_pairs_list, 2, Int(length(coil_pairs_list)/2)))
  strengths = permutedims(reshape(coil_kicks_list, 2, Int(length(coil_pairs_list)/2)))

  writedlm("$(path)/bumps$(str_phi).txt", hcat(coil_pairs, strengths), ',')
end

coils = readdlm("$(path)/bumps$(str_phi).txt", ',')
coil_pairs = coils[:,1:2]
strengths = coils[:,3:4]

mkdir("$(path)/responses_$(str_kick)")
# All of these bumps are separate group knobs, the individual coils have opposite strengths
# Now build response matrix of dn/ddelta at each sbend (sampled at beginning and ends of bends)
tao_cmd = open("$(path)/BAGELS1.cmd", "w")
println(tao_cmd, "show -write $(path)/responses_$(str_kick)/baseline.txt lat sbend::* multipole::* -at spin_dn_dpz.x@f20.16 -at spin_dn_dpz.y@f20.16 -at spin_dn_dpz.z@f20.16")

for i=1:length(coil_pairs[:,1])
  coil1 = coil_pairs[i,1]
  coil2 = coil_pairs[i,2]
  strength1 = strengths[i,1]
  strength2 = strengths[i,2]
  println(tao_cmd, "set ele $(coil1) kick = $(strength1 + kick)")
  println(tao_cmd, "set ele $(coil2) kick = $(strength2 - kick)")
  println(tao_cmd, "show -write $(path)/responses_$(str_kick)/$(coil1)_$(coil2).txt lat sbend::* multipole::* -at spin_dn_dpz.x@f20.16 -at spin_dn_dpz.y@f20.16 -at spin_dn_dpz.z@f20.16")
  println(tao_cmd, "set ele $(coil1) kick = $(strength1)")
  println(tao_cmd, "set ele $(coil2) kick = $(strength2)")
end
close(tao_cmd)

#run(`tao -lat $lattice -noplot`)
end





end
