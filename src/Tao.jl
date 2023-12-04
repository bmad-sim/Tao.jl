module Tao
using DelimitedFiles
using LinearAlgebra
using Printf

export  metadata_path,
        BAGELS_1,
        BAGELS_2

# Returns empty string if lattice not found
function metadata_path(lat)
  if isfile(pwd() * "/$(lat)")
    path = homedir() * "/.tao_jl" * pwd()[length(homedir())+1:end] * "/" * lat
    if !ispath(path)
      mkpath(path)
    end
    return path
  else
    return ""
  end
end

# --- Polarization routines ---

"""
    BAGELS_1(lat, phi_start, phi_step, sgn, kick=1e-5, tol=1e-8)

Best Adjustment Groups for ELectron Spin (BAGELS) method step 1: calculates the response of 
dn/ddelta with all combinations of closed orbit 2-bumps. Coil pairs with phase separation of 
dPhi are included when `rem(dPhi - phi_start, phi_step, RoundNearest) < tol`. `sgn` specifies 
the sign of the first kick times the sign of the second kick. E.g., for only pi-bumps, use 
`phi_start` = pi, `phi_step` = (something large), and `sgn` = -1. For all orthogonal kicks 
(equal kicks 2*pi*N apart), use `phi_start` = 0, `phi_step` = 2*pi, `sgn` = 1.

### Input
- `lat`       -- lat file name
- `phi_start` -- phi_start as described above
- `phi_step`  -- phi_step as described above
- `sgn`       -- sgn as described above
- `kick`      -- (Optional) Coil kick, default is 1e-5
- `tol`       -- (Optional) Tolerance for difference in phase, default is 1e-8
"""
function BAGELS_1(lat, phi_start, phi_step, sgn, kick=1e-5, tol=1e-8)
  path = metadata_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  str_phi = @sprintf("%1.2e", phi_start) * "_" * @sprintf("%1.2e", phi_step)
  str_kick = @sprintf("%1.2e", kick)

  # Generate directory in lattice metadata path for these phi_start and phi_step
  if !isdir("$(path)/BAGELS_$(str_phi)")
    mkdir("$(path)/BAGELS_$(str_phi)")
  end

  path = "$(path)/BAGELS_$(str_phi)"

  # First, obtain all combinations of bumps with desired phase advance
  if !isfile("$(path)/bumps.txt")
    if !isfile("$(path)/vkickers.txt")
      run(`tao -lat $lat -noplot -command "set ele * kick = 0; show -write $(path)/vkickers.txt lat vkicker::* -at phi_b@f20.16; exit"`)
    end
    if !isfile("$(path)/kicks.txt")
      run(`tao -lat $lat -noplot -command "show -write $(path)/kicks.txt lat vkicker::* -at phi_b@f20.16 -at kick@f20.16; exit"`)
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
        if abs(rem(coil_phis[j] - coil_phis[i] - phi_start, phi_step, RoundNearest)) < tol
          push!(coil_pairs_list, coil_names[i])
          push!(coil_pairs_list, coil_names[j])
          push!(coil_kicks_list, coil_kicks[i])
          push!(coil_kicks_list, coil_kicks[j])
        end
      end
    end

    coil_pairs = permutedims(reshape(coil_pairs_list, 2, Int(length(coil_pairs_list)/2)))
    strengths = permutedims(reshape(coil_kicks_list, 2, Int(length(coil_pairs_list)/2)))

    writedlm("$(path)/bumps.txt", hcat(coil_pairs, strengths), ',')
  end

  coils = readdlm("$(path)/bumps.txt", ',')
  coil_pairs = coils[:,1:2]
  strengths = coils[:,3:4]
  if !isdir("$(path)/responses_$(str_kick)")
    mkdir("$(path)/responses_$(str_kick)")
  end
  # All of these bumps are separate group knobs, the individual coils have opposite strengths
  # Now build response matrix of dn/ddelta at each sbend (sampled at beginning and ends of bends)
  tao_cmd = open("$(path)/responses_$(str_kick)/BAGELS_1.cmd", "w")
  println(tao_cmd, "show -write $(path)/responses_$(str_kick)/baseline.txt lat sbend::* multipole::* -at spin_dn_dpz.x@f20.16 -at spin_dn_dpz.y@f20.16 -at spin_dn_dpz.z@f20.16")

  for i=1:length(coil_pairs[:,1])
    coil1 = coil_pairs[i,1]
    coil2 = coil_pairs[i,2]
    strength1 = strengths[i,1]
    strength2 = strengths[i,2]
    println(tao_cmd, "set ele $(coil1) kick = $(strength1 + kick)")
    println(tao_cmd, "set ele $(coil2) kick = $(strength2 + sgn*kick)")
    println(tao_cmd, "show -write $(path)/responses_$(str_kick)/$(coil1)_$(coil2).txt lat sbend::* multipole::* -at spin_dn_dpz.x@f20.16 -at spin_dn_dpz.y@f20.16 -at spin_dn_dpz.z@f20.16")
    println(tao_cmd, "set ele $(coil1) kick = $(strength1)")
    println(tao_cmd, "set ele $(coil2) kick = $(strength2)")
  end
  println(tao_cmd, "exit")
  close(tao_cmd)

  run(`tao -lat $lat -noplot -command "call $(path)/responses_$(str_kick)/BAGELS_1.cmd"`)
end

"""
    BAGELS_2(lat, phi_start, phi_step, N_knobs; suffix="", outf="BAGELS.bmad", kick=1e-5))

Best Adjustment Groups for ELectron Spin (BAGELS) method step 2: peform an SVD of the 
response matrix to obtain the best adjustment groups, based on the settings of step 1. 
E.g., for only pi-bumps, use `phi_start` = pi, `phi_step` = (something large), and 
`sgn` = -1. For all orthogonal kicks (equal kicks 2*pi*N apart), use `phi_start` = 0, 
`phi_step` = 2*pi, `sgn` = 1.


### Input
- `lat`       -- lat file name
- `phi_start` -- phi_start as described above
- `phi_step`  -- phi_step as described above
- `N_knobs`   -- Number of knobs to generate in group element, written in Bmad format
- `suffix`    -- (Optional) Suffix to append to group elements generated for knobs in Bmad format
- `outf`      -- (Optional) Output file name with group elements constructed from BAGELS, default is "BAGELS.bmad"
- `kick`      -- (Optional) Coil kick, default is 1e-5
"""
function BAGELS_2(lat, phi_start, phi_step, N_knobs; suffix="", outf="BAGELS.bmad", kick=1e-5)
  path = metadata_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  
  str_phi = @sprintf("%1.2e", phi_start) * "_" * @sprintf("%1.2e", phi_step)
  str_kick = @sprintf("%1.2e", kick)

  path = "$(path)/BAGELS_$(str_phi)"
  coil_pairs = readdlm("$(path)/bumps.txt", ',')[:,1:2]
  eletypes = readdlm("$(path)/responses_$(str_kick)/baseline.txt", skipstart=2)[1:end-2,3]
  dn_dpz0 = readdlm("$(path)/responses_$(str_kick)/baseline.txt", skipstart=2)[1:end-2,6:8]

  # Get number of Sbends to only sample at Sbends
  idx_bends = findall(i-> i=="Sbend", eletypes)
  dn_dpz0_bends = dn_dpz0[idx_bends,:]

  # We now will have a response matrix that is N_bends x N_coil_pairs to multiply with vector N_coil_pairs x 1
  delta_dn_dpz = zeros(length(idx_bends), length(coil_pairs[:,1]))
  for i=1:length(coil_pairs[:,1])
    coil1 = coil_pairs[i,1]
    coil2 = coil_pairs[i,2]
    dn_dpz_bends = readdlm("$(path)/responses_$(str_kick)/$(coil1)_$(coil2).txt", skipstart=2)[1:end-2,6:8][idx_bends,:]
    #dn_dpz_bends_amp = [norm(dn_dpz_bends[j,:]) for j=1:length(dn_dpz_bends[:,1])]
    #delta_dn_dpz_amp = dn_dpz_bends_amp .- dn_dpz0_bends_amp
    #delta_dn_dpz_amp = [norm(delta_dn_dpz[j,:]) for j=1:length(delta_dn_dpz[:,1])]
    # For now, BAGELS only uses x component of dn/ddelta
    delta_dn_dpz[:,i] = (dn_dpz_bends[:,1] .- dn_dpz0_bends[:,1])./kick
  end
  
  # Now do SVD to get the principal components
  F = svd(delta_dn_dpz)

  # Get first N_knobs principal directions
  V = F.V[:,1:N_knobs]

  fmt = "%0$(length(string(N_knobs)))i"
  @eval mysprintf(x) = @sprintf($fmt, x)

  # Create a group element with these as knobs  
  for i=1:N_knobs
    knob = V[:,i]
    raw = Matrix{Any}(undef, length(coil_pairs), 2) #zeros(length(coil_pairs), 2)
    # First column of raw contains coil name, second is knob value
    for j=1:length(coil_pairs[:,1])
      coil1 = coil_pairs[j,1]
      coil2 = coil_pairs[j,2]
      raw[2*j-1, 1] = coil1
      raw[2*j-1, 2] = knob[j]
      raw[2*j, 1] = coil2
      raw[2*j, 2] = -knob[j]
    end

    # Now make knobs for each
    unique_coils = unique(coil_pairs)

    if i == 1
      knob_out = open(outf, "w")
    else
      knob_out = open(outf, "a")
    end
    println(knob_out, "BAGELS$(suffix)$(mysprintf(i)): group = {")

    for coil in unique_coils
      strength = 0.
      idxs_coil = findall(i->i==coil, raw[:,1])
      strength = sum(raw[idxs_coil,2])
      println(knob_out, "$(coil)[kick]: $(strength)*X,")
    end
    println(knob_out, "}, var = {X}")
    close(knob_out)
  end
end

end
