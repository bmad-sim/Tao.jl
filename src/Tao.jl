module Tao
using DelimitedFiles
using LinearAlgebra
using Printf
using NLsolve
using Interpolations
using DataFrames, StatsBase, GLM
using Random, Distributions

export  data_path,
        PolData,
        BAGELS_1,
        BAGELS_2,
        calc_P_t,
        calc_T,
        pol_scan,
        get_pol_data,
        run_map_tracking,
        run_pol_scan,
        download_pol_scan,
        read_pol_scan,
        get_pol_track_data,
        PolTrackData,
        misalign,
        groups_to_one_knob,
        run_pol_tune_scan,
        download_pol_tune_scan,
        read_pol_tune_scan,
        get_pol_tune_data



# Returns empty string if lattice not found
function data_path(lat)
  if isfile(lat)
    path = dirname(abspath(lat)) * "/tao_jl_" * basename(lat)
    if !ispath(path)
      mkpath(path)
    end
    return path
  else
    return ""
  end
end


"""
    groups_to_one_knob(groupsf, name, outf)

This function converts all of the groups defined in the file `groupsf`, with their corresponding 
`X` vars set, to a single group that one can scale up and down. As of now, this only works for 1
variable in each group (which must be named `X`) and one attribute set per element (an element 
cannot have both `kick` and `k1` set in the groups, for example). E.g. if `groupsf` contains
```
G1: group = {CV01[kick]: 1*X, CV02[kick] = 1.5*X}, var = {X}, X = 2
G2: group = {CV01[kick]: 0.5*X, CV02[kick] = 1*X}, var = {X}, X = -1
```

and `name = G3`, then this is written in `outf` as

`G3: group = {CV01[kick]: (2 - 0.5)*X, CV02[kick]: (3 - 1)*X}, X = 0`

Any `oldX` in the group elements are ignored. This method is useful when creating one knob out of 
a BAGELS solution consisting of many different knobs, for example.
"""
function groups_to_one_knob(groupsf, name, outf)
  txt = read(groupsf, String)

  # Get all the groups
  group_idxs = findall("group", txt)
  # Elements
  eles = []
  # Corresponding attribute for each element
  atts = []

  # New expressions for element in one group
  exprs = [] 

  for i=1:length(group_idxs)
    if i != length(group_idxs)
      grouptxt = txt[group_idxs[i][end]:group_idxs[i+1][begin]]
    else
      grouptxt = txt[group_idxs[i][end]:end]
    end

    # Get the X for the group: remove whitespace, find X=<number>,
    X = parse(Float64, replace(grouptxt, " " => "")[findfirst(r"X=(.*?)(?=,|\n|$)\b", replace(grouptxt, " " => ""))][3:end])

    # Substitute the X in the expressions
    grouptxtnew = replace(grouptxt, "X" => "$(X)")

    # Now go through each expression in the group
    for m in eachmatch(r"\w*\[\w*\](.+?)(?=,|})", grouptxtnew)
      ele = m.match[findfirst(r"\w*\[", m.match)][1:end-1]
      ele_idx = findfirst(x->x==ele, eles)
      if isnothing(ele_idx)
        push!(eles, ele)
        push!(exprs, "")
        push!(atts, strip(m.match[findfirst('[', m.match)+1:findlast(']', m.match)-1]))
        ele_idx = length(eles)
      end
      exprs[ele_idx] = exprs[ele_idx] * "+(" * strip(m.match[findlast(':', m.match)+1:end]) * ")"
    end
  end

  # now write the group element
  knob_out = open(outf, "w")
  println(knob_out, "$(name): group = {")
  for i=1:length(eles)
    if i != length(eles)
      println(knob_out, "$(eles[i])[$(atts[i])]: ($(exprs[i]))*X,")
    else
      println(knob_out, "$(eles[i])[$(atts[i])]: ($(exprs[i]))*X")
    end
  end
  println(knob_out, "}, var = {X}")
  close(knob_out)
end



# --- Generate misalignments ---

"""
This method is NOT generalized yet:
Generates misalignments Bmad files for each element for the ESR of the EIC
"""
function misalign(lat, seed, outf = "misalign-seed-$(seed).bmad")
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  
  Random.seed!(seed)
  high_beta_dipoles = ["D1EF_6", "D1EF_8", "D2ER_6", "D2ER_8"]
  FF_quads = ["Q0EF_6", "Q0EF_8", "Q1EF_6", "Q1EF_8", "Q1ER_6", "Q1ER_8", "Q2ER_6", "Q2ER_8"]

  # Get file of all elements and their types
  if !isfile("$(path)/eles.txt")
    run(`tao -lat $lat -noplot -noinit -nostart -command "show -write $(path)/eles.txt lat * -at s@f20.16; exit"`)
  end
  eles_txt = readdlm("$(path)/eles.txt", skipstart=2)
  row_end = findlast(x->x=="END", eles_txt[:,2])
  ele_names = eles_txt[1:row_end,2]
  ele_types = eles_txt[1:row_end,3]

  out = open(outf, "w")
  i =  1
  while i<length(ele_types)
    if ele_types[i] in high_beta_dipoles
      x_offset = rand(Normal(0,0.0002)) # 0.2mm
      y_offset = rand(Normal(0,0.0002)) # 0.2mm
      roll = rand(Normal(0,0.5e-3))     # 0.5 mrad
      dBpB = rand(Normal(0,0.0005))     # 0.05% field error 
      println(out, "$(ele_names[i])[x_offset] = $(x_offset)")
      println(out, "$(ele_names[i])[y_offset] = $(y_offset)")
      println(out, "$(ele_names[i])[roll] = $(roll)")
      println(out, "$(ele_names[i])[dg] = $(ele_names[i])[g]*$(dBpB)")
      i = i+1
    elseif ele_types[i] in FF_quads
      x_offset = rand(Normal(0,0.0001)) # 0.1mm
      y_offset = rand(Normal(0,0.0001)) # 0.1mm
      roll = rand(Normal(0,0.5e-3))     # 0.5 mrad
      dBpB = rand(Normal(0,0.0005))     # 0.05% field error 
      println(out, "$(ele_names[i])[x_offset] = $(x_offset)")
      println(out, "$(ele_names[i])[y_offset] = $(y_offset)")
      println(out, "$(ele_names[i])[tilt] = $(roll)")
      println(out, "$(ele_names[i])[k1] = $(ele_names[i])[k1]*$(dBpB+1)")
      i = i+1
    elseif ele_types[i] == "Multipole"      # All multipoles in the lattice are edges of bends
      x_offset = rand(Normal(0,0.0002)) # 0.2mm
      y_offset = rand(Normal(0,0.0002)) # 0.2mm
      roll = rand(Normal(0,0.5e-3))     # 0.5 mrad
      dBpB = rand(Normal(0,0.001))      # 0.1% field error 
      println(out, "$(ele_names[i])[x_offset] = $(x_offset)")
      println(out, "$(ele_names[i])[y_offset] = $(y_offset)")
      println(out, "$(ele_names[i])[tilt] = $(roll)")
      println(out, "$(ele_names[i])[k1l] = $(ele_names[i])[k1l]*$(dBpB+1)")
      
      println(out, "$(ele_names[i+1])[x_offset] = $(x_offset)")
      println(out, "$(ele_names[i+1])[y_offset] = $(y_offset)")
      println(out, "$(ele_names[i+1])[roll] = $(roll)")
      println(out, "$(ele_names[i+1])[dg] = $(ele_names[i+1])[g]*$(dBpB)")

      println(out, "$(ele_names[i+2])[x_offset] = $(x_offset)")
      println(out, "$(ele_names[i+2])[y_offset] = $(y_offset)")
      println(out, "$(ele_names[i+2])[tilt] = $(roll)")
      println(out, "$(ele_names[i+2])[k1l] = $(ele_names[i+2])[k1l]*$(dBpB+1)")
      i = i+3
    elseif ele_types[i] == "Sbend"
      x_offset = rand(Normal(0,0.0002)) # 0.2mm
      y_offset = rand(Normal(0,0.0002)) # 0.2mm
      roll = rand(Normal(0,0.5e-3))     # 0.5 mrad
      dBpB = rand(Normal(0,0.001))      # 0.1% field error 
      println(out, "$(ele_names[i])[x_offset] = $(x_offset)")
      println(out, "$(ele_names[i])[y_offset] = $(y_offset)")
      println(out, "$(ele_names[i])[roll] = $(roll)")
      println(out, "$(ele_names[i])[dg] = $(ele_names[i])[g]*$(dBpB)")
      i = i+1
    elseif ele_types[i] == "Quadrupole"
      x_offset = rand(Normal(0,0.0002)) # 0.2mm
      y_offset = rand(Normal(0,0.0002)) # 0.2mm
      roll = rand(Normal(0,0.5e-3))     # 0.5 mrad
      dBpB = rand(Normal(0,0.001))      # 0.1% field error 
      println(out, "$(ele_names[i])[x_offset] = $(x_offset)")
      println(out, "$(ele_names[i])[y_offset] = $(y_offset)")
      println(out, "$(ele_names[i])[tilt] = $(roll)")
      println(out, "$(ele_names[i])[k1] = $(ele_names[i])[k1]*$(dBpB+1)")
    elseif ele_types[i] == "Sextupole"
      x_offset = rand(Normal(0,0.0002)) # 0.2mm
      y_offset = rand(Normal(0,0.0002)) # 0.2mm
      roll = rand(Normal(0,0.5e-3))     # 0.5 mrad
      dBpB = rand(Normal(0,0.002))      # 0.2% field error 
      println(out, "$(ele_names[i])[x_offset] = $(x_offset)")
      println(out, "$(ele_names[i])[y_offset] = $(y_offset)")
      println(out, "$(ele_names[i])[tilt] = $(roll)")
      println(out, "$(ele_names[i])[k2] = $(ele_names[i])[k2]*$(dBpB+1)")
    end
    i = i+1
  end
  close(out)
end


# --- Polarization routines ---
"""
Structure to store important polarization quantities calculated 
for a lattice.

"""
struct PolData
  agamma0
  spin_tune
  P_st 
  P_dk 
  P_dk_a 
  P_dk_b 
  P_dk_c 
  P_dk_2a
  P_dk_2b
  P_dk_2c
  tau_bks
  tau_dep
  tau_dep_a
  tau_dep_b
  tau_dep_c
  tau_dep_2a 
  tau_dep_2b 
  tau_dep_2c 
  tau_eq 
  T
  T_du 
  T_dd 
  P_t
  T_du_t 
  T_dd_t 
end

"""
Structure to store important polarization quantities calculated 
from tracking (so nonlinear tau_dep) for a lattice.

"""
struct PolTrackData
  agamma0
  P_st
  P_dk
  tau_bks
  tau_dep
  tau_eq
  T
  T_du 
  T_dd 
  P_t
  T_du_t 
  T_dd_t 
end

"""
Struct for storing the tau_depol from tracking tune scan
"""
struct DepolTuneData
  Qx
  Qy
  Qz
  tau_dep # In Minutes  
  emit_a
  emit_b
  emit_c
end

"""
    BAGELS_1(lat, unit_bump, kick=1e-5, tol=1e-8)

Best Adjustment Groups for ELectron Spin (BAGELS) method step 1: calculates the response of 
dn/ddelta with all combinations of the inputted vertical closed orbit bump types as the 
"unit bumps". The types are:
1. `pi` bump              -- Delocalized coupling, delocalized dispersion
2. `n2pi` bump            -- Localized coupling, delocalized dispersion
3. `n2pi_cancel_eta` bump -- Localized coupling, localized dispersion
4. `2pi` bump             -- Localized coupling, delocalized dispersion
5. `pi_cancel_eta` bump   -- Delocalized coupling, localized dispersion

### Input
- `lat`       -- lat file name
- `unit_bump` -- Type of unit closed orbit bump. Options are: (1) `pi`, (2) `n2pi`, (3) `n2pi_cancel_eta`, (4) `2pi`, (5) `pi_cancel_eta`
- `kick`      -- (Optional) Coil kick, default is 1e-5
- `tol`       -- (Optional) Tolerance for difference in phase, default is 1e-8
"""
function BAGELS_1(lat, unit_bump, kick=1e-5, tol=1e-8)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end

  str_kick = @sprintf("%1.2e", kick) 

  if unit_bump == 1     # pi_bump
    str_bump = "pi"
  elseif unit_bump == 2 # n2pi_bump
    str_bump = "n2pi"
  elseif unit_bump == 3 # n2pi_cancel_eta_bump
    str_bump = "n2pi_cancel_eta"
  elseif unit_bump == 4 # 2pi_bump
    str_bump = "2pi"
  elseif unit_bump == 5 # pi_cancel_eta_bump
    str_bump = "pi_cancel_eta"
  else
    println("Unit bump type not defined!")
    return
  end

  # Generate directory in lattice data path for these phi_start and phi_step
  if !isdir("$(path)/BAGELS_$(str_bump)")
    mkdir("$(path)/BAGELS_$(str_bump)")
  end

  path = "$(path)/BAGELS_$(str_bump)"

  # First, obtain all combinations of bumps with desired phase advance
  if !isfile("$(path)/groups.txt")
    if !isfile("$(path)/vkickers.txt")
      run(`tao -lat $lat -noplot -noinit -nostart -command "set ele * spin_tracking_method = sprint; set ele * kick = 0; show -write $(path)/vkickers.txt lat vkicker::* -at phi_b@f20.16; exit"`)
    end
    if !isfile("$(path)/kicks.txt")
      run(`tao -lat $lat -noplot -noinit -nostart -command "set ele * spin_tracking_method = sprint; show -write $(path)/kicks.txt lat vkicker::* -at phi_b@f20.16 -at kick@f20.16; exit"`)
    end

    # Read coil data and extract valid pairs
    coil_data = readdlm("$(path)/vkickers.txt", skipstart=2)
    coil_names = coil_data[1:end-2, 2]
    coil_phis = coil_data[1:end-2, 6]
    coil_kicks = readdlm("$(path)/kicks.txt", skipstart=2)[1:end-2,7]

    unit_groups_list = []
    unit_curkicks_list = []
    unit_sgns_list = []

    # Loop through coil names and determine the unit_groups
    if unit_bump == 1     # pi_bump
      for i=1:length(coil_names)-1
        for j=i+1:length(coil_names)
          if abs(coil_phis[j] - coil_phis[i] - pi) < tol
            push!(unit_groups_list, coil_names[i])
            push!(unit_groups_list, coil_names[j])
            push!(unit_sgns_list, +1.)
            push!(unit_sgns_list, +1.)  # Coils in pi bumps have same kick sign and strength
            push!(unit_curkicks_list, coil_kicks[i])
            push!(unit_curkicks_list, coil_kicks[j])
          end
        end
      end
      n_per_group = 2
    elseif unit_bump == 2 # n2pi_bump
      for i=1:length(coil_names)-1
        for j=i+1:length(coil_names)
          if abs(rem(coil_phis[j] - coil_phis[i], 2*pi, RoundNearest)) < tol
            push!(unit_groups_list, coil_names[i])
            push!(unit_groups_list, coil_names[j]) 
            push!(unit_sgns_list, +1.)
            push!(unit_sgns_list, -1.)  # Coils in 2npi bumps have opposite kick sign but same strength
            push!(unit_curkicks_list, coil_kicks[i])
            push!(unit_curkicks_list, coil_kicks[j])
          end
        end
      end
      n_per_group = 2
    elseif unit_bump == 3 # n2pi_cancel_eta_bump
      for i=1:length(coil_names)-3
        for j=i+1:length(coil_names)-2
          for k=j+1:length(coil_names)-1
            for l=k+1:length(coil_names)
              # The second two coils can be either in phase (2pin apart) and opposite sign, or out of 
              # phase with same sign)
              # First set and second set must be separate n2pi bump:
              if abs(rem(coil_phis[j] - coil_phis[i], 2*pi, RoundNearest)) < tol && abs(rem(coil_phis[l] - coil_phis[k], 2*pi, RoundNearest)) < tol
                # Now check if either in phase exactly or out of phase exactly:
                if abs(rem(coil_phis[j] -coil_phis[k], 2*pi, RoundNearest)) < tol   # In phase exactly
                  push!(unit_groups_list, coil_names[i])
                  push!(unit_groups_list, coil_names[j])
                  push!(unit_groups_list, coil_names[k])
                  push!(unit_groups_list, coil_names[l])
                  push!(unit_sgns_list, +1.)
                  push!(unit_sgns_list, -1.)  
                  push!(unit_sgns_list, -1.)  
                  push!(unit_sgns_list, +1.)  
                  push!(unit_curkicks_list, coil_kicks[i])
                  push!(unit_curkicks_list, coil_kicks[j])
                  push!(unit_curkicks_list, coil_kicks[k])
                  push!(unit_curkicks_list, coil_kicks[l])
                elseif abs(rem(coil_phis[j] - coil_phis[k] - pi,2*pi, RoundNearest)) < tol   # Out of phase exactly
                  push!(unit_groups_list, coil_names[i])
                  push!(unit_groups_list, coil_names[j])
                  push!(unit_groups_list, coil_names[k])
                  push!(unit_groups_list, coil_names[l])
                  push!(unit_sgns_list, +1.)
                  push!(unit_sgns_list, -1.) 
                  push!(unit_sgns_list, +1.) 
                  push!(unit_sgns_list, -1.) 
                  push!(unit_curkicks_list, coil_kicks[i])
                  push!(unit_curkicks_list, coil_kicks[j])
                  push!(unit_curkicks_list, coil_kicks[k])
                  push!(unit_curkicks_list, coil_kicks[l])
                end
              end
            end
          end
        end
      end
      n_per_group = 4
    elseif unit_bump == 4 # 2pi bump
      for i=1:length(coil_names)-1
        for j=i+1:length(coil_names)
          if abs(coil_phis[j] - coil_phis[i] - 2*pi) < tol
            push!(unit_groups_list, coil_names[i])
            push!(unit_groups_list, coil_names[j]) 
            push!(unit_sgns_list, +1.)
            push!(unit_sgns_list, -1.)  # Coils in 2pi bumps have opposite kick sign but same strength
            push!(unit_curkicks_list, coil_kicks[i])
            push!(unit_curkicks_list, coil_kicks[j])
          end
        end
      end
      n_per_group = 2
    elseif unit_bump == 5 # pi_cancel_eta bump
      for i=1:length(coil_names)-3
        for j=i+1:length(coil_names)-2
          for k=j+1:length(coil_names)-1
            for l=k+1:length(coil_names)
              # First set and second set must be separate pi bump:
              if abs(coil_phis[j] - coil_phis[i]- pi) < tol && abs(coil_phis[l] - coil_phis[k] - pi) < tol
                # Now check if either 2*pi apart
                if abs(rem(coil_phis[j] - coil_phis[k], 2*pi, RoundNearest)) < tol  
                  push!(unit_groups_list, coil_names[i])
                  push!(unit_groups_list, coil_names[j])
                  push!(unit_groups_list, coil_names[k])
                  push!(unit_groups_list, coil_names[l])
                  push!(unit_sgns_list, +1.)
                  push!(unit_sgns_list, +1.)  
                  push!(unit_sgns_list, +1.)  
                  push!(unit_sgns_list, +1.)  
                  push!(unit_curkicks_list, coil_kicks[i])
                  push!(unit_curkicks_list, coil_kicks[j])
                  push!(unit_curkicks_list, coil_kicks[k])
                  push!(unit_curkicks_list, coil_kicks[l])
                end
              end
            end
          end
        end
      end
      n_per_group = 4
    end

    # Info to store is: each coil in the group, each of their current strengths, and each of their kick sgns/mags
    # Just write to three separate files for now
  
    unit_groups = permutedims(reshape(unit_groups_list, n_per_group, Int(length(unit_groups_list)/n_per_group)))
    unit_sgns = permutedims(reshape(unit_sgns_list, n_per_group, Int(length(unit_sgns_list)/n_per_group)))
    unit_curkicks = permutedims(reshape(unit_curkicks_list, n_per_group, Int(length(unit_curkicks_list)/n_per_group)))
    
    writedlm("$(path)/groups.txt", unit_groups, ',')
    writedlm("$(path)/sgns.txt", unit_sgns, ',')
    writedlm("$(path)/curkicks.txt", unit_curkicks, ',')
  end

  unit_groups = readdlm("$(path)/groups.txt", ',')
  unit_sgns =  readdlm("$(path)/sgns.txt", ',')
  unit_curkicks = readdlm("$(path)/curkicks.txt", ',')

  if !isdir("$(path)/responses_$(str_kick)")
    mkdir("$(path)/responses_$(str_kick)")
  end
  # All of these bumps are separate group knobs, the individual coils have opposite strengths
  # Now build response matrix of dn/ddelta at each sbend (sampled at beginning and ends of bends)
  tao_cmd = open("$(path)/responses_$(str_kick)/BAGELS_1.cmd", "w")
  println(tao_cmd, "set ele * spin_tracking_method = sprint")
  println(tao_cmd, "set bmad_com spin_tracking_on = T")
  println(tao_cmd, "show -write $(path)/responses_$(str_kick)/baseline.txt lat sbend::* multipole::* -at spin_dn_dpz.x@f20.16 -at spin_dn_dpz.y@f20.16 -at spin_dn_dpz.z@f20.16")

  for i=1:length(unit_groups[:,1])
    str_coils = ""
    for j=1:length(unit_groups[i,:])
      coil = unit_groups[i,j]
      sgn = unit_sgns[i,j]
      curkick = unit_curkicks[i,j]
      println(tao_cmd, "set ele $(coil) kick = $(curkick + sgn*kick)")
      str_coils = str_coils * coil * "_"
    end
    str_coils = str_coils[1:end-1] * ".txt"
    println(tao_cmd, "show -write $(path)/responses_$(str_kick)/$(str_coils) lat sbend::* multipole::* -at spin_dn_dpz.x@f20.16 -at spin_dn_dpz.y@f20.16 -at spin_dn_dpz.z@f20.16")
    # Reset coils to original strengths
    for j=1:length(unit_groups[i,:])
      coil = unit_groups[i,j]
      curkick = unit_curkicks[i,j]
      println(tao_cmd, "set ele $(coil) kick = $(curkick)")
    end
  end
  println(tao_cmd, "exit")
  close(tao_cmd)

  run(`tao -lat $lat -noplot -noinit -nostart -command "call $(path)/responses_$(str_kick)/BAGELS_1.cmd"`)
end

"""
    BAGELS_2(lat, unit_bump; suffix="", outf="\$(lat)_BAGELS.bmad", kick=1e-5))

Best Adjustment Groups for ELectron Spin (BAGELS) method step 2: peform an SVD of the 
response matrix to obtain the best adjustment groups, based on the settings of step 1. 
The vertical closed orbit unit bump types are:
1. `pi` bump              -- Delocalized coupling, delocalized dispersion
2. `n2pi` bump            -- Localized coupling, delocalized dispersion
3. `n2pi_cancel_eta` bump -- Localized coupling, localized dispersion
4. `2pi` bump             -- Localized coupling, delocalized dispersion
5. `pi_cancel_eta` bump   -- Delocalized coupling, localized dispersion

### Input
- `lat`       -- lat file name
- `unit_bump` -- Type of unit closed orbit bump. Options are: (1) `pi`, (2) `n2pi`, (3) `n2pi_cancel_eta`, (4) `2pi`, (5) `pi_cancel_eta`
- `coil_regex` -- (Optional) Regex of coils to match to (e.g. for all coils ending in `_7` or `_11`, use `r".*_(?:7|11)\\b"`)
- `suffix`     -- (Optional) Suffix to append to group elements generated for knobs in Bmad format
- `outf`       -- (Optional) Output file name with group elements constructed from BAGELS, default is `\$(lat)_BAGELS.bmad`
- `kick`       -- (Optional) Coil kick, default is 1e-5
"""
function BAGELS_2(lat, unit_bump; coil_regex=r".*", suffix="", outf="$(lat)_BAGELS.bmad", kick=1e-5)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end

  str_kick = @sprintf("%1.2e", kick)
  if unit_bump == 1     # pi_bump
    str_bump = "pi"
  elseif unit_bump == 2 # n2pi_bump
    str_bump = "n2pi"
  elseif unit_bump == 3 # n2pi_cancel_eta_bump
    str_bump = "n2pi_cancel_eta"
  elseif unit_bump == 4 # 2pi_bump
    str_bump = "2pi"
  elseif unit_bump == 5 # pi_cancel_eta_bump
    str_bump = "pi_cancel_eta"
  else
    println("Unit bump type not defined!")
    return
  end
  # Generate directory in lattice data path for these phi_start and phi_step
  if !isdir("$(path)/BAGELS_$(str_bump)")
    mkdir("$(path)/BAGELS_$(str_bump)")
  end

  path = "$(path)/BAGELS_$(str_bump)"

  unit_groups_raw = readdlm("$(path)/groups.txt", ',')
  unit_sgns_raw =  readdlm("$(path)/sgns.txt", ',')

  eletypes = readdlm("$(path)/responses_$(str_kick)/baseline.txt", skipstart=2)[1:end-2,3]
  dn_dpz0 = readdlm("$(path)/responses_$(str_kick)/baseline.txt", skipstart=2)[1:end-2,6:8]

  # Get number of Sbends to only sample at Sbends
  idx_bends = findall(i-> i=="Sbend", eletypes)
  dn_dpz0_bends = dn_dpz0[idx_bends,:]

  # Only use unit groups where all coils in the group match the regex
  unit_groups_list = []
  unit_sgns_list = []
  n_per_group = length(unit_groups_raw[1,:])
  for i=1:length(unit_groups_raw[:,1]) # for each group
    use = true
    for j=1:n_per_group
      if !occursin(coil_regex, unit_groups_raw[i,j])
        use = false
      end
    end
    if use
      for j=1:n_per_group
        push!(unit_groups_list, unit_groups_raw[i,j])
        push!(unit_sgns_list, unit_sgns_raw[i,j])
      end
    end
  end

  unit_groups = permutedims(reshape(unit_groups_list, (n_per_group,floor(Int, length(unit_groups_list)/n_per_group))))
  unit_sgns = permutedims(reshape(unit_sgns_list, (n_per_group,floor(Int, length(unit_sgns_list)/n_per_group))))

  # We now will have a response matrix that is N_bends x N_groups to multiply with vector N_groups x 1
  delta_dn_dpz = zeros(length(idx_bends), length(unit_groups[:,1]))
  for i=1:length(unit_groups[:,1])
    str_coils = ""
    for j=1:length(unit_groups[i,:])
      coil = unit_groups[i,j]
      str_coils = str_coils * coil * "_"
    end
    str_coils = str_coils[1:end-1] * ".txt"
    dn_dpz_bends = readdlm("$(path)/responses_$(str_kick)/$(str_coils)", skipstart=2)[1:end-2,6:8][idx_bends,:]
    #dn_dpz_bends_amp = [norm(dn_dpz_bends[j,:]) for j=1:length(dn_dpz_bends[:,1])]
    #delta_dn_dpz_amp = dn_dpz_bends_amp .- dn_dpz0_bends_amp
    #delta_dn_dpz_amp = [norm(delta_dn_dpz[j,:]) for j=1:length(delta_dn_dpz[:,1])]
    # For now, BAGELS only uses x component of dn/ddelta
    delta_dn_dpz[:,i] = (dn_dpz_bends[:,1] .- dn_dpz0_bends[:,1])./kick
  end
  
  # Now do SVD to get the principal components
  F = svd(delta_dn_dpz)

  # Get first N_knobs principal directions
  V = F.V

  # Create a group element with these as knobs  
  for i=1:length(F.S)
    knob = V[:,i]
    # Create matrix containing the strength of individual coils based on the 
    # the strength of that coil in each unit group
    raw = Matrix{Any}(undef, length(unit_groups), 2)
    # First column of raw contains coil name, second is knob value
    # RAW CONTAINS DUPLICATES INTENTIONALLY!!! This will be accounted for later
    for j=1:length(unit_groups[:,1])
      for k=1:n_per_group
        coil = unit_groups[j,k]
        sgn = unit_sgns[j,k]
        raw[n_per_group*j-n_per_group+k, 1] = coil
        raw[n_per_group*j-n_per_group+k, 2] = sgn*knob[j]
      end
    end

    # Now make knobs for each
    unique_coils = unique(unit_groups)

    if i == 1
      knob_out = open(outf, "w")
    else
      knob_out = open(outf, "a")
    end
    
    println(knob_out, "BAGELS$(suffix)$(Printf.format(Printf.Format("%0$(length(string(length(F.S))))i"), i)): group = {\t\t\t ! Singular value = $(F.S[i])")
    
    coil = unique_coils[1]
    strength = 0.
    idxs_coil = findall(i->i==coil, raw[:,1])
    strength = sum(raw[idxs_coil,2])
    print(knob_out, "$(coil)[kick]: $(strength)*X")

    for coil in Iterators.drop(unique_coils, 1)
      strength = 0.
      idxs_coil = findall(i->i==coil, raw[:,1])
      strength = sum(raw[idxs_coil,2])
      print(knob_out, ",\n$(coil)[kick]: $(strength)*X")
    end
    println(knob_out, "}, var = {X}")
    close(knob_out)
  end

  print("\nSingular values are: ")
  print(F.S)
  return F
end


"""
    calc_P_t(P_dk, tau_eq; P_0 = 0.85, T = 4.8)

Calculates the time-averaged e-polarization one could maintain in a lattice with 
equal bunches parallel and antiparallel to the arc dipole fields, assuming an 
average bunch replacement rate `T` and initial injected polarization `P_0`. Defaults 
are values used in the ESR of the EIC (`P_0` = 85%, `T` = 4.8 min)

### Input
- `P_dk`   -- Array of DK polarizations
- `tau_eq` -- Array of corresponding equilibrium times
- `P_0`    -- (Optional) Initial bunch polarization at injection
- `T`      -- (Optional) Average bunch replacement rate

### Output
- `P_t`    -- Time-averaged polarization maintainable in ring
- `T_du_t` -- How often the antiparallel bunches need replacement
- `T_dd_t` -- How often the parallel bunches need replacement
"""
function calc_P_t(P_dk, tau_eq; P_0 = 0.85, T = 4.8)
  P_dd = x -> (-P_0.*tau_eq .- P_dk.*tau_eq .+ P_dk.*x .+ (P_0.*tau_eq .+ P_dk.*tau_eq).*exp.(-x./tau_eq))./x
  P_du = x -> (P_0.*tau_eq .- P_dk.*tau_eq .+ P_dk.*x .+ (-P_0.*tau_eq .+ P_dk.*tau_eq).*exp.(-x./tau_eq))./x
  P = T_du -> (P_dd((T_du.*T./(2 .*T_du.-T))) + P_du(T_du)).^2

  T_du_t = nlsolve(P, 10. .*ones(length(P_dk))).zero
  T_dd_t = (T_du_t.*T./(2 .*T_du_t.-T))
  P_t = P_du(T_du_t)

  return P_t, T_du_t, T_dd_t
end


"""
    calc_T(P_dk, tau_eq; P_0 = 0.85, P_min_avg = 0.7)

Calculates how often the bunches would need to be replaced in a lattice with 
equal bunches parallel and antiparallel to the arc dipole fields to maintain 
a minimum time-averaged polarization of `P_min_avg` Defaults are values used 
in the ESR of the EIC (`P_0` = 85%, `P_min_avg` = 70%)

### Input
- `P_dk`      -- Array of DK polarizations
- `tau_eq`    -- Array of corresponding equilibrium times
- `P_0`       -- (Optional) Initial bunch polarization at injection
- `P_min_avg` -- (Optional) Minimum-allowable time-averaged polarization

### Output
- `T`    -- Average bunch replacement time to maintain `P_min_avg`
- `T_du` -- Antiparallel bunch replacement time to maintain `P_min_avg`
- `T_dd` -- Parallel bunch replacement time to maintain `P_min_avg`
"""
function calc_T(P_dk, tau_eq; P_0 = 0.85, P_min_avg = 0.7)
  P_dd = x -> (-P_0.*tau_eq .- P_dk.*tau_eq .+ P_dk.*x .+ (P_0.*tau_eq .+ P_dk.*tau_eq).*exp.(-x./tau_eq))./x .+ P_min_avg
  P_du = x -> (P_0.*tau_eq .- P_dk.*tau_eq .+ P_dk.*x .+ (-P_0.*tau_eq .+ P_dk.*tau_eq).*exp.(-x./tau_eq))./x .- P_min_avg
  T_dd = nlsolve(P_dd, 10. .*ones(length(P_dk))).zero
  T_du = nlsolve(P_du, 10. .*ones(length(P_dk))).zero
  
  T=2*T_dd.*T_du./(T_dd + T_du)
  return T, T_du, T_dd
end



"""
    pol_scan(lat, agamma0)

Performs a polarization scan across the agamma0 range specified, calculates 
important quantities and stores them for fast reference in the data. 
The data can be retrieved using the `get_pol_data` method.

### Input
- `lat`      -- lat file name
- `agamma0`  -- range of `agamma0` to scan over

### Output 
- `pol_data` -- PolData struct containing polarization quantities for lattice
"""
function pol_scan(lat, agamma0)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end

  # Generate tao command script
  tao_cmd = open("$(path)/spin.tao", "w")
  a_e = 0.00115965218128
  m_e = 0.51099895e6
  println(tao_cmd, "set ele * spin_tracking_method = sprint")
  println(tao_cmd, "set bmad_com spin_tracking_on = T")
  println(tao_cmd, "set ele 0 e_tot = $(agamma0[1]*m_e/a_e)")
  println(tao_cmd, "python -write $(path)/spin.dat spin_polarization")
  for i in Iterators.drop(eachindex(agamma0), 1)
    println(tao_cmd, "set ele 0 e_tot = $(agamma0[i]*m_e/a_e)")
    println(tao_cmd, "python -append $(path)/spin.dat spin_polarization")
  end

  println(tao_cmd, "exit")
  close(tao_cmd)

  # Execute the scan
  run(`tao -lat $lat -noplot -noinit -nostart -command "call $(path)/spin.tao"`)

  # Calculate important quantities and store in data
  data = readdlm("$(path)/spin.dat", ';')[:,4]
  data = permutedims(reshape(data, (18,Int(length(data)/18))))
  agamma0    = data[:,1]
  spin_tune  = data[:,2]
  P_st       = data[:,3]
  P_dk       = data[:,4]
  P_dk_a     = data[:,5]
  P_dk_b     = data[:,6]
  P_dk_c     = data[:,7]
  P_dk_2a    = data[:,8]
  P_dk_2b    = data[:,9]
  P_dk_2c    = data[:,10]
  tau_bks    = 1 ./data[:,11] ./ 60
  tau_dep    = 1 ./data[:,12] ./ 60
  tau_dep_a  = 1 ./data[:,13] ./ 60
  tau_dep_b  = 1 ./data[:,14] ./ 60
  tau_dep_c  = 1 ./data[:,15] ./ 60
  tau_dep_2a = 1 ./data[:,16] ./ 60
  tau_dep_2b = 1 ./data[:,17] ./ 60
  tau_dep_2c = 1 ./data[:,18] ./ 60
  tau_eq = (tau_dep.^-1+tau_bks.^-1).^-1
  T, T_du, T_dd = calc_T(P_dk,tau_eq)
  P_t, T_du_t, T_dd_t = calc_P_t(P_dk,tau_eq)

  pol_data = PolData( agamma0,
                      spin_tune,
                      P_st,
                      P_dk,
                      P_dk_a,
                      P_dk_b,
                      P_dk_c,
                      P_dk_2a,
                      P_dk_2b,
                      P_dk_2c,
                      tau_bks,
                      tau_dep,
                      tau_dep_a,
                      tau_dep_b,
                      tau_dep_c,
                      tau_dep_2a,
                      tau_dep_2b,
                      tau_dep_2c,
                      tau_eq,
                      T,
                      T_du,
                      T_dd,
                      P_t,
                      T_du_t,
                      T_dd_t)
  names =["agamma0",
          "spin_tune",
          "P_st",
          "P_dk",
          "P_dk_a",
          "P_dk_b",
          "P_dk_c",
          "P_dk_2a",
          "P_dk_2b",
          "P_dk_2c",
          "tau_bks",
          "tau_dep",
          "tau_dep_a",
          "tau_dep_b",
          "tau_dep_c",
          "tau_dep_2a",
          "tau_dep_2b",
          "tau_dep_2c",
          "tau_eq",
          "T",
          "T_du",
          "T_dd",
          "P_t",
          "T_du_t",
          "T_dd_t"]
  pol_data_dlm = permutedims(hcat(names, permutedims(hcat(agamma0,
                                                            spin_tune,
                                                            P_st,
                                                            P_dk,
                                                            P_dk_a,
                                                            P_dk_b,
                                                            P_dk_c,
                                                            P_dk_2a,
                                                            P_dk_2b,
                                                            P_dk_2c,
                                                            tau_bks,
                                                            tau_dep,
                                                            tau_dep_a,
                                                            tau_dep_b,
                                                            tau_dep_c,
                                                            tau_dep_2a,
                                                            tau_dep_2b,
                                                            tau_dep_2c,
                                                            tau_eq,
                                                            T,
                                                            T_du,
                                                            T_dd,
                                                            P_t,
                                                            T_du_t,
                                                            T_dd_t))))
  writedlm("$(path)/pol_data.dlm", pol_data_dlm, ';')
  return pol_data
end


"""
    get_pol_data(lat)

Gets the polarization data for the specified lattice.
"""
function get_pol_data(lat)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  if !isfile("$(path)/pol_data.dlm")
    println("First-order polarization data not generated for lattice $(lat). Please use the pol_scan method to generate the data.")
  end
  pol_data_dlm = readdlm("$(path)/pol_data.dlm", ';')[2:end,:]
  

  return PolData( pol_data_dlm[:,1],
                  pol_data_dlm[:,2],
                  pol_data_dlm[:,3],
                  pol_data_dlm[:,4],
                  pol_data_dlm[:,5],
                  pol_data_dlm[:,6],
                  pol_data_dlm[:,7],
                  pol_data_dlm[:,8],
                  pol_data_dlm[:,9],
                  pol_data_dlm[:,10],
                  pol_data_dlm[:,11],
                  pol_data_dlm[:,12],
                  pol_data_dlm[:,13],
                  pol_data_dlm[:,14],
                  pol_data_dlm[:,15],
                  pol_data_dlm[:,16],
                  pol_data_dlm[:,17],
                  pol_data_dlm[:,18],
                  pol_data_dlm[:,19],
                  pol_data_dlm[:,20],
                  pol_data_dlm[:,21],
                  pol_data_dlm[:,22],
                  pol_data_dlm[:,23],
                  pol_data_dlm[:,24],
                  pol_data_dlm[:,25])
end


"""
    run_pol_tune_scan(lat,  n_particles, n_turns, order, Qx_range, Qy_range, Qz_range, sh)

Runs a tune scan for the grid defined in Qx and Qy with 1st order map
tracking with radiation.

"""
function run_pol_tune_scan(lat,  n_particles, n_turns, order, Qx_range, Qy_range, Qz_range, sh)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  if order == 1
    track_path = "$(path)/pol_tune_scan/1st_order_map"
  elseif order == 2
    track_path = "$(path)/pol_tune_scan/2nd_order_map"
  elseif order == 3
    track_path = "$(path)/pol_tune_scan/3rd_order_map"
  elseif order == 0
    track_path = "$(path)/pol_tune_scan/bmad"
  else
    error("only order < 3 or Bmad tracking supported right now")
  end
  if !ispath(track_path)
    mkpath(track_path)
  end

  # Create subdirectories for all tunes
  for Qx in Qx_range
    subdirname1 = Printf.format(Printf.Format("%02.4f"), Qx)
    for Qy in Qy_range
      subdirname2 = Printf.format(Printf.Format("%02.4f"), Qy)
      for Qz in Qz_range
        subdirname3 = Printf.format(Printf.Format("%02.4f"), Qz)
        # ONLY CREATE NEW LAT FILES IF THEY DON'T EXIST YET
        temp_lat = "$(track_path)/$(subdirname1)/$(subdirname2)/$(subdirname3)/$(lat)"
        tao_cmd = open("$(path)/spin_tune_scan.tao", "w")
        println(tao_cmd, "set tune $Qx $Qy -mask *,~hq%_*_*; set z_tune $Qz")
        println(tao_cmd, "write bmad -form one_file $(temp_lat)")
        println(tao_cmd, "exit")
        close(tao_cmd)
        if !isfile(temp_lat)
          mkpath("$(track_path)/$(subdirname1)/$(subdirname2)/$(subdirname3)")
          cp(lat,temp_lat)
          run(`tao -lat $lat -noplot -noinit -nostart -command "call $(path)/spin_tune_scan.tao"`)
        end
        # Run the tracking
        run_map_tracking(temp_lat, n_particles, n_turns, order; use_data_path=false,sh = sh)
      end
    end
  end
end


"""
  run_map_tracking(lat, n_particles, n_turns, order; use_data_path=true, sh=nothing)

NOTE: THIS FUNCTION ONLY WORKS WHEN LOGGED INTO A COMPUTER ON THE CLASSE VPN!
This routine sets up 3rd order map tracking with the bends split for radiation, 
starting the distribution with the equilibrium emittances of the lattice. The 
job is then submitted to the CLASSE cluster for parallel evaluation and stored 
in a directory on the CLASSE machine in ~/trackings_jl/. First, 1 particle is 
tracked for 1 turn using only 1 thread, and then n_particles are tracked for 
n_turns using the maximum number of threads (32). This is the most efficient 
way of tracking on the CLASSE cluster.

IMPORTANT: Before first running this function, a softlink should be set up in a 
user's home directory on the CLASSE machine that points to the user's directory 
/nfs/acc/user/<NetID>. E.g.,

cd /nfs/acc/user/<NetID>
mkdir trackings_jl
ln -s /nfs/acc/user/<NetID> ~/trackings_jl

This must be done because storage is limited in the users' home directory, but not 
in /nfs/acc/user/<NetID>
"""
function run_map_tracking(lat, n_particles, n_turns, order; use_data_path=true, sh=nothing)
  if (use_data_path)
    path = data_path(lat)
    if path == ""
      println("Lattice file $(lat) not found!")
      return
    end
    if order == 1
      track_path = "$(path)/1st_order_map"
    elseif order == 2
      track_path = "$(path)/2nd_order_map"
    elseif order == 3
      track_path = "$(path)/3rd_order_map"
    elseif order == 0
      track_path = "$(path)/bmad"
    else
      error("only order < 3 or Bmad tracking supported right now")
    end
    if !ispath(track_path)
      mkpath(track_path)
    end
  else
    path = dirname(abspath(lat))
    track_path = path
  end

  if order == 1
    method = "MAP"
  elseif order == 2
    method = "MAP"
  elseif order == 3
    method = "MAP"
  elseif order == 0
    method = "BMAD"
    order=3
  else
    error("only order < 3 or Bmad tracking supported right now")
  end
  

  

  # In the tracking directory, we must create the long_term_tracking.init, run.sh, and qtrack.sh, 
  # which will then be scp-ed to the equivalent directory on the remote machine. On the remote 
  # machine, tracking info is stored in ~/trackings_jl/, where the full path to the lattice on 
  # this machine will be written and the tracking folder dropped there.
  # Old executable: /home/mgs255/mgs255/master/production/bin/long_term_tracking_mpi
  run1_sh =    """
                cd \$1
                mpirun -np 1 /home/mgs255/mgs255/master/production/bin/long_term_tracking_mpi long_term_tracking1.init
              """
  run32_sh =    """
              cd \$1
              mpirun -np 32 /home/mgs255/mgs255/master/production/bin/long_term_tracking_mpi long_term_tracking.init
            """
  qtrack_sh = """
                set -x
                p1=\$(pwd)
                p3=\$(basename \$PWD)
                qsub -q all.q -pe sge_pe 32 -N $(replace(basename(lat), "."=>"_") * "_32") -o \${p1}/out.txt -e \${p1}/err.txt -hold_jid \$(qsub -terse -q all.q -pe sge_pe 1 -N $(replace(basename(lat), "."=>"_") * "_1") -o \${p1}/out.txt -e \${p1}/err.txt \${p1}/run1.sh \${p1}) \${p1}/run32.sh \${p1}
              """
  #= No longer necessary - use negative value for equilibrium emittance
  # We need to get the equilibrium emittances to start the tracking with those:
  run(`tao -lat $lat -noplot -command "show -write $(path)/uni.txt uni ; exit"`)
  
  uni = readdlm("$(path)/uni.txt")
  emit_a = uni[26,3]
  emit_b = uni[26,5]
  sig_z = uni[33,3]
  sig_pz = uni[32,3]
  =#
  long_term_tracking1_init = """
                            &params
                            ltt%lat_file = '$(basename(lat))'         ! Lattice file
                            ltt%ele_start = '0'                   ! Where to start in the lattice
                            ltt%ele_stop = ''                     

                            ltt%averages_output_file = 'data'
                            ltt%beam_binary_output_file = ''
                            ltt%sigma_matrix_output_file = ''
                            ltt%averages_output_every_n_turns = 1
                            ltt%averaging_window = 1
                            ltt%core_emit_cutoff = 1,0.95,0.9,0.85,0.80,0.78,0.76,0.74,0.72,0.70,0.68,0.66,0.64,0.62,0.60,0.58,0.56,0.54,0.52,0.50,0.48,0.46,0.42,0.40,0.35,0.3,0.25,0.2,0.15,0.1

                            ltt%simulation_mode = 'BEAM'
                            ltt%tracking_method = '$(method)'   !
                            ltt%n_turns = 1                 ! Number of turns to track
                            ltt%rfcavity_on = T
                            ltt%map_order = $(order) ! increase see effects
                            ltt%split_bends_for_stochastic_rad = T ! for map tracking = T
                            ltt%random_seed = 1                     ! Random number seed. 0 => use system clock.
                            ltt%timer_print_dtime = 300
                            ltt%add_closed_orbit_to_init_position = T

                            bmad_com%spin_tracking_on = T         ! See Bmad manual for bmad_com parameters.
                            bmad_com%radiation_damping_on = T
                            bmad_com%radiation_fluctuations_on = T

                            beam_init%n_particle =  1
                            beam_init%spin = 0, 0, 0

                            beam_init%a_emit = -1.
                            beam_init%b_emit = -1.
                            beam_init%sig_z = -1.
                            beam_init%sig_pz = -1. 
                            /
                            """

  long_term_tracking_init = """
                            &params
                            ltt%lat_file = '$(basename(lat))'         ! Lattice file
                            ltt%ele_start = '0'                   ! Where to start in the lattice
                            ltt%ele_stop = ''                     

                            ltt%averages_output_file = 'data'
                            ltt%beam_binary_output_file = ''
                            ltt%sigma_matrix_output_file = ''
                            ltt%averages_output_every_n_turns = 1
                            ltt%averaging_window = 1
                            ltt%core_emit_cutoff = 1,0.95,0.9,0.85,0.80,0.78,0.76,0.74,0.72,0.70,0.68,0.66,0.64,0.62,0.60,0.58,0.56,0.54,0.52,0.50,0.48,0.46,0.42,0.40,0.35,0.3,0.25,0.2,0.15,0.1

                            ltt%simulation_mode = 'BEAM'
                            ltt%tracking_method = '$(method)'   !
                            ltt%n_turns = $(n_turns)                 ! Number of turns to track
                            ltt%rfcavity_on = T
                            ltt%map_order = $(order) ! increase see effects
                            ltt%split_bends_for_stochastic_rad = T ! for map tracking = T
                            ltt%random_seed = 1                     ! Random number seed. 0 => use system clock.
                            ltt%timer_print_dtime = 300
                            ltt%add_closed_orbit_to_init_position = T

                            bmad_com%spin_tracking_on = T         ! See Bmad manual for bmad_com parameters.
                            bmad_com%radiation_damping_on = T
                            bmad_com%radiation_fluctuations_on = T

                            beam_init%n_particle =  $(n_particles)
                            beam_init%spin = 0, 0, 0

                            beam_init%a_emit = -1.
                            beam_init%b_emit = -1.
                            beam_init%sig_z = -1.
                            beam_init%sig_pz = -1. 
                            /
                            """

  remote_path = "~/trackings_jl" * track_path
  
  if isnothing(sh)
    write("$(track_path)/run1.sh", run1_sh)
    write("$(track_path)/run32.sh", run32_sh)
    write("$(track_path)/qtrack.sh", qtrack_sh)
    write("$(track_path)/long_term_tracking.init", long_term_tracking_init)
    write("$(track_path)/long_term_tracking1.init", long_term_tracking1_init)
    # Create directories on host:
    run(`ssh lnx4200 "mkdir -p $(remote_path)"`)
    # Copy lattice file (use rsync so it remains unchanged if the files are equivalent)
    run(`rsync -t $(lat) lnx4200:$(remote_path)`)
    # Copy files over:
    run(`scp $(track_path)/run1.sh $(track_path)/run32.sh $(track_path)/qtrack.sh $(track_path)/long_term_tracking.init $(track_path)/long_term_tracking1.init lnx4200:$(remote_path)`)
    # Submit the tracking on host
    run(`ssh lnx4200 "cd $(remote_path) sh qtrack.sh"`)
  else
    # Create directories on host:
    println(sh, "mkdir -p $(remote_path)")
    lat_txt = ""
    for line in readlines(lat)
      lat_txt = lat_txt * line * "\n"
    end
    lat_txt = replace(lat_txt, "\"" => "\\\"")
    run1_sh = replace(run1_sh, "\"" => "\\\"")
    run32_sh = replace(run32_sh, "\"" => "\\\"")
    qtrack_sh = replace(qtrack_sh, "\"" => "\\\"")
    long_term_tracking1_init = replace(long_term_tracking1_init, "\"" => "\\\"")
    long_term_tracking_init = replace(long_term_tracking_init, "\"" => "\\\"")

    lat_txt = replace(lat_txt, "\$" => "\\\$")
    run1_sh = replace(run1_sh, "\$" => "\\\$")
    run32_sh = replace(run32_sh, "\$" => "\\\$")
    qtrack_sh = replace(qtrack_sh, "\$" => "\\\$")
    long_term_tracking1_init = replace(long_term_tracking1_init, "\$" => "\\\$")
    long_term_tracking_init = replace(long_term_tracking_init, "\$" => "\\\$")

    if (length("echo \"lat_txt\" > $(remote_path)") > 2621440)
      println("ERROR: lnx4200 terminal command line characters exceeded")'
      return
    end

    println(sh, "echo \"$(lat_txt)\" > $(remote_path)/$(basename(lat))")
    println(sh, "echo \"$(run1_sh)\" > $(remote_path)/run1.sh")
    println(sh, "echo \"$(run32_sh)\" > $(remote_path)/run32.sh")
    println(sh, "echo \"$(qtrack_sh)\" > $(remote_path)/qtrack.sh")
    println(sh, "echo \"$(long_term_tracking_init)\" > $(remote_path)/long_term_tracking.init")
    println(sh, "echo \"$(long_term_tracking1_init)\" > $(remote_path)/long_term_tracking1.init")
    println(sh, "cd $(remote_path)")
    println(sh, "sh qtrack.sh")
  end
end


"""
    run_pol_scan(lat, n_particles, n_turns, order, agamma0)

IMPORTANT: Please follow the setup instructions in the `run_map_tracking`
documentation before running this routine.

This routine sets up and submits the parallel 3rd order map tracking 
jobs to the CLASSE cluster for the range of `agamma0` specified. 
"""
function run_pol_scan(lat, n_particles, n_turns, order, agamma0, sh)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  if order == 1
    track_path = "$(path)/1st_order_map"
  elseif order == 2
    track_path = "$(path)/2nd_order_map"
  elseif order == 3
    track_path = "$(path)/3rd_order_map"
  elseif order == 0
    track_path = "$(path)/bmad"
  else
    error("only order < 3 or Bmad tracking supported right now")
  end
  if !ispath(track_path)
    mkpath(track_path)
  end

  # Create subdirectories with name equal to agamma0
  for i in eachindex(agamma0)
    subdirname = Printf.format(Printf.Format("%0$(length(string(floor(maximum(agamma0))))).2f"), agamma0[i])
    
    # ONLY CREATE NEW FILES IF IT DOESN'T EXIST YET
    temp_lat = "$(track_path)/$(subdirname)/$(lat)_$(subdirname)"
    if !isfile(temp_lat)
      mkpath("$(track_path)/$(subdirname)")
      cp(lat,"$(track_path)/$(subdirname)/$(lat)_$(subdirname)")
      latf = open(temp_lat, "a")
      write(latf, "\nparameter[e_tot] = $(agamma0[i])/anom_moment_electron*m_electron")
      close(latf)
    end

    run_map_tracking(temp_lat, n_particles, n_turns, order; use_data_path=false,sh = sh)
  end
end


"""
    download_pol_scan(lat, order, sh)

This routine reads the results of run_pol_scan from the CLASSE 
computer and creates a PolTrackData for this lattice.
"""
function download_pol_scan(lat, order, sh)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  if order == 1
    track_path = "$(path)/1st_order_map"
  elseif order == 2
    track_path = "$(path)/2nd_order_map"
  elseif order == 3
    track_path = "$(path)/3rd_order_map"
  elseif order == 0
    track_path = "$(path)/bmad"
  else
    error("only order < 3 or Bmad tracking supported right now")
  end
  if !ispath(track_path)
    mkpath(track_path)
  end
  remote_path = "~/trackings_jl" * track_path
  println(sh, "cd $(remote_path)")
  println(sh, "find . -name \"data.ave\" -o -name \"data.emit\" -o -name \"data.sigma\" | find  -name \"data.ave\" -o -name \"data.emit\" -o -name \"data.sigma\" | tar -czvf data.tar.gz -T -")
  println(sh, "scp data.tar.gz \${SSH_CONNECTION%% *}:$(track_path)")
end

function read_pol_scan(lat, order, n_damp)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  if order == 1
    track_path = "$(path)/1st_order_map"
  elseif order == 2
    track_path = "$(path)/2nd_order_map"
  elseif order == 3
    track_path = "$(path)/3rd_order_map"
  elseif order == 0
    track_path = "$(path)/bmad"
  else
    error("only order < 3 or Bmad tracking supported right now")
  end
  if !ispath(track_path)
    mkpath(track_path)
  end
  run(`tar -xzvf $(track_path)/data.tar.gz -C $(track_path)`)

  # Get the tracking subdirs
  subdirs = filter(isdir, readdir(track_path, join=true))

  # Get tracking agamma0
  agamma0 = [readdlm(subdir * "/data.ave")[findfirst(x->x=="anom_moment_times_gamma", readdlm(subdir * "/data.ave")) + CartesianIndex(0,2)] for subdir in subdirs]

  # Now calculate the PolTrackData for this lattice
  # Use first-order P_st and tau_bks to calculate polarization data with nonlinear tau_dep
  pol_data = get_pol_data(lat)
  if isnothing(pol_data)
    # First order scan has not been performed - must be performed
    # Use the agamma0 we have above
    println("First order polarization data not found - performing scan with tracking agamma0...")
    pol_data = pol_scan(lat, agamma0)
  end

  # Tracking energies not particularly at linear evaluation energies, so interpolate
  P_st_interp = linear_interpolation(pol_data.agamma0, pol_data.P_st)
  tau_bks_interp = linear_interpolation(pol_data.agamma0, pol_data.tau_bks)

  tau_dep = Float64[]
  for subdir in subdirs
    data_ave = readdlm(subdir * "/data.ave")
    row_start = findlast(x->occursin("#",string(x)), data_ave[:,1])[1] + 1
    t = Float64.(data_ave[row_start+n_damp:end,3])
    P = Float64.(data_ave[row_start+n_damp:end,4])
    # Linear regression to get depol time
    data = DataFrame(X=t, Y=P)
    tau_dep_track = -coef(lm(@formula(Y ~ X), data))[2]^-1/60
    push!(tau_dep, tau_dep_track)
  end


  P_st = P_st_interp.(agamma0)
  tau_bks = tau_bks_interp.(agamma0)
  tau_eq = (tau_dep.^-1+tau_bks.^-1).^-1
  P_dk = P_st.*tau_bks.^-1 ./(tau_bks.^-1 .+tau_dep.^-1)
  T, T_du, T_dd = calc_T(P_dk,tau_eq)
  P_t, T_du_t, T_dd_t = calc_P_t(P_dk,tau_eq)
  pol_track_data = PolTrackData(agamma0,
                                P_st,
                                P_dk,
                                tau_bks,
                                tau_dep,
                                tau_eq,
                                T,
                                T_du,
                                T_dd,
                                P_t,
                                T_du_t,
                                T_dd_t)
  names =["agamma0",
          "P_st",
          "P_dk",
          "tau_bks",
          "tau_dep",
          "tau_eq",
          "T",
          "T_du",
          "T_dd",
          "P_t",
          "T_du_t",
          "T_dd_t"]
  pol_track_data_dlm = permutedims(hcat(names, permutedims(hcat(agamma0,
                                                                  P_st,
                                                                  P_dk,
                                                                  tau_bks,
                                                                  tau_dep,
                                                                  tau_eq,
                                                                  T,
                                                                  T_du,
                                                                  T_dd,
                                                                  P_t,
                                                                  T_du_t,
                                                                  T_dd_t))))
  writedlm("$(track_path)/pol_track_data.dlm", pol_track_data_dlm, ';')

  return pol_track_data
end

"""
    download_pol_tune_scan(lat, order, sh)

"""
function download_pol_tune_scan(lat, order, sh)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  if order == 1
    track_path = "$(path)/pol_tune_scan/1st_order_map"
  elseif order == 2
    track_path = "$(path)/pol_tune_scan/2nd_order_map"
  elseif order == 3
    track_path = "$(path)/pol_tune_scan/3rd_order_map"
  elseif order == 0
    track_path = "$(path)/pol_tune_scan/bmad"
  else
    error("only order < 3 or bmad tracking (0) supported right now")
  end
  if !ispath(track_path)
    mkpath(track_path)
  end

  remote_path = "~/trackings_jl" * track_path
  println(sh, "cd $(remote_path)")
  println(sh, "find . -name \"data.ave\" -o -name \"data.emit\" -o -name \"data.sigma\" | find  -name \"data.ave\" -o -name \"data.emit\" -o -name \"data.sigma\" | tar -czvf data.tar.gz -T -")
  println(sh, "scp data.tar.gz \${SSH_CONNECTION%% *}:$(track_path)")
end

function read_pol_tune_scan(lat, order, n_damp)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  if order == 1
    track_path = "$(path)/pol_tune_scan/1st_order_map"
  elseif order == 2
    track_path = "$(path)/pol_tune_scan/2nd_order_map"
  elseif order == 3
    track_path = "$(path)/pol_tune_scan/3rd_order_map"
  elseif order == 0
    track_path = "$(path)/pol_tune_scan/bmad"
  else 
    error("only order < 3 or bmad tracking (0) supported right now")
  end
  run(`tar -xzvf $(track_path)/data.tar.gz -C $(track_path)`)

  # Empty DepolTuneData struct which we will fill:
  Qx_data = Float64[]
  Qy_data = Float64[]
  Qz_data = Float64[]
  tau_dep_data = Float64[]
  emit_a_data = Float64[]
  emit_b_data = Float64[]
  emit_c_data = Float64[]
  depol_tune_data = DepolTuneData(Qx_data, Qy_data, Qz_data, tau_dep_data, emit_a_data, emit_b_data, emit_c_data)

  # Get the tracking subdirs
  Qx_subdirs = filter(isdir, readdir(track_path, join=true))
  for Qx_subdir in Qx_subdirs
    Qx = parse(Float64, basename(Qx_subdir))
    Qy_subdirs = filter(isdir, readdir(Qx_subdir, join=true))
    for Qy_subdir in Qy_subdirs
      Qy = parse(Float64, basename(Qy_subdir))
      Qz_subdirs = filter(isdir, readdir(Qy_subdir, join=true))
      for Qz_subdir in Qz_subdirs
        # Calculate tau_depol
        Qz = parse(Float64, basename(Qz_subdir))
        if isfile(Qz_subdir * "/data.ave")
          data_ave = readdlm(Qz_subdir * "/data.ave")
          row_start = findlast(x->occursin("#",string(x)), data_ave[:,1])[1] + 1
          t = Float64.(data_ave[row_start+n_damp:end,3])
          P = Float64.(data_ave[row_start+n_damp:end,4])
          # Linear regression to get depol time
          data = DataFrame(X=t, Y=P)
          tau_dep_track = -coef(lm(@formula(Y ~ X), data))[2]^-1/60
          push!(depol_tune_data.Qx, Qx)
          push!(depol_tune_data.Qy, Qy)
          push!(depol_tune_data.Qz, Qz)
          push!(depol_tune_data.tau_dep, tau_dep_track)
          data_emit = readdlm(Qz_subdir * "/data.emit")
          row_start = findlast(x->occursin("#",string(x)), data_emit[:,1])[1] + 1
          emit_a = Float64.(data_emit[row_start+n_damp:end,4])
          emit_b = Float64.(data_emit[row_start+n_damp:end,5])
          emit_c = Float64.(data_emit[row_start+n_damp:end,6])
          push!(depol_tune_data.emit_a, mean(emit_a))
          push!(depol_tune_data.emit_b, mean(emit_b))
          push!(depol_tune_data.emit_c, mean(emit_c))
        end
      end
    end
  end

  names =["Qx",
          "Qy",
          "Qz",
          "tau_dep",
          "emit_a",
          "emit_b",
          "emit_c"]
  depol_tune_data_dlm = permutedims(hcat(names, permutedims(hcat(depol_tune_data.Qx,
                                                                  depol_tune_data.Qy,
                                                                  depol_tune_data.Qz,
                                                                  depol_tune_data.tau_dep,
                                                                  depol_tune_data.emit_a,
                                                                  depol_tune_data.emit_b,
                                                                  depol_tune_data.emit_c))))
  writedlm("$(track_path)/depol_tune_data.dlm", depol_tune_data_dlm, ';')

  return depol_tune_data
end

function get_pol_tune_data(lat, order)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  if order == 1
    track_path = "$(path)/pol_tune_scan/1st_order_map"
  elseif order == 2
    track_path = "$(path)/pol_tune_scan/2nd_order_map"
  elseif order == 3
    track_path = "$(path)/pol_tune_scan/3rd_order_map"
  else
    error("only order < 3 supported right now")
  end
  if !isfile("$(track_path)/depol_tune_data.dlm")
    println("Tracking polarization tune scan not generated for lattice $(lattice).")
  end
  depol_tune_data_dlm = readdlm("$(track_path)/depol_tune_data.dlm", ';')[2:end,:]
  

  return DepolTuneData( depol_tune_data_dlm[:,1],
                  depol_tune_data_dlm[:,2],
                  depol_tune_data_dlm[:,3],
                  depol_tune_data_dlm[:,4],
                  depol_tune_data_dlm[:,5],
                  depol_tune_data_dlm[:,6],
                  depol_tune_data_dlm[:,7])
end


"""
    get_pol_track_data(lat, order)

Gets the polarization tracking data for the specified lattice.
"""
function get_pol_track_data(lat, order)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  if order == 1
    track_path = "$(path)/1st_order_map"
  elseif order == 2
    track_path = "$(path)/2nd_order_map"
  elseif order == 3
    track_path = "$(path)/3rd_order_map"
    # for backwards compatibility
  elseif order == 0
    track_path = "$(path)/bmad"
  else
    error("only order < 3 or Bmad tracking supported right now")
  end
  if !ispath(track_path)
    mkpath(track_path)
  end
  if !isfile("$(track_path)/pol_track_data.dlm")
    println("Tracking polarization data not generated for lattice $(lattice). Please use the read_pol_scan method to generate the data.")
  end
  pol_track_data_dlm = readdlm("$(track_path)/pol_track_data.dlm", ';')[2:end,:]
  

  return PolTrackData( pol_track_data_dlm[:,1],
                  pol_track_data_dlm[:,2],
                  pol_track_data_dlm[:,3],
                  pol_track_data_dlm[:,4],
                  pol_track_data_dlm[:,5],
                  pol_track_data_dlm[:,6],
                  pol_track_data_dlm[:,7],
                  pol_track_data_dlm[:,8],
                  pol_track_data_dlm[:,9],
                  pol_track_data_dlm[:,10],
                  pol_track_data_dlm[:,11],
                  pol_track_data_dlm[:,12])
end

end
