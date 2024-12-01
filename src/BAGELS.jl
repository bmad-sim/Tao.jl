#=

Routines for the BAGELS method.

=#

struct UnitBump
  type::Int
  names::Vector{String}
  sgns::Vector{Int}
end

"""
    BAGELS_1(lat, unit_bump; kick=5e-7, tol=1e-6, responses=Tao.DEPOL, at="sbend::*")

Best Adjustment Groups for ELectron Spin (BAGELS) method step 1: calculates the `responses` of 
something (default is ∂n/∂δ) at certain elements in Tao format (default is `sbend::*`) with all 
combinations of the inputted vertical closed orbit bump types as the "unit bumps". The types are:

1. `pi` bump              -- Delocalized coupling, delocalized dispersion
2. `2pin` bump            -- Localized coupling, delocalized dispersion
3. `opposing_2pin` bumps  -- Localized coupling, localized dispersion
4. `2pi` bump             -- Localized coupling, delocalized dispersion
5. `opposing_pi` bumps    -- Delocalized coupling, localized dispersion
6. `opposing_2pi` bumps   -- Localized coupling, localized dispersion
7. `quad shit bumps` bumps
8. `BAGELS` bumps

### Input
- `lat`       -- lat file name
- `unit_bump` -- Type of unit closed orbit bump. Options are: (1) `pi`, (2) `2pin`, (3) `opposing_2pin`, (4) `2pi`, (5) `opposing_pi`
- `kick`      -- (Optional) Coil kick, default is 5e-7
- `tol`       -- (Optional) Tolerance for difference in phase, default is 1e-6
"""
function BAGELS_1(lat, unit_bump; kick=5e-7, tol=1e-6, responses=Tao.DEPOL, at="sbend::*")
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end

  str_kick = @sprintf("%1.2e", kick) 

  if unit_bump < 0 || unit_bump > 7
    println("Unit bump type not defined!")
    return
  end

  # Generate directory in lattice data path for this unit bump
  if !ispath("$(path)/BAGELS_UB$(unit_bump)/$(str_kick)")
    mkpath("$(path)/BAGELS_UB$(unit_bump)/$(str_kick)")
  end

  path = "$(path)/BAGELS_UB$(unit_bump)/$(str_kick)"

  # First, obtain all combinations of bumps with desired phase advance
  if !isfile("$(path)/groups.dlm")
    if !isfile("$(path)/vkickers.dlm")
      (tmppath, tao_cmd) = mktemp()
      println(tao_cmd, "set ele * kick = 0")
      println(tao_cmd, "set ele * spin_tracking_method = sprint")
      println(tao_cmd, "show -write $(path)/vkickers.dlm lat -python vkicker::0:end -at phi_b@e23.16 -at kick@e23.16")
      println(tao_cmd, "exit")
      close(tao_cmd)
      run(`tao -lat $lat -noplot -noinit -nostart -command "call $tmppath"`)
    end

    # Read coil data and extract valid pairs
    coil_data = readdlm("$(path)/vkickers.dlm", ';', skipstart=2)
    coil_names = coil_data[:, 2]
    coil_phis = coil_data[:, 6]
    coil_kicks = coil_data[:,7]

    unit_groups_list = []
    unit_sgns_list = []
    unit_curkicks_list = []

    # Loop through coil names and determine the unit_groups
    if unit_bump == 0 # no bump just coils
      for i=1:length(coil_names)
        push!(unit_groups_list, coil_names[i])
        push!(unit_sgns_list, +1.)
        push!(unit_curkicks_list, coil_kicks[i])
      end
      n_per_group = 1
    elseif unit_bump == 1     # pi bump
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
    elseif unit_bump == 2 # 2pin bump
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
    elseif unit_bump == 3 # opposing_2pin bumps
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
    elseif unit_bump == 5 # opposing_pi bumps
      for i=1:length(coil_names)
        for j=i:length(coil_names)
          for k=j:length(coil_names)
            for l=k:length(coil_names)
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
    elseif unit_bump == 6 # opposing_2pi bumps
      for i=1:length(coil_names)-3
        for j=i+1:length(coil_names)-2
          for k=j+1:length(coil_names)-1
            for l=k+1:length(coil_names)
              # The second two coils can be either in phase (2pin apart) and opposite sign, or out of 
              # phase with same sign)
              if abs(coil_phis[j] - coil_phis[i] - 2*pi) < tol && abs(coil_phis[l] - coil_phis[k] - 2*pi) < tol
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
    elseif unit_bump == 7 # quad pi bumps
      # we have to allow overlapping here now
      # but max number of coils per group is indeed 8
      for i=1:length(coil_names)
        for j=i+1:length(coil_names)
          if abs(coil_phis[j] - coil_phis[i] - pi) < tol
            println("Found pi bump: $(coil_names[i]) and $(coil_names[j])")
            for k=j:length(coil_names)
              if abs(rem(coil_phis[k] -coil_phis[j], 2*pi, RoundNearest)) < tol
                println("These coils are 2*pi*n apart: $(coil_names[j]) and $(coil_names[k])")
                for l=k+1:length(coil_names)
                  if abs(coil_phis[l] - coil_phis[k] - pi) < tol
                    println("And it is the start of another pi bump!: $(coil_names[k]) and $(coil_names[l])")
                    println("OK we have one pair of pi bumps which cancel dispersion, let's look for a second pair in/out of phase")
                    for m=l:length(coil_names)
                      if coil_names[m] != coil_names[k] && abs(rem(coil_phis[m] -coil_phis[j], 2*pi, RoundNearest)) < tol
                        println("This coil is 2*pi*n from end of first pi bump: $(coil_names[j]) and $(coil_names[m])")
                        for n=m+1:length(coil_names)
                          print("Checking if $(coil_names[m]) and $(coil_names[n]) are a pi bump... ")
                          if abs(coil_phis[n] - coil_phis[m] - pi) < tol
                            println("Found second pi bump: $(coil_names[m]) and $(coil_names[n])")
                            for o=n:length(coil_names)
                              if coil_names[o] != coil_names[k] && abs(rem(coil_phis[o] -coil_phis[n], 2*pi, RoundNearest)) < tol # && abs(rem(coil_phis[j] -coil_phis[m], 2*pi, RoundNearest)) < tol
                                println("This coil is 2*pi*n apart from end of second pi bump: $(coil_names[n]) and $(coil_names[o])")
                                for p=o+1:length(coil_names)
                                  print("Checking if $(coil_names[o]) and $(coil_names[p]) are a pi bump... ")
                                  if abs(coil_phis[p] - coil_phis[o] - pi) < tol
                                  # Now we know that i,j,k,l are positive and m,n,o,p are negative
                                    println("Found combination: $([coil_names[i], coil_names[j], coil_names[k], coil_names[l], coil_names[m], coil_names[n], coil_names[o], coil_names[p]])")
                                    push!(unit_groups_list, coil_names[i])
                                    push!(unit_groups_list, coil_names[j])
                                    push!(unit_groups_list, coil_names[k])
                                    push!(unit_groups_list, coil_names[l])
                                    push!(unit_groups_list, coil_names[m])
                                    push!(unit_groups_list, coil_names[n])
                                    push!(unit_groups_list, coil_names[o])
                                    push!(unit_groups_list, coil_names[p])
                                    push!(unit_sgns_list, +1.)
                                    push!(unit_sgns_list, +1.)  
                                    push!(unit_sgns_list, +1.)  
                                    push!(unit_sgns_list, +1.) 
                                    push!(unit_sgns_list, -1.)
                                    push!(unit_sgns_list, -1.)  
                                    push!(unit_sgns_list, -1.)  
                                    push!(unit_sgns_list, -1.) 
                                    push!(unit_curkicks_list, coil_kicks[i])
                                    push!(unit_curkicks_list, coil_kicks[j])
                                    push!(unit_curkicks_list, coil_kicks[k])
                                    push!(unit_curkicks_list, coil_kicks[l])
                                    push!(unit_curkicks_list, coil_kicks[m])
                                    push!(unit_curkicks_list, coil_kicks[n])
                                    push!(unit_curkicks_list, coil_kicks[o])
                                    push!(unit_curkicks_list, coil_kicks[p])
                                  else
                                    print("nope\n")
                                  end
                                end
                              end
                            end
                          else
                            print("Nope\n")
                          end
                        end
                      end
                    end
                  end
                end
              end
            end
          end
        end
      end
      n_per_group = 8
    end

    # Info to store is: each coil in the group, each of their current strengths, and each of their kick sgns/mags
    # Just write to three separate files for now
  
    unit_groups = permutedims(reshape(unit_groups_list, n_per_group, Int(length(unit_groups_list)/n_per_group)))
    unit_sgns = permutedims(reshape(unit_sgns_list, n_per_group, Int(length(unit_sgns_list)/n_per_group)))
    unit_curkicks = permutedims(reshape(unit_curkicks_list, n_per_group, Int(length(unit_curkicks_list)/n_per_group)))

    writedlm("$(path)/groups.dlm", unit_groups, ';')
    writedlm("$(path)/sgns.dlm", unit_sgns, ';')
    writedlm("$(path)/curkicks.dlm", unit_curkicks, ';')
  end

  unit_groups = readdlm("$(path)/groups.dlm", ';')
  unit_sgns =  readdlm("$(path)/sgns.dlm", ';')
  unit_curkicks =  readdlm("$(path)/curkicks.dlm", ';')


  # Now find the response for each 
  (tmppath, tao_cmd) = mktemp()# open("$(path)/responses_$(str_kick)/BAGELS_1.cmd", "w")
  println(tao_cmd, "set ele * spin_tracking_method = sprint")
  println(tao_cmd, "set bmad_com spin_tracking_on = T")
  #println(tao_cmd, "set bmad_com rel_tol_tracking = 1e-12")
  #println(tao_cmd, "set bmad_com abs_tol_tracking = 1e-16")

  # First the baseline:
  for response in responses
    str_response = base64encode(response)
    if !isdir("$(path)/$(str_response)")
      mkdir("$(path)/$(str_response)")
    end
    println(tao_cmd, "show -write $(path)/$(str_response)/baseline.dlm lat -python $at -at ($response)@e23.16")
  end
  
  # Now go through each unit bump:

  for i=1:length(unit_groups[:,1])
    str_coils = ""
    for j=1:length(unit_groups[i,:])
      coil = unit_groups[i,j]
      sgn = unit_sgns[i,j]
      curkick = unit_curkicks[i,j]
      println(tao_cmd, "change ele $(coil) kick $(sgn*kick)")
      str_coils = str_coils * coil * "_"
    end
    str_coils = str_coils[1:end-1] * ".dlm"
    println(tao_cmd, "scale")
    for response in responses
      str_response = base64encode(response)
      println(tao_cmd, "show -write $(path)/$(str_response)/$str_coils lat -python $at -at ($response)@e23.16")
    end
    # Reset coils to original strengths
    for j=1:length(unit_groups[i,:])
      coil = unit_groups[i,j]
      sgn = unit_sgns[i,j]
      curkick = unit_curkicks[i,j]
      println(tao_cmd, "change ele $(coil) kick $(-sgn*kick)")
    end
  end
  println(tao_cmd, "exit")
  close(tao_cmd)

  run(`tao -lat $lat -noplot -noinit -nostart -command "call $tmppath"`)
end

"""
    BAGELS_2(lat, unit_bump; kick=5e-7, solve_knobs=0, prefix="BAGELS", suffix="", outf="BAGELS.bmad", coil_regex=r".*",
              A=[Tao.DEPOL], eps_A=0, B=[["orbit.y"]], eps_B=0 ,
              w_A=ones(length(A)), w_B=zeros(length(B))) 

Best Adjustment Groups for ELectron Spin (BAGELS) method step 2: peform an SVD of the 
response matrix to obtain the best adjustment groups, based on the settings of step 1. 
The vertical closed orbit unit bump types are:

1. `pi` bump              -- Delocalized coupling, delocalized dispersion
2. `2pin` bump            -- Localized coupling, delocalized dispersion
3. `opposing_2pin` bumps  -- Localized coupling, localized dispersion
4. `2pi` bump             -- Localized coupling, delocalized dispersion
5. `opposing_pi` bumps    -- Delocalized coupling, localized dispersion
6. `opposing_2pi` bumps   -- Localized coupling, localized dispersion

### Input
- `lat`       -- lat file name
- `unit_bump` -- Type of unit closed orbit bump. Options are: (1) `pi`, (2) `n2pi`, (3) `n2pi_cancel_eta`, (4) `2pi`, (5) `pi_cancel_eta`
- `coil_regex` -- (Optional) Regex of coils to match to (e.g. for all coils ending in `_7` or `_11`, use `r".*_(?:7|11)\\b"`)
- `bend_regex` -- (Optional) Regex of bends to match to and include in SVD 
- `suffix`     -- (Optional) Prefix to append to group elements generated for knobs in Bmad format
- `suffix`     -- (Optional) Suffix to append to group elements generated for knobs in Bmad format
- `outf`       -- (Optional) Output file name with group elements constructed from BAGELS, default is `\$(lat)_BAGELS.bmad`
- `kick`       -- (Optional) Coil kick, default is 1e-5
- `print_sol`  -- (Optional) Prints group elements which solves (in a least squares sense) the matrix equation for dn/ddelta=0
- `do_amp`     -- (Optional) If true, calculates response matrix of |dn/ddelta| instead of dn/ddelta.x, which is the default 
- `solve_knobs` -- (Optional) If set > 0, will print the knob settings for the first `solve_knobs` BAGELS knobs to minimize in a least squares sense dn/ddelta
"""
function BAGELS_2(lat, unit_bump; kick=5e-7, solve_knobs=0, prefix="BAGELS", suffix="", outf="BAGELS.bmad", coil_regex=r".*",
                                  A=[Tao.DEPOL], eps_A=0, B=[["orbit.y"]], eps_B=0 ,
                                  w_A=ones(length(A)), w_B=zeros(length(B)), include_knobs=-1) 



  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end

  str_kick = @sprintf("%1.2e", kick) 

  if unit_bump < 0 || unit_bump > 7
    println("Unit bump type not defined!")
    return
  end

  # Generate directory in lattice data path for this unit bump
  if !ispath("$(path)/BAGELS_UB$(unit_bump)/$(str_kick)")
    mkpath("$(path)/BAGELS_UB$(unit_bump)/$(str_kick)")
  end

  path = "$(path)/BAGELS_UB$(unit_bump)/$(str_kick)"

  unit_groups_raw = readdlm("$(path)/groups.dlm", ';')
  unit_sgns_raw =  readdlm("$(path)/sgns.dlm", ';')
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
  n_unit_bumps = length(unit_groups[:,1])

  function construct_response_matrix(path, F, unit_groups; normalize_submatrices=true, weights=ones(length(F)))
    n_unit_bumps = length(unit_groups[:,1])
    delta_F = Matrix{Float64}(undef, 0, n_unit_bumps)

    curidx = 1
    F0 = Float64[]
    length(F) == length(weights) || error("Incorrect length for weights array!")
    for (rv,weight) in zip(F, weights)
      n_rv_ele = -1
      for response in rv
        str_response = base64encode(response)
        f0 = readdlm("$path/$str_response/baseline.dlm", ';', skipstart=2)[:,end]
        if n_rv_ele == -1
          n_rv_ele = length(f0)
        else
          n_rv_ele == length(f0) || error("Invalid number of evaluated elements for $response in $rv ! If these quantities are to be considered on the same scale then please use BAGELS_1 with the same `at` for all.")
        end
  
        # Add another subsubmatrix to delta_A
        new_subsub = zeros(n_rv_ele, n_unit_bumps)
  
        # now get response for each coil to construct row of matrix:
        for i=1:n_unit_bumps
          str_coils = ""
          for j=1:length(unit_groups[i,:])
            coil = unit_groups[i,j]
            str_coils = str_coils * coil * "_"
          end
          str_coils = str_coils[1:end-1] * ".dlm"
  
          fi = readdlm("$path/$str_response/$str_coils", ';', skipstart=2)[:,end]
          n_rv_ele == length(fi) || error("Number of evaluated elements for $response for the unit bump $str_coil disagrees with the baseline for this unit bump ! This is a weird, hard to reach error so congrats on reaching it. Please repeat BAGELS_1 with the desired `at` for all.")
          new_subsub[:,i] = (fi .- f0)./kick
        end
  
        delta_F = vcat(delta_F, new_subsub)
        F0 = vcat(F0, f0)
      end
      # here we normalize the submatrix and continue
      if normalize_submatrices
        delta_F[curidx:curidx-1+n_rv_ele*length(rv),:] .= delta_F[curidx:curidx-1+n_rv_ele*length(rv),:]/norm(delta_F[curidx:curidx-1+n_rv_ele*length(rv),:])
      end
      # Multiply the weight
      delta_F[curidx:curidx-1+n_rv_ele*length(rv),:] .= weight*delta_F[curidx:curidx-1+n_rv_ele*length(rv),:]
      curidx = n_rv_ele*length(rv)+1
    end

    return delta_F, F0
  end

  normed_delta_A, A0 = construct_response_matrix(path, A, unit_groups)
  normed_delta_B, B0 = construct_response_matrix(path, B, unit_groups)

  
  # Now do SVD to get the principal components
  ATA = transpose(normed_delta_A)*normed_delta_A
  BTB = transpose(normed_delta_B)*normed_delta_B
  #return normed_delta_B #ATA, BTB
  #ATA = sparse(ATA)
  #BTB = sparse(BTB)
  #decomp, history = partialschur(A, nev=10, tol=1e-6, which=:SR);
  #return BTB/norm(BTB)
  ATA = Symmetric(ATA/norm(ATA) + eps_A*I)
  BTB = Symmetric(BTB/norm(BTB) + eps_B*I)


  println("Condition number of A'A = $(cond(ATA))")
  println("Condition number of B'B = $(cond(BTB))")
  #return ATA,BTB
  #return ATA,BTB
  F = eigen(ATA, BTB, sortby=t->-abs(t))
  #F = svd(delta_A)
  #return normed_delta_B, B0, F.vectors
  # Get first N_knobs principal directions
  #V = F.V
  V = F.vectors
  vals = F.values

  if solve_knobs > 0 # calculate least squares solution using BAGELS knobs
    # construct new response matrix as combined 
    #SVD_delta = zeros(size(normed_delta_A,1), solve_knobs)
    SVD_delta = zeros(size(normed_delta_A,1)+size(normed_delta_B,1), solve_knobs)
    delta_A, ___ = construct_response_matrix(path, A, unit_groups; normalize_submatrices=false, weights=w_A)
    delta_B, ___ = construct_response_matrix(path, B, unit_groups; normalize_submatrices=false, weights=w_B)
    for i=1:solve_knobs
      SVD_delta[1:size(normed_delta_A,1),i] =  delta_A*V[:,i]
      SVD_delta[size(normed_delta_A,1)+1:end,i] = delta_B*V[:,i]
    end
    #strengths = SVD_delta\float.(-A0)
    strengths = SVD_delta\float.(-vcat(A0,B0))
    println("Norm of solution: $(norm(vcat(A0,B0)-SVD_delta*strengths))")
  end

  # Create a group element with these as knobs  
  for i=1:length(vals)
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
      println(knob_out, "! BAGELS method knobs")
      println(knob_out, "! A = $A")
      println(knob_out, "! B = $B")
      println(knob_out, "! eps_A = $eps_A")
      println(knob_out, "! eps_B = $eps_B")
      println(knob_out, "! w_A = $w_A, w_B = $w_B")
      println(knob_out, "! coil_regex = $coil_regex")
    else
      knob_out = open(outf, "a")
    end
    if include_knobs == i-1
      println(knob_out, "end_file")
    end
    
    println(knob_out, "$(prefix)$(Printf.format(Printf.Format("%0$(length(string(length(vals))))i"), i))$(suffix): group = {\t\t\t ! Eigenvalue = $(vals[i])")
    
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
    print(knob_out, "}, var = {X}")

    if i <= solve_knobs
      print(knob_out, ", X = $(strengths[i])")
    end

    
    print(knob_out, "\n ")
    close(knob_out)
    
  end
  println("Solution written to file $outf")
  #print("\nEigenvalues are: ")
  #print(vals)

  return F
end
