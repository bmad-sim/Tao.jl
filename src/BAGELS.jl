#=

Routines for the BAGELS method.

=#

struct UnitBump
  type::Int
  names::Vector{String}
  sgns::Vector{Int}
end

"""
    BAGELS_1(lat, unit_bump; kick=1e-5, tol=1e-8)

Best Adjustment Groups for ELectron Spin (BAGELS) method step 1: calculates the `responses` of 
something (default is ∂n/∂δ) at certain elements in Tao format (default is `sbend::*`) with all 
combinations of the inputted vertical closed orbit bump types as the "unit bumps". The types are:

1. `pi` bump              -- Delocalized coupling, delocalized dispersion
2. `2pin` bump            -- Localized coupling, delocalized dispersion
3. `opposing_2pin` bumps  -- Localized coupling, localized dispersion
4. `2pi` bump             -- Localized coupling, delocalized dispersion
5. `opposing_pi` bumps    -- Delocalized coupling, localized dispersion

### Input
- `lat`       -- lat file name
- `unit_bump` -- Type of unit closed orbit bump. Options are: (1) `pi`, (2) `2pin`, (3) `opposing_2pin`, (4) `2pi`, (5) `opposing_pi`
- `kick`      -- (Optional) Coil kick, default is 5e-7
- `tol`       -- (Optional) Tolerance for difference in phase, default is 1e-8
"""
function BAGELS_1(lat, unit_bump; kick=5e-7, tol=1e-8, responses=["spin_dn_dpz.x", "spin_dn_dpz.y", "spin_dn_dpz.z"], at="sbend::*")
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end

  str_kick = @sprintf("%1.2e", kick) 

  if unit_bump < 1 || unit_bump > 5
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
      println(tao_cmd, "show -write $(path)/vkickers.dlm lat -python vkicker::0:end -at phi_b@e23.16")
      println(tao_cmd, "exit")
      close(tao_cmd)
      run(`tao -lat $lat -noplot -noinit -nostart -command "call $tmppath"`)
    end

    # Read coil data and extract valid pairs
    coil_data = readdlm("$(path)/vkickers.dlm", ';', skipstart=2)
    coil_names = coil_data[:, 2]
    coil_phis = coil_data[:, 6]

    unit_groups_list = []
    unit_sgns_list = []

    # Loop through coil names and determine the unit_groups
    if unit_bump == 1     # pi bump
      for i=1:length(coil_names)-1
        for j=i+1:length(coil_names)
          if abs(coil_phis[j] - coil_phis[i] - pi) < tol
            push!(unit_groups_list, coil_names[i])
            push!(unit_groups_list, coil_names[j])
            push!(unit_sgns_list, +1.)
            push!(unit_sgns_list, +1.)  # Coils in pi bumps have same kick sign and strength
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
                elseif abs(rem(coil_phis[j] - coil_phis[k] - pi,2*pi, RoundNearest)) < tol   # Out of phase exactly
                  push!(unit_groups_list, coil_names[i])
                  push!(unit_groups_list, coil_names[j])
                  push!(unit_groups_list, coil_names[k])
                  push!(unit_groups_list, coil_names[l])
                  push!(unit_sgns_list, +1.)
                  push!(unit_sgns_list, -1.) 
                  push!(unit_sgns_list, +1.) 
                  push!(unit_sgns_list, -1.) 
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
    
    writedlm("$(path)/groups.dlm", unit_groups, ';')
    writedlm("$(path)/sgns.dlm", unit_sgns, ';')
  end

  unit_groups = readdlm("$(path)/groups.dlm", ';')
  unit_sgns =  readdlm("$(path)/sgns.dlm", ';')


  # Now find the response for each 
  (tmppath, tao_cmd) = mktemp()# open("$(path)/responses_$(str_kick)/BAGELS_1.cmd", "w")
  println(tao_cmd, "set ele * kick = 0")
  println(tao_cmd, "set ele * spin_tracking_method = sprint")
  println(tao_cmd, "set bmad_com spin_tracking_on = T")

  # First the baseline:
  for response in responses
    if !isdir("$(path)/$(response)")
      mkdir("$(path)/$(response)")
    end
    println(tao_cmd, "show -write $(path)/$(response)/baseline.dlm lat -python $at -at $response@e23.16")
  end
  
  # Now go through each unit bump:

  for i=1:length(unit_groups[:,1])
    str_coils = ""
    for j=1:length(unit_groups[i,:])
      coil = unit_groups[i,j]
      sgn = unit_sgns[i,j]
      println(tao_cmd, "set ele $(coil) kick = $(sgn*kick)")
      str_coils = str_coils * coil * "_"
    end
    str_coils = str_coils[1:end-1] * ".dlm"
    for response in responses
      println(tao_cmd, "show -write $(path)/$(response)/$str_coils lat -python $at -at $response@e23.16")
    end
    # Reset coils to original strengths
    for j=1:length(unit_groups[i,:])
      coil = unit_groups[i,j]
      println(tao_cmd, "set ele $(coil) kick = 0")
    end
  end
  println(tao_cmd, "exit")
  close(tao_cmd)

  run(`tao -lat $lat -noplot -noinit -nostart -command "call $tmppath"`)
end

"""
    BAGELS_2(lat, unit_bump; prefix="BAGELS", coil_regex=r".*", bend_regex=r".*", suffix="", outf="\$(lat)_BAGELS.bmad", kick=1e-5, print_sol=false, do_amp=false)

Best Adjustment Groups for ELectron Spin (BAGELS) method step 2: peform an SVD of the 
response matrix to obtain the best adjustment groups, based on the settings of step 1. 
The vertical closed orbit unit bump types are:

1. `pi` bump              -- Delocalized coupling, delocalized dispersion
2. `2pin` bump            -- Localized coupling, delocalized dispersion
3. `opposing_2pin` bumps  -- Localized coupling, localized dispersion
4. `2pi` bump             -- Localized coupling, delocalized dispersion
5. `opposing_pi` bumps    -- Delocalized coupling, localized dispersion

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
function BAGELS_2(lat, unit_bump; kick=5e-7, solve_knobs=0, prefix="BAGELS", suffix="", outf="$(lat)_BAGELS2.bmad", coil_regex=r".*",
                                  A=["spin_dn_dpz.x", "spin_dn_dpz.y", "spin_dn_dpz.z"], A_regex=r".*", 
                                  B=nothing, B_regex=r".*"  ) #prefix="BAGELS", coil_regex=r".*", bend_regex=r".*", suffix="", outf="$(lat)_BAGELS.bmad", kick=1e-5, print_sol=false, do_amp=false, solve_knobs=0)



  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end

  str_kick = @sprintf("%1.2e", kick) 

  if unit_bump < 1 || unit_bump > 5
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

  #eletypes = readdlm("$(path)/responses_$(str_kick)/baseline.dlm", skipstart=2)[1:end-2,3]
  A_elenames = readdlm("$(path)/$(first(A))/baseline.dlm", ';', skipstart=2)[:,2]
  for a in A
    if A_elenames != readdlm("$(path)/$a/baseline.dlm", ';', skipstart=2)[:,2]
      error("Cannot combine responses in A! Different element names per calculated response. Please use BAGELS_1 with same specified `at`.")
    end
  end

  fA0 = mapreduce(a->readdlm("$path/$a/baseline.dlm", ';', skipstart=2)[:,end], hcat,A)
  idx_A = findall(t->occursin(A_regex, t), A_elenames)
  fA0 = fA0[idx_A,:]
  fA0 = reshape(transpose(fA0), (length(A)*length(fA0[:,1]),1)) # Flatten

  if !isnothing(B)
    B_elenames = readdlm("$(path)/$(first(B))/baseline.dlm", ';', skipstart=2)[:,2]
    for b in B
      if B_elenames != readdlm("$(path)/$b/baseline.dlm", ';', skipstart=2)[:,2]
        error("Cannot combine responses in B! Different element names per calculated response. Please use BAGELS_1 with same specified `at`.")
      end
    end

    fB0 = mapreduce(b->readdlm("$path/$b/baseline.dlm", ';', skipstart=2)[:,end], hcat, B)
    idx_B = findall(t->occursin(B_regex, t), B_elenames)
    fB0 = fB0[idx_B,:]
    fB0 = reshape(transpose(fB0), (length(B)*length(fB0[:,1]),1)) # Flatten
  end


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

  # We now will have a response matrix that is N_A x N_groups to multiply with vector N_groups x 1
  delta_fA = zeros(length(A)*length(idx_A), length(unit_groups[:,1]))
  if !isnothing(B) delta_fB = zeros(length(B)*length(idx_B), length(unit_groups[:,1])) end

  for i=1:length(unit_groups[:,1])
    str_coils = ""
    for j=1:length(unit_groups[i,:])
      coil = unit_groups[i,j]
      str_coils = str_coils * coil * "_"
    end
    str_coils = str_coils[1:end-1] * ".dlm"

    # for each response to A, we get the response:
    fA = mapreduce(a->readdlm("$path/$a/$str_coils", ';', skipstart=2)[:,end], hcat,A)[idx_A,:]
    fA = reshape(transpose(fA), (length(A)*length(fA[:,1]),1)) # Flatten
    delta_fA[:,i] = (fA .- fA0)./kick

    if !isnothing(B)
      fB = mapreduce(b->readdlm("$path/$b/$str_coils", ';', skipstart=2)[:,end], hcat,B)[idx_B,:]
      fB = reshape(transpose(fB), (length(B)*length(fB[:,1]),1)) # Flatten
      delta_fB[:,i] = (fB .- fB0)./kick
    end
    #dn_dpz_bends = readdlm("$(path)/responses_$(str_kick)/$(str_coils)", skipstart=2)[1:end-2,6:8][idx_bends,:]
    
    #return dn_dpz_bends
    #=if do_amp
      dn_dpz_bends_amp = [norm(dn_dpz_bends[j,:]).^2 for j=1:length(dn_dpz_bends[:,1])]
      #delta_dn_dpz_amp = dn_dpz_bends_amp .- dn_dpz0_bends_amp
      #delta_dn_dpz_amp = [norm(delta_dn_dpz[j,:]) for j=1:length(delta_dn_dpz[:,1])]
      delta_dn_dpz[:,i] = (dn_dpz_bends_amp .- dn_dpz0_bends_amp)./kick #(dn_dpz_bends[:,1] .- dn_dpz0_bends[:,1])./kick
    else
      dn_dpz_bends = reshape(transpose(dn_dpz_bends), (3*length(dn_dpz_bends[:,1]),1))
      delta_dn_dpz[:,i] = (dn_dpz_bends .- dn_dpz0_bends)./kick
    end =#
  end

  #=
  if print_sol
    strengths =  do_amp ? delta_dn_dpz\-dn_dpz0_bends_amp : delta_dn_dpz\float.(-dn_dpz0_bends)
    #return delta_dn_dpz, -dn_dpz0_bends_amp, strengths
    for i=1:size(unit_groups, 1)
      print("BAGELS_SOLVE$i: group = {")
      for j=1:size(unit_groups, 2)
        print("$(unit_groups[i,j])[kick]: $(unit_sgns[i,j])*X")
        if j != size(unit_groups, 2)
          print(", ")
        end
      end
      print("}, var = {X}, X = $(strengths[i])\n")
    end
  end
  =#
  
  #return delta_fB
  # Now do SVD to get the principal components
  ATA = transpose(delta_fA)*delta_fA
  if isnothing(B)
    BTB = I(length(unit_groups[:,1]))
  else
    BTB = transpose(delta_fB)*delta_fB
  end
  F = eigen(ATA, BTB, sortby=t->-abs(t))
  #F = svd(delta_fA)

  # Get first N_knobs principal directions
  #V = F.V
  V = F.vectors
  vals = F.values

  if solve_knobs > 0 # calculate least squares solution using BAGELS knobs
    # construct new response matrix
    SVD_delta_fA = zeros( length(A)*length(idx_A), solve_knobs)
    for i=1:solve_knobs
      SVD_delta_fA[:,i] = delta_fA*V[:,i]
    end
    strengths = SVD_delta_fA\float.(-fA0)
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
    else
      knob_out = open(outf, "a")
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

  print("\nEigenvalues are: ")
  print(vals)

  return F
end
