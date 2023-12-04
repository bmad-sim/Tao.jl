module Tao
using DelimitedFiles
using LinearAlgebra
using Printf
using NLsolve

export  metadata_path,
        PolData,
        BAGELS_1,
        BAGELS_2,
        calc_refill_sensitivity,
        calc_refill_time,
        pol_scan,
        get_pol_data

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
Structure to store important polarization quantities calculated 
for a lattice.

"""
struct PolData
  agamma0::Float64
  spin_tune::Float64
  P_st::Float64
  P_dk::Float64
  P_dk_a::Float64
  P_dk_b::Float64
  P_dk_c::Float64
  P_dk_2a::Float64
  P_dk_2b::Float64
  P_dk_2c::Float64
  tau_bks::Float64
  tau_dep::Float64
  tau_dep_a::Float64
  tau_dep_b::Float64
  tau_dep_c::Float64
  tau_dep_2a::Float64
  tau_dep_2b::Float64
  tau_dep_2c::Float64
  tau_eq::Float64
  T::Float64
  T_du::Float64
  T_dd::Float64
  P_t::Float64
  T_du_t::Float64
  T_dd_t::Float64
end

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
    
    println(knob_out, "BAGELS$(suffix)$(Printf.format(Printf.Format("%0$(length(string(N_knobs)))i"), i)): group = {")

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

function calc_refill_sensitivity(P_dk, tau_eq; P_0 = 0.85, T = 4.8)
  P_dd = x -> (-P_0.*tau_eq .- P_dk.*tau_eq .+ P_dk.*x .+ (P_0.*tau_eq .+ P_dk.*tau_eq).*exp.(-x./tau_eq))./x
  P_du = x -> (P_0.*tau_eq .- P_dk.*tau_eq .+ P_dk.*x .+ (-P_0.*tau_eq .+ P_dk.*tau_eq).*exp.(-x./tau_eq))./x
  P = T_du -> (P_dd((T_du.*T./(2 .*T_du.-T))) + P_du(T_du)).^2

  T_du_t = nlsolve(P, 10. .*ones(length(P_dk))).zero
  T_dd_t = (T_du_t.*T./(2 .*T_du_t.-T))
  
  return P_du(T_du_t), T_du_t, T_dd_t
end

function calc_refill_time(P_dk, tau_eq; P_0 = 0.85, P_min_avg = 0.7)
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
important quantities and stores them for fast reference in the metadata.

### Input
- `lat`      -- lat file name
- `agamma0`  -- range of a*gamma_0 to scan over

### Output 
- `pol_data` -- PolData struct containing polarization quantities for lattice
"""
function pol_scan(lat, agamma0)
  path = metadata_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end

  # Generate tao command script
  tao_cmd = open("$(path)/spin.tao", "w")
  a_e = 1.15965218e-3
  m_e = 0.51099895e6
  println(tao_cmd, "set ele * spin_tracking_method = sprint")
  println(tao_cmd, "set bmad_com spin_tracking_on = T")
  println(tao_cmd, "set ele 0 e_tot = $(agamma0[1]*m_e/a_e)")
  println(tao_cmd, "python -write $(path)/spin.dat spin_polarization")
  for i = 1:length(agamma0)
    println(tao_cmd, "set ele 0 e_tot = $(agamma0[i]*m_e/a_e)")
    println(tao_cmd, "python -append $(path)/spin.dat spin_polarization")
  end

  println(tao_cmd, "exit")
  close(tao_cmd)

  # Execute the scan
  run(`tao -lat $lat -noplot -command "call $(path)/spin.tao"`)

  # Calculate important quantities and store in metadata
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
  T, T_du, T_dd = calc_refill_time(P_dk,tau_eq)
  P_t, T_du_t, T_dd_t = calc_refill_sensitivity(P_dk,tau_eq)

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
  pol_data_dlm = hcat(names, permutedims(hcat(agamma0,
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
                                  T_dd_t)))
  writedlm("$(path)/pol_data.dlm", pol_data_dlm, ';')
  return pol_data
end


"""
    get_pol_data(lat)

Gets the polarization data for the specified lattice.
"""
function get_pol_data(lat)
  path = metadata_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end

  pol_data_dlm = readlm("$(path)/pol_data.dlm", pol_data_dlm, ';')[:,2:end]
  
  return PolData( pol_data_dlm[1,:],
                  pol_data_dlm[2,:],
                  pol_data_dlm[3,:],
                  pol_data_dlm[4,:],
                  pol_data_dlm[5,:],
                  pol_data_dlm[6,:],
                  pol_data_dlm[7,:],
                  pol_data_dlm[8,:],
                  pol_data_dlm[9,:],
                  pol_data_dlm[10,:],
                  pol_data_dlm[11,:],
                  pol_data_dlm[12,:],
                  pol_data_dlm[13,:],
                  pol_data_dlm[14,:],
                  pol_data_dlm[15,:],
                  pol_data_dlm[16,:],
                  pol_data_dlm[17,:],
                  pol_data_dlm[18,:],
                  pol_data_dlm[19,:],
                  pol_data_dlm[20,:],
                  pol_data_dlm[21,:],
                  pol_data_dlm[22,:],
                  pol_data_dlm[23,:],
                  pol_data_dlm[24,:],
                  pol_data_dlm[25,:])
end

end
