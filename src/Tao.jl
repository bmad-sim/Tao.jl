module Tao
using DelimitedFiles
using LinearAlgebra
using Printf
using NLsolve
using Interpolations
using DataFrames, StatsBase, GLM

export  data_path,
        PolData,
        BAGELS_1,
        BAGELS_2,
        calc_P_t,
        calc_T,
        pol_scan,
        get_pol_data,
        run_3rd_order_map_tracking,
        run_pol_scan_3rd_order,
        read_pol_scan_3rd_order,
        get_pol_track_data

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
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  str_phi = @sprintf("%1.2e", phi_start) * "_" * @sprintf("%1.2e", phi_step)
  str_kick = @sprintf("%1.2e", kick)

  # Generate directory in lattice data path for these phi_start and phi_step
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
  path = data_path(lat)
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
  for i in eachindex(agamma0)
    println(tao_cmd, "set ele 0 e_tot = $(agamma0[i]*m_e/a_e)")
    println(tao_cmd, "python -append $(path)/spin.dat spin_polarization")
  end

  println(tao_cmd, "exit")
  close(tao_cmd)

  # Execute the scan
  run(`tao -lat $lat -noplot -command "call $(path)/spin.tao"`)

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
  pol_data_dlm = permute_dims(hcat(names, permutedims(hcat(agamma0,
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
    println("First-order polarization data not generated for lattice $(lattice). Please use the pol_scan method to generate the data.")
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
  run_3rd_order_map_tracking(lat, n_particles, n_turns; use_data_path=true)

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
function run_3rd_order_map_tracking(lat, n_particles, n_turns; use_data_path=true)
  if (use_data_path)
    path = data_path(lat)
    if path == ""
      println("Lattice file $(lat) not found!")
      return
    end
    track_path = "$(path)/3rd_order_map"
    if !ispath(track_path)
      mkpath(track_path)
    end
  else
    path = dirname(abspath(lat))
    track_path = path
  end
  

  

  # In the tracking directory, we must create the long_term_tracking.init, run.sh, and qtrack.sh, 
  # which will then be scp-ed to the equivalent directory on the remote machine. On the remote 
  # machine, tracking info is stored in ~/trackings_jl/, where the full path to the lattice on 
  # this machine will be written and the tracking folder dropped there.

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
                            ltt%tracking_method = 'MAP'   !
                            ltt%n_turns = 1                 ! Number of turns to track
                            ltt%rfcavity_on = T
                            ltt%map_order = 3 ! increase see effects
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
                            ltt%tracking_method = 'MAP'   !
                            ltt%n_turns = $(n_turns)                 ! Number of turns to track
                            ltt%rfcavity_on = T
                            ltt%map_order = 3 ! increase see effects
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
  run(`scp -r $(track_path)/\\\{run1.sh,run32.sh,qtrack.sh,long_term_tracking.init,long_term_tracking1.init\\\} lnx4200:$(remote_path)`)


  # Submit the tracking on host
  run(`ssh lnx4200 "cd $(remote_path); sh qtrack.sh"`)
end


"""
    run_pol_scan_3rd_order(lat, n_particles, n_turns, agamma0)

IMPORTANT: Please follow the setup instructions in the `run_3rd_order_map_tracking`
documentation before running this routine.

This routine sets up and submits the parallel 3rd order map tracking 
jobs to the CLASSE cluster for the range of `agamma0` specified. 
"""
function run_pol_scan_3rd_order(lat, n_particles, n_turns, agamma0)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  
  if !ispath("$(path)/3rd_order_map")
    mkpath("$(path)/3rd_order_map")
  end

  # Create subdirectories with name equal to agamma0
  for i in eachindex(agamma0)
    subdirname = Printf.format(Printf.Format("%0$(length(string(floor(maximum(agamma0))))).2f"), agamma0[i])
    
    # ONLY CREATE NEW FILES IF IT DOESN'T EXIST YET
    temp_lat = "$(path)/3rd_order_map/$(subdirname)/$(lat)_$(subdirname)"
    if !isfile(temp_lat)
      mkpath("$(path)/3rd_order_map/$(subdirname)")
      cp(lat,"$(path)/3rd_order_map/$(subdirname)/$(lat)_$(subdirname)")
      latf = open(temp_lat, "a")
      write(latf, "\nparameter[e_tot] = $(agamma0[i])/anom_moment_electron*m_electron")
      close(latf)
    end

    run_3rd_order_map_tracking(temp_lat, n_particles, n_turns; use_data_path=false)
  end
end


"""
    read_pol_scan_3rd_order(lat, n_damp)

This routine reads the results of run_pol_scan_3rd_order from the CLASSE 
computer and creates a PolTrackData for this lattice.
"""
function read_pol_scan_3rd_order(lat, n_damp)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end    
  
  track_path = "$(path)/3rd_order_map"
  if !ispath(track_path)
    mkpath(track_path)
  end
  remote_path = "~/trackings_jl" * track_path
  run(`ssh lnx4200 "cd $(remote_path); find . -name "data.ave" -o -name "data.emit" -o -name "data.sigma" | find  -name "data.ave" -o -name "data.emit" -o -name "data.sigma" | tar -czvf data.tar.gz -T -"`)
  run(`scp lnx4200:$(remote_path)/data.tar.gz $(track_path)`)
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
    t = data_ave[row_start+n_damp:end,3]
    P = data_ave[row_start+n_damp:end,4]
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
  pol_track_data_dlm = permute_dims(hcat(names, permutedims(hcat(agamma0,
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
  writedlm("$(path)/pol_track_data.dlm", pol_track_data_dlm, ';')

  return pol_track_data
end

"""
    get_pol_track_data(lat)

Gets the polarization tracking data for the specified lattice.
"""
function get_pol_track_data(lat)
  path = data_path(lat)
  if path == ""
    println("Lattice file $(lat) not found!")
    return
  end
  if !isfile("$(path)/pol_track_data.dlm")
    println("Tracking polarization data not generated for lattice $(lattice). Please use the read_pol_scan_3rd_order method to generate the data.")
  end
  pol_track_data_dlm = readdlm("$(path)/pol_track_data.dlm", ';')[2:end,:]
  

  return PolData( pol_track_data_dlm[:,1],
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
