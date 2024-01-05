# AutoNR: automatic calculation of ISC/IC non-radiative rate

# 2023-06-12
# xujiawei@fjirsm.ac.cn

# customize your commands of running QM programs here
rung16='g16'
runqchem='qchem'
rundalton='dalton'
runfcclasses='fcclasses3'

# constants
bohr=0.52917721067
c=3.0E8
NA=6.0221407E23
planck=6.62607015E-34

import os,sys,getopt,numpy

def print_help():
    print('''
    Usage: python autonr.py [options]

    Current workflow:
        Gaussian (opt/freq)
        DALTON (spin-orbit coupling, required by ISC)
        Q-Chem (derivative coupling, required by IC)
        FCClasses3 (non-radiative rate calculation)
        MOKIT (transfer wavefunction between QM programs)

    Available options:
        -h: Print help information and exit.
        -b: PBS/LSF/Slurm. Generate job script and submit.
        -n: Number of parallel threads.
        -a: Default: mol. This is name of the system.
            This requires mol.gjf, mol.inp, mol.dal and mol.mol as input templates.

        -q: Charge of current system.
        -r: Multiplicity of ground state.
            If state no.0 specified, state multiplicity of state no.0 will be used.
        -s: State number. (should be specified twice for state1 and state2, respectively)
        -m: State multiplicity. (should be specified twice for state1 and state2, respectively)
        -x: Input structure. (can be specified twice for state1 and state2, respectively)
            Should be formatted as xyz format with one structure for each file.
            If only one structure is given, -o and -f will be forced to be True.
            Ignored when -f being set to False.
        -v: Default: True. Also calculate reversed IC/ISC process.

        -o: Default: True. Whether to perform optimization task.
        -f: Default: True. Whether to perform frequency task.
            If -f being set to False, you must provide .fchk files for both states.
        -F: Provide two .fchk files.
        -c: Default: True. Whether to perform SOC or NAC calculation.
            If -c being set to False, you must provide a file containing related information.
        -C: Provide file that contains SOC or NAC information.
            Should be specified twice for state1 and state2, respectively, if -v Ture.
            File formats:
                SOC file: one float number
                NAC file: one matrix (3*n_atoms)
        -t: Default: False. Whether to use TD-DFT for lowest state with different multiplicity to ground state.
            e.g. UKS is recommanded for T1 state (ground state: closed-shell S0).
                 -t True will switch UKS into TD-DFT.

    Examples:
    >> python autonr.py -n 32 -r 1 -s 1 -m 1 -x S1.xyz -s 1 -m 3 T1.xyz
        Above will calculate (r)ISC rate between S1 and T1 states.
        There will be following steps:
        (1) Gaussian16 opt+freq
        (2) DALTON response calculation
        (3) FCClasses3 analysis
    >> python autonr.py -n 32 -r 1 -s 1 -m 1 -x S1.xyz -s 2 -m 1 S2.xyz
        Above will calculate (r)IC rate between S1 and S2 states.
        There will be following steps:
        (1) Gaussian16 opt+freq
        (2) Q-Chem derivative coupling calculation
        (3) FCClasses3 analysis
''')
    sys.exit()
    return

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def idelem(id):
    Element=['H' ,                                                                                                                                                      'He',
             'Li','Be',                                                                                                                        'B' ,'C' ,'N' ,'O' ,'F' ,'Ne',
             'Na','Mg',                                                                                                                        'Al','Si','P' ,'S' ,'Cl','Ar',
             'K' ,'Ca',                                                                      'Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
             'Rb','Sr',                                                                      'Y' ,'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I' ,'Xe',
             'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W' ,'Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
             'Fr','Ra','Ac','Th','Pa','U' ,'Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn','Nh','Fl','Mc','Lv','Ts','Og']
    return Element[id-1]

def td(state,mult):
    global ref_mult,flag_td
    if (mult==ref_mult and state>=1) or\
       (mult!=ref_mult and state==1 and flag_td) or\
       (mult!=ref_mult and state>=2):
        return True
    else:
        return False

def format_atom(atom):
    line=f'     {atom[0].lower().capitalize()}'.ljust(4)
    for i in range(3):
        line+=f'{atom[1+i]:.10f}'.rjust(16)
    return line+'\n'

def next_item(list,item):
    for i in list:
        if item==i:
            flag=True
            continue
        if flag:
            return i

def parse_mult(mult):
    mults=['S','D','T','Q']
    if mult<=len(mults):
        return mults[mult-1]
    else:
        return mult

def read_geom_from_xyz(xyz):
    geom=[]
    with open(xyz,'r') as read:
        lines=read.readlines()
        natoms=int(lines[0])
        for line in lines[2:]:
            l=line.split()
            if len(l)!=0:
                geom.append([l[0],float(l[1]),float(l[2]),float(l[3])])
            if len(geom)==natoms:
                break
    return geom

def read_geom_from_fch(fch):
    global bohr
    atm,xyz=[],[]
    with open(fch,'r') as read:
        flag1,flag2=False,False
        for line in read.readlines():
            l=line.split()
            if 'Nuclear charges' in line:
                flag1=False
                continue
            if 'Number of symbols' in line:
                break
            if 'Current cartesian coordinates' in line:
                flag2=True
                continue
            if 'Atomic numbers' in line:
                flag1=True
                continue
            if flag1:
                for i in range(len(l)):
                    atm.append(idelem(int(l[i])))
            if flag2:
                for i in range(len(l)):
                    xyz.append(float(l[i])*bohr)
    geom=[]
    for i in range(len(atm)):
        geom.append([atm[i],xyz[0+3*i],xyz[1+3*i],xyz[2+3*i]])
    return geom

def gen_g16_input(geom,state,mult,opt=True,freq=True):
    global nproc,sysname,chg,ref_mult
    flag,case1,case2=False,False,False
    with open(f'{sysname}.gjf','r') as tmp:
        with open(f'{sysname}_{parse_mult(mult)}{state}.gjf','w') as gen:
            gen.write(f'%nprocshared={nproc}\n%chk={sysname}_{parse_mult(mult)}{state}.chk\n')
            for line in tmp.readlines():
                if len(line.split())!=0:
                    if '%nproc' in line.lower() or '%chk' in line.lower():
                        continue
                    elif line.split()[0][0]=='#':
                        flag=True
                if flag:
                    if len(line.split())==0:
                        flag=False
                        if case1==False and opt:
                            gen.write('opt ')
                        if case2==False and freq:
                            gen.write('freq\n')
                        gen.write(f'\n{parse_mult(mult)}{state}\n\n{chg} {ref_mult if td(state,mult) else mult}\n')
                        for i in geom:
                            gen.write(format_atom(i))
                        gen.write('\n')
                        continue
                    else:
                        for i in line.split():
                            if '#' in i:
                                gen.write('#p nosymm int=nobasistransform scf(maxcycle=512,xqc)')
                            elif i.lower() not in ['opt','freq']:
                                if 'opt' in i.lower():
                                    if opt:
                                        case1=True
                                    else:
                                        continue
                                if 'freq' in i.lower():
                                    case2=True
                                gen.write(f' {i}')
                            elif 'td' in i.lower:
                                if td(state,mult):
                                    for item in ['=',',','(',')']:
                                        i=i.replace(item,' ')
                                    l=i.lower().split()
                                    nstates=state+3; root=state
                                    if 'nstate' in l:
                                        nstates=max(nstates,int(next_item(l,'nstate')))
                                    elif 'nstates' in l:
                                        nstates=max(nstates,int(next_item(l,'nstates')))
                                    if mult==1:
                                        gen.write(f' TD(nstates={nstates},root={state})')
                                    elif mult==3:
                                        gen.write(f' TD(nstates={nstates},root={state},triplet)')
                                else:
                                    continue
                    gen.write('\n')
                else:
                    gen.write(line)
    return

def run_g16(state,mult):
    global rung16,sysname
    run_command(f'{rung16} {sysname}_{parse_mult(mult)}{state}.gjf {sysname}_{parse_mult(mult)}{state}.log')
    # check g16 normal termination
    ene,gibbs=0.,0.
    with open(f'{sysname}_{parse_mult(mult)}{state}.log','r') as read:
        lines=read.readlines()
        if 'Normal termination' in lines[-1]:
            print(lines[-1])
        else:
            autonr_exit('Error termination of Gaussian16 detected. Please check!')
    # check imaginary frequency and energy output
        for line in lines:
            if 'Frequencies --' in line:
                if float(line.split()[2])<0:
                    autonr_exit('Imaginary frequency detected. Please check!')
            if 'SCF Done' in line:
                ene=float(line.split()[4])
            if 'E(TD-HF/TD-DFT)' in line:
                ene=float(line.split()[4])
            if 'Sum of electronic and thermal Free Energies=' in line:
                gibbs=float(line.split()[-1])
    print(' No imaginary frequency detected. Minimum point reached.')
    print(f' Converged single point energy for {parse_mult(mult)}{state} state: {ene:.8f} a.u.')
    print(f' Sum of electronic and thermal Free Energy:  {gibbs:.6f}   a.u.\n')
    # generate .fchk file
    run_command(f'formchk {sysname}_{parse_mult(mult)}{state}.chk {sysname}_{parse_mult(mult)}{state}.fchk')
    geom=read_geom_from_fch(f'{sysname}_{parse_mult(mult)}{state}.fchk')
    print('\n Current cartesian coordinates (Angstrom):')
    for i in geom:
        print(format_atom(i),end='')
    print('')
    return geom,ene

def gen_dalton_input(geom,state,mult):
    global sysname,chg
    # transfer wavefunction from Gaussian16 to DALTON
    recalcref=False
    # DALTON does not support UHF/UKS
    import linecache
    if mult!=1 and linecache.getline(f'{sysname}_{parse_mult(mult)}{state}.fchk',2).lower().split()[1][0]=='u':
        recalcref=True
        print(''' Warning: DALTON does not support UHF/UKS!
 Switch to RHF/RKS reference state. Recalculate reference state by Gaussian16.
''')
    if recalcref:
        gen_g16_input(geom,-1,1,False,False)
        global rung16
        run_command(f'{rung16} {sysname}_S-1.gjf {sysname}_S-1.log')
        run_command(f'formchk {sysname}_S-1.chk {sysname}_S-1.fchk')
        with os.popen(f'fch2dal {sysname}_S-1.fchk') as fch2dal:
            null=fch2dal.read()
        os.system(f'mv {sysname}_S-1.dal {sysname}_{parse_mult(mult)}{state}.dal')
        os.system(f'mv {sysname}_S-1.mol {sysname}_{parse_mult(mult)}{state}.mol')
    else:
        with os.popen(f'fch2dal {sysname}_{parse_mult(mult)}{state}.fchk') as fch2dal:
            null=fch2dal.read()
    if os.path.exists(f'{sysname}_{parse_mult(mult)}{state}.mol') and os.path.exists(f'{sysname}_{parse_mult(mult)}{state}.dal'):
        print(' Successfully transferred wavefunction from Gaussian16 to DALTON.')
    else:
        autonr_exit('Error in transferring wavefunction from Gaussian16 to DALTON!')
    # modify .mol input
    os.system(f'sed -i "s/Charge=0/Charge={chg}/g" {sysname}_{parse_mult(mult)}{state}.mol')
    # modify .dal input
    os.system(f'sed -i "s/.HF/.DFT/g" {sysname}_{parse_mult(mult)}{state}.dal')
    xc='CAM-B3LYP'
    xcinfo=linecache.getline(f'{sysname}_{parse_mult(mult)}{state}.fchk',2).lower()
    if 'cam-b3lyp' not in xcinfo and 'b3lyp' in xcinfo:
        xc='B3LYP'
    with open(f'{sysname}.dal','r') as tmp:
        with open(f'{sysname}_{parse_mult(mult)}{state}.dal','r') as read:
            with open(f'gen','w') as gen:
                gen.write(f'''\
**DALTON INPUT
.RUN RESPONS
**INTEGRALS
.SPIN-ORBIT
**WAVE FUNCTIONS
.DFT
 {xc}
*ORBITAL INPUT
.MOSTART
FORM18
.PUNCHOUTPUTORBITALS
''')
                flag=False
                for line in read.readlines():
                    if '**MOLORB' in line:
                        flag=True
                    elif '**END' in line:
                        flag=False
                        break
                    if flag:
                        gen.write(line)
                gen.write(f'''
**RESPONS
*QUADRATIC
.DOUBLE RESIDUE
.ISPABC
1 0 1
.PROPRT
X1SPNORB
.PROPRT
Y1SPNORB
.PROPRT
Z1SPNORB
.ROOTS
 {state+3}
''')
                for line in tmp.readlines():
                    if line.lstrip()[:2]=='**':
                        if line.split()[0] not in ['**DALTON','**WAVE','**MOLORB','**RESPONS','**INTEGRALS','**END']:
                            flag=True
                        else:
                            flag=False
                    if flag:
                        gen.write(line)
                gen.write('**END OF INPUT')
    os.system(f'mv gen {sysname}_{parse_mult(mult)}{state}.dal')
    return

def run_dalton(state1,mult1,state2,mult2):
    global rundalton,sysname,nproc
    run_command(f'{rundalton} -N {nproc} -o {sysname}_{parse_mult(mult1)}{state1}.out {sysname}_{parse_mult(mult1)}{state1} {sysname}_{parse_mult(mult1)}{state1}')
    # check DALTON normal termination
    with open(f'{sysname}_{parse_mult(mult1)}{state1}.out','r') as read:
        flag=False
        for line in read.readlines():
            if 'End of Dynamic Property Section (RESPONS)' in line:
                flag=True
            if flag:
                if 'Total wall time used in DALTON:' in line:
                    print(f' Normal termination of DALTON. {line.strip()}.')
        if flag==False:
            autonr_exit('Error termination of DALTON detected. Please check!')
    # read spin-orbit coupling integrals from DALTON output
    os.system(f'grep "B excited state no., symmetry, spin:             {state1 if mult1==1 else state2}" {sysname}_{parse_mult(mult1)}{state1}.out -A 3 -B 2 > null1')
    os.system(f'grep "C excited state no., symmetry, spin:             {state1 if mult1==3 else state2}" null1 -A 3 -B 2 > null2')
    socxyz=[]
    with open('null2','r') as read:
        for line in read.readlines():
            if 'B and C excitation energies, moment:' in line:
                socxyz.append(float(line.split()[-1]))
    os.system('rm null1 null2')
    [socx,socy,socz]=socxyz
    soc=numpy.linalg.norm(numpy.array(socxyz))
    print(f'''
 Spin-orbit coupling matrix elements:
     |SOC_x| = {abs(socx):.8f} a.u.
     |SOC_y| = {abs(socy):.8f} a.u.
     |SOC_z| = {abs(socz):.8f} a.u.
      SOC    = {soc:.8f} a.u.
''')
    return soc

def gen_qchem_input(geom,state,mult):
    

def run_qchem(state1,mult1,state2,mult2):

    # read non-adiabatic coupling from Q-Chem output
    global natoms
    nac=[]
    with open() as read:
        flag=False
        for line in read.readlines():
            if 'derivative coupling from response' in line:
                flag=True
            if flag:
                l=line.split()
                if len(l)==4 and is_number(l[0]):
                    nac.append([float(l[1]),float(l[2]),float(l[3])])
                if len(nac)==natoms:
                    break
    return nac

def run_fcc(states,mults,enes,coup):
    global sysname,reverse
    # transfer .fchk files into .fcc inputs
    for i in range(2):
        with os.popen(f'gen_fcc_state -i {sysname}_{parse_mult(mults[i])}{states[i]}.fchk') as fchk2fcc:
            null=fchk2fcc.read()
        with open(f'{sysname}_{parse_mult(mults[i])}{states[i]}.fcc','a') as addene:
            addene.write(f'''
ENER      UNITS=AU
 {enes[i]:.8E}
''')
    # FCClasses3 input for ISC rate calculation
    if mults[0]!=mults[1]:
        with open(f'{sysname}_{parse_mult(mults[0])}{states[0]}_{parse_mult(mults[1])}{states[1]}_fcc.inp','w') as gen:
            gen.write(f'''$$$; FCClasses3 input file generated by AutoNR
PROPERTY     =   NR0
MODEL        =   AH
DE           =   {27.2114*(enes[1]-enes[0]):.8f}
NR0_COUPL    =   {coup:.10f}
TEMP         =   298.15
BROADFUN     =   GAU
HWHM         =   0.06
METHOD       =   TD
ROT          =   1
;VIBRATIONAL ANALYSIS
NORMALMODES  =   COMPUTE
COORDS       =   CARTESIAN
;INPUT DATA FILES
STATE1_FILE  =   {sysname}_{parse_mult(mults[0])}{states[0]}.fcc
STATE2_FILE  =   {sysname}_{parse_mult(mults[1])}{states[1]}.fcc''')
    # FCClasses3 input for IC rate calculation
    else:
        with open(f'{sysname}_{parse_mult(mults[0])}{states[0]}_{parse_mult(mults[1])}{states[1]}_fcc.inp','w') as gen:
            gen.write(f'''$$$; FCClasses3 input file generated by AutoNR
;GENERAL OPTIONS:
PROPERTY     =   IC
MODEL        =   AH
TEMP         =   298.15
BROADFUN     =   GAU
HWHM         =   0.010
METHOD       =   TD
;VIBRATIONAL ANALYSIS
COORDS       =   CARTESIAN
;INPUN DATA FILES
STATE1_FILE  =   {sysname}_{parse_mult(mults[0])}{states[0]}.fcc
STATE2_FILE  =   {sysname}_{parse_mult(mults[1])}{states[1]}.fcc
NAC_FILE     =   {sysname}_{parse_mult(mults[0])}{states[0]}_{parse_mult(mults[1])}{states[1]}.coup
;VERBOSE LEVEL
VERBOSE      =   1''')
        # write NAC input file
        with open(f'{sysname}_{parse_mult(mults[0])}{states[0]}_{parse_mult(mults[1])}{states[1]}.coup','w') as gen:
            global natoms
            for i in range(natoms):
                gen.write(f'{coup[0+3*i]}  {coup[1+3*i]}  {coup[2+3*i]}\n')
    # run FCClasses3
    global runfcclasses
    run_command(f'{runfcclasses} {sysname}_{parse_mult(mults[0])}{states[0]}_{parse_mult(mults[1])}{states[1]}_fcc.inp',True)
    proc_fcc_res(f'{sysname}_{parse_mult(mults[0])}{states[0]}_{parse_mult(mults[1])}{states[1]}_fcc.out')
    return

def proc_fcc_res(fccout='fcc.out'):
    # read frequencies and rate from fcc output
    rate=0.; wavenum=[]
    with open(fccout,'r') as read:
        flag1,flag2=False,False
        for line in read.readlines():
            if 'Non-radiative rate constant' in line:
                rate=float(line.split()[-1])
                continue
            if 'VIBRATIONAL ANALYSIS ON STATE2' in line:
                flag1=True
                continue
            if flag1 and 'FREQUENCIES (cm-1)' in line:
                flag2=True
                continue
            if 'Orthogonalizing normal mode vector' in line:
                flag1,flag2=False,False
                continue
            if flag2:
                l=line.split()
                if len(l)==2:
                    wavenum.append(float(l[1]))
    print(f'\n Rate constant: {rate:.3E} s^-1')
    # read Huang-Rhys factors
    hr=[]
    with open('HuangRhys.dat','r') as read:
        for line in read.readlines():
            l=line.split()
            if len(l)==2:
                hr.append(float(l[1]))
    # read displacement vector
    disp=[]
    with open('displacement.dat') as read:
        for line in read.readlines():
            l=line.split()
            if len(l)==1:
                disp.append(float(l[0]))
    # output Origin plot file for Huang-Rhys factors and reorganization energy decomposition
    reorg=0.
    print(f'\n Huang-Rhys factor output for Origin plot: {fccout[:-4]}_huangrhys.plot')
    global c,NA,planck
    with open(f'{fccout[:-4]}_huangrhys.plot','w') as gen:
        gen.write('Frequency (cm^-1)    Huang-Rhys factor    Reorganization energy (cm^-1)\n')
        print('--------------------------------------------------------------------------------')
        print('    Frequency (cm^-1)    Huang-Rhys factor    Reorganization energy (cm^-1)')
        gen.write(f'{0:>18.4f}     {0:>16.4f}     {0:>16.4f}\n')
        for i in range(len(wavenum)):
            # calculate normal mode contribution to total reorganization energy
            tmp=hr[i]*wavenum[i]
            reorg+=tmp
            # generate plot information
            gen.write(f'{wavenum[i]:>18.4f}     {0:>16.4f}     {0:>16.4f}\n')
            gen.write(f'{wavenum[i]:>18.4f}     {hr[i]:>16.4f}     {tmp:>16.4f}\n')
            print(f'    {wavenum[i]:>18.4f}     {hr[i]:>16.4f}     {tmp:>16.4f}')
            gen.write(f'{wavenum[i]:>18.4f}     {0:>16.4f}     {0:>16.4f}\n')
        gen.write(f'{4000:>18.4f}     {0:>16.4f}     {0:>16.4f}\n')
        print('--------------------------------------------------------------------------------')
        print(f' Total reorganization energy: {reorg:.8f} cm^-1 / {(100*reorg*c*planck*NA)/2625500*27.2114:.8f} eV')
    # read duschinsky matrix
    duschinsky=numpy.zeros((len(wavenum),len(wavenum)))
    with open('duschinsky.dat') as read:
        counter=0
        for line in read.readlines():
            l=line.split()
            if len(l)==1:
                duschinsky[int(counter/len(wavenum)),counter%len(wavenum)]=float(l[0])
                counter+=1
    # output Origin plot file for Duschinsky matrix
    print(f'\n Duschinsky matrix output for Origin plot: {fccout[:-4]}_duschinsky.plot\n')
    with open(f'{fccout[:-4]}_duschinsky.plot','w') as gen:
        for i in range(len(wavenum)):
            for j in range(len(wavenum)):
                gen.write(f'{i:>5}  {j:>5}  {duschinsky[i,j]:>16.8f}\n')
    return

def main():
    # running parameters
    global nproc,sysname,submit
    nproc,sysname,submit=1,'mol',None
    global chg,ref_mult,flag_td,reverse
    chg=0; ref_mult=None; flag_td=False; state,mult,geom=[],[],[]
    opt,freq,coup,reverse=True,True,True,True; fchk,coupfile=[],[]
    # get options
    try:
        opts,args=getopt.getopt(sys.argv[1:],'hb:n:a:q:r:s:m:x:v:o:f:F:c:C:t:')
    except getopt.GetoptError:
        print_help()
    for opt,arg in opts:
        if opt=='-h':
            print_help()
        if opt=='-b':
            submit=arg.lower()
        elif opt=='-n':
            nproc=int(arg)
        elif opt=='-a':
            sysname=arg
        # system options
        elif opt=='-q':
            chg=int(arg)
        elif opt=='-r':
            ref_mult=int(arg)
        elif opt=='-s':
            state.append(int(arg))
        elif opt=='-m':
            mult.append(int(arg))
        elif opt=='-x':
            geom.append(arg)
        elif opt=='-v':
            if arg.lower()=='false':
                reverse=False
        # workflow options
        elif opt=='-o':
            if arg.lower()=='false':
                opt=False
        elif opt=='-f':
            if arg.lower()=='false':
                freq=False
        elif opt=='-F':
            fchk.append(arg)
        elif opt=='-c':
            if arg.lower()=='false':
                coup=False
        elif opt=='-C':
            coupfile=arg
        elif opt=='-t':
            if arg.lower()=='true':
                flag_td=True
    print(' Entering AutoNR program.\n')
    # check options
    print(' Checking input parameters...')
    import multiprocessing as mp
    if nproc > mp.cpu_count():
        autonr_exit(f'Required {nproc} threads but only {multiprocessing.cpu_count()} available!')
    # state options
    if len(state)!=2:
        autonr_exit(f'Required two number of states specified! Current: {len(state)}')
    if len(mult)!=2:
        autonr_exit(f'Required two multiplicities specified! Current: {len(mult)}')
    for i in range(2):
        if state[i]==0:
            if ref_mult==None:
                ref_mult=mult[i]
            elif ref_mult!=mult[i]:
                print(f' System multiplicity is not equal to that of state No.0. Change system multiplicity into {mult[i]}.')
                ref_mult=mult[i]
    if ref_mult==None:
        autonr_exit('Unable to determine the ground state multiplicity. Please specify in input options.')
    # workflow options: opt/freq
    if freq==False:
        if opt:
            print(' Optimization without frequency task is not allowed! Forced setting freq=True.')
            freq=True
        elif len(fchk)!=2:
            autonr_exit(f'Required two .fchk files specified! Current: {len(fchk)}')
    # workflow options: SOC/NAC
    if len(coupfile)==0:
        if coup==False:
            print(f' No SOC/NAC results provided. Forced setting coup=True.')
            coup=True
    elif len(coupfile)>=1:
        if coup==True:
            print(f' SOC/NAC results provided. Forced setting coup=False.')
            coup=False
        if reverse and len(coupfile)==1:
            autonr_exit('Two files containing SOC/NAC result should be provided. Current: 1')
        elif reverse==False and len(coupfile)==2:
            print(f' Two files provided while "-v False". The second file "{coupfile[1]}" is ignored.')
            del coupfile[1]
    # geometry options
    if len(fchk)==2 and len(geom)!=0:
        print(' Two .fchk files provided and inpuit geometry ignored.')
        geom=[]
    if len(geom)==1:
        geom.append(geom[0])
        opt,freq=True,True
    elif len(geom)==2 or (len(fchk)==2 and len(geom)==0):
        pass
    else:
        autonr_exit(f'Required one or two structure(s) specified! Current: {len(geom)}')
    check_pass_info(state,mult,geom,opt,freq,coup,fchk,coupfile)
    # check passed, now run standard workflow
    # go to queuing system if required
    if submit in ['pbs','lsf','slurm']:
        submit_to_queue_system(submit)
        autonr_exit(None,False)
    elif submit!=None:
        autonr_exit(f'Unsupported queuing system "{submit}"! Available: PBS/LSF/Slurm')
    # directly running
    # read input structures
    if len(geom)==2:
        for i in range(2):
            geom[i]=read_geom_from_xyz(geom[i])
    else:
        for i in range(2):
            geom.append(read_geom_from_fch(fchk[i]))
    # check number of atoms
    global natoms
    if len(geom[0])==len(geom[1]):
        natoms=len(geom[0])
    else:
        autonr_exit('Two structures have different numbers of atoms!')
    # run standard workflow to calculate (r)ISC/(r)IC processes
    enes,soc=[None,None],None
    if freq:
        for i in range(2):
            gen_g16_input(geom[i],state[i],mult[i],opt)
            geom[i],enes[i]=run_g16(state[i],mult[i])
    coupint=[]
    if coup: # NAC/SOC is calculated based on the structure of initial state minima
        if mult[0]==mult[1]: # (r)IC
            gen_qchem_input(geom[0],state[0],mult[0])
            coupint.append(run_qchem(state[0],mult[0]))
            if reverse:
                gen_qchem_input(geom[1],state[1],mult[1])
                coupint.append(run_qchem(state[1],mult[1]))
        else: # (r)ISC
            gen_dalton_input(geom[0],state[0],mult[0])
            coupint.append(run_dalton(state[0],mult[0],state[1],mult[1]))
            if reverse:
                gen_dalton_input(geom[1],state[1],mult[1])
                coupint.append(run_dalton(state[1],mult[1],state[0],mult[0]))
    else:
        def read_coupfile(coupfile,filetype='SOC'):
            if filetype=='SOC':
                with open(coupfile,'r') as read:
                    for line in read.readlines():
                        l=line.split()
                        if len(l)!=0:
                            soc=float(l[0])
                            break
                print(f' Spin-orbit coupling integral read from {coupfile}: {soc:.8f} a.u.')
                return soc
            elif filetype=='NAC':
                global natoms
                nac=[]
                with open(coupfile,'r') as read:
                    for line in read.readlines():
                        l=line.split()
                        if len(l)==3:
                            for i in range(3):
                                nac.append(float(l[i]))
                        if len(nac)==3*natoms:
                            break
                print(f' Non-adiabatic coupling integral read from {coupfile}:')
                for i in range(natoms):
                    print(f'{nac[0+3*i]:.10f}  {nac[1+3*i]:.10f}  {nac[2+3*i]:.10f}')
                return nac
        # read NAC/SOC from user-provided files
        if mult[0]==mult[1]: # (r)IC
            coupint.append(read_coupfile(coupfile[0],filetype='NAC'))
            if reverse:
                coupint.append(read_coupfile(coupfile[1],filetype='NAC'))
        else: # (r)ISC
            coupint.append(read_coupfile(coupfile[0],filetype='SOC'))
            if reverse:
                coupint.append(read_coupfile(coupfile[1],filetype='SOC'))
    print(f'{"> Intersystem Crossing Process <" if mult[0]!=mult[1] else "> Internal Conversion Process <" :-^80}')
    run_fcc(state,mult,enes,coupint[0])
    if reverse:
        print(f'{"> Reversed Intersystem Crossing Process <" if mult[0]!=mult[1] else "> Reversed Internal Conversion Process <" :-^80}')
        run_fcc(list(reversed(state)),list(reversed(mult)),list(reversed(enes)),coupint[1])
    autonr_exit(None,False)
    return

def autonr_exit(info=None,error=True):
    if info!=None:
        print(f' {info}')
    if error:
        print(' Error termination of AutoNR.')
    else:
        print(' Normal termination of AutoNR.')
    sys.exit()

def check_pass_info(state,mult,geom,opt,freq,coup,fchk,coupfile):
    global sysname,chg,ref_mult,reverse
    print(f'''
 Initial check passed.
     system name:          {sysname}
     system charge:        {chg}
     system multiplicity:  {ref_mult}

 Calculate non-radiative rate for {'(r)' if reverse else ''}{'IC' if mult[0]==mult[1] else 'ISC'} process of:
 state | no.| multiplicity | geom/fchk            | {'NAC' if mult[0]==mult[1] else 'SOC'}
     1 |  {state[0]} |   {mult[0]}          | {geom[0] if len(geom)==2 else fchk[0] :<20} | {coupfile[0] if len(coupfile)!=0 else None}
     2 |  {state[1]} |   {mult[1]}          | {geom[1] if len(geom)==2 else fchk[1] :<20} | {coupfile[1] if len(coupfile)==2 else None}
''')
    return

def run_command(command,printnull=False):
    print(f' $ {command}')
    with os.popen(command) as run:
        null=run.read()
        if printnull:
            print(null)
    return

def submit_to_queue_system(submit):
    print(' Job submitted!')
    return

if __name__=='__main__':
    main()
