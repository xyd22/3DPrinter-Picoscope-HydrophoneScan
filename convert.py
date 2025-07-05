import math
import sys

FR = 0.5e6
V_Pa = 2.078E-007

def print_mpa(pa):
    print(f"{pa/1e6:.3f} Mpa")

def print_vpp(vpp):
    print(f"{vpp:.3f} Vpp")
    
def vpp2prms(Vpp):
    # convert VPP to prms 
    Vrms = Vpp/(2*math.sqrt(2))
    return Vrms / V_Pa
    
def vpp2ppeak(Vpp):
    # convert VPP to prms 
    Vpeak = Vpp / 2
    return Vpeak / V_Pa

def ppeak2vpp(ppeak):
    return ppeak * V_Pa * 2 

Vpp_re = 1.568

if len(sys.argv) > 1:
    Vpp_re = float(sys.argv[1])

print_mpa(vpp2prms(Vpp_re))
print_mpa(vpp2prms(Vpp_re)*math.sqrt(2))
print_vpp(ppeak2vpp(400000))
