# Octopus

Modified 3D printer hydrophone scanning system that does (https://www.ondacorp.com/scanning-tank/) but 300x cheaper, excluding hydrophone costs. 

```bash python measure.py 30 50 30 0.5``` to manually position hydrophone in transducer space origin, and sample pressure field in x=30, y=50, z=30 volume in 30/0.5 x 50/0.5 x 30/0.5 total sampling steps 

Then run ```bash python visualize_live.py``` to see pressure field populated in real time 

# Requirements

- Hydrophone. With luck you'll find used ones on ebay for cheap. Unfortunately calibration will cost you ($1.5k for Onda HNR-0500 for 0.25-1Mhz range). 

- 3D printer with step size preferrably 3x smaller than the diameter of your hydrophone. In my setup I use Onda's HNR-0500 with 500µm diameter and Ender 3D V2 with minimum 50µm step size 

- Oscilloscope (used DS1202)

- Function generator (used JDS6600) 

Including the hydrophone + calibration, the total cost was $2,600.  


