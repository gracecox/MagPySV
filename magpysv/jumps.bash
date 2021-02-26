#!/bin/bash
# Condense jump records from (private) WDC Edinburgh OAM database.
# Get all lines containing jumps and the following one line
grep -A 1 "    0    0" ~worldobs/data/oam/oamfile.dat > jumps.txt
# Remove the line with dashes
cat jumps.txt | sed '/^--/d' > jumps2.txt
# Extract the observatory name and year from the line below the jump values
obs=( $(cat jumps2.txt | sed -n 'n;p' | cut -c11-13))
year=( $(cat jumps2.txt | sed -n 'n;p' | cut -c15-18))
# Extract the x, y and z jump values from the jump record
x=( $(cat jumps2.txt | sed -n 'p;n' | cut -c46-51))
y=( $(cat jumps2.txt | sed -n 'p;n' | cut -c53-58))
z=( $(cat jumps2.txt | sed -n 'p;n' | cut -c60-65))
# Write jumps to file
for i in "${!obs[@]}"
do printf "%s,%s,%s,%s,%s\n" "${obs[i]}" "${year[i]}" "${x[i]}" "${y[i]}" "${z[i]}"
done > baseline_records
rm jumps.txt jumps2.txt
