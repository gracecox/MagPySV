#!/bin/bash
# Condense jump records from (private) WDC Edinburgh OAM database.
# Get all lines containing jumps and the following one line
grep -A 1 "    0    0" ~worldobs/data/oam/oamfile.dat > tmp
# Remove the dashes on every third line
awk 'NR%3' tmp > temp
# Extract the observatory name and year from the line below the jump values
obs=( $(cat temp | sed -n 'n;p' | cut -c11-13))
year=( $(cat temp | sed -n 'n;p' | cut -c15-18))
# Extract the x, y and z jump values from the jump record
x=( $(cat temp | sed -n 'p;n' | cut -c46-51))
y=( $(cat temp | sed -n 'p;n' | cut -c53-58))
z=( $(cat temp | sed -n 'p;n' | cut -c60-65))
# Write jumps to file
for i in "${!obs[@]}"; do printf "%s,%s,%s,%s,%s\n" "${obs[i]}" "${year[i]}" "${x[i]}" "${y[i]}" "${z[i]}"; done > baseline_records
# Remove flagged rows
sed -i '/99999,99999,99999$/d' baseline_records
rm temp
rm tmp
