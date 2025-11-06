head -n 1 TARGRT_DATA/TARGET_1/merged.csv > TARGRT_DATA/combined.csv
tail -n +2 -q TARGRT_DATA/TARGET_1/merged.csv TARGRT_DATA/TARGET_2/merged.csv TARGRT_DATA/TARGET_3/merged.csv >> TARGRT_DATA/combined.csv
