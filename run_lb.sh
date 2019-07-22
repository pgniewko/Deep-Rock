#! /bin/bash -x 
LB_PATH="/Users/pawel/Desktop/LB/"
SF="./src/lb/porous-2d"
L=64
LxLxL=192
ka=2
kb=4

for INDEX in `seq 1 10000`; do
    FILE="/Users/pawel/Projects/Deep-Rock/output/LATTICE/"$L"_"$ka"_"$INDEX".lattice"
    if [ -f "$FILE" ]
    then
        $SF $LxLxL $L $FILE $L $ka $INDEX $LB_PATH
    else
        echo 0.0 0.0 > $LB_PATH"/"$L"_"$ka"_"$INDEX".dat"
    fi

    FILE="/Users/pawel/Projects/Deep-Rock/output/LATTICE/"$L"_"$kb"_"$INDEX".lattice"
    if [ -f "$FILE" ]
    then
        $SF $LxLxL $L $FILE $L $kb $INDEX $LB_PATH
    else
        echo 0.0 0.0 > $LB_PATH"/"$L"_"$kb"_"$INDEX".dat"
    fi
done
