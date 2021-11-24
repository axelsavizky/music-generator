FILES="./InputSongs/*"
for f in $FILES; do
    for i in 1 10 50 100 300 500 1000; do
        model=$(basename -- "$1")
        filename=$(basename -- "$f")
        name=$model$i$filename
        python3 ../generate_song.py $1 $2 $f $i $name
    done
done
