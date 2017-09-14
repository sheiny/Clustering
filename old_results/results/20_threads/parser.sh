for i in gpu parallel
do
  touch $i.csv
  cat $i.dat | grep -w superblue18 | awk '{print $2",\t "}' > $i.csv
  for j in superblue5 superblue16 superblue1 superblue3 superblue4 superblue10
  do
    cat $i.dat | grep -w $j | awk '{print $2",\t"}' > temp.txt
    paste $i.csv temp.txt > temp.csv
    cat temp.csv > $i.csv
  done
  cat $i.dat | grep -w superblue7 | awk '{print $2}' > temp.txt
  paste $i.csv temp.txt > temp.csv
  echo "superblue18, superblue5, superblue16, superblue1, superblue3, superblue4, superblue10, superblue7" > $i.csv
  cat temp.csv >> $i.csv
done
rm temp.txt
rm temp.csv
