dir=${1}

if [ -d "$dir" ]; then 
    echo "Directory $dir already exists."
else
    mkdir "$dir"
    echo "Directory $dir created. Adding simple .gnu file..."
    touch $dir/$dir.gnu 
    echo "load \"../style/default.gnu\"" >> $dir/$dir.gnu
    echo "load \"../style/term.gnu\"" >> $dir/$dir.gnu
    echo "set out \"$dir.tex\"" >> $dir/$dir.gnu
fi 