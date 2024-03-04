###
# We are now ready to run pytype on our full corpus
# This scrpit should be run on docker container typilus:env !!!
# Working path: /usr/data
##
# export SITE_PACKAGES=/usr/local/lib/python3.6/dist-packages
# for repo in ./repos/*; do
#      echo Running: pytype -V3.6 --keep-going -o ./pytype -P $SITE_PACKAGES:$repo infer $repo
#      pytype -V3.6 --keep-going -o ./pytype -P $SITE_PACKAGES:$repo infer $repo

#      files=$(find $repo -name "*.py")
#      for f in $files
#      do
#          f_stub=$f"i"
#          f_stub="./pytype/pyi"${f_stub#"$repo"}
#          if [ -f $f_stub ]; then
#              echo Running: merge-pyi -i $f $f_stub
#              merge-pyi -i $f $f_stub
#          fi
#      done
#  done

export SITE_PACKAGES=/usr/local/lib/python3.6/dist-packages
input_file="./typilus/repo_list.txt"
repo_dir="./typilus/repos/"

while IFS= read -r repo; do
    repo=$repo_dir$repo
    echo Running: pytype -V3.6 --keep-going -o ./pytype -P $SITE_PACKAGES:$repo infer $repo
    pytype -V3.6 --keep-going -o ./pytype -P $SITE_PACKAGES:$repo infer $repo

    files=$(find $repo -name "*.py")
    for f in $files
    do
        f_stub=$f"i"
        f_stub="./pytype/pyi"${f_stub#"$repo"}
        if [ -f $f_stub ]; then
            echo Running: merge-pyi -i $f $f_stub
            merge-pyi -i $f $f_stub
        fi
    done
done < "$input_file"