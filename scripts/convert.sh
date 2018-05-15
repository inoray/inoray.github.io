#!/bin/bash

# example use:
# [~/gitrepos/dkmehrmann.github.io/_ipynb]$ ../scripts/convert.sh google_maps.ipynb 
# 

BASE_DIR="/home/shkim/blog/inoray.github.io"
BUILD_DIR=$BASE_DIR"/_ipynb/"
POST_DIR=$BASE_DIR"/_posts/"

# Generate a filename with today's date.
ipynb_fname="$1"
md_fname="${ipynb_fname/ipynb/md}"
dt=`date +%Y-%m-%d`
post_fname="$dt-$md_fname"
echo "file name changed from $1 to $post_fname"

# Jupyter will put all the assets associated with the notebook in a folder with this naming convention.
# The folder will be in the same output folder as the generated markdown file.
filename="${ipynb_fname/.ipynb}"
dt_filename="$dt-$filename"
dt_files_dir=$dt_filename"_files"

# use nbconvert on the file
jupyter nbconvert --to markdown $1 --output-dir=$POST_DIR --output=$dt_filename --config $BASE_DIR/scripts/jekyll.py

# Move the images from the jupyter-generated folder to the images folder.
echo "Moving images..."
mkdir $BASE_DIR/assets/images/$dt_filename
mv $POST_DIR$dt_files_dir/* $BASE_DIR/assets/images/$dt_filename

# Remove the now empty folder.
rmdir $POST_DIR$dt_files_dir

# Go through the markdown file and rewrite image paths.
# NB: this sed command works on OSX, it might need to be tweaked for other platforms.
echo "Rewriting image paths..."
sed -i "s/$dt_files_dir/\/assets\/images\/$dt_filename/g" $POST_DIR$post_fname

# adds the date to the file
#dt2=`date +"%b %d, %Y"`
#sed -i "3i date: $dt2" $POST_DIR$post_fname
#echo "added date $dt2 to line 3"

# Gets the title of the post
#echo "What's the title of this post going to be?"
#read ttl
#sed -i "4i title: \"$ttl\"" $POST_DIR$post_fname
#echo "added title $ttl in line 4"

# if the current version is newer than the version in _posts
#if [[ $1 -nt $POST_DIR$post_fname ]]; then
#  mv $BUILD_DIR$post_fname $POST_DIR$post_fname
#  echo "moved $post_fname from $BUILD_DIR to $POST_DIR"
#  echo -e "\e[32m Process Completed Successfully \e[0m"
#else
#  echo -e "\e[31m $1 older than the version in $POST_DIR, not overwriting $POST_DIR$post_fname \e[0m"
#fi
