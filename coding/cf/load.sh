#!/bin/bash
url=$1

tmpfile=data_$url
tmpfile="${tmpfile//\//}"

declare -i tot=15
declare -i line_width=75

for i in {1..$tot}
do
    curl --proto '=https' -k --tlsv1.3 -sSf $url 2>&1 > $tmpfile
    if [ "$?" == "0" ]; then
        break
    fi
done

if [ "$?" != "0" ]; then
    echo "curl failed after retried $tot times"
    exit 1
fi

round=$(cat $tmpfile | grep 'href="/contest/' | perl -nle 'm/\/contest\/\d+\">(.*)<\/a/; print $1' | head -n 1)
title=$(cat $tmpfile | grep 'problem-statement' | grep '<div class="title">' \
    | perl -nle 'm/<div class="header"><div class="title">(.*)<\/div><div class="time-limit">/; print $1' | head -n 1)
description=$(cat $tmpfile | grep 'problem-statement' | grep '<div class="title">' \
    | perl -nle 'm/<div>(.*)<\/div><div class="input-specification">/; print $1' \
    | sed -E 's/<\/?span[^>]*>//g' \
    | sed -E 's/<i>([^<>]*)<\/i>/\1/g' \
    | sed -E 's/ *<[a-zA-Z]*>//g' \
    | sed -E 's/<sub class="lower-index">([^<>]*)<\/sub>/_\1/g' \
    | sed -E 's/&quot;/"/g' \
    | sed -E 's/<br \/>/\n/g' \
    | sed -E 's/<pre class="verbatim">//g' \
    | sed 's/\$\$\$//g' | sed -E 's/<\/[a-zA-Z]+>/\n\n/g' \
    | sed 's/&gt;/>/g' \
    | fold -s -w $line_width | sed -e :a -e '/^\n*$/{$d;N;ba' -e '}')
folder=$(echo $round | tr -d " " | grep -E '[[:upper:]|[:digit:]]' -o \
    | tr -d '\n' | tr '[:upper:]' '[:lower:]')
filename=$(echo $title | cut -d' ' -f1 | cut -d'.' -f1).cpp
echo "folder/filename: " `pwd`/$folder/$filename
mkdir -p $folder 
touch $folder/data.in

echo "#include<bits/stdc++.h>
using namespace std;
using ll=long long;
/*
$url

$round $title 

$description
*/" > $folder/$filename
echo 'void run(){
    // Welcome, your majesty.
}
int main(){
#ifdef WINE
    freopen("data.in","r",stdin);
#endif
    int T;scanf("%d",&T);
    while(T--){
        run();
    }
}' >> $folder/$filename

rm -rf $tmpfile
