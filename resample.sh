find . -name "*.flac" -print | xargs -I {} sox {} -r16k -c1 {}.16k.flac