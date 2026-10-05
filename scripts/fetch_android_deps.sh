#!/usr/bin/env bash
set -eu
cd ~/android
[ -x gradle-9.2.1/bin/gradle ] || { curl -sSL -o g.zip https://services.gradle.org/distributions/gradle-9.2.1-bin.zip && python3 -m zipfile -e g.zip . && chmod +x gradle-9.2.1/bin/gradle && rm g.zip; }
mkdir -p fonts && cd fonts
for f in IBMPlexMono-Regular IBMPlexMono-Medium IBMPlexMono-SemiBold; do
  [ -s $f.ttf ] || curl -sSL -o $f.ttf "https://github.com/google/fonts/raw/main/ofl/ibmplexmono/$f.ttf"; done
[ -s PixelifySans.ttf ] || curl -sSL -o PixelifySans.ttf "https://github.com/google/fonts/raw/main/ofl/pixelifysans/PixelifySans%5Bwght%5D.ttf"
[ -s NotoSansDevanagari.ttf ] || curl -sSL -o NotoSansDevanagari.ttf "https://github.com/google/fonts/raw/main/ofl/notosansdevanagari/NotoSansDevanagari%5Bwdth%2Cwght%5D.ttf"
file *.ttf; echo DEPS_DONE
