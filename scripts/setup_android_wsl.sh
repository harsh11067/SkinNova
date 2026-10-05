#!/usr/bin/env bash
# Linux Android toolchain in WSL home (C: is full). JDK 17 (Temurin) + cmdline-tools + platform/build-tools.
set -eu
A=~/android; mkdir -p $A && cd $A
if [ ! -x jdk/bin/java ]; then
  curl -sSL -o jdk.tgz "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
  mkdir -p jdk && tar xzf jdk.tgz -C jdk --strip-components=1 && rm jdk.tgz
fi
export JAVA_HOME=$A/jdk PATH=$A/jdk/bin:$PATH
if [ ! -x sdk/cmdline-tools/latest/bin/sdkmanager ]; then
  curl -sSL -o clt.zip https://dl.google.com/android/repository/commandlinetools-linux-13114758_latest.zip
  mkdir -p sdk/cmdline-tools && python3 -m zipfile -e clt.zip sdk/cmdline-tools/ && mv sdk/cmdline-tools/cmdline-tools sdk/cmdline-tools/latest && rm clt.zip
  chmod +x sdk/cmdline-tools/latest/bin/*
fi
export ANDROID_HOME=$A/sdk
yes | sdk/cmdline-tools/latest/bin/sdkmanager --licenses >/dev/null
sdk/cmdline-tools/latest/bin/sdkmanager "platform-tools" "platforms;android-36" "build-tools;36.0.0"
java -version 2>&1 | head -1; ls sdk
echo ANDROID_SETUP_DONE
