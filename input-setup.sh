#!/bin/bash

app_name=$(basename "$0")
output_monitor=$1

print_usage() {
  echo "Usage:"
  echo "  $app_name OUTPUT_MONITOR"
}

if [[ -z "$output_monitor" ]]; then
  pactl list sources short | cut -d ' ' -f 1
  echo "Choose an audio output to monitor"
  echo
  print_usage
  exit
fi

pactl load-module module-remap-source \
  source_name=loopback-hdmi-output master="$output_monitor" \
  source_properties=device.description="Loopback-HDMI-Output"
