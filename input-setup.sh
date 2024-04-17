#!/bin/bash

app_name=$(basename "$0")
output_monitor=$1

print_usage() {
  echo "Usage:"
  echo "  $app_name OUTPUT_MONITOR"
}

if [[ -z "$output_monitor" ]]; then
  echo "Choose an audio output to monitor:"
  pactl list sources short | cut -d ' ' -f 1
  echo
  print_usage
  exit
fi

pactl load-module module-remap-source \
  source_name=transcribe-output-loopback master="$output_monitor" \
  source_properties=device.description="Transcribe-Output-Loopback" >/dev/null
if [[ $? -eq 0 ]]; then
  echo "Loopback device created successfully"
fi
