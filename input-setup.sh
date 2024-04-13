#!/bin/bash

pactl load-module module-remap-source \
  source_name=loopback-hdmi-output master=alsa_output.pci-0000_01_00.1.hdmi-stereo.monitor \
  source_properties=device.description="Loopback-HDMI-Output"
