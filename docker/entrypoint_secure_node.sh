#!/bin/sh
set -e

tor -f /etc/tor/torrc &
sleep 2

if [ -f /var/lib/tor/aegis_storage/hostname ]; then
  cat /var/lib/tor/aegis_storage/hostname
fi

exec aegis-storage-node

