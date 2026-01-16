#include <tunables/global>

profile secure-node flags=(attach_disconnected,mediate_deleted) {
  network,
  capability,
  file,
  umount,
  deny mount,
  deny /proc/** w,
  deny /sys/** w,
  /entrypoint.sh rix,
  /usr/bin/tor rix,
  /usr/lib/** mr,
  /lib/** mr,
  /etc/tor/torrc r,
  /var/lib/tor/** rwk,
  /data/** rwk,
  /app/** r,
}

