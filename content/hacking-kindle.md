Title: Hacking Kindle PW3
SubTitle: UART is love UART is life
Date: 2017-02-06 02:52
Modified: 2020-07-25 15:30
Category: Electronics
Tags: Root, Kindle, Hacking
Slug: hacking-kindle-pw3
Authors: Schlerp
Cover: /images/hackingkindle/close_up_pins_banner.jpg
Thumbnail: /images/hackingkindle/close_up_pins_banner.jpg


I love getting root access on devices and running things that were not designed for them and just the general troubleshooting involved with porting software to low power devices. The kindle is a prime example of exactly that! After about 6 months I got sick of waiting for the 5.6.5 Kindle Jail Break (JB) to be released so I chose to take the more invasive approach and open her up to bang her bits.

It's a fairly well known fact that you can hack a Kindle via the serial port on the board as they are still exposed in the final products (much to my/the communities delight). I used the guide/various posts on the mobile read forum to formulate my exact plan. I initially had a PW2 however I managed to delaminate one of the pads which rended that UART port completely useless. I was having trouble soldering with such small pads (and a much shittier soldering iron than I now have).

Find the Mobileread forums [here](http://www.mobileread.com/forums/forumdisplay.php?f=150).

Basically the process is to:

1. Copy the Jail break by NiLuJe (JailBreak v1.14.N) to the kindle data partition
2. Open Kindle case and locate the serial ports
3. Atttach the USB UART (USB to Serial) adapter to the serial ports
![USB UART](/images/hackingkindle/usbuart.jpg)
I did this by getting an eraser and cutting it up to use the blocks of rubber as pseudo clamps to hold the pins for me. The reason being rubber is firm and non conductive.
Here you can see the pins maming contact:
![Close Up Pins](/images/hackingkindle/close_up_pins.jpg)
This shows the whole set up:
![Far Pins](/images/hackingkindle/far_pins.jpg)
You can get earth from the metal shielding:
![Earth Pin](/images/hackingkindle/earth_pin.jpg)
4. Boot the kindle into recovery mode.
    1. Reboot the kindle form the menu
    2. Spam keys on the keyboard connect via serial
    3. This should pause the boot loader at the uboot menu, if not reboot and spam again until you get it.
    4. Enter this into the uboot menu:

            bootm 0xE41000

    5. Choose Exit or Quit, then when prompted drop the shell rather than rebooting.

5. Using the python code below calculate your factory root passwd.  

        #!/usr/bin/env python
        import hashlib
        serial_md5_utf8 = "**YOURSERIAL NUMBER**\n".encode('utf-8')
        serial_md5 = hashlib.md5(serial_md5_utf8)
        hex_digest = serial_md5.hexdigest()[7:11]
        print("fiona{}".format(hex_digest))

    You need the root passwd discovered in previous step in order to mount the partition and delete the offending entry in the `/etc/passwd` file.

6. From recovery mode, mount the system as read/write.  

        #!/bin/sh
        mount /dev/mmcblk0p1 /mnt/mmc

7. Remove the `x` in the password field of the `/mnt/mmc/etc/passwd` file. This will mean there is a blank pass word for the root account in the real system.
8. Reboot the Kindle to normal user mode while still attached to the serial port (eg. reboot and dont interupt boot process this time)
9. Run the Jail Break Script you copied on initially as root from the serial terminal. (it can be found at `/mnt/us/where_you_put_it`)

As you can see its actually not that hard once you have the concepts down. As someone who fucks with a lot of ARM devices I find I am often using serial interfaces (lots of the time they are root too!) to execute commands on devices I don't have proper root access to.

> UART is love UART is life.
