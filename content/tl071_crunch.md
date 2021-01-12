Title: TL071 Crunch
SubTitle: mhmmmmm crunchy
Date: 2017-02-07 08:45
Modified: 2020-07-25 15:30
Category: Electronics
Tags: Guitar, Pedals, Op Amp, Overdrive
Slug: tl071-crunch
Authors: Schlerp
Cover: /images/tl071/tl071_physical_banner.jpg
Thumbnail: /images/tl071/tl071_physical_banner.jpg


It's been a while since my last foray into making my own effects pedals. Most of my designs so far have been either very simple, or overly complicated at no extra bonus to tone. I decided to take it back to basics and build a single gain stage overdrive/booster pedal.

![tl071crunch](/images/tl071/tl071_schematic.jpg)

This isn't a new opamp, its been around the block for quite a while and you probably have an effects pedal or two with either this, or its big brother, the TL072 (dual channel TL071). They use a JFET for the input tranny, it apparently sounds better and tbh they are a rather nice sounding opamp so id be inclined to agree.

TL071's are marketed as low noise, JFET input, general purpose opamp. I chose to use a TL071 mainly because I wanted a single gain stage circuit and a had a few of them lying around. I'm glad I did!

I went with fairly standard values for input and output caps, using a 0.1uF for input and a 10uF for output. these were picked after looking at a few schematics on the net, I tested a few different values for each and decided that these suited my guitar and amp nicely. I've used a greencap for the input cap and a low voltage electrolytic cap for the output.

Originally I was just using the opamp to clip by driving the shit out of it at about 100x gain (a la RAT). This actually sounded surprisingly good, but was a bit fucked when it came to trying to tame the controls. I decided to up the gain range setting Resistor from 1k to 10k (I only have 100k pots atm, so 100x gain down to 10x gain). This sounded really nice as a boost pedal (those input/output cap values just sound so good!) but I wanted to get a bit more breakup from it.

I decided to add some diode clipping in. In the past I had mostly used a softer clipping on the feedback path of the opamp but this time I just wanted to dump these peaks to ground. Lately I have been mucking around a bit with different types of clipping using Ge/Si signal diodes, FET's and some power diodes. I have been really liking the asymmetrical sound of two 4148's on one side and a single 4001 on the other. These are all Si diodes, they sound great and I can contest they hold up against FET's and Ge diodes for clipping, they all have subtle differences, and maybe its just my amp and guitar, but I feel that Si just sound better in this application.

![tl071crunch](/images/tl071/tl071_physical.jpg)

This pedal is not very loud, I'm directly dumping output to ground when I clip the signal like this and it would give me a maximum swing of about `(0.6v+0.6v) + 0.5v = 1.7v`. Some of my other pedals have gain stages after the clipping to boost the output back up. Ill get the scope on it when I get a chance, I'm curious to see if I'm understanding this circuit properly.

Rather than using a voltage divider directly I could add another opamp to buffer the output of the voltage divider, this apparently gives a more stable power supply because any other resistance from the input doesn't change the value of the voltage divider. I haven't tried this before but I've heard it used to good results from people on various gear building forums around the net.

I want to add a switch in and put some sockets on the outside of the pedal to allow me to swap between other clipping setups, or even build one on the fly using the external sockets (yeah... that's making me pretty hard just thinking about it!).

Add some selector switches to swap between higher/lower values for input/output caps. I think this would be a very a fun addition to a pedal and make it much more versatile between different amp/guitar set ups.
