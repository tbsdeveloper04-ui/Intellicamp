# Intellicamp feature orbit (hero animation)

Open `index.html` in a browser. Everything it needs is in `assets/`.

**What it does (12 s loop, one GSAP master timeline):**
`initial` → `swipeStart` (finger flick, light trail) → `orbitRotation` (all six cards turn clockwise on one circle, upright, centres always exactly on the drawn ring, making one full 360° turn and easing to a stop with Smart Study Material back beside the fingertip) → `orbitStop` (other cards dim) → `fingerTap` (ripple, 1 → 0.97 → 1.07 → 1 press, cyan selection glow, icon pulse) → `selectionComplete` → `finalHold` → `reset`. The fingertip glow appears only while swiping and tapping.

**Layers:** `background.jpg` (gradient, glow, logo, headline, button) · SVG orbit ring, nodes and travelling light · `student-body.png` · six cards (`card-*.png`, the original artwork from the high-resolution image, with `icon-*.png` on top for the icon pulse) · `student-hand.png` (wrist pivot for the gesture) · fingertip glow and ripple.

**Built in:** reduced motion shows the finished state with Smart Study Material selected; the timeline pauses when scrolled off-screen; the whole artboard (1198×1264) scales to its container. Set `CONFIG.loop` in the script to turn looping off.

**Source and limits:** the student comes from the original transparent artwork in the Figma file. Two areas were hidden behind cards in that artwork and were reconstructed: the forearm/elbow under *Performance Tracking* and the shirt edge under *Progress Analytics*. They're visible only while those cards are away. Cards are cut from the high-resolution artwork; orbit lines, his arm showing through the glass, and the touch rings on Smart Study Material were cleaned off.

**Email version:** `../email/hero-orbit.gif` is this animation rendered at 600 px, 10 fps (about 1.7 MB). Outlook on Windows shows only the first frame, which matches the static hero.

GSAP is © GreenSock, used under its standard licence. Inter is under the SIL Open Font License.
