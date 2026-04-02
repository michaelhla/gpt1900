"""Generate GPT-1900 banner image.

Draws the box frame manually with PIL rectangles to avoid Unicode width issues.
Only uses font rendering for the inner figlet text.
"""

from PIL import Image, ImageDraw, ImageFont

# The inner figlet text only (no box frame)
FIGLET = """\
 ██████╗ ██████╗ ████████╗    ███╗    █████╗  ██████╗  ██████╗
██╔════╝ ██╔══██╗╚══██╔══╝   ████║   ██╔══██╗██╔═████╗██╔═████╗
██║  ███╗██████╔╝   ██║      ╚═██║   ╚█████╔╝██║██╔██║██║██╔██║
██║   ██║██╔═══╝    ██║        ██║    ╚═══██╗████╔╝██║████╔╝██║
╚██████╔╝██║        ██║        ██║   █████╔╝ ╚██████╔╝╚██████╔╝
 ╚═════╝ ╚═╝        ╚═╝        ╚═╝   ╚════╝   ╚═════╝  ╚═════╝"""

BG = (30, 30, 30)
FG = (204, 204, 204)
BORDER = (204, 204, 204)
DOT = (100, 100, 100)
FONT_SIZE = 26

# Find monospace font
for font_path in [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.dfont",
    "/System/Library/Fonts/SFMono-Regular.otf",
]:
    try:
        font = ImageFont.truetype(font_path, FONT_SIZE)
        break
    except:
        continue
else:
    font = ImageFont.load_default()

# Measure figlet text
tmp = Image.new("RGB", (1, 1))
td = ImageDraw.Draw(tmp)
bbox = td.textbbox((0, 0), FIGLET, font=font)
text_w = bbox[2] - bbox[0]
text_h = bbox[3] - bbox[1]

# Layout
BORDER_W = 8       # border line thickness
DOT_BAND = 14      # dotted inner border band
TEXT_PAD_X = 30     # padding between dot band and text
TEXT_PAD_Y = 16
OUTER_PAD = 20      # padding outside the box

# Calculate box dimensions
box_inner_w = text_w + 2 * TEXT_PAD_X
box_inner_h = text_h + 2 * TEXT_PAD_Y
box_w = box_inner_w + 2 * DOT_BAND + 2 * BORDER_W
box_h = box_inner_h + 2 * DOT_BAND + 2 * BORDER_W

img_w = box_w + 2 * OUTER_PAD
img_h = box_h + 2 * OUTER_PAD

img = Image.new("RGB", (img_w, img_h), BG)
draw = ImageDraw.Draw(img)

bx = OUTER_PAD
by = OUTER_PAD

# Outer border rectangle (solid)
draw.rectangle([bx, by, bx + box_w - 1, by + box_h - 1], fill=BORDER)

# Inner area (dark background)
inner_x = bx + BORDER_W + DOT_BAND
inner_y = by + BORDER_W + DOT_BAND
inner_w = box_inner_w
inner_h = box_inner_h
draw.rectangle([inner_x, inner_y, inner_x + inner_w - 1, inner_y + inner_h - 1], fill=BG)

# Dot band: fill the band between border and inner area with dotted pattern
# Top band
for y in range(by + BORDER_W, by + BORDER_W + DOT_BAND):
    for x in range(bx + BORDER_W, bx + box_w - BORDER_W):
        if (x + y) % 3 == 0:
            img.putpixel((x, y), DOT)
        else:
            img.putpixel((x, y), BG)

# Bottom band
for y in range(by + box_h - BORDER_W - DOT_BAND, by + box_h - BORDER_W):
    for x in range(bx + BORDER_W, bx + box_w - BORDER_W):
        if (x + y) % 3 == 0:
            img.putpixel((x, y), DOT)
        else:
            img.putpixel((x, y), BG)

# Left band
for y in range(by + BORDER_W + DOT_BAND, by + box_h - BORDER_W - DOT_BAND):
    for x in range(bx + BORDER_W, bx + BORDER_W + DOT_BAND):
        if (x + y) % 3 == 0:
            img.putpixel((x, y), DOT)
        else:
            img.putpixel((x, y), BG)

# Right band
for y in range(by + BORDER_W + DOT_BAND, by + box_h - BORDER_W - DOT_BAND):
    for x in range(bx + box_w - BORDER_W - DOT_BAND, bx + box_w - BORDER_W):
        if (x + y) % 3 == 0:
            img.putpixel((x, y), DOT)
        else:
            img.putpixel((x, y), BG)

# Draw the figlet text centered in the inner area
text_x = inner_x + TEXT_PAD_X - bbox[0]
text_y = inner_y + TEXT_PAD_Y - bbox[1]
draw.text((text_x, text_y), FIGLET, fill=FG, font=font)

out_path = "figures/gpt1900_banner.png"
img.save(out_path)
print(f"Saved to {out_path} ({img_w}x{img_h})")
