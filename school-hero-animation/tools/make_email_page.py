"""Step 7: email-record.html = index.html in email mode with the email cover's logo/headline layer on top,
framed exactly like the student email's 600 x 380 cover (used only to record the email GIF)."""
ROOT = __file__.rsplit('/tools/', 1)[0]
s = open(f'{ROOT}/index.html').read()
rec = s.replace('<div class="stage">', '<div class="eframe"><div class="stage">', 1).replace(
    '</div>\n\n<button class="replay"', '</div><img class="eover" src="assets/email-overlay.png" alt=""></div>\n\n<button class="replay"', 1)
css = """
  body{ padding:0!important; display:block!important; min-height:0!important; }
  .eframe{ position:relative; width:600px; height:380px; overflow:hidden; background:#fff; }
  .eframe .stage{ position:absolute; left:-1px; top:-15.5px; width:602px; }
  .eover{ position:absolute; left:0; top:0; width:600px; height:380px; z-index:10; pointer-events:none; }
  .replay{ display:none!important; }
"""
rec = rec.replace('</style>', css + '</style>', 1)
assert 'eover' in rec and 'eframe' in rec
open(f'{ROOT}/email-record.html', 'w').write(rec)
print('ok')
