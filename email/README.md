# Intellicamp launch email

- `index.html`: the email template, ready to paste into your sending tool (Resend, Mailchimp, Brevo and so on).
- Images in this folder are linked from the template via raw.githubusercontent.com. Keep this repo **public** and don't rename or move them, or the images in sent emails will break.

## Before sending
- Replace `{{unsubscribe_url}}` with your sending tool's unsubscribe merge tag
  (in Resend broadcasts the unsubscribe tag is `{{{RESEND_UNSUBSCRIBE_URL}}}`).
- Buttons link to intellicamp.in/signup, /book and /how-it-works, with UTM tags for tracking.
- Layout, sizes and copy follow the Figma design. The four "Text here" placeholders and the step labels use wording from intellicamp.in.
- The site doesn't mention a 7-day trial; confirm that offer before sending.
