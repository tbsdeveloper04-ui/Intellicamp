# Intellicamp launch email

- `index.html`: the email template, ready to paste into your sending tool (Resend, Mailchimp, Brevo and so on).
- Images in this folder are linked from the template via raw.githubusercontent.com. Keep this repo **public** and don't rename or move them, or the images in sent emails will break.

## Before sending
- Replace `{{company_address}}` and `{{unsubscribe_url}}` with your sending tool's merge tags
  (in Resend broadcasts the unsubscribe tag is `{{{RESEND_UNSUBSCRIBE_URL}}}`).
- Point the button links at your real sign-up and demo pages (they currently go to intellicamp.in with UTM tags).
- `workflow.png` is still the Recrutly mockup from the Figma file; replace it with an Intellicamp screenshot (same name, 564×464 px).
